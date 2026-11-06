"""Stage-2 refiner: re-draw each selected instance's mask from a crop around it (filseg v4 reported +0.006 OOF).

1. build (CPU): instances = the hysteresis rule (RULE) on the stage-1 probability (PROB_DIR, averaged if several).
   For each instance, a square window (bbox + PAD, at least MIN_SIDE) is resized to R² with three channels:
   disk-normalised image, stage-1 probability, the instance's own seed mask. The target is the mean of the GT masks of
   the annotator records whose best match has IoU > 0.5 (instances that no record matches are kept, flagged
   unmatched; TRAIN_UNMATCHED=1 trains them towards an empty mask).
2. train (GPU): per fold k, a U-Net (REF_ENCODER) on instances of the other folds' images (their stage-1 maps are
   out of fold too), D4 augmentation, BCE + 0.5 Dice; then predicts fold k's instances (D4 TTA).
3. score: OOF PQ of the refined masks vs the seeds (same instances, same images), for a few output thresholds.
4. submit: test instances from test_prob, the mean of the fold refiners, → WORK_DIR/submission.csv.
Refined masks are pasted back at 2048²; pixels already claimed by an earlier (higher mean p) instance are removed,
so instances never overlap. Instances whose refined mask is empty or below min_area are dropped.

Env: MODE (build,train,score,submit — comma list), DATA_DIR, WORK_DIR (work/refiner), PROB_DIR (outputs/kaggle/
     convnext-native5), RULE ("0.4,3,0.99,0.85,200"), R (256), PAD (32), MIN_SIDE (128), WORKERS (8),
     REF_ENCODER (resnet34), STEPS (4000), BATCH (32), LR (3e-4), TRAIN_UNMATCHED (0), FOLDS (0,1,2,3,4),
     VRAM_FRACTION (0.85), THRESH (output threshold for submit, 0.5).
"""
import glob
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from filament import H, W  # noqa: E402
from filament.data import FilamentData, load_gray  # noqa: E402
from filament.instances import InstanceParams, components, keep_mask, upsample  # noqa: E402
from filament.metric import pq_score  # noqa: E402
from filament.unet import preprocess  # noqa: E402

env = os.environ.get
R, PAD, MIN_SIDE, WORKERS = int(env("R", 256)), int(env("PAD", 32)), int(env("MIN_SIDE", 128)), int(env("WORKERS", 8))
t_lo, close, t_hi, mn, area = map(float, env("RULE", "0.4,3,0.99,0.85,200").split(","))
RULE = InstanceParams(t_lo=t_lo, t_hi=t_hi, close_px=int(close), min_area=int(area), min_mean=mn)
WORK = Path(env("WORK_DIR", "work/refiner"))
WORK.mkdir(parents=True, exist_ok=True)
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)  # noqa: E731
data = FilamentData(env("DATA_DIR", "data"))


def load_probs(pattern):
    dirs = [Path(sorted(glob.glob(p))[0]) for p in env("PROB_DIR", "outputs/kaggle/convnext-native5").split(",")]
    models = []
    for d in dirs:
        probs = {}
        for f in sorted(d.rglob(pattern)):
            probs.update(pickle.load(open(f, "rb")))
        models.append(probs)
    stems = sorted(set.intersection(*(set(m) for m in models)))
    log(f"{pattern}: {len(stems)} images from {[str(d) for d in dirs]}")
    if len(models) == 1:
        return {s: models[0][s] for s in stems}
    return {s: np.mean([upsample(np.asarray(m.pop(s), np.float32)) for m in models], 0).astype(np.float16)
            for s in stems}


def window(box):
    """Instance bbox (x, y, w, h) → square window (x0, y0, side) centred on it."""
    x, y, w, h = box
    side = max(int(max(w, h)) + 2 * PAD, MIN_SIDE)
    return int(x + w / 2 - side / 2), int(y + h / 2 - side / 2), side


def cut(arr, x0, y0, side, interp=cv2.INTER_AREA):
    """Window of a 2048² array (zero outside the image) resized to R²."""
    out = np.zeros((side, side), arr.dtype)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + side, W), min(y0 + side, H)
    out[ya - y0:yb - y0, xa - x0:xb - x0] = arr[ya:yb, xa:xb]
    return cv2.resize(out, (R, R), interpolation=interp)


def instances(args):
    """One image → crops (n, 3, R, R) float16, targets (n, R, R) uint8 (×255), meta rows, seed RLEs."""
    stem, path, record_ids, prob = args
    prob = upsample(np.asarray(prob, np.float32))
    img = preprocess(load_gray(path), H)
    labels, ids, stats, boxes = components(prob, RULE.t_lo, RULE.close_px, RULE.min_area, RULE.t_hi)
    keep = keep_mask(stats, RULE)
    ids, stats, boxes = ids[keep], stats[keep], boxes[keep]
    order = np.argsort(-stats[:, 2]) if len(ids) else np.zeros(0, int)  # higher mean p claims pixels first
    gts = [list(data.record_rles(r)) for r in record_ids] if record_ids is not None else []
    xs, ys, meta, seeds = [], [], [], []
    canvas = np.zeros((H, W), np.uint8, order="F")
    for j in order:
        i, box = ids[j], boxes[j]
        x, y, w, h = box
        canvas[y:y + h, x:x + w] = labels[y:y + h, x:x + w] == i
        seed_rle = mask_utils.encode(canvas)
        x0, y0, side = window(box)
        seed = canvas.astype(np.float32)
        xs.append(np.stack([cut(img, x0, y0, side), cut(prob, x0, y0, side), cut(seed, x0, y0, side)]))
        canvas[y:y + h, x:x + w] = 0
        tgt, n_match = np.zeros((R, R), np.float32), 0
        for g in gts:
            if not g:
                continue
            iou = np.asarray(mask_utils.iou([seed_rle], g, [0] * len(g)))[0]
            if iou.max() > 0.5:
                tgt += cut(mask_utils.decode(g[int(iou.argmax())]).astype(np.float32), x0, y0, side)
                n_match += 1
        ys.append(np.round(255 * tgt / max(n_match, 1)).astype(np.uint8))
        seeds.append(seed_rle)
        meta.append(dict(stem=stem, x0=x0, y0=y0, side=side, mean=stats[j, 2], n_match=n_match,
                         n_records=len(gts)))
    if not xs:
        return np.zeros((0, 3, R, R), np.float16), np.zeros((0, R, R), np.uint8), [], []
    return np.stack(xs).astype(np.float16), np.stack(ys), meta, seeds


def build(split):
    cache = WORK / f"crops_{split}.pkl"
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    if split == "oof":
        probs = load_probs("oof_fold*.pkl")
        pickle.dump(set(probs), open(WORK / "oof_stems.pkl", "wb"))  # images without any instance still count in PQ
        phys = data.physical[data.physical.stem.isin(probs)]
        jobs = ((s, data.train_path(f), list(r), probs.pop(s))
                for s, f, r in zip(phys.stem, phys.file_name, phys.record_ids))
    else:
        probs = load_probs("test_prob.pkl")
        files = {p.stem: p for p in data.test_files}
        jobs = ((s, files[s], None, probs.pop(s)) for s in list(probs))
    xs, ys, meta, seeds = [], [], [], []
    with ProcessPoolExecutor(WORKERS) as pool:
        for x, y, m, sd in pool.map(instances, jobs, chunksize=2):
            xs.append(x), ys.append(y), meta.extend(m), seeds.extend(sd)
    meta = pd.DataFrame(meta)
    if split == "oof":
        meta["fold"] = meta.stem.map(dict(zip(data.physical.stem, data.physical.fold)))
    out = (np.concatenate(xs), np.concatenate(ys), meta, seeds)
    pickle.dump(out, open(cache, "wb"), protocol=4)
    log(f"{split}: {len(meta)} instances over {meta.stem.nunique() if len(meta) else 0} images; "
        f"matched {(meta.n_match > 0).mean() if len(meta) else 0:.1%}")
    return out


def predict(models, x, device, batch=64):
    """Mean sigmoid over models and the 8 D4 transforms → (n, R, R) float16."""
    import torch
    from filament.unet import d4, d4_inverse
    out = np.zeros((len(x), R, R), np.float16)
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        for a in range(0, len(x), batch):
            xb = torch.from_numpy(x[a:a + batch].astype(np.float32)).to(device)
            acc = torch.zeros(xb.shape[0], 1, R, R, device=device)
            for m in models:
                for k in range(8):
                    acc += d4_inverse(torch.sigmoid(m(d4(xb, k)).float()), k)
            out[a:a + batch] = (acc / (8 * len(models)))[:, 0].cpu().numpy()
    return out


def train_fold(x, y, w, device, seed):
    import segmentation_models_pytorch as smp
    import torch
    from filament.unet import d4, loss_fn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = smp.Unet(env("REF_ENCODER", "resnet34"), encoder_weights="imagenet", in_channels=3, classes=1).to(device)
    steps, batch, lr = int(env("STEPS", 4000)), int(env("BATCH", 32)), float(env("LR", 3e-4))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler()
    model.train()
    for step in range(steps):
        idx = np.sort(rng.choice(len(x), batch, p=w))
        xb = torch.from_numpy(x[idx].astype(np.float32)).to(device, non_blocking=True)
        yb = torch.from_numpy(y[idx]).to(device, non_blocking=True).float().div_(255).unsqueeze(1)
        k = int(rng.integers(8))
        xb, yb = d4(xb, k), d4(yb, k)
        with torch.autocast("cuda", torch.float16):
            loss = loss_fn(model(xb).float(), yb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 500 == 0 or step == steps - 1:
            log(f"  step {step}/{steps} loss {loss.item():.4f}")
    return model.eval()


def train(oof):
    import torch
    x, y, meta, _ = oof
    device = "cuda"
    torch.cuda.set_per_process_memory_fraction(float(env("VRAM_FRACTION", 0.85)))
    torch.backends.cudnn.benchmark = True
    use = (meta.n_match > 0).to_numpy() | (env("TRAIN_UNMATCHED", "0") == "1")
    for k in [int(f) for f in env("FOLDS", "0,1,2,3,4").split(",")]:
        weights, pred_path = WORK / f"ref_fold{k}.pt", WORK / f"ref_oof{k}.npy"
        if pred_path.exists():
            continue
        tr = (meta.fold != k).to_numpy() & use
        log(f"fold {k}: training on {tr.sum()} instances")
        w = tr / tr.sum()
        model = train_fold(x, y, w, device, seed=k)
        torch.save(model.state_dict(), weights)
        te = np.flatnonzero((meta.fold == k).to_numpy())
        np.save(pred_path, predict([model], x[te], device))
        log(f"fold {k}: predicted {len(te)} hold-out instances")
        del model
        torch.cuda.empty_cache()


def paste(meta, probs, thresh):
    """Refined R² probabilities → per-stem RLE lists at 2048², earlier instances (higher mean p) claim pixels."""
    out = {}
    for stem, g in meta.groupby("stem", sort=False):
        claimed = np.zeros((H, W), bool)
        rles = []
        for r in g.itertuples():
            m = cv2.resize(probs[r.Index].astype(np.float32), (r.side, r.side), interpolation=cv2.INTER_LINEAR) >= thresh
            full = np.zeros((H, W), bool)
            xa, ya, xb, yb = max(r.x0, 0), max(r.y0, 0), min(r.x0 + r.side, W), min(r.y0 + r.side, H)
            full[ya:yb, xa:xb] = m[ya - r.y0:yb - r.y0, xa - r.x0:xb - r.x0]
            full &= ~claimed
            if full.sum() < RULE.min_area:
                continue
            claimed |= full
            rles.append(mask_utils.encode(np.asfortranarray(full.astype(np.uint8))))
        out[stem] = rles
    return out


def score(oof):
    x, y, meta, seeds = oof
    probs = np.zeros((len(meta), R, R), np.float16)
    for k in sorted(meta.fold.unique()):
        te = np.flatnonzero((meta.fold == k).to_numpy())
        probs[te] = np.load(WORK / f"ref_oof{k}.npy")
    phys = data.physical[data.physical.stem.isin(pickle.load(open(WORK / "oof_stems.pkl", "rb")))]
    records = [r for rs in phys.record_ids for r in rs]
    base = {s: [] for s in phys.stem}
    for s, rle in zip(meta.stem, seeds):
        base[s].append(rle)
    pq, st = pq_score(base, records, data.record_rles)
    rows = [dict(method="seeds", thresh=np.nan, PQ=pq, **st)]
    log(f"seeds: PQ {pq:.4f} {st}")
    for t in (0.4, 0.45, 0.5, 0.55, 0.6):
        preds = {s: [] for s in phys.stem}
        preds.update(paste(meta, probs, t))
        pq, st = pq_score(preds, records, data.record_rles)
        rows.append(dict(method="refined", thresh=t, PQ=pq, **st))
        log(f"refined @ {t}: PQ {pq:.4f} {st}")
    table = pd.DataFrame(rows).sort_values("PQ", ascending=False)
    table.to_csv(WORK / "refiner.csv", index=False)
    print(table.to_string(index=False), flush=True)


def submit():
    import segmentation_models_pytorch as smp
    import torch
    x, _, meta, _ = build("test")
    models = []
    for f in sorted(WORK.glob("ref_fold*.pt")):
        m = smp.Unet(env("REF_ENCODER", "resnet34"), encoder_weights=None, in_channels=3, classes=1)
        m.load_state_dict(torch.load(f, map_location="cpu"))
        models.append(m.cuda().eval())
    log(f"test: {len(meta)} instances, {len(models)} refiners")
    probs = predict(models, x, "cuda")
    preds = paste(meta, probs, float(env("THRESH", 0.5)))
    rows = [{"filament_id": f"{s}_{k}", "segmentation_rle": r["counts"].decode("ascii")}
            for s, rles in preds.items() for k, r in enumerate(rles, 1)]
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(WORK / "submission.csv", index=False)
    log(f"submission.csv: {len(rows)} filaments over {len(preds)} images")


if __name__ == "__main__":
    modes = env("MODE", "build,train,score").split(",")
    if "build" in modes or "train" in modes or "score" in modes:
        oof = build("oof")
    if "train" in modes:
        train(oof)
    if "score" in modes:
        score(oof)
    if "submit" in modes:
        submit()
