"""Learned instance selection on a threshold ladder (replaces the fixed hysteresis keep rule).

1. Candidates: components of p ≥ t (closing + hole filling) for every t in LEVELS. The sets are nested — a component
   at level t lies inside exactly one component at the next lower level — so the candidates of an image form a forest.
2. Each candidate gets descriptors (level, size, probability statistics, shape, limb distance, image contrast, parent /
   children) and a target y = mean over the image's annotator records of IoU(best GT) if that IoU > 0.5, else 0:
   the expected numerator a candidate adds to PQ. Selecting any candidate adds 0.5 to the denominator (TP replaces an
   FN, or one FP is added), so a candidate pays iff y > PQ/2: one global threshold τ.
3. A gradient-boosted regressor predicts y out of fold (date-grouped folds); an exact tree DP picks, per image, the
   antichain (no candidate together with its ancestor) maximising Σ (ŷ − τ). τ is swept on OOF PQ and compared with
   the hysteresis rule on the same images.

Env: DATA_DIR, WORK_DIR (work/scorer), PROB_DIR (comma list of globs, one model directory each; averaged at 2048),
     WORKERS (8), LIMIT (0 = all; first N images per fold for a smoke run), BASE ("0.4,3,0.99,0.85,200": hysteresis
     t_lo, close, t_hi, min_mean, min_area for the baseline).
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
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from filament.data import FilamentData, load_gray  # noqa: E402
from filament.instances import components, encode_components, upsample  # noqa: E402
from filament.metric import pq_score  # noqa: E402
from filament.unet import preprocess  # noqa: E402

env = os.environ.get
LEVELS = (0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8)
CLOSE, MIN_AREA, WORKERS = 3, 100, int(env("WORKERS", 8))
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)  # noqa: E731
data = FilamentData(env("DATA_DIR", "data"))
WORK = Path(env("WORK_DIR", "work/scorer"))
WORK.mkdir(parents=True, exist_ok=True)


def candidates(args):
    """One image → (descriptor table, RLEs) of every candidate on the ladder, with targets if records are given."""
    stem, path, record_ids, prob = args
    prob = upsample(np.asarray(prob, np.float32))
    gray = load_gray(path)
    img = preprocess(gray, prob.shape[0])
    ys, xs = np.nonzero(gray > 0.1 * gray.max())
    xc, yc, R = xs.mean(), ys.mean(), np.sqrt(len(xs) / np.pi)
    rows, rles, prev = [], [], None  # prev: (labels, index of label id → row) of the level below
    for li, t in enumerate(LEVELS):
        labels, ids, stats, boxes = components(prob, t, CLOSE, MIN_AREA)
        index = {}
        for i, (x, y, w, h), (area, peak, mean) in zip(ids, boxes, stats):
            pad = 8
            x0, y0, x1, y1 = max(x - pad, 0), max(y - pad, 0), min(x + w + pad, prob.shape[1]), min(y + h + pad, prob.shape[0])
            sub = labels[y0:y1, x0:x1] == i
            ring = cv2.dilate(sub.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~sub
            p = prob[y0:y1, x0:x1][sub]
            m = cv2.moments(sub.astype(np.uint8), True)
            a, b, c = m["mu20"], m["mu11"], m["mu02"]
            l1, l2 = np.sort(np.linalg.eigvalsh([[a, b], [b, c]]))[::-1]
            perim = cv2.arcLength(max(cv2.findContours(sub.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_NONE)[0], key=len), True)
            cx, cy = x0 + m["m10"] / m["m00"], y0 + m["m01"] / m["m00"]
            parent = -1
            if prev is not None:
                py, px = np.unravel_index(np.argmax(np.where(sub, prob[y0:y1, x0:x1], -1)), sub.shape)
                parent = prev[1][prev[0][y0 + py, x0 + px]]
            index[i] = len(rows)
            rows.append(dict(level=t, area=area, peak=peak, mean=mean, p10=np.percentile(p, 10),
                             p50=np.median(p), p90=np.percentile(p, 90), elong=np.sqrt(l1 / max(l2, 1e-6)),
                             width=2 * area / max(perim, 1), length=perim / 2, bw=w, bh=h,
                             dr=np.hypot(cx - xc, cy - yc) / R, dx=abs(cx - xc) / R,
                             img_in=float(img[y0:y1, x0:x1][sub].mean()),
                             contrast=float(img[y0:y1, x0:x1][ring].mean() - img[y0:y1, x0:x1][sub].mean()),
                             p_ring=float(prob[y0:y1, x0:x1][ring].mean()), parent=parent))
        rles += encode_components(labels, ids, boxes)
        prev = (labels, np.vectorize(lambda k: index.get(k, -1), otypes=[int])(np.arange(labels.max() + 1)))
    df = pd.DataFrame(rows)
    df["stem"] = stem
    if len(df):
        df["n_children"] = df.index.map(df.parent.value_counts()).fillna(0).to_numpy()
        par = df.parent.to_numpy()
        df["parent_area_ratio"] = np.where(par >= 0, df.area / df.area.reindex(par).to_numpy(), 1.0)
        df["parent_mean"] = np.where(par >= 0, df["mean"].reindex(par).to_numpy(), 0.0)
        df["n_level"] = df.groupby("level")["level"].transform("size")
        y = np.zeros(len(df))
        for rid in record_ids:
            gt = list(data.record_rles(rid))
            if gt:
                best = np.asarray(mask_utils.iou(rles, gt, [0] * len(gt))).max(1)
                y += np.where(best > 0.5, best, 0)
        df["y"] = y / max(len(record_ids), 1)
    return df, rles


def select(df, score, tau):
    """Exact DP on the candidate forest: the antichain maximising Σ (score − τ). → selected row positions."""
    v = score - tau
    children = [[] for _ in range(len(df))]
    for k, p in enumerate(df.parent.to_numpy()):
        if p >= 0:
            children[p].append(k)
    best, take = np.zeros(len(df)), np.zeros(len(df), bool)
    for k in range(len(df) - 1, -1, -1):  # rows are ordered by level, so children come after their parent
        below = sum(best[c] for c in children[k])
        take[k] = v[k] > max(below, 0)
        best[k] = max(v[k], below, 0)
    out, stack = [], [k for k in range(len(df)) if df.parent.iat[k] < 0]
    while stack:
        k = stack.pop()
        if take[k]:
            out.append(k)
        else:
            stack += children[k]
    return out


FEATURES = ["level", "area", "peak", "mean", "p10", "p50", "p90", "elong", "width", "length", "bw", "bh", "dr", "dx",
            "img_in", "contrast", "p_ring", "n_children", "parent_area_ratio", "parent_mean", "n_level"]


def load_probs(pattern):
    dirs = [Path(sorted(glob.glob(p, recursive=True))[0]) for p in env("PROB_DIR", "outputs/kaggle/b4-native5").split(",")]
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
    return {s: np.mean([upsample(np.asarray(m.pop(s), np.float32)) for m in models], 0).astype(np.float16) for s in stems}


if __name__ == "__main__":
    from sklearn.ensemble import HistGradientBoostingRegressor
    oof = load_probs("oof_fold*.pkl")
    phys = data.physical[data.physical.stem.isin(oof)]
    if int(env("LIMIT", 0)):
        phys = phys.groupby("fold").head(int(env("LIMIT"))).reset_index(drop=True)
    records = [r for rs in phys.record_ids for r in rs]
    cache = WORK / f"cands_{len(phys)}.pkl"
    if cache.exists():
        cand, rles = pickle.load(open(cache, "rb"))
    else:
        with ProcessPoolExecutor(WORKERS) as pool:
            jobs = ((s, data.train_path(f), list(r), oof.pop(s)) for s, f, r in zip(phys.stem, phys.file_name, phys.record_ids))
            out = list(pool.map(candidates, jobs, chunksize=2))
        cand = {s: df for s, (df, _) in zip(phys.stem, out)}
        rles = {s: r for s, (_, r) in zip(phys.stem, out)}
        pickle.dump((cand, rles), open(cache, "wb"))
    del oof
    allc = pd.concat([d.assign(row=np.arange(len(d))) for d in cand.values()], ignore_index=True)
    allc["fold"] = allc.stem.map(dict(zip(phys.stem, phys.fold)))
    log(f"{len(phys)} images, {len(records)} records, {len(allc)} candidates; y>0 for {(allc.y > 0).mean():.1%}")

    allc["s"] = 0.0
    for k in sorted(allc.fold.unique()):
        tr, te = allc.fold != k, allc.fold == k
        reg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                                            l2_regularization=1.0, random_state=k)
        reg.fit(allc.loc[tr, FEATURES], allc.loc[tr, "y"])
        allc.loc[te, "s"] = reg.predict(allc.loc[te, FEATURES])
    corr = np.corrcoef(allc.s, allc.y)[0, 1]
    log(f"OOF score vs target: r = {corr:.3f}")

    # Baseline: the hysteresis rule on the same images (its t_lo must be one of LEVELS, same closing).
    t_lo, close, t_hi, mn, area = map(float, env("BASE", "0.4,3,0.99,0.85,200").split(","))
    assert t_lo in LEVELS and close == CLOSE
    base = {s: [rles[s][k] for k in d.index[(d.level == t_lo) & (d.peak >= t_hi) & (d["mean"] >= mn) & (d.area >= area)]]
            if len(d) else [] for s, d in cand.items()}
    pq, st = pq_score(base, records, data.record_rles)
    rows = [dict(method="hysteresis", tau=np.nan, PQ=pq, **st)]
    log(f"hysteresis baseline: PQ {pq:.4f} {st}")
    by_stem = {s: g.s.to_numpy() for s, g in allc.groupby("stem")}
    for tau in (0.12, 0.15, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.33):
        preds = {s: [rles[s][k] for k in select(d, by_stem[s], tau)] if len(d) else [] for s, d in cand.items()}
        pq, st = pq_score(preds, records, data.record_rles)
        rows.append(dict(method="scorer", tau=tau, PQ=pq, **st))
        log(f"scorer tau {tau}: PQ {pq:.4f} {st}")
    table = pd.DataFrame(rows).sort_values("PQ", ascending=False)
    table.to_csv(WORK / "scorer.csv", index=False)
    allc.drop(columns=["row"]).to_csv(WORK / "candidates.csv", index=False)
    print(table.to_string(index=False), flush=True)
