"""Out-of-fold evaluation for the report: PQ, matched IoU / Dice distributions, splits and merges.

Methods compared on the same 707 OOF images (stage-1 = ConvNeXt-Base native 5-fold, convnext-native5):
  seeds    — hysteresis instances (RULE)
  refined  — the same instances redrawn by the stage-2 refiner (work/refiner/ref_oof*.npy, threshold THRESH)

Per annotator record (the metric's unit):
  * TP pairs (IoU > 0.5, one-to-one): IoU and Dice = 2·IoU / (1 + IoU);
  * split (one-to-many): a GT instance of which ≥ 2 predictions each lie mostly inside (|P∩G| ≥ 0.5·|P|);
  * merge (many-to-one): a prediction that covers most of ≥ 2 GT instances (|P∩G| ≥ 0.5·|G|).
Uncertainty: PQ per date-grouped fold, and a bootstrap over images (B = 2000) for the PQ 95% interval and the
paired refiner gain.
Outputs: report/stats.csv (summary), report/uncertainty.csv, report/figures/{iou_dice.pdf, qualitative.pdf}.
Env: DATA_DIR, THRESH (0.45), WORKERS (8).
"""
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import refiner  # noqa: E402  (WORK_DIR default work/refiner, RULE = ConvNeXt best)

OUT = ROOT / "report"
(OUT / "figures").mkdir(parents=True, exist_ok=True)
data = refiner.data


def record_stats(args):
    """(method, record id, predicted RLEs) → per-record counts and the IoUs of its TP pairs."""
    method, rid, preds = args
    gts = list(data.record_rles(rid))
    row = dict(method=method, record=rid, n_pred=len(preds), n_gt=len(gts), tp=0, iou_sum=0.0, split=0, merge=0)
    ious = []
    if preds and gts:
        iou = np.asarray(mask_utils.iou(preds, gts, [0] * len(gts)))
        pa = mask_utils.area(preds).astype(float)
        ga = mask_utils.area(gts).astype(float)
        inter = iou * (pa[:, None] + ga[None, :]) / (1 + iou)  # |P∩G| from IoU and the two areas
        tp = iou > 0.5
        row["tp"] = int(tp.sum())
        ious = iou[tp].tolist()
        row["iou_sum"] = float(iou[tp].sum())
        row["split"] = int(((inter >= 0.5 * pa[:, None]).sum(0) >= 2).sum())
        row["merge"] = int(((inter >= 0.5 * ga[None, :]).sum(1) >= 2).sum())
    row["fp"], row["fn"] = row["n_pred"] - row["tp"], row["n_gt"] - row["tp"]
    return row, ious


def main():
    x, _, meta, seeds = pickle.load(open(refiner.WORK / "crops_oof.pkl", "rb"))
    stems = pickle.load(open(refiner.WORK / "oof_stems.pkl", "rb"))
    phys = data.physical[data.physical.stem.isin(stems)]
    probs = np.zeros((len(meta), refiner.R, refiner.R), np.float16)
    for k in sorted(meta.fold.unique()):
        probs[np.flatnonzero((meta.fold == k).to_numpy())] = np.load(refiner.WORK / f"ref_oof{k}.npy")
    methods = {"seeds": {s: [] for s in phys.stem}, "refined": {s: [] for s in phys.stem}}
    for s, r in zip(meta.stem, seeds):
        methods["seeds"][s].append(r)
    methods["refined"].update(refiner.paste(meta, probs, float(os.environ.get("THRESH", 0.45))))
    if os.environ.get("QUAL_ONLY") == "1":
        return qualitative(phys, methods["refined"])
    jobs = [(m, rid, preds[s]) for m, preds in methods.items() for s, rids in zip(phys.stem, phys.record_ids)
            for rid in rids]
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", 8))) as pool:
        out = list(pool.map(record_stats, jobs, chunksize=16))
    rec = pd.DataFrame([r for r, _ in out])
    ious = {m: [] for m in methods}
    for (r, i) in out:
        ious[r["method"]] += i
    rows = []
    for m, g in rec.groupby("method", sort=False):
        tp, fp, fn = g.tp.sum(), g.fp.sum(), g.fn.sum()
        iu = np.asarray(ious[m])
        dice = 2 * iu / (1 + iu)
        rows.append(dict(method=m, PQ=iu.sum() / (tp + 0.5 * fp + 0.5 * fn), SQ=iu.mean(), RQ=tp / (tp + 0.5 * fp + 0.5 * fn),
                         TP=tp, FP=fp, FN=fn, iou_median=np.median(iu), dice_mean=dice.mean(), dice_median=np.median(dice),
                         splits=g.split.sum(), merges=g["merge"].sum(), gt_instances=g.n_gt.sum(),
                         pred_per_record=g.n_pred.mean(), gt_per_record=g.n_gt.mean()))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "stats.csv", index=False)
    rec.to_csv(OUT / "records.csv", index=False)
    print(table.T.to_string(), flush=True)
    uncertainty(rec, phys)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
    bins = np.linspace(0.5, 1, 26)
    for m, c in (("seeds", "#9aa0a6"), ("refined", "#1a73e8")):
        iu = np.asarray(ious[m])
        ax[0].hist(iu, bins, histtype="step", lw=1.5, color=c, label=f"{m} (median {np.median(iu):.3f})")
        ax[1].hist(2 * iu / (1 + iu), np.linspace(0.66, 1, 26), histtype="step", lw=1.5, color=c, label=m)
    ax[0].set(xlabel="IoU of matched pairs", ylabel="pairs")
    ax[1].set(xlabel="Dice of matched pairs")
    ax[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "iou_dice.pdf")
    qualitative(phys, methods["refined"])


def uncertainty(rec, phys, B=2000):
    """PQ per fold and image-bootstrap 95% intervals (images resampled; all records of an image move together)."""
    stem_of = {rid: s for s, rids in zip(phys.stem, phys.record_ids) for rid in rids}
    rec = rec.assign(stem=rec.record.map(stem_of), fold=rec.record.map(stem_of).map(dict(zip(phys.stem, phys.fold))))
    per_img = rec.groupby(["method", "stem"])[["iou_sum", "tp", "fp", "fn"]].sum()
    stems = phys.stem.to_numpy()
    rng = np.random.default_rng(0)
    draws = rng.integers(0, len(stems), (B, len(stems)))
    pq = lambda d: d.iou_sum / (d.tp + 0.5 * d.fp + 0.5 * d.fn)  # noqa: E731
    boot, rows = {}, []
    for m in ("seeds", "refined"):
        a = per_img.loc[m].reindex(stems, fill_value=0).to_numpy()  # columns iou_sum, tp, fp, fn
        tot = np.stack([a[d].sum(0) for d in draws])
        boot[m] = tot[:, 0] / (tot[:, 1] + 0.5 * tot[:, 2] + 0.5 * tot[:, 3])
        folds = rec[rec.method == m].groupby("fold")[["iou_sum", "tp", "fp", "fn"]].sum()
        fq = pq(folds)
        rows.append(dict(method=m, PQ=pq(pd.Series(a.sum(0), ["iou_sum", "tp", "fp", "fn"])),
                         ci_lo=np.percentile(boot[m], 2.5), ci_hi=np.percentile(boot[m], 97.5),
                         fold_mean=fq.mean(), fold_std=fq.std(ddof=1), **{f"fold{k}": v for k, v in fq.items()}))
    gain = boot["refined"] - boot["seeds"]
    rows.append(dict(method="gain (refined - seeds)", PQ=rows[1]["PQ"] - rows[0]["PQ"], ci_lo=np.percentile(gain, 2.5),
                     ci_hi=np.percentile(gain, 97.5), p_le0=(gain <= 0).mean()))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "uncertainty.csv", index=False)
    print(table.T.to_string(), flush=True)


def qualitative(phys, preds, n=3, win=1024):
    """Rows of 1024² windows around an annotated filament: image, annotator (medoid record), ours (refined)."""
    import cv2
    import matplotlib.pyplot as plt
    from filament.data import load_gray
    rng = np.random.default_rng(int(os.environ.get("QUAL_SEED", 3)))
    pick = phys[[len(data.record_rles(r)) >= 4 for r in phys.train_record]]
    pick = pick.iloc[rng.choice(len(pick), n, replace=False)]
    fig, ax = plt.subplots(n, 3, figsize=(7, 2.35 * n))
    for i, r in enumerate(pick.itertuples()):
        gray = load_gray(data.train_path(r.file_name))
        gts = list(data.record_rles(r.train_record))
        x, y, w, h = mask_utils.toBbox(gts[int(np.argmax(mask_utils.area(gts)))])
        x0 = int(np.clip(x + w / 2 - win / 2, 0, gray.shape[1] - win))
        y0 = int(np.clip(y + h / 2 - win / 2, 0, gray.shape[0] - win))
        panels = [[], gts, preds[r.stem]]
        for j, rles in enumerate(panels):
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            for k, rle in enumerate(rles):
                m = mask_utils.decode(rle).astype(bool)
                col = np.array(plt.cm.tab10(k % 10)[:3]) * 255
                rgb[m] = (0.35 * rgb[m] + 0.65 * col).astype(np.uint8)
            ax[i, j].imshow(rgb[y0:y0 + win, x0:x0 + win])
            ax[i, j].set_xticks([]), ax[i, j].set_yticks([])
            if i == 0:
                ax[i, j].set_title(("H-alpha image", "annotator", "ours (refined)")[j], fontsize=9)
        ax[i, 0].set_ylabel(r.stem, fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "figures" / "qualitative.pdf", dpi=170)


if __name__ == "__main__":
    main()
