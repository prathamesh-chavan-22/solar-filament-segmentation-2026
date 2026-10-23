"""Learned keep rule: predict each candidate component's PQ value, keep those above a tuned threshold.

Candidates = hysteresis components of the probability map (loose rules). Target = mean over the image's annotator
records of the candidate's matched IoU (IoU if > 0.5 with some GT filament, else 0) — exactly what it adds to the PQ
numerator; every kept candidate adds ½ per record to the denominator, so keeping it pays off when target is high.

A gradient-boosted regressor is trained with the same date-grouped folds as the U-Net (each candidate is scored by a
model that never saw its image), the threshold is tuned by exact PQ, and the model refit on all candidates scores test.

    python experiments/iou_scorer.py outputs/cv1        # needs oof_fold*.pkl, test_prob.pkl there
"""
import gc
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from filament import H, W  # noqa: E402
from filament.data import FilamentData, load_gray  # noqa: E402
from filament.instances import components, encode_components, upsample  # noqa: E402
from filament.metric import pq_score  # noqa: E402
from filament.unet import preprocess  # noqa: E402

T_LO, CLOSE, MIN_AREA, MIN_PEAK = 0.4, 3, 100, 0.5  # loose candidate generation (OOF-best rule keeps a subset)
RING = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))


def candidate_features(prob, gray):
    """→ (DataFrame of per-component features, list of RLEs) for one image."""
    prob = upsample(np.asarray(prob, np.float32))
    img = preprocess(gray, H)  # disk-normalised, off-disk 0
    labels, ids, stats, boxes = components(prob, T_LO, CLOSE, MIN_AREA, MIN_PEAK)
    rows = []
    for i, (x, y, w, h), (area, peak, mean) in zip(ids, boxes, stats):
        pad = 8
        x0, y0, x1, y1 = max(x - pad, 0), max(y - pad, 0), min(x + w + pad, W), min(y + h + pad, H)
        m = labels[y0:y1, x0:x1] == i
        p, g = prob[y0:y1, x0:x1][m], img[y0:y1, x0:x1]
        ring = cv2.dilate(m.astype(np.uint8), RING).astype(bool) & ~m
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perim = sum(cv2.arcLength(c, True) for c in cnts)
        (_, _), (rw, rh), _ = cv2.minAreaRect(np.concatenate(cnts))
        hull = cv2.contourArea(cv2.convexHull(np.concatenate(cnts)))
        cy, cx = y + h / 2, x + w / 2
        rows.append(dict(
            area=area, log_area=np.log(area), peak=peak, mean=mean, std=p.std(),
            p10=np.percentile(p, 10), p50=np.median(p), p90=np.percentile(p, 90),
            f50=(p > 0.5).mean(), f70=(p > 0.7).mean(), f90=(p > 0.9).mean(),
            bw=w, bh=h, extent=area / (w * h), elong=max(rw, rh) / max(min(rw, rh), 1), length=max(rw, rh),
            width=area / max(max(rw, rh), 1), compact=4 * np.pi * area / max(perim, 1) ** 2, solidity=area / max(hull, 1),
            n_parts=len(cnts), radius=np.hypot(cx - W / 2, cy - H / 2) / (W / 2),
            inten=g[m].mean(), ring_inten=g[ring].mean() if ring.any() else 0.0,
            contrast=(g[ring].mean() if ring.any() else 0.0) - g[m].mean(),
        ))
    df = pd.DataFrame(rows)
    if len(df):
        df["n_cand"] = len(df)
        df["area_rank"] = df.area.rank(ascending=False, pct=True)
        df["mean_rank"] = df["mean"].rank(ascending=False, pct=True)
    return df, encode_components(labels, ids, boxes)


def matched_iou(rles, record_ids, data):
    """Mean over records of each candidate's matched IoU (0 when no GT overlaps it by IoU > 0.5)."""
    out = np.zeros(len(rles))
    if not len(rles):
        return out
    for rid in record_ids:
        gt = list(data.record_rles(rid))
        if gt:
            iou = np.asarray(mask_utils.iou(rles, gt, [0] * len(gt))).max(1)
            out += np.where(iou > 0.5, iou, 0)
    return out / len(record_ids)


def build(run_dir, data):
    oof_path, test_path = run_dir / "cand_oof.pkl", run_dir / "cand_test.pkl"
    if not oof_path.exists():
        t0, frames, rles = time.time(), [], []
        phys = data.physical.set_index("stem")
        for pk in sorted(run_dir.glob("oof_fold*.pkl")):
            probs = pickle.load(open(pk, "rb"))
            for stem, prob in probs.items():
                row = phys.loc[stem]
                df, r = candidate_features(prob, load_gray(data.train_path(row.file_name)))
                df["stem"], df["fold"] = stem, row.fold
                df["target"] = matched_iou(r, row.record_ids, data)
                frames.append(df)
                rles += r
            del probs
            gc.collect()
            print(f"{pk.name}: {sum(map(len, frames))} candidates so far ({time.time() - t0:.0f}s)", flush=True)
        pickle.dump((pd.concat(frames, ignore_index=True), rles), open(oof_path, "wb"))
    if not test_path.exists():
        frames, rles = [], []
        test_by_stem = {p.stem: p for p in data.test_files}
        for stem, prob in pickle.load(open(run_dir / "test_prob.pkl", "rb")).items():
            df, r = candidate_features(prob, load_gray(test_by_stem[stem]))
            df["stem"] = stem
            frames.append(df)
            rles += r
        pickle.dump((pd.concat(frames, ignore_index=True), rles), open(test_path, "wb"))
        print(f"test: {sum(map(len, frames))} candidates", flush=True)
    return pickle.load(open(oof_path, "rb")), pickle.load(open(test_path, "rb"))


def make_model():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=0)


def select(df, rles, score, tau):
    keep = score >= tau
    out = {s: [] for s in df.stem.unique()}
    for s, r in zip(df.stem[keep], np.asarray(rles, dtype=object)[keep]):
        out[s].append(r)
    return out


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    data = FilamentData("data")
    (oof, oof_rles), (test, test_rles) = build(run_dir, data)
    feats = [c for c in oof.columns if c not in ("stem", "fold", "target")]
    records = [r for rs in data.physical.record_ids for r in rs]
    print(f"{len(oof)} OOF candidates, {len(feats)} features, mean target {oof.target.mean():.3f}", flush=True)

    # Baseline: the hand-tuned rule (OOF-best: mean ≥ 0.7, area ≥ 300) on the same candidates.
    rule = ((oof["mean"] >= 0.7) & (oof.area >= 300)).to_numpy().astype(float)
    print("rule baseline PQ %.4f" % pq_score(select(oof, oof_rles, rule, 0.5), records, data.record_rles)[0])

    pred = np.zeros(len(oof))
    for k in sorted(oof.fold.unique()):
        tr, va = oof.fold != k, oof.fold == k
        pred[va] = make_model().fit(oof.loc[tr, feats], oof.target[tr]).predict(oof.loc[va, feats])
    res = []
    for tau in np.arange(0.10, 0.46, 0.02):
        pq, st = pq_score(select(oof, oof_rles, pred, tau), records, data.record_rles)
        res.append(dict(tau=round(tau, 2), PQ=pq, **st))
    res = pd.DataFrame(res).sort_values("PQ", ascending=False)
    res.to_csv(run_dir / "scorer_sweep.csv", index=False)
    print(res.head(8).to_string(index=False))
    tau = float(res.tau.iloc[0])

    model = make_model().fit(oof[feats], oof.target)
    score = model.predict(test[feats])
    rows = []
    for stem, rles in sorted(select(test, test_rles, score, tau).items()):
        rows += [{"filament_id": f"{stem}_{k}", "segmentation_rle": r["counts"].decode("ascii")}
                 for k, r in enumerate(rles, 1)]
    sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    assert sub.filament_id.is_unique
    assert all(mask_utils.area({"size": [H, W], "counts": c.encode()}) > 0 for c in sub.segmentation_rle)
    sub.to_csv(run_dir / "submission_scorer.csv", index=False)
    print(f"tau {tau}: submission_scorer.csv {len(sub)} filaments over {sub.filament_id.str.rsplit('_', n=1).str[0].nunique()} images")
