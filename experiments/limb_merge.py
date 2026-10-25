"""Two post-processing ideas on saved hold-out probabilities (CPU only), both scored by OOF PQ against a plain baseline.

1. Limb cut-off — MAGFiLO annotators skip filaments beyond ~±70° from the central meridian (Ahmadzadeh et al., Sci. Data
   2024), so components whose centroid lies far from the disk centre (|dx|/R or r/R above a cut-off) are dropped.
2. Learned fragment merging — pairs of nearby components are labelled "same GT filament" from the annotations, a
   gradient-boosted classifier is trained on pair features (gap, orientation alignment, probability along the
   connecting segment, …) with the date-grouped folds, and pairs whose out-of-fold probability exceeds τ are merged
   (union + a bridge along the gap). A hand rule (gap ≤ G and bridge probability ≥ L) is the non-learned comparison.

Env: DATA_DIR, WORK_DIR, PROB_DIR (glob of one model directory with oof_fold*.pkl), WORKERS (4), T_LO (0.4), CLOSE (3),
     LIMIT (0 = all; else first N images per fold, for a smoke run),
     SUBMIT ("" = experiment; "tau,t_hi,min_mean,min_area", e.g. "0.7,0.99,0.85,200": fit the pair classifier on all
     OOF pairs, merge the test components (PROB_DIR/test_prob.pkl) with it and write WORK_DIR/submission.csv).
"""
import glob
import itertools
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
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from filament import H, W  # noqa: E402
from filament.data import FilamentData, load_gray  # noqa: E402
from filament.instances import components, encode_components, upsample  # noqa: E402
from filament.metric import pq_score  # noqa: E402

env = os.environ.get
T_LO, CLOSE, WORKERS = float(env("T_LO", 0.4)), int(env("CLOSE", 3)), int(env("WORKERS", 4))
MAX_GAP = 150  # px; farther pairs are never merge candidates
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)  # noqa: E731
data = FilamentData(env("DATA_DIR", "data"))
WORK = Path(env("WORK_DIR", "work/limb_merge"))
WORK.mkdir(parents=True, exist_ok=True)


def disk(gray):
    """Solar disk centre (x, y) and radius from the thresholded image."""
    ys, xs = np.nonzero(gray > 0.1 * gray.max())
    return xs.mean(), ys.mean(), np.sqrt(len(xs) / np.pi)


def analyse(args):
    """One image → component table, their RLEs, candidate pairs (+ features, label) and nearest points for bridging."""
    stem, path, record_ids, prob = args
    prob = upsample(np.asarray(prob, np.float32))
    labels, ids, stats, boxes = components(prob, T_LO, CLOSE, 50, 0.5)
    xc, yc, R = disk(load_gray(path))
    comps, pts = [], []
    for i, (x, y, w, h) in zip(ids, boxes):
        sub = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
        m = cv2.moments(sub, True)
        cx, cy = x + m["m10"] / m["m00"], y + m["m01"] / m["m00"]
        a, b, c = m["mu20"], m["mu11"], m["mu02"]
        theta = 0.5 * np.arctan2(2 * b, a - c)
        l1, l2 = np.sort(np.linalg.eigvalsh([[a, b], [b, c]]))[::-1]
        contour = max(cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=len)[:, 0]
        pts.append(contour[::2] + [x, y])
        comps.append(dict(cx=cx, cy=cy, theta=theta, elong=np.sqrt(l1 / max(l2, 1e-6)),
                          width=2 * m["m00"] / max(len(contour), 1), dx=abs(cx - xc) / R, dr=np.hypot(cx - xc, cy - yc) / R))
    comps = pd.DataFrame(comps, columns=["cx", "cy", "theta", "elong", "width", "dx", "dr"])
    comps["area"], comps["peak"], comps["mean"] = stats[:, 0], stats[:, 1], stats[:, 2]
    rles = encode_components(labels, ids, boxes)

    # Which GT instance (per annotator record) covers each component: crowd IoU = intersection / component area.
    owner = np.full((len(record_ids), len(ids)), -1)
    for r, rid in enumerate(record_ids):
        gt = list(data.record_rles(rid))
        if gt and rles:
            cover = np.asarray(mask_utils.iou(rles, gt, [1] * len(gt))).reshape(len(rles), len(gt))
            owner[r] = np.where(cover.max(1) > 0.5, cover.argmax(1), -1)

    pairs = []
    trees = [cKDTree(p) for p in pts]
    for i, j in itertools.combinations(range(len(ids)), 2):
        (xi, yi, wi, hi), (xj, yj, wj, hj) = boxes[i], boxes[j]
        if xi - MAX_GAP > xj + wj or xj - MAX_GAP > xi + wi or yi - MAX_GAP > yj + hj or yj - MAX_GAP > yi + hi:
            continue
        d, k = trees[j].query(pts[i])
        a = int(np.argmin(d))
        gap, p, q = float(d[a]), pts[i][a], pts[j][k[a]]
        if gap > MAX_GAP:
            continue
        n = max(int(gap), 2)
        line = prob[np.linspace(p[1], q[1], n).round().astype(int), np.linspace(p[0], q[0], n).round().astype(int)]
        ci, cj = comps.iloc[i], comps.iloc[j]
        link = np.arctan2(q[1] - p[1], q[0] - p[0])
        fold = lambda t: abs((t + np.pi / 2) % np.pi - np.pi / 2)  # noqa: E731  angle difference folded to [0, π/2]
        small, big = sorted([ci.area, cj.area])
        same = (owner[:, i] == owner[:, j]) & (owner[:, i] >= 0)
        pairs.append(dict(i=i, j=j, gap=gap, gap_rel=gap / np.sqrt(small), log_small=np.log(small), log_big=np.log(big),
                          mean_lo=min(ci["mean"], cj["mean"]), mean_hi=max(ci["mean"], cj["mean"]),
                          peak_lo=min(ci.peak, cj.peak), dtheta=fold(ci.theta - cj.theta),
                          align_i=fold(link - ci.theta) if gap > 1 else 0, align_j=fold(link - cj.theta) if gap > 1 else 0,
                          elong_lo=min(ci.elong, cj.elong), line_mean=float(line.mean()), line_min=float(line.min()),
                          dr_max=max(ci.dr, cj.dr), cdist=np.hypot(ci.cx - cj.cx, ci.cy - cj.cy) / np.sqrt(big),
                          width=min(ci.width, cj.width), px=int(p[0]), py=int(p[1]), qx=int(q[0]), qy=int(q[1]),
                          label=float(same.mean()) if len(record_ids) else 0.0))
    return stem, comps, rles, pd.DataFrame(pairs)


FEATURES = ["gap", "gap_rel", "log_small", "log_big", "mean_lo", "mean_hi", "peak_lo", "dtheta", "align_i", "align_j",
            "elong_lo", "line_mean", "line_min", "dr_max", "cdist", "width"]


def merged_instances(comps, rles, edges):
    """Union-find over ``edges`` (pair rows) → (stats table, RLEs) of the merged instances, bridges drawn along gaps."""
    parent = list(range(len(comps)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for e in edges.itertuples():
        parent[find(e.i)] = find(e.j)
    groups = {}
    for a in range(len(comps)):
        groups.setdefault(find(a), []).append(a)
    rows, out = [], []
    for members in groups.values():
        c = comps.iloc[members]
        if len(members) == 1:
            out.append(rles[members[0]])
            area = c.area.iloc[0]
        else:
            canvas = np.zeros((H, W), np.uint8)
            for a in members:
                canvas |= mask_utils.decode(rles[a])
            for e in edges[edges.i.isin(members)].itertuples():
                cv2.line(canvas, (e.px, e.py), (e.qx, e.qy), 1, max(3, int(round(e.width))))
            out.append(mask_utils.encode(np.asfortranarray(canvas)))
            area = float(mask_utils.area(out[-1]))
        wsum = c.area.sum()
        rows.append(dict(area=area, peak=c.peak.max(), mean=(c["mean"] * c.area).sum() / wsum,
                         dx=(c.dx * c.area).sum() / wsum, dr=(c.dr * c.area).sum() / wsum))
    return pd.DataFrame(rows, columns=["area", "peak", "mean", "dx", "dr"]), out


def _init(instances, records):
    global INSTANCES, RECORDS
    INSTANCES, RECORDS = instances, records


def score(args):
    """(instances by stem, rule) → PQ row, in a worker set up by ``_init`` (works with spawn, i.e. on Windows)."""
    option, (t_hi, mn, area, limb) = args
    preds = {}
    for stem, (st, rles) in INSTANCES[option].items():
        keep = (st.area >= area) & (st.peak >= t_hi) & (st["mean"] >= mn)
        if limb[0] != "none":
            keep &= st[limb[0]] <= limb[1]
        preds[stem] = [r for r, k in zip(rles, keep) if k]
    pq, s = pq_score(preds, RECORDS, data.record_rles)
    return dict(option=option, t_hi=t_hi, min_mean=mn, min_area=area, limb=f"{limb[0]}<={limb[1]}", PQ=pq, **s)


if __name__ == "__main__":
    prob_dir = Path(sorted(glob.glob(env("PROB_DIR", "work/unet_cv"), recursive=True))[0])
    oof = {}
    for p in sorted(prob_dir.rglob("oof_fold*.pkl")):
        oof.update(pickle.load(open(p, "rb")))
    phys = data.physical[data.physical.stem.isin(oof)]
    if int(env("LIMIT", 0)):
        phys = phys.groupby("fold").head(int(env("LIMIT"))).reset_index(drop=True)
    RECORDS = [r for rs in phys.record_ids for r in rs]
    log(f"{len(oof)} OOF images from {prob_dir}, {len(RECORDS)} records")

    with ProcessPoolExecutor(WORKERS) as pool:
        jobs = ((s, data.train_path(f), list(r), oof.pop(s)) for s, f, r in zip(phys.stem, phys.file_name, phys.record_ids))
        results = {stem: (c, r, p) for stem, c, r, p in pool.map(analyse, jobs, chunksize=4)}
    log(f"components: {sum(len(c) for c, _, _ in results.values())}, "
        f"candidate pairs: {sum(len(p) for _, _, p in results.values())}")

    # Learned merging: out-of-fold pair probabilities with the date-grouped folds.
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    fold_of = dict(zip(phys.stem, phys.fold))
    allp = pd.concat([p.assign(stem=s) for s, (_, _, p) in results.items() if len(p)], ignore_index=True)
    allp["y"] = (allp.label >= 0.5).astype(int)
    allp["fold"] = allp.stem.map(fold_of)
    allp["p"] = 0.0
    for k in sorted(allp.fold.unique()):
        tr, te = allp.fold != k, allp.fold == k
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=k)
        clf.fit(allp.loc[tr, FEATURES], allp.loc[tr, "y"])
        allp.loc[te, "p"] = clf.predict_proba(allp.loc[te, FEATURES])[:, 1]
        if allp.loc[te, "y"].nunique() == 2:
            log(f"fold {k}: pair AUC {roc_auc_score(allp.loc[te, 'y'], allp.loc[te, 'p']):.3f}")
    log(f"pairs: {len(allp)}, positives {allp.y.sum()} ({allp.y.mean():.1%}), "
        f"overall AUC {roc_auc_score(allp.y, allp.p):.3f}")
    allp.to_csv(WORK / "pairs.csv", index=False)

    if env("SUBMIT"):
        tau, t_hi, mn, area = map(float, env("SUBMIT").split(","))
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0)
        clf.fit(allp[FEATURES], allp.y)
        test = pickle.load(open(next(prob_dir.rglob("test_prob.pkl")), "rb"))
        paths = {f.stem: f for f in data.test_files}
        with ProcessPoolExecutor(WORKERS) as pool:
            tested = list(pool.map(analyse, ((s, paths[s], [], test.pop(s)) for s in sorted(test)), chunksize=4))
        rows, merges = [], 0
        for stem, comps, rles, pairs in tested:
            edges = pairs[clf.predict_proba(pairs[FEATURES])[:, 1] > tau] if len(pairs) else pairs
            merges += len(edges)
            st, rles = merged_instances(comps, rles, edges) if len(edges) else (comps, rles)
            keep = (st.area >= area) & (st.peak >= t_hi) & (st["mean"] >= mn)
            # Bridges can cross a neighbouring component; the scorer rejects overlapping masks, so earlier ones win.
            taken, k = np.zeros((H, W), bool), 0
            for rle in (r for r, ok in zip(rles, keep) if ok):
                m = mask_utils.decode(rle).astype(bool) & ~taken
                if not m.any():
                    continue
                taken |= m
                k += 1
                rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
                rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": rle["counts"].decode("ascii")})
        sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
        assert sub.filament_id.is_unique
        assert all(mask_utils.area({"size": [H, W], "counts": c.encode()}) > 0 for c in sub.segmentation_rle)
        sub.to_csv(WORK / "submission.csv", index=False)
        log(f"submission.csv: {len(sub)} filaments over {len(tested)} test images, {merges} merges")
        sys.exit(0)

    options = {"none": lambda p: p.iloc[:0]}
    options |= {f"gbm>{tau}": (lambda p, tau=tau: p[p.p > tau]) for tau in (0.3, 0.5, 0.7, 0.85)}
    options |= {f"gap<={g},line>={lm}": (lambda p, g=g, lm=lm: p[(p.gap <= g) & (p.line_mean >= lm)])
                for g, lm in ((40, 0.2), (80, 0.2), (80, 0.1))}
    by_stem = dict(tuple(allp.groupby("stem")))
    INSTANCES = {}
    for name, pick in options.items():
        INSTANCES[name] = {}
        for stem, (comps, rles, _) in results.items():
            edges = pick(by_stem[stem]) if stem in by_stem else pd.DataFrame(columns=allp.columns)
            INSTANCES[name][stem] = merged_instances(comps, rles, edges) if len(edges) else (comps, rles)
        log(f"{name}: {sum(len(r) for _, r in INSTANCES[name].values())} instances")

    # Stage 1: limb cut-offs on the unmerged components; stage 2: merge options with and without the best limb rule.
    limbs = [("none", 0)] + [(k, c) for k in ("dx", "dr") for c in (0.97, 0.94, 0.9, 0.85)]
    keep_grid = list(itertools.product((0.7, 0.99), (0.75, 0.8, 0.85, 0.9), (200, 300, 500)))
    with ProcessPoolExecutor(WORKERS, initializer=_init, initargs=(INSTANCES, RECORDS)) as pool:
        stage1 = pd.DataFrame(pool.map(score, [("none", (*k, l)) for k in keep_grid for l in limbs], chunksize=4))
    stage1 = stage1.sort_values("PQ", ascending=False)
    print(stage1.groupby("limb").head(1).to_string(index=False), flush=True)
    best_limb = next(l for l in limbs if f"{l[0]}<={l[1]}" == stage1.limb.iloc[0])
    with ProcessPoolExecutor(WORKERS, initializer=_init, initargs=(INSTANCES, RECORDS)) as pool:
        stage2 = pd.DataFrame(pool.map(score, [(o, (*k, l)) for o in options if o != "none" for k in keep_grid
                                               for l in {("none", 0), best_limb}], chunksize=4))
    table = pd.concat([stage1, stage2]).sort_values("PQ", ascending=False)
    table.to_csv(WORK / "limb_merge.csv", index=False)
    print(table.groupby(["option", "limb"]).head(1).to_string(index=False), flush=True)
    log("done")
