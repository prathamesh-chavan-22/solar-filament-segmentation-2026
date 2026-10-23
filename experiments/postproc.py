"""Error analysis + post-processing sweep on saved hold-out predictions (no GPU).

Usage: python experiments/postproc.py outputs/v1/val_raw.pkl
"""
import itertools
import os
import pickle
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DATA_DIR", str(ROOT / "data"))
os.environ.setdefault("WORK_DIR", str(ROOT / "work"))
src = (ROOT / "src" / "notebook.py").read_text(encoding="utf-8")
cells = [c.split("\n", 1)[1] for c in re.split(r"^# %%", src, flags=re.M)[1:] if "[markdown]" not in c.split("\n", 1)[0]]
g = {"display": print, "__name__": "nb"}
for cell in cells[1:5]:                      # imports, config, data, masks/medoid/folds
    exec(cell, g)
pq_cell = next(c for c in cells if "def pq_score" in c)
exec(pq_cell[:pq_cell.index("val_records =")], g)
pq_score, record_rles, H, W = g["pq_score"], g["record_rles"], g["H"], g["W"]

d = pickle.load(open(sys.argv[1], "rb"))
raw, val_records = d["raw"], d["val_records"]
gt = {r: record_rles(r) for r in val_records}
PAD = 32                                     # crop margin ≥ largest morph/merge radius


def to_crop(rle):
    """RLE at 2048² → (y0, x0, bool crop) around its bbox, padded."""
    x, y, w, h = mask_utils.toBbox(rle).astype(int)
    y0, x0 = max(y - PAD, 0), max(x - PAD, 0)
    y1, x1 = min(y + h + PAD + 1, H), min(x + w + PAD + 1, W)
    return y0, x0, mask_utils.decode(rle)[y0:y1, x0:x1].astype(bool)


crops = {s: [(to_crop(r), c) for r, c in v if c >= 0.3] for s, v in raw.items()}


def enc(y0, x0, m):
    full = np.zeros((H, W), np.uint8)
    full[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
    return mask_utils.encode(np.asfortranarray(full))


def overlap(a, b):
    """Do two crop masks share any pixel?"""
    (ay, ax, am), (by, bx, bm) = a, b
    y0, x0 = max(ay, by), max(ax, bx)
    y1, x1 = min(ay + am.shape[0], by + bm.shape[0]), min(ax + am.shape[1], bx + bm.shape[1])
    if y1 <= y0 or x1 <= x0:
        return False
    return bool((am[y0 - ay:y1 - ay, x0 - ax:x1 - ax] & bm[y0 - by:y1 - by, x0 - bx:x1 - bx]).any())


def postprocess(items, conf, min_area=100, morph=0, merge_px=0, keep_frac=0.5):
    """items: [((y0, x0, crop), conf)] → list of RLEs."""
    items = sorted([it for it in items if it[1] >= conf], key=lambda t: -t[1])
    if morph:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(morph) + 1,) * 2)
        op = cv2.dilate if morph > 0 else cv2.erode
        items = [((y0, x0, op(m.astype(np.uint8), k).astype(bool)), c) for (y0, x0, m), c in items]
    claimed = np.zeros((H, W), bool)
    kept = []
    for (y0, x0, m), c in items:             # greedy de-overlap by confidence
        view = claimed[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
        mm = m & ~view
        if mm.sum() == 0 or mm.sum() < keep_frac * m.sum():
            continue
        view |= mm
        kept.append((y0, x0, mm))
    if merge_px and len(kept) > 1:           # union instances whose gap is <= merge_px
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * merge_px + 1,) * 2)
        grown = [(y0, x0, cv2.dilate(m.astype(np.uint8), k).astype(bool)) for y0, x0, m in kept]
        parent = list(range(len(kept)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i, j in itertools.combinations(range(len(kept)), 2):
            if overlap(grown[i], kept[j]):
                parent[find(i)] = find(j)
        groups = {}
        for i in range(len(kept)):
            groups.setdefault(find(i), []).append(i)
        rles = []
        for idx in groups.values():
            if sum(int(kept[i][2].sum()) for i in idx) >= min_area:
                rles.append(mask_utils.merge([enc(*kept[i]) for i in idx]))
        return rles
    return [enc(*t) for t in kept if t[2].sum() >= min_area]


def score(**kw):
    return pq_score({s: postprocess(v, **kw) for s, v in crops.items()}, val_records)


# ---- error analysis on the v1 setting ----
base = {s: postprocess(v, conf=0.35) for s, v in crops.items()}
print("baseline (expect ~0.3133)", score(conf=0.35), flush=True)
buckets, merged, fragmented = {"0": 0, "0-0.3": 0, "0.3-0.5": 0, ">0.5": 0}, 0, 0
for rid in val_records:
    preds = base.get(rid.split("-", 1)[1], [])
    if not gt[rid]:
        continue
    if not preds:
        buckets["0"] += len(gt[rid]); continue
    iou = np.asarray(mask_utils.iou(gt[rid], preds, [0] * len(preds)))
    best = iou.max(1)
    for i, b in enumerate(best):
        buckets["0" if b == 0 else "0-0.3" if b <= .3 else "0.3-0.5" if b <= .5 else ">0.5"] += 1
        if 0 < b <= .5:
            j = iou[i].argmax()
            merged += (iou[:, j] > 0.05).sum() > 1      # best pred also covers other GTs
            fragmented += (iou[i] > 0.05).sum() > 1     # GT touched by several preds
print("GT best-IoU buckets:", buckets, "| near-miss GTs whose pred spans >1 GT:", merged,
      "| near-miss GTs split across >1 pred:", fragmented)

# ---- sweep ----
rows = []
for conf, morph, merge_px, keep_frac in itertools.product([0.3, 0.35, 0.4, 0.45], [-1, 0, 1, 2], [0, 8, 24], [0.3, 0.5]):
    pq, st = score(conf=conf, morph=morph, merge_px=merge_px, keep_frac=keep_frac)
    rows.append(dict(conf=conf, morph=morph, merge_px=merge_px, keep_frac=keep_frac, PQ=round(pq, 4),
                     **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()}))
    print(rows[-1], flush=True)
res = pd.DataFrame(rows).sort_values("PQ", ascending=False)
res.to_csv(Path(sys.argv[1]).with_name("postproc_sweep.csv"), index=False)
print(res.head(10).to_string(index=False))
