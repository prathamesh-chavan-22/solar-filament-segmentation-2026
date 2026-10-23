"""Sanity-check tiled mode without training: label fidelity + tiled inference/merge/PQ path.

Usage: python experiments/check_tiled.py outputs/v1/fold0_best.pt
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(DATA_DIR=str(ROOT / "data"), WORK_DIR=str(ROOT / "work" / "check_tiled"), TILED="1", DEBUG="1")
src = (ROOT / "src" / "notebook.py").read_text(encoding="utf-8")
cells = [c.split("\n", 1)[1] for c in re.split(r"^# %%", src, flags=re.M)[1:] if "[markdown]" not in c.split("\n", 1)[0]]
g = {"display": print, "__name__": "nb"}
for cell in cells[1:5]:
    exec(cell, g)
ds = next(c for c in cells if "def build_dataset" in c)
exec(ds[:ds.index("val_df = physical")], g)
inf = next(c for c in cells if "def predict_tiled" in c)
exec(inf, g)
pq = next(c for c in cells if "def pq_score" in c)
exec(pq[:pq.index("val_records =")], g)

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from ultralytics import YOLO

physical, TILE = g["physical"], g["TILE"]

# 1) label fidelity: rasterise the YOLO tile polygons back and compare with the cropped GT union
ious = []
for row in physical.head(10).itertuples():
    gt = [mask_utils.decode(r) for r in g["record_rles"](row.train_record)]
    union = np.logical_or.reduce(gt) if gt else np.zeros((2048, 2048), bool)
    for ty, tx in g["TRAIN_TILE_ORIGINS"]:
        canvas = np.zeros((TILE, TILE), np.uint8)
        for line in g["tile_lines"](row.train_record, ty, tx):
            pts = (np.array(line.split()[1:], float).reshape(-1, 2) * TILE).round().astype(np.int32)
            cv2.fillPoly(canvas, [pts], 1)
        crop = union[ty:ty + TILE, tx:tx + TILE]
        if crop.sum() > 0:
            ious.append((canvas.astype(bool) & crop).sum() / (canvas.astype(bool) | crop).sum())
print(f"tile label IoU vs GT: mean={np.mean(ious):.3f} min={np.min(ious):.3f} over {len(ious)} tiles")

# 2) tiled inference path end-to-end (v1 global weights → scale mismatch, so only the code path is checked)
model = YOLO(sys.argv[1])
rows = list(physical[physical.fold == g["VAL_FOLD"]].itertuples())[:3]
raw = g["predict_raw"](model, [g["TRAIN_IMG_DIR"] / r.file_name for r in rows])
for stem, (rles, confs) in raw.items():
    inst = g["postprocess"](rles, confs, 0)
    print(stem, "raw", len(rles), "after postprocess", len(inst), "top conf", np.round(confs[:3], 2) if len(confs) else [])
records = [rid for r in rows for rid in r.record_ids]
sel = {s: g["select"](g["postprocess"](*v, 0), 0.3, 100) for s, v in raw.items()}
print("PQ on 3 images (code-path check only):", g["pq_score"](sel, records))
