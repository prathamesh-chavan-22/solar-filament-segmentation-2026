"""Predict the fold-0 hold-out locally and save raw instances (pre de-overlap) for post-processing experiments.

Usage: python experiments/dump_val_preds.py outputs/v1/fold0_best.pt outputs/v1/val_raw.pkl
"""
import os
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DATA_DIR", str(ROOT / "data"))
os.environ.setdefault("WORK_DIR", str(ROOT / "work"))

# Reuse the notebook's code: run every code cell up to (not including) dataset building / training.
src = (ROOT / "src" / "notebook.py").read_text(encoding="utf-8")
cells = [c.split("\n", 1)[1] for c in re.split(r"^# %%", src, flags=re.M)[1:] if "[markdown]" not in c.split("\n", 1)[0]]
g = {"display": print, "__name__": "nb"}
for cell in cells:
    if "def build_dataset" in cell:
        break
    exec(cell, g)
dataset_cell = next(c for c in cells if "def build_dataset" in c)
exec(dataset_cell[dataset_cell.index("def load_resized"):dataset_cell.index("def build_dataset")], g)

import numpy as np
import torch
from pycocotools import mask as mask_utils
from tqdm import tqdm
from ultralytics import YOLO

physical, H, W = g["physical"], g["H"], g["W"]
val_df = physical[physical.fold == g["VAL_FOLD"]].reset_index(drop=True)
model = YOLO(sys.argv[1])
raw = {}
for row in tqdm(list(val_df.itertuples()), desc="predict"):
    r = model.predict([g["load_resized"](g["TRAIN_IMG_DIR"] / row.file_name)], imgsz=g["IMGSZ"], conf=0.05, iou=0.5,
                      retina_masks=True, max_det=100, verbose=False, half=True)[0]
    inst = []
    if r.masks is not None:
        probs = torch.nn.functional.interpolate(r.masks.data.float()[None], size=(H, W), mode="bilinear")[0]
        for m, c in zip((probs > 0.5).cpu().numpy(), r.boxes.conf.cpu().numpy()):
            if m.any():
                inst.append((mask_utils.encode(np.asfortranarray(m.astype(np.uint8))), float(c)))
    raw[row.stem] = inst
pickle.dump({"raw": raw, "val_records": [r for ids in val_df.record_ids for r in ids],
             "train_record": dict(zip(val_df.stem, val_df.train_record))}, open(sys.argv[2], "wb"))
print("saved", sum(map(len, raw.values())), "instances over", len(raw), "images")
