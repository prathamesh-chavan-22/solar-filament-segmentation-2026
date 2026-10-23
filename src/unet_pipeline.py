"""Semantic U-Net pipeline: train on folds ≠ VAL_FOLD → tune instance rules on the hold-out → (optionally) refit on all
images and write submission.csv.

Env: DATA_DIR, WORK_DIR, SIZE (1024), CROP (512), STEPS (6000), BATCH (8), LR (3e-4), ENCODER (resnet34),
     VAL_FOLD (0), FULL_FIT (1), FOLD_WEIGHTS (skip fold training), DEBUG (0), TTA (8).
"""
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/root")
from filament import H, W  # noqa: E402
from filament.data import FilamentData, load_gray  # noqa: E402
from filament.instances import prob_to_rles  # noqa: E402
from filament import tune, unet  # noqa: E402

env = os.environ.get
DEBUG = env("DEBUG", "0") == "1"
SIZE, CROP = int(env("SIZE", 1024)), int(env("CROP", 512))
STEPS, BATCH, LR = int(env("STEPS", 6000)), int(env("BATCH", 8)), float(env("LR", 3e-4))
ENCODER, VAL_FOLD, TTA = env("ENCODER", "resnet34"), int(env("VAL_FOLD", 0)), int(env("TTA", 8))
FULL_FIT, FOLD_WEIGHTS = env("FULL_FIT", "1") == "1", env("FOLD_WEIGHTS", "")
WORK = Path(env("WORK_DIR", "work/unet"))
WORK.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEBUG:
    STEPS, TTA = 40, 2

t0 = time.time()
data = FilamentData(env("DATA_DIR", "data"))
phys = data.physical
if DEBUG:
    phys = pd.concat([phys[phys.fold == VAL_FOLD].head(6), phys[phys.fold != VAL_FOLD].head(10)])
print(f"{len(phys)} physical images, device {DEVICE}, {time.time() - t0:.0f}s", flush=True)

# ---- inputs and soft targets (cached) -------------------------------------------------------------------------------
cache = WORK / f"cache_{SIZE}{'_debug' if DEBUG else ''}.npz"
if cache.exists():
    z = np.load(cache)
    images, targets = list(z["images"]), list(z["targets"])
else:
    images = [unet.preprocess(load_gray(data.train_path(f)), SIZE).astype(np.float16) for f in phys.file_name]
    targets = [data.soft_target(r, SIZE).astype(np.float16) for r in phys.record_ids]
    np.savez(cache, images=np.stack(images), targets=np.stack(targets))
print(f"inputs ready {time.time() - t0:.0f}s", flush=True)
is_val = (phys.fold == VAL_FOLD).to_numpy()


def fit(idx, tag):
    model = unet.build_model(ENCODER)
    unet.train(model, [images[i] for i in idx], [targets[i] for i in idx], steps=STEPS, batch=BATCH, crop=CROP, lr=LR,
               device=DEVICE, log_every=max(STEPS // 30, 1))
    path = WORK / f"{tag}.pt"
    torch.save(model.state_dict(), path)
    return model, path


def load(path):
    model = unet.build_model(ENCODER, pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(DEVICE).eval()


# ---- fold model + hold-out probabilities ----------------------------------------------------------------------------
model = load(FOLD_WEIGHTS) if FOLD_WEIGHTS else fit(np.flatnonzero(~is_val), f"fold{VAL_FOLD}")[0]
val = phys[is_val]
prob_path = WORK / "val_prob.pkl"
if FOLD_WEIGHTS and prob_path.exists():  # resuming a run whose fold model already predicted the hold-out
    val_prob = pickle.load(open(prob_path, "rb"))
else:
    val_prob = {s: unet.predict_prob(model, images[i].astype(np.float32), DEVICE, TTA).astype(np.float16)
                for s, i in zip(val.stem, np.flatnonzero(is_val))}
    pickle.dump(val_prob, open(prob_path, "wb"))
print(f"hold-out probabilities {time.time() - t0:.0f}s", flush=True)

# ---- instance-rule sweep on the hold-out -----------------------------------------------------------------------------
val_records = [r for rs in val.record_ids for r in rs]
table = tune.sweep(val_prob, val_records, data.record_rles, log=lambda m: print(m, flush=True))
table.to_csv(WORK / "sweep.csv", index=False)
print(table.head(10).to_string(index=False), flush=True)
BEST = tune.best_params(table)
print(f"best {BEST} → hold-out PQ {table.PQ[0]:.4f}  ({time.time() - t0:.0f}s)", flush=True)

# ---- final model + submission ----------------------------------------------------------------------------------------
if FULL_FIT:
    model = fit(np.arange(len(phys)), "full")[0]
test_files = data.test_files[:4] if DEBUG else data.test_files
rows = []
for f in test_files:
    prob = unet.predict_prob(model, unet.preprocess(load_gray(f), SIZE), DEVICE, TTA)
    for k, rle in enumerate(prob_to_rles(prob, BEST), 1):
        rows.append({"filament_id": f"{f.stem}_{k}", "segmentation_rle": rle["counts"].decode("ascii")})
sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
assert sub.filament_id.is_unique
assert all(mask_utils.area({"size": [H, W], "counts": c.encode()}) > 0 for c in sub.segmentation_rle)
sub.to_csv(WORK / "submission.csv", index=False)
print(f"submission.csv: {len(sub)} filaments over {sub.filament_id.str.rsplit('_', n=1).str[0].nunique()}/"
      f"{len(test_files)} images  ({time.time() - t0:.0f}s)", flush=True)
