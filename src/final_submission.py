# %% [markdown]
# # Solar Filament Segmentation 2026 — final submission (inference only)
#
# Reproduces our final `submission.csv` from the trained weights. Training is done by `src/unet_cv.py` (stage 1)
# and `experiments/refiner.py` (stage 2); see the README and the report for the full pipeline.
#
# **Pipeline**
# 1. **Stage 1 — semantic probability.** U-Net with a ConvNeXt-Base encoder on the disk-normalised 1-channel image at
#    native 2048², trained on the soft annotator-mean mask (BCE + ½ Dice). Two training seeds × date-grouped folds
#    (5 + 4 models); D4 test-time augmentation; the two seeds' fold means are averaged.
# 2. **Instances.** Components of `p ≥ 0.4` after a 3 px closing and hole filling; a component is kept if its peak
#    `p ≥ 0.7`, mean `p ≥ 0.85` and area ≥ 200 px (rule tuned on out-of-fold Panoptic Quality).
# 3. **Stage 2 — refiner.** Every kept instance is redrawn by a ResNet-34 U-Net from a 256² crop (image, stage-1 `p`,
#    the instance's own mask), mean of 5 fold refiners with D4 TTA, threshold 0.45. Pixels already claimed by an
#    instance with a higher mean `p` are removed, so instances never overlap.
#
# **Inputs (Kaggle):** the competition data; `xerneas/solar-filament-code` (this repository's `filament/`,
# `experiments/`); stage-1 weights from the notebooks `xerneas/solar-filament-convnext-native5` (seed 0, folds 0-4)
# and `xerneas/solar-filament-convnext-seed2` (seed 1, folds 0-3); refiner weights `xerneas/solar-filament-refiner`.
# Locally, set `DATA_DIR`, `CODE_DIR`, `WEIGHTS_DIR` instead. GPU: 2×T4 ≈ 1 h (one GPU works, twice as long).

# %%
import importlib.util
import subprocess
import sys

missing = [p for m, p in (("segmentation_models_pytorch", "segmentation-models-pytorch==0.5.0"),
                          ("pycocotools", "pycocotools"), ("timm", "timm")) if not importlib.util.find_spec(m)]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=False)

# %%
import glob
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_utils

ON_KAGGLE = Path("/kaggle/input").exists()
DATA_DIR = os.environ.get("DATA_DIR", "/kaggle/input" if ON_KAGGLE else "data")
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/kaggle/input" if ON_KAGGLE else "outputs/kaggle")
OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working" if ON_KAGGLE else "work/final"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
# Our code: the repository root (local) or the code dataset (Kaggle), found by the package marker.
CODE_DIR = os.environ.get("CODE_DIR") or str(Path(sorted(glob.glob(
    "/kaggle/input/**/filament/__init__.py" if ON_KAGGLE else "filament/__init__.py", recursive=True))[0]).parent.parent)
sys.path[:0] = [CODE_DIR, os.path.join(CODE_DIR, "experiments")]

# Settings of the final submission (all tuned on out-of-fold PQ; see the report).
os.environ.update(DATA_DIR=DATA_DIR, WORK_DIR=str(OUT_DIR), RULE="0.4,3,0.7,0.85,200", R="256", PAD="32",
                  MIN_SIDE="128", WORKERS=str(min(os.cpu_count(), 8)))
REFINER_THRESH = 0.45
STAGE1 = {"seed0": "convnext-native5", "seed1": "convnext-seed2"}  # weight folders (Kaggle notebook slugs)
ENCODER, TTA = "tu-convnext_base", 8

from filament.data import load_gray  # noqa: E402
from filament.unet import build_model, predict_prob, preprocess  # noqa: E402
import refiner  # noqa: E402  (experiments/refiner.py: crops, refiner inference, paste-back)

t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:6.0f}s] {m}", flush=True)  # noqa: E731
test_files = refiner.data.test_files
gpus = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] or [torch.device("cpu")]
log(f"code {CODE_DIR}; {len(test_files)} test images; devices {gpus}")


def weight_files(folder, pattern):
    files = sorted(glob.glob(f"{WEIGHTS_DIR}/**/*{folder}*/**/{pattern}", recursive=True)
                   or glob.glob(f"{WEIGHTS_DIR}/{folder}/{pattern}"))
    assert files, f"no {pattern} under {WEIGHTS_DIR} for {folder}"
    return files


# %% [markdown]
# ## Stage 1: ConvNeXt-Base U-Net probabilities (two seeds × folds, D4 TTA)
# Each GPU predicts every model on its half of the test images; models are loaded one at a time.

# %%
images = {f.stem: preprocess(load_gray(f), 2048) for f in test_files}
stems = list(images)
seed_files = {s: weight_files(folder, "fold*.pt") for s, folder in STAGE1.items()}
for s, files in seed_files.items():
    log(f"{s}: {len(files)} fold models")
prob = {s: {stem: np.zeros((2048, 2048), np.float32) for stem in stems} for s in seed_files}


def stage1_worker(device, my_stems):
    for s, files in seed_files.items():
        for f in files:
            model = build_model(ENCODER, pretrained=False)
            model.load_state_dict(torch.load(f, map_location="cpu"))
            model.to(device).eval()
            with torch.no_grad():
                for stem in my_stems:
                    prob[s][stem] += predict_prob(model, images[stem], device, TTA, chunk=1) / len(files)
            del model
            torch.cuda.empty_cache()
            log(f"{device}: {s} {Path(f).name} done")


threads = [threading.Thread(target=stage1_worker, args=(d, stems[i::len(gpus)])) for i, d in enumerate(gpus)]
for t in threads:
    t.start()
for t in threads:
    t.join()
test_prob = {stem: np.mean([prob[s][stem] for s in prob], 0).astype(np.float16) for stem in stems}
del prob, images
log("stage 1 done")

# %% [markdown]
# ## Instances (hysteresis rule) → stage-2 refiner → `submission.csv`

# %%
import segmentation_models_pytorch as smp  # noqa: E402

files = {f.stem: f for f in test_files}
crops, meta = [], []
for stem in stems:
    x, _, m, _ = refiner.instances((stem, files[stem], None, test_prob[stem]))
    crops.append(x)
    meta += m
crops, meta = np.concatenate(crops), pd.DataFrame(meta)
log(f"{len(meta)} instances over {meta.stem.nunique()} images")

refiners = []
for f in weight_files("refiner", "ref_fold*.pt"):
    m = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(f, map_location="cpu"))
    refiners.append(m.to(gpus[0]).eval())
refined = refiner.predict(refiners, crops, gpus[0])
preds = {stem: [] for stem in stems}
preds.update(refiner.paste(meta, refined, REFINER_THRESH))

rows = [{"filament_id": f"{stem}_{k}", "segmentation_rle": r["counts"].decode("ascii")}
        for stem, rles in preds.items() for k, r in enumerate(rles, 1)]
sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
sub.to_csv(OUT_DIR / "submission.csv", index=False)
log(f"submission.csv: {len(sub)} filaments over {sum(map(bool, preds.values()))} of {len(stems)} images")

# %%
# Sanity checks: unique ids and no overlapping instances (overlaps make the scorer fail).
assert sub.filament_id.is_unique
for stem, rles in preds.items():
    if len(rles) > 1:
        assert mask_utils.area(mask_utils.merge(rles)) == mask_utils.area(rles).sum(), stem
print("ok: unique ids, no overlaps")
sub.head()
