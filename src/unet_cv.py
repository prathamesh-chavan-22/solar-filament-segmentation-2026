"""5-fold U-Net (GPU part): train each fold (skipped when WORK/fold{k}.pt exists), save its hold-out probabilities, then
average all fold models over the test set.  Rule tuning + submission happen on CPU in src/unet_submit.py.

Outputs in WORK_DIR: fold{k}.pt, oof_fold{k}.pkl ({stem: (S, S) float16}), test_prob.pkl ({stem: (S, S) float16}).
Env: DATA_DIR, WORK_DIR, SIZE (1024), CROP (512), STEPS (5000), BATCH (8), LR (3e-4), ENCODER (resnet34),
     FOLDS ("0,1,2,3,4"), TTA (8), DEBUG (0), CHANNELS_LAST (0), COMPILE (0), AMP (fp16|bf16), PRED_CHUNK (8).
Local RTX 3050 Ti (4 GB), EfficientNet-B4: BATCH=4 STEPS=10000 CHANNELS_LAST=1 COMPILE=1 PRED_CHUNK=2
(compiled NHWC: 23 crops/s, 2.2 GB vs 19 crops/s, 3.1 GB eager).
"""
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/root")
from filament import unet  # noqa: E402
from filament.data import FilamentData, load_gray  # noqa: E402

env = os.environ.get
DEBUG = env("DEBUG", "0") == "1"
SIZE, CROP = int(env("SIZE", 1024)), int(env("CROP", 512))
STEPS, BATCH, LR = int(env("STEPS", 5000)), int(env("BATCH", 8)), float(env("LR", 3e-4))
ENCODER, TTA = env("ENCODER", "resnet34"), int(env("TTA", 8))
FOLDS = [int(k) for k in env("FOLDS", "0,1,2,3,4").split(",")]
# Kernel-level speedups (see filament.unet.speedups): NHWC convs, Triton-compiled graph, AMP dtype; TTA views per pass.
CHANNELS_LAST, COMPILE = env("CHANNELS_LAST", "0") == "1", env("COMPILE", "0") == "1"
AMP_DTYPE = {"fp16": torch.float16, "bf16": torch.bfloat16}[env("AMP", "fp16")]
PRED_CHUNK = int(env("PRED_CHUNK", 8))
WORK = Path(env("WORK_DIR", "work/unet_cv"))
WORK.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEBUG:
    STEPS, TTA = 20, 2

t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)  # noqa: E731
data = FilamentData(env("DATA_DIR", "data"))
phys = data.physical
if DEBUG:
    phys = phys.groupby("fold").head(3).reset_index(drop=True)

cache = WORK / f"cache_{SIZE}{'_debug' if DEBUG else ''}.npz"
if cache.exists():
    z = np.load(cache)
    images, targets = list(z["images"]), list(z["targets"])
else:
    images = [unet.preprocess(load_gray(data.train_path(f)), SIZE).astype(np.float16) for f in phys.file_name]
    targets = [data.soft_target(r, SIZE).astype(np.float16) for r in phys.record_ids]
    np.savez(cache, images=np.stack(images), targets=np.stack(targets))
log(f"{len(phys)} images ready, device {DEVICE}")

models = []
for k in FOLDS:
    weights = WORK / f"fold{k}.pt"
    model = unet.build_model(ENCODER, pretrained=not weights.exists())
    if weights.exists():
        model.load_state_dict(torch.load(weights, map_location="cpu"))
        model.to(DEVICE).eval()
        log(f"fold {k}: loaded {weights.name}")
    else:
        idx = np.flatnonzero(phys.fold.to_numpy() != k)
        unet.train(model, [images[i] for i in idx], [targets[i] for i in idx], steps=STEPS, batch=BATCH, crop=CROP,
                   lr=LR, device=DEVICE, log_every=max(STEPS // 10, 1), seed=k, channels_last=CHANNELS_LAST,
                   compile_model=COMPILE, amp_dtype=AMP_DTYPE)
        torch.save(model.state_dict(), weights)
        log(f"fold {k}: trained on {len(idx)} images")
    oof = WORK / f"oof_fold{k}.pkl"
    if not oof.exists():
        val = np.flatnonzero(phys.fold.to_numpy() == k)
        pickle.dump({phys.stem.iat[i]: unet.predict_prob(model, images[i].astype(np.float32), DEVICE, TTA, PRED_CHUNK)
                     .astype(np.float16) for i in val}, open(oof, "wb"))
        log(f"fold {k}: hold-out probabilities for {len(val)} images")
    models.append(model)

test_files = data.test_files[:3] if DEBUG else data.test_files
test_prob = {}
for f in test_files:
    x = unet.preprocess(load_gray(f), SIZE)
    test_prob[f.stem] = np.mean([unet.predict_prob(m, x, DEVICE, TTA, PRED_CHUNK) for m in models], 0).astype(np.float16)
pickle.dump(test_prob, open(WORK / "test_prob.pkl", "wb"))
log(f"test probabilities: {len(test_prob)} images × {len(models)} models")
