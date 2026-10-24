"""5-fold U-Net (CPU part): tune the instance rules on all out-of-fold probabilities, then turn the fold-averaged
test probabilities into submission.csv.

Env: DATA_DIR, WORK_DIR (outputs), PROB_DIR (WORK_DIR; comma list of glob patterns, one directory per model, each
     searched recursively for oof_fold*.pkl and test_prob.pkl from src/unet_cv.py; models are averaged), WORKERS (os.cpu_count()), GRID_T_LO / GRID_CLOSE / GRID_T_HI / GRID_MEAN / GRID_AREA (comma lists
     overriding the filament.tune grids).
"""
import glob
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, "/root")
from filament import H, W, tune  # noqa: E402
from filament.data import FilamentData  # noqa: E402
from filament.instances import prob_to_rles  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    WORK = Path(os.environ.get("WORK_DIR", "work/unet_cv"))
    # One directory (glob pattern) per model; Kaggle mounts inputs at varying depths, e.g. /kaggle/input/**/<slug>.
    PROBS = []
    for pat in os.environ.get("PROB_DIR", str(WORK)).split(","):
        found = sorted(glob.glob(pat, recursive=True))
        if not found:
            sys.exit(f"PROB_DIR: nothing matches {pat}")
        PROBS.append(Path(found[0]))
    print(f"models: {[str(d) for d in PROBS]}", flush=True)
    grid = {f"{k}_grid": [cast(v) for v in os.environ[f"GRID_{k.upper()}"].split(",")]
            for k, cast in (("t_lo", float), ("close", int), ("t_hi", float), ("mean", float), ("area", int))
            if f"GRID_{k.upper()}" in os.environ}
    workers = int(os.environ.get("WORKERS", os.cpu_count()))
    data = FilamentData(os.environ.get("DATA_DIR", "data"))

    def load(pattern):
        """{stem: prob} from every PROB_DIR model (files matching ``pattern``), averaged over models."""
        models = []
        for d in PROBS:
            probs = {}
            for p in sorted(d.rglob(pattern)):
                probs.update(pickle.load(open(p, "rb")))
            models.append(probs)
        stems = sorted(set.intersection(*(set(m) for m in models)))
        return {s: np.mean([m[s].astype(np.float32) for m in models], 0).astype(np.float16) for s in stems}

    oof = load("oof_fold*.pkl")
    records = [r for rs in data.physical[data.physical.stem.isin(oof)].record_ids for r in rs]
    print(f"OOF: {len(oof)} images, {len(records)} annotator records, {workers} workers", flush=True)

    table = tune.sweep(oof, records, data.record_rles, workers=workers, **grid, log=lambda m: print(m, flush=True))
    table.to_csv(WORK / "sweep_oof.csv", index=False)
    print(table.head(10).to_string(index=False), flush=True)
    best = tune.best_params(table)
    print(f"best {best} → OOF PQ {table.PQ[0]:.4f} ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for stem, prob in load("test_prob.pkl").items():
        for k, rle in enumerate(prob_to_rles(prob.astype("float32"), best), 1):
            rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": rle["counts"].decode("ascii")})
    sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    assert sub.filament_id.is_unique
    assert all(mask_utils.area({"size": [H, W], "counts": c.encode()}) > 0 for c in sub.segmentation_rle)
    sub.to_csv(WORK / "submission.csv", index=False)
    print(f"submission.csv: {len(sub)} filaments over {sub.filament_id.str.rsplit('_', n=1).str[0].nunique()} "
          f"images ({time.time() - t0:.0f}s)", flush=True)
