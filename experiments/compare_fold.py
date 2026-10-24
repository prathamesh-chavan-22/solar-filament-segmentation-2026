"""Compare models on one hold-out fold: rule sweep (filament.tune grid) per model and for their probability average.

    python experiments/compare_fold.py r34=outputs/cv1/oof_fold0.pkl b4=outputs/b4/oof_fold0.pkl [--workers 4]
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from filament import tune  # noqa: E402
from filament.data import FilamentData  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="name=path/to/oof_foldK.pkl")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    probs = {name: pickle.load(open(path, "rb")) for name, path in (r.split("=", 1) for r in args.runs)}
    stems = sorted(set.intersection(*(set(p) for p in probs.values())))
    if len(probs) > 1:
        probs["+".join(probs)] = {s: np.mean([p[s].astype(np.float32) for p in probs.values()], 0).astype(np.float16)
                                  for s in stems}
    data = FilamentData("data")
    records = [r for rs in data.physical[data.physical.stem.isin(stems)].record_ids for r in rs]
    print(f"{len(stems)} common hold-out images, {len(records)} annotator records", flush=True)

    rows = []
    for name, p in probs.items():
        table = tune.sweep({s: p[s] for s in stems}, records, data.record_rles, workers=args.workers, log=None)
        best = table.iloc[0]
        rows.append(dict(model=name, PQ=best.PQ, SQ=best.SQ, RQ=best.RQ, TP=best.TP, FP=best.FP, FN=best.FN,
                         rule=str(tune.best_params(table))))
        print(f"{name:12s} PQ {best.PQ:.4f}  (SQ {best.SQ:.3f}, RQ {best.RQ:.3f})", flush=True)
    print(pd.DataFrame(rows).to_string(index=False))
