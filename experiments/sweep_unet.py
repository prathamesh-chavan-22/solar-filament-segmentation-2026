"""Tune the U-Net instance rules on saved hold-out probabilities (CPU only).

    python experiments/sweep_unet.py outputs/u1/val_prob.pkl
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from filament import tune  # noqa: E402
from filament.data import FilamentData  # noqa: E402

prob_path = Path(sys.argv[1])
val_prob = pickle.load(open(prob_path, "rb"))
data = FilamentData("data")
records = [r for rs in data.physical[data.physical.stem.isin(val_prob)].record_ids for r in rs]
print(f"{len(val_prob)} hold-out images, {len(records)} annotator records", flush=True)
table = tune.sweep(val_prob, records, data.record_rles, log=lambda m: print(m, flush=True))
table.to_csv(prob_path.with_name("sweep.csv"), index=False)
print(table.head(15).to_string(index=False))
print("best", tune.best_params(table))
