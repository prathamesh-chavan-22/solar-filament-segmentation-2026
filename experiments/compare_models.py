"""Score several models (and their average) on the same out-of-fold images: one src/unet_submit.py sweep per variant.

Env: VARIANTS ("name=glob1+glob2;name2=glob3": each variant averages the listed PROB_DIR patterns), OOF_GLOB
     (e.g. oof_fold[01].pkl), WORK_DIR, plus anything unet_submit.py reads (GRID_*, WORKERS).
Outputs: WORK_DIR/<name>/sweep_oof.csv and submission.csv per variant, WORK_DIR/compare.csv (best rule per variant).
"""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

SUBMIT = Path(__file__).resolve().parent.parent / "src" / "unet_submit.py"
WORK = Path(os.environ.get("WORK_DIR", "work/compare"))
rows = []
for spec in os.environ["VARIANTS"].split(";"):
    name, globs = spec.split("=", 1)
    out = WORK / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== {name}: {globs}", flush=True)
    rc = subprocess.run([sys.executable, "-u", str(SUBMIT)],
                        env={**os.environ, "PROB_DIR": globs.replace("+", ","), "WORK_DIR": str(out)}).returncode
    if rc:
        print(f"{name}: failed ({rc})", flush=True)
        continue
    rows.append({"variant": name, **pd.read_csv(out / "sweep_oof.csv").iloc[0].to_dict()})
table = pd.DataFrame(rows)
table.to_csv(WORK / "compare.csv", index=False)
print(table.to_string(index=False), flush=True)
