"""Train + predict on Modal; Kaggle is only used for scoring.

One-time data upload (from a local `kaggle competitions download` into ./data):
    MODAL_PROFILE=third modal volume create solar-filament-data
    MODAL_PROFILE=third modal volume put solar-filament-data data/MAGFiLO_1.0_Kaggle_2026 /MAGFiLO_1.0_Kaggle_2026

Run (detached, survives closing the terminal):
    MODAL_PROFILE=third modal run --detach modal_app.py --run-name v1 [--gpu L4|L40S] [--epochs 40] [--full-fit] [--fold-weights /runs/v1/runs/fold0/weights/best.pt]
        [--tiled --init-weights /runs/v1/runs/fold0/weights/best.pt] [--debug]

Fetch outputs and score:
    MODAL_PROFILE=third modal volume get solar-filament-runs /v1/submission.csv outputs/v1/
    kaggle competitions submit filament-segmentation-2026 -f outputs/v1/submission.csv -m "v1"
"""
import os
import runpy
import threading
from pathlib import Path

import modal

app = modal.App("solar-filament-seg")
data_vol = modal.Volume.from_name("solar-filament-data", create_if_missing=True)
runs_vol = modal.Volume.from_name("solar-filament-runs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("numpy==1.26.4", "torch==2.5.1", "torchvision==0.20.1", "ultralytics==8.3.40", "pycocotools==2.0.8",
                 "pandas==2.2.3", "matplotlib==3.9.2", "pyyaml==6.0.2", "tqdm==4.67.1")
    .add_local_file(Path(__file__).parent / "src" / "notebook.py", "/root/notebook.py")
)


def _commit_every(seconds):
    """Persist checkpoints periodically so a timeout/preemption doesn't lose the run."""
    def loop():
        while True:
            threading.Event().wait(seconds)
            try:
                runs_vol.commit()
            except Exception as e:  # noqa: BLE001 — best effort
                print("volume commit failed:", e)
    threading.Thread(target=loop, daemon=True).start()


def _run(run_name, debug, epochs, full_fit, fold_weights="", tiled=False, init_weights=""):
    os.environ.update(DATA_DIR="/data", WORK_DIR=f"/runs/{run_name}", DEBUG="1" if debug else "0", MPLBACKEND="Agg",
                      EPOCHS=str(epochs), FULL_FIT="1" if full_fit else "0", FOLD_WEIGHTS=fold_weights,
                      TILED="1" if tiled else "0", INIT_WEIGHTS=init_weights)
    _commit_every(600)
    runpy.run_path("/root/notebook.py", init_globals={"display": print}, run_name="__main__")
    runs_vol.commit()


# Budget-sized: the account has ~$5, so cap wall time hard (L4 ≈ $1.1/h all-in).
common = dict(image=image, volumes={"/data": data_vol, "/runs": runs_vol}, timeout=2 * 3600, cpu=4, memory=16384)


@app.function(gpu="L40S", **common)
def train_l40s(run_name: str, debug: bool, epochs: int, full_fit: bool, fold_weights: str, tiled: bool, init_weights: str):
    _run(run_name, debug, epochs, full_fit, fold_weights, tiled, init_weights)


@app.function(gpu="L4", **common)
def train_l4(run_name: str, debug: bool, epochs: int, full_fit: bool, fold_weights: str, tiled: bool, init_weights: str):
    _run(run_name, debug, epochs, full_fit, fold_weights, tiled, init_weights)


@app.local_entrypoint()
def main(run_name: str = "v1", gpu: str = "L4", debug: bool = False, epochs: int = 40, full_fit: bool = False,
         fold_weights: str = "", tiled: bool = False, init_weights: str = ""):
    fn = {"L40S": train_l40s, "L4": train_l4}[gpu.upper()]
    fn.remote(run_name, debug, epochs, full_fit, fold_weights, tiled, init_weights)
    print(f"done → modal volume get solar-filament-runs /{run_name}/submission.csv outputs/{run_name}/")
