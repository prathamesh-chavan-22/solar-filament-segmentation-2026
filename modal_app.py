"""Train + predict on Modal; Kaggle is only used for scoring.

One-time data upload (from a local `kaggle competitions download` into ./data):
    MODAL_PROFILE=third modal volume create solar-filament-data
    MODAL_PROFILE=third modal volume put solar-filament-data data/MAGFiLO_1.0_Kaggle_2026 /MAGFiLO_1.0_Kaggle_2026

Run (detached, survives closing the terminal):
    MODAL_PROFILE=third modal run --detach modal_app.py --run-name v1 [--gpu L4] [--debug]

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


def _run(run_name, debug):
    os.environ.update(DATA_DIR="/data", WORK_DIR=f"/runs/{run_name}", DEBUG="1" if debug else "0", MPLBACKEND="Agg")
    _commit_every(600)
    runpy.run_path("/root/notebook.py", init_globals={"display": print}, run_name="__main__")
    runs_vol.commit()


common = dict(image=image, volumes={"/data": data_vol, "/runs": runs_vol}, timeout=8 * 3600, cpu=8, memory=32768)


@app.function(gpu="L40S", **common)
def train_l40s(run_name: str, debug: bool = False):
    _run(run_name, debug)


@app.function(gpu="L4", **common)
def train_l4(run_name: str, debug: bool = False):
    _run(run_name, debug)


@app.local_entrypoint()
def main(run_name: str = "v1", gpu: str = "L40S", debug: bool = False):
    fn = {"L40S": train_l40s, "L4": train_l4}[gpu.upper()]
    fn.remote(run_name, debug)
    print(f"done → modal volume get solar-filament-runs /{run_name}/submission.csv outputs/{run_name}/")
