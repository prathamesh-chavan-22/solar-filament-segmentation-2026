"""Run one of our scripts as a Kaggle kernel (free compute) — nothing heavy runs locally.

    python tools/kaggle_run.py sync-code                       # create/version dataset xerneas/solar-filament-code
    python tools/kaggle_run.py push <name> <script> [--gpu L4|T4|none] [--input owner/dataset ...]
                                    [--kernel-input owner/kernel ...] [--env KEY=VALUE ...] [--pip 'pkg==ver']
    python tools/kaggle_run.py status <name>
    python tools/kaggle_run.py fetch <name> [--files REGEX]        # → outputs/kaggle/<name>/

<script> is a path under src/ or experiments/ (e.g. src/unet_cv.py), or several comma-separated, run in order. The kernel gets the competition data at /kaggle/input/...,
our code at /kaggle/input/solar-filament-code, writes results to /kaggle/working (= kernel output) and keeps
caches in /tmp. Extra inputs (datasets / other kernels' outputs) appear under /kaggle/input/<slug>.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWNER = "xerneas"
COMPETITION = "filament-segmentation-2026"
CODE_DATASET = f"{OWNER}/solar-filament-code"
ACCELERATORS = {"L4": "NvidiaL4X1", "T4": "NvidiaTeslaT4"}  # P100 is retired on Kaggle
PIP = "segmentation-models-pytorch==0.5.0 pycocotools"

RUNNER = '''import glob, json, os, subprocess, sys
subprocess.run("pip install -q {pip}", shell=True, check=False)
# Mount layout of inputs varies; locate our code by its package marker.
code = os.path.dirname(os.path.dirname(sorted(glob.glob("/kaggle/input/**/filament/__init__.py", recursive=True))[0]))
print("code:", code, flush=True)
os.environ.update({{"DATA_DIR": "/kaggle/input", "WORK_DIR": "/kaggle/working", "CACHE_DIR": "/tmp", "MPLBACKEND": "Agg"}})
os.environ.update(json.loads({env!r}))
print("env:", {{k: os.environ[k] for k in json.loads({env!r})}}, flush=True)
for script in {script!r}.split(","):  # several scripts run one after another, stopping at the first failure
    rc = subprocess.run([sys.executable, "-u", os.path.join(code, script)], cwd="/tmp").returncode
    if rc:
        sys.exit(rc)
'''


def kaggle(*args, check=True):
    print("$ kaggle", " ".join(args), flush=True)
    return subprocess.run(["kaggle", *args], check=check, text=True)


def sync_code():
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "solar-filament-code"
        shutil.copytree(ROOT / "filament", stage / "filament", ignore=shutil.ignore_patterns("__pycache__"))
        for d in ("src", "experiments"):
            shutil.copytree(ROOT / d, stage / d, ignore=shutil.ignore_patterns("__pycache__"))
        json.dump({"title": "solar-filament-code", "id": CODE_DATASET, "licenses": [{"name": "CC0-1.0"}]},
                  open(stage / "dataset-metadata.json", "w"))
        exists = subprocess.run(["kaggle", "datasets", "status", CODE_DATASET], capture_output=True, text=True)
        if exists.returncode == 0 and "ready" in exists.stdout.lower():
            kaggle("datasets", "version", "-p", str(stage), "-m", "sync", "-r", "zip", "-q")
        else:
            kaggle("datasets", "create", "-p", str(stage), "-r", "zip", "-q")


def push(name, script, gpu, inputs, kernel_inputs, env, pip=""):
    env = dict(e.split("=", 1) for e in env)
    # Git Bash rewrites "/kaggle/..." args into "C:/Program Files/Git/kaggle/..."; run with MSYS_NO_PATHCONV=1.
    mangled = [k for k, v in env.items() if "/Git/" in v.replace("\\", "/")]
    if mangled:
        sys.exit(f"MSYS path conversion mangled {mangled}; rerun with MSYS_NO_PATHCONV=1 or from PowerShell")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "run.py").write_text(RUNNER.format(pip=f"{PIP} {pip}".strip(), env=json.dumps(env), script=script))
        json.dump({
            "id": f"{OWNER}/solar-filament-{name}", "title": f"solar-filament-{name}", "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": gpu != "none", "enable_internet": True,
            "dataset_sources": [CODE_DATASET, *inputs], "competition_sources": [COMPETITION],
            "kernel_sources": kernel_inputs,
        }, open(d / "kernel-metadata.json", "w"), indent=1)
        args = ["kernels", "push", "-p", str(d)]
        if gpu != "none":
            args += ["--accelerator", ACCELERATORS[gpu]]
        kaggle(*args)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sync-code")
    p = sub.add_parser("push")
    p.add_argument("name")
    p.add_argument("script")
    p.add_argument("--gpu", default="T4", choices=[*ACCELERATORS, "none"])
    p.add_argument("--input", nargs="*", default=[])
    p.add_argument("--kernel-input", nargs="*", default=[])
    p.add_argument("--env", nargs="*", default=[])
    p.add_argument("--pip", default="", help="extra pinned packages, e.g. timm==1.0.27")
    s = sub.add_parser("status")
    s.add_argument("name")
    f = sub.add_parser("fetch")
    f.add_argument("name")
    f.add_argument("--files", default="", help="regex on output filenames")
    a = ap.parse_args()
    if a.cmd == "sync-code":
        sync_code()
    elif a.cmd == "push":
        push(a.name, a.script, a.gpu, a.input, a.kernel_input, a.env, a.pip)
    elif a.cmd == "status":
        kaggle("kernels", "status", f"{OWNER}/solar-filament-{a.name}")
    else:
        out = ROOT / "outputs" / "kaggle" / a.name
        out.mkdir(parents=True, exist_ok=True)
        args = ["kernels", "output", f"{OWNER}/solar-filament-{a.name}", "-p", str(out), "-o"]
        if a.files:
            args += ["--file-pattern", a.files]
        kaggle(*args)


if __name__ == "__main__":
    sys.exit(main())
