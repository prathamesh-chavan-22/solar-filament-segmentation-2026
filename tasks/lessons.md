# Lessons

- **Check the metric date before quoting a public score.** The metric switched Dice → Panoptic Quality on 2026-08-07;
  notebooks/LB numbers from before that (e.g. "LB 0.70" on 2026-08-05) are not comparable. Current PQ top = 0.69 (2026-10-23).
- **"model" may mean Modal.** Ask when a request names a compute provider ambiguously; GPU training runs on Modal (profile `third`, ~$5 budget), Kaggle is only for scoring.
- **Don't probe APIs with side effects.** `kaggle kernels push --no-run` creates a visible version per attempt.

## 2026-10-24 — No local compute
- Rule from user: nothing heavy runs on the laptop (training, inference, sweeps, benchmarks). It crashed twice under load.
- Default compute is Kaggle kernels (free CPU + weekly GPU quota); Modal L4 only when paid GPU is truly needed (estimate cost first).
- Local = edit code, git, launch/monitor remote jobs, download small results.

## 2026-10-24 — Git Bash mangles /kaggle paths
- `python tools/kaggle_run.py push ... --env PROB_DIR=/kaggle/input` from Git Bash sent `C:/Program Files/Git/kaggle/input` (MSYS path conversion); the kernel ran on zero inputs.
- Rule: launch Kaggle pushes from PowerShell, or `export MSYS_NO_PATHCONV=1` in bash. kaggle_run.py now refuses mangled env values.

## 2026-10-24 — No automatic push-retry loops on Kaggle
- A 5-minute loop retrying GPU kernel pushes against a full session quota kept failing ("Maximum batch GPU session count of 2 reached") and the user saw it as constant cancelling. They asked to stop it.
- Rule: push a Kaggle kernel once. If it is refused for quota, tell the user and wait for them to free a slot. Never retry in the background.
