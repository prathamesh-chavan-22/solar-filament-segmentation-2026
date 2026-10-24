# Lessons

- **Check the metric date before quoting a public score.** The metric switched Dice → Panoptic Quality on 2026-08-07;
  notebooks/LB numbers from before that (e.g. "LB 0.70" on 2026-08-05) are not comparable. Current PQ top = 0.69 (2026-10-23).
- **"model" may mean Modal.** Ask when a request names a compute provider ambiguously; GPU training runs on Modal (profile `third`, ~$5 budget), Kaggle is only for scoring.
- **Don't probe APIs with side effects.** `kaggle kernels push --no-run` creates a visible version per attempt.

## 2026-10-24 — No local compute
- Rule from user: nothing heavy runs on the laptop (training, inference, sweeps, benchmarks). It crashed twice under load.
- Default compute is Kaggle kernels (free CPU + weekly GPU quota); Modal L4 only when paid GPU is truly needed (estimate cost first).
- Local = edit code, git, launch/monitor remote jobs, download small results.
