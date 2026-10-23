# Lessons

- **Check the metric date before quoting a public score.** The metric switched Dice → Panoptic Quality on 2026-08-07;
  notebooks/LB numbers from before that (e.g. "LB 0.70" on 2026-08-05) are not comparable. Current PQ top = 0.69 (2026-10-23).
- **"model" may mean Modal.** Ask when a request names a compute provider ambiguously; GPU training runs on Modal (profile `third`, ~$5 budget), Kaggle is only for scoring.
- **Don't probe APIs with side effects.** `kaggle kernels push --no-run` creates a visible version per attempt.
