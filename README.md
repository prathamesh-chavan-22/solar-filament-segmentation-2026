# Solar Filament Segmentation 2026

Starter pipeline for the Kaggle competition [filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(metric: Panoptic Quality, TP when IoU > 0.5, deadline 2026-11-15).

## Approach
- **YOLO11-seg** instance segmentation (`m` on L4/T4, `l` on L40S) at 1024² on the 2048² GONG H-alpha disks.
- One **medoid annotator record** per physical image for training; hold-out **grouped by observation date**.
- Local **Panoptic Quality** mirroring the organisers' self-evaluation notebook (each annotator record scored separately).
- Post-processing: upsample to 2048², greedy confidence de-overlap, sweep `conf` × `min_area` on the hold-out.
- Optional full-data refit for the final submission.

## Layout
| Path | Purpose |
|---|---|
| `src/notebook.py` | Source of truth (percent-format cells) |
| `tools/py2nb.py` | Builds the `.ipynb` from the source |
| `modal_app.py` | Modal training/inference app (GPU L40S or L4) |
| `notebooks/` | Built notebook (for reading/local runs) |

## Run (train on Modal, score on Kaggle)
GPU work runs on Modal (profile `third`); Kaggle is used only to download data and submit.
```bash
kaggle competitions download filament-segmentation-2026 -p data   # then unzip into data/
MODAL_PROFILE=third modal volume put solar-filament-data data/MAGFiLO_1.0_Kaggle_2026 /MAGFiLO_1.0_Kaggle_2026
MODAL_PROFILE=third modal run --detach modal_app.py --run-name v1 --gpu L4 --epochs 40
MODAL_PROFILE=third modal volume get solar-filament-runs /v1/submission.csv outputs/v1/
kaggle competitions submit filament-segmentation-2026 -f outputs/v1/submission.csv -m "v1"
```

## Notes
- Results (hold-out = date-grouped fold 0, 141 images / 240 annotator records):

  | run | model | hold-out PQ | public LB |
  |---|---|---|---|
  | v1 | YOLO11m-seg, 40 ep | 0.313 | 0.28 |
  | v2 | v1 + full fit + 1 px erosion | 0.331 | 0.30 |
  | u1 | semantic U-Net r34@1024, soft annotator-mean target, D4 TTA, hysteresis components (`src/unet_pipeline.py`) | 0.407 | **0.35** |
  | cv1 | 5-fold u1 ensemble, rules tuned on all 707 OOF images (`src/unet_cv.py` + `src/unet_submit.py`) | 0.417 (OOF, all folds) | **0.35** |
  | cv1+scorer | cv1 candidates kept by a learned IoU regressor (`experiments/iou_scorer.py`) | 0.420 (OOF) | **0.35** |
  | cv1-strict | cv1 with a wider, stricter rule grid (t_lo 0.5, close 8, mean ≥ 0.80, area ≥ 300), swept on a Kaggle CPU kernel | 0.421 (OOF) | 0.36 |
  | b4 | EfficientNet-B4 U-Net, 5 folds on Kaggle 2×T4 (fold per GPU), strict grid (t_lo 0.4, close 3, mean ≥ 0.80, area ≥ 200) | 0.422 (OOF) | 0.36 |
  | r34+b4 | probability average of cv1 (ResNet34) and b4, strict grid (t_lo 0.5, close 8, t_hi 0.9, mean ≥ 0.80, area ≥ 300) | 0.426 (OOF) | **0.37** |
- Inter-annotator PQ is low (~0.45–0.6 on samples), so the label noise caps the score; the public LB top is 0.69 under PQ (pre-Aug-7 scores like "0.70" were Dice and are not comparable).
- A known leaderboard exploit (partial / empty submissions) exists; this pipeline always predicts on all 180 test images.
