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
  | r34+b4+rx50 | adds a ResNeXt-50 U-Net (5 folds, 2×T4) to the average; strict grid picks t_lo 0.4, close 8, t_hi 0.9, mean ≥ 0.75, area ≥ 200 | 0.429 (OOF) | 0.36 |
  | b4-native (fold 0) | B4 trained and predicted at native 2048 (1024 crops, batch 4); fold-0 PQ 0.421 vs 0.415 for B4 at 1024 under the same grid | — | — |
  | b4-native | B4 at native 2048, 5 folds on 2×T4 (GPU-resident uint8 crops), strict grid (t_lo 0.4, close 3, t_hi 0.99, mean ≥ 0.85, area ≥ 200) | 0.433 (OOF) | **0.37** |
  | r34+b4+native | 1024 models and b4-native averaged at 2048 (t_lo 0.4, close 3, t_hi 0.7, mean ≥ 0.80, area ≥ 200) | 0.433 (OOF) | **0.37** |
  | b4-native + merge | b4-native + learned fragment merging (pair GBM > 0.7, overlaps resolved) | 0.435 (OOF) | 0.37 |
  | convnext-native | ConvNeXt-Base (tu-convnext_base, 88M) at native 2048, 5 folds, batch 2, 10k steps, lr 1e-4; strict grid | 0.448 (OOF) | **0.39** |
  | convnext+b4-native | convnext-native and b4-native averaged (t_lo 0.35, t_hi 0.7, mean ≥ 0.80) | 0.447 (OOF) | **0.39** |
  | b4-native + aug (folds 0-1) | GpuCrops augmentation: any-angle rotation, scale 0.85–1.15, contrast/brightness, blur, noise. Same settings otherwise; OOF on folds 0-1 vs 0.431 without. Dropped | 0.409 (f0-1) | — |
  | convnext-dino5 | ConvNeXt-Base with DINOv3 weights (tu-convnext_base.dinov3_lvd1689m), native 5-fold, same recipe | 0.430 (OOF) | 0.37 |
  | dino+convnext(+b4) | dino5 averaged with convnext-native (and b4-native); t_lo 0.35, t_hi 0.7, mean ≥ 0.80 | 0.444 (OOF) | 0.38 |
  | convnext-native + refiner | stage-2 refiner (experiments/refiner.py): U-Net R34 on 256² crops around each kept instance (image, p, seed), target = mean of matched GT masks, 5-fold OOF, D4; output threshold 0.45. Seeds 0.4478 | 0.4505 (OOF) | **0.39** |
- Inter-annotator PQ is low (~0.45–0.6 on samples), so the label noise caps the score; the public LB top is 0.69 under PQ (pre-Aug-7 scores like "0.70" were Dice and are not comparable).
- A known leaderboard exploit (partial / empty submissions) exists; this pipeline always predicts on all 180 test images.
