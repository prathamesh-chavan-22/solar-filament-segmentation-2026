# Solar Filament Segmentation 2026

Solution to the Kaggle challenge [filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026):
instance segmentation of solar filaments in GONG H-alpha images, scored by Panoptic Quality (PQ).

**Result:** out-of-fold PQ **0.451** on all 707 training images (95% bootstrap interval 0.442–0.460), public
leaderboard **0.39**. No data beyond the competition training set was used (see *Data use* below).
Technical report: [`report/template/main.pdf`](report/template/main.pdf).

## Pipeline
```
H-alpha 2048² ─► disk normalisation ─► Stage 1: U-Net (ConvNeXt-Base), native 2048², 2 seeds × folds, D4 TTA
             ─► probability p ─► hysteresis instances (p ≥ 0.4, close 3 px, peak ≥ 0.7, mean ≥ 0.85, area ≥ 200)
             ─► Stage 2: crop refiner (ResNet-34 U-Net on 256² crops: image, p, seed mask) ─► non-overlapping RLEs
```
1. **Targets and validation.** Soft target = mean of the annotator records' masks. Five folds grouped by observation
   date (the GONG sites observe the same Sun minutes apart). Every choice is made on out-of-fold (OOF) PQ, computed
   like the organisers' self-evaluation notebook (each annotator record scored separately).
2. **Stage 1** (`src/unet_cv.py`, `filament/unet.py`): smp U-Net, ImageNet ConvNeXt-Base encoder, 1-channel input,
   random 1024² crops of the native image + D4, BCE + ½ Dice, AdamW 1e-4, 10k steps, batch 2 (Kaggle 2×T4, one fold
   per GPU). Whole-image inference with 8 D4 views.
3. **Instances** (`filament/instances.py`, rule tuned by `src/unet_submit.py`): connected components with hysteresis
   and mean-probability filters; disjoint by construction.
4. **Stage 2** (`experiments/refiner.py`): each instance redrawn from a 256² crop by a ResNet-34 U-Net trained out of
   fold on the matched annotator masks; mean of 5 fold models × D4, threshold 0.45; pixels claimed by a stronger
   instance first, so outputs never overlap.

## Reproduce the submission
`notebooks/final_submission.ipynb` (source: `src/final_submission.py`) rebuilds `submission.csv` from the trained
weights, without training. The weights (9 ConvNeXt U-Nets, 5 refiners) are attached to the GitHub release
[`weights-v1`](../../releases/tag/weights-v1); the notebook downloads them and checks their SHA-256
(`src/weights.sha256`). About 1 h on 2×T4 (Kaggle), longer on one GPU. Verified on Kaggle 2×T4 (58 min): the
notebook's output matches the submitted file instance for instance (same 1,130 ids, mean IoU 0.99985, 88% of masks
pixel-identical; the rest differ by fp16 GPU arithmetic).
```bash
pip install -r requirements.txt
kaggle competitions download filament-segmentation-2026 -p data && unzip -q data/*.zip -d data
jupyter nbconvert --to notebook --execute notebooks/final_submission.ipynb   # → work/final/submission.csv
```

## Train from scratch
```bash
# Stage 1, one run per seed (GPU; on Kaggle via tools/kaggle_run.py push ... src/unet_cv.py)
ENCODER=tu-convnext_base SIZE=2048 CROP=1024 BATCH=2 STEPS=10000 LR=1e-4 GPU_DATA=1 python src/unet_cv.py
# Instance rule on OOF probabilities (CPU)
PROB_DIR=work/unet_cv GRID_T_LO=0.35,0.4,0.45 GRID_CLOSE=3 GRID_T_HI=0.7,0.99 GRID_MEAN=0.75,0.8,0.85,0.9   GRID_AREA=200,300 python src/unet_submit.py
# Stage 2 refiner: crops (CPU), 5-fold training (GPU, ~20 min/fold on 4 GB), OOF score, test submission
PROB_DIR=work/unet_cv MODE=build,train,score python experiments/refiner.py
PROB_DIR=work/unet_cv MODE=submit THRESH=0.45 python experiments/refiner.py
# Report statistics and figures
python experiments/report_stats.py
```

## Layout
| Path | Purpose |
|---|---|
| `filament/` | Library: data index and soft targets (`data.py`), U-Net training/inference (`unet.py`), instance extraction (`instances.py`), PQ metric (`metric.py`) |
| `src/unet_cv.py` | Stage-1 k-fold training, OOF and test probabilities |
| `src/unet_submit.py` | Instance-rule grid search on OOF PQ and submission writer |
| `src/final_submission.py` | Inference notebook source (built with `tools/py2nb.py`) |
| `experiments/refiner.py` | Stage-2 refiner |
| `experiments/report_stats.py` | OOF statistics (IoU/Dice, splits/merges, fold + bootstrap uncertainty) and figures |
| `experiments/` (other) | Ideas that were tested and dropped (learned scorer, fragment merging, …) |
| `tools/kaggle_run.py` | Runs our scripts as Kaggle kernels (2×T4) |
| `report/` | Technical report (ACM template), statistics and figures |

## Data use
Only the competition training data and ImageNet-pretrained encoders were used. The public MAGFiLO release also
contains annotations for the challenge's test images; we never downloaded or used external annotations.

## Development history
Public leaderboard scores are rounded to two decimals by Kaggle.

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
  | convnext 2 seeds + refiner | convnext-native (5f) + convnext-seed2 (SEED=1, 4f, 8k steps) averaged, rule 0.4/3/0.7/0.85/200, then the refiner (0.45). OOF on folds 0-3: seed1 0.4474, seed2 0.4458, avg 0.4496 | 0.4496 (OOF f0-3, before refiner) | **0.39** |
  | refiner, unmatched → empty | refiner also trained to erase unmatched instances (TRAIN_UNMATCHED=1); removes too many true positives (TP 5172 → 4868). Dropped | 0.442 (OOF) | — |
