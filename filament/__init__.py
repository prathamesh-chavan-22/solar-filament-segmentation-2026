"""Solar filament segmentation (Kaggle filament-segmentation-2026).

Modules
-------
data       COCO annotations → physical images, annotator records, medoid record, date-grouped folds, soft targets
metric     Panoptic Quality exactly as the organisers' self-evaluation notebook (TP ⇔ IoU > 0.5, pooled over records)
instances  semantic probability map → filament instances (hysteresis components + keep rules) → COCO RLE
unet       disk-normalised input, crop dataset, smp U-Net training and D4 test-time augmentation
"""
H = W = 2048
