# Todo
- [x] Research competition (metric = PQ, COCO polygons, 707 images / 1154 annotator records)
- [x] Build YOLO11-seg train → PQ hold-out → submission notebook
- [x] Local smoke test (labels, metric 1.0/0.0 sanity, inference + RLE)
- [x] Private GitHub repo
- [x] Native B4 5-fold (OOF 0.4329, LB 0.37); ConvNeXt-B f0-1 + B4 (OOF f0-1 0.4424, LB 0.38)
- [x] Limb cut-off (no gain), learned fragment merging (+0.002 OOF, LB 0.37)

## Resume here (stopped 2026-10-25 evening; start morning of 2026-10-26)
State when stopped:
- Kaggle `convnext-native5` (ConvNeXt-B folds 2,3,4 + INIT folds 0,1, native 2048) was RUNNING on 2×T4, started
  ~09:52 UTC, expected done ~13:30 UTC. Only ~1.5 h of Kaggle GPU quota is left this week, so no more GPU kernels.
- Local: `outputs/kaggle/b4-native5/` has oof_fold0-4.pkl + test_prob.pkl (native B4, 7.4 GB).
- `experiments/scorer.py` (learned instance selection) passed a 15-image smoke test; the full run was stopped before finishing
  (rerun: `WORKERS=8 python -u experiments/scorer.py > work/scorer.log 2>&1`, ~5 min, CPU only).
- Kaggle CLI token expires 2026-10-26 00:45 UTC → refresh first (see tasks/lessons.md, "Kaggle CLI Authentication").
- Submissions used today: 5/5. Best public 0.38 = ConvNeXt f0-1 + B4 native avg.

Morning steps:
1. [ ] Refresh the Kaggle token; check `kaggle kernels status xerneas/solar-filament-convnext-native5` (expect COMPLETE).
   Download its outputs to outputs/kaggle/convnext-native5 (oof_fold0-4.pkl, test_prob.pkl).
2. [ ] Sweep 5-fold ConvNeXt alone and ConvNeXt+B4 avg (src/unet_submit.py with PROB_DIR=both dirs, locally on CPU or
   Kaggle CPU via experiments/compare_models.py). Submit the best (check overlaps = 0 before submitting).
3. [ ] Run experiments/scorer.py on B4 OOF, then on PROB_DIR=convnext5,b4 (ensemble). Adopt it if the gain over hysteresis is above 0.004;
   then add a SUBMIT mode (test candidates → DP selection → submission.csv).
4. [ ] Stage-2 refiner on the local GPU (only when the user says so): see step 4 below.

## Plan (2026-10-25): local RTX 3050 Ti + Kaggle
Reference: filseg v4 notebook (tushhaaarrrr, public 0.40) — learned instance selection and a stage-2 refiner were the
two gains that moved the public board (0.437 → 0.449 OOF, 0.37 → 0.39).

1. [ ] Tomorrow, after the 00:00 UTC reset: submit the 5-fold ConvNeXt + B4 average (sweep on Kaggle CPU or locally).
2. [ ] Download the OOF and test probability maps to local disk:
   - native B4 and ConvNeXt, about 6 GB each as float16 pickles;
   - average them into one ensemble map per image (memmap on disk).
3. [ ] Learned instance scorer (CPU, local, 75% cap):
   - Build candidates from a threshold ladder t ∈ {0.3 … 0.7} over the ensemble map: nested components with closing.
   - Describe each candidate: area, mean/peak/quantiles of p, elongation, width, limb distance, contrast in the image, parent/child stats.
   - Target: crowd match rate, i.e. the fraction of records where the candidate has IoU > 0.5 with a GT instance.
   - Model: HistGBM, out-of-fold by fold.
   - Selection: pick the non-overlapping set greedily by score, along the tree, with one global τ tuned on OOF.
   - Compare with the hysteresis rule on the same OOF. Adopt only if it gains more than 0.004 (filseg's seed-noise floor).
4. [ ] Stage-2 refiner (local GPU):
   - Input: a 256² crop around each selected instance, channels = image, p, seed mask. Target: the matched annotators' mean mask. Model: U-Net R34.
   - Train with the GPU-resident crop set and VRAM capped at 0.85.
   - Use torch.compile (reduce-overhead / CUDA graphs) with a static batch; fp16, channels_last.
   - Check memory with a smoke run that includes an optimizer step and a save. Launch the long run detached (Start-Process + log).
   - Score: OOF PQ with the refined masks replacing the seeds (out-of-fold refiner).
5. [ ] Report + public repo + reproducible notebook by Nov 15.

Not doing:
- skeleton-recall / clDice: −0.045 in filseg;
- pseudo-labelled GONG frames: +0.003, within noise;
- public MAGFiLO: contains the test labels.
