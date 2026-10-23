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
| `kaggle/` | Kaggle kernel bundle (`kernel-metadata.json` + notebook) |

## Run on Kaggle
```bash
python tools/py2nb.py src/notebook.py kaggle/solar-filament-yolo-seg.ipynb
kaggle kernels push -p kaggle --accelerator NvidiaL4   # private, GPU + internet
kaggle kernels output xerneas/solar-filament-yolo11-seg-pq -p out
kaggle competitions submit filament-segmentation-2026 -f out/submission.csv -m "yolo11-seg"
```

## Notes
- Inter-annotator PQ is low (~0.45–0.6 on samples), so the label noise caps the score; the public LB top is ~0.69.
- A known leaderboard exploit (partial / empty submissions) exists; this pipeline always predicts on all 180 test images.
