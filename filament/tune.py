"""Grid search of the probability → instance rules against hold-out PQ."""
import itertools

import numpy as np
import pandas as pd

from .instances import InstanceParams, components, encode_components, keep_mask, upsample
from .metric import pq_score

T_LO_GRID, CLOSE_GRID = [0.3, 0.4, 0.5], [0, 3]
T_HI_GRID, MEAN_GRID, AREA_GRID = [0.5, 0.7, 0.85, 0.95], [0.0, 0.5, 0.6, 0.7], [50, 200, 500, 1000]


def sweep(prob_by_stem, record_ids, gt_rles, t_lo_grid=T_LO_GRID, close_grid=CLOSE_GRID, t_hi_grid=T_HI_GRID,
          mean_grid=MEAN_GRID, area_grid=AREA_GRID, log=print):
    """→ DataFrame of every rule combination with PQ/TP/FP/FN/SQ/RQ, best first.

    Components are extracted and encoded once per (t_lo, close_px); the keep rules are cheap filters on top.
    """
    rows = []
    for t_lo, close in itertools.product(t_lo_grid, close_grid):
        cand = {}
        for stem, prob in prob_by_stem.items():
            labels, ids, stats, boxes = components(upsample(np.asarray(prob, np.float32)), t_lo, close,
                                                   min(area_grid), min(t_hi_grid))
            cand[stem] = (stats, encode_components(labels, ids, boxes))
        for t_hi, mn, area in itertools.product(t_hi_grid, mean_grid, area_grid):
            p = InstanceParams(t_lo, t_hi, close, area, mn)
            preds = {s: [r for r, k in zip(rles, keep_mask(st, p)) if k] for s, (st, rles) in cand.items()}
            pq, st = pq_score(preds, record_ids, gt_rles)
            rows.append(dict(t_lo=t_lo, close_px=close, t_hi=t_hi, min_mean=mn, min_area=area, PQ=pq, **st))
        if log:
            log(f"t_lo={t_lo} close={close}: best PQ so far {max(r['PQ'] for r in rows):.4f}")
    return pd.DataFrame(rows).sort_values("PQ", ascending=False).reset_index(drop=True)


def best_params(table):
    b = table.iloc[0]
    return InstanceParams(float(b.t_lo), float(b.t_hi), int(b.close_px), int(b.min_area), float(b.min_mean))
