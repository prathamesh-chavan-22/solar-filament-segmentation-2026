"""Panoptic Quality as implemented in the organisers' self-evaluation notebook.

Each annotator record ``"<annotator>-<stem>"`` is scored against all predictions for ``<stem>``; a pair is a true positive
when IoU > 0.5; FP/FN are counted per record and everything is pooled:  PQ = Σ IoU(TP) / (|TP| + ½|FP| + ½|FN|).
"""
import numpy as np
from pycocotools import mask as mask_utils


def pq_score(pred_by_stem, record_ids, gt_rles):
    """``pred_by_stem``: {stem: [rle, ...]}; ``gt_rles``: callable record_id → sequence of GT RLEs.

    Returns (PQ, stats) with stats = TP, FP, FN, SQ (mean IoU of matches), RQ.
    """
    tp_iou, fp, fn = [], 0, 0
    for rid in record_ids:
        gt = list(gt_rles(rid))
        pred = list(pred_by_stem.get(rid.split("-", 1)[1], []))
        if not gt:
            fp += len(pred)
            continue
        if not pred:
            fn += len(gt)
            continue
        iou = np.asarray(mask_utils.iou(gt, pred, [0] * len(pred)))
        hit = iou > 0.5
        tp_iou.extend(iou[hit].tolist())
        fp += int((hit.sum(0) == 0).sum())
        fn += int((hit.sum(1) == 0).sum())
    tp = len(tp_iou)
    denom = tp + 0.5 * fp + 0.5 * fn
    pq = sum(tp_iou) / denom if denom else 0.0
    return pq, dict(TP=tp, FP=fp, FN=fn, SQ=float(np.mean(tp_iou)) if tp else 0.0,
                    RQ=tp / denom if denom else 0.0)
