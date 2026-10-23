"""Semantic filament probability → instances.

1. Upsample the probability map to 2048² (bilinear) so instance boundaries are drawn at native resolution.
2. Foreground = p ≥ ``t_lo``, optionally closed (bridges small gaps inside one filament) and hole-filled.
3. 8-connected components are candidate filaments.
4. A component is kept only if it is confidently a filament: peak p ≥ ``t_hi`` (hysteresis), mean p ≥ ``min_mean``,
   area ≥ ``min_area``. PQ rewards dropping doubtful instances, so these filters matter more than the threshold.

Components never overlap, so there are no one-to-many / many-to-one conflicts between predictions.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from scipy import ndimage

from . import H, W


@dataclass(frozen=True)
class InstanceParams:
    t_lo: float = 0.4
    t_hi: float = 0.7
    close_px: int = 0
    min_area: int = 200
    min_mean: float = 0.0


def upsample(prob):
    return prob if prob.shape == (H, W) else cv2.resize(prob.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)


def components(prob2048, t_lo, close_px, min_area=0, min_peak=0.0):
    """→ (labels, ids, stats, boxes) for components with area ≥ ``min_area`` and peak ≥ ``min_peak``.

    ``stats`` rows are (area, peak, mean probability); ``boxes`` rows are (x, y, w, h).
    """
    fg = (prob2048 >= t_lo).astype(np.uint8)
    if close_px:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1,) * 2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    fg = ndimage.binary_fill_holes(fg).astype(np.uint8)
    n, labels, cc, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    ids = np.arange(1, n)
    ids = ids[cc[ids, cv2.CC_STAT_AREA] >= max(min_area, 1)]
    if not len(ids):
        return labels, ids, np.zeros((0, 3)), np.zeros((0, 4), int)
    peak = np.asarray(ndimage.maximum(prob2048, labels, ids))
    ids = ids[peak >= min_peak]
    stats = np.stack([cc[ids, cv2.CC_STAT_AREA], ndimage.maximum(prob2048, labels, ids),
                      ndimage.mean(prob2048, labels, ids)], 1) if len(ids) else np.zeros((0, 3))
    return labels, ids, stats, cc[ids, :4]


def keep_mask(stats, p: InstanceParams):
    area, peak, mean = stats.T if len(stats) else (np.zeros(0),) * 3
    return (area >= p.min_area) & (peak >= p.t_hi) & (mean >= p.min_mean)


def encode_components(labels, ids, boxes):
    """Component ids → compressed RLEs; only each bounding box is compared, the 2048² canvas is reused."""
    canvas = np.zeros((H, W), np.uint8, order="F")
    out = []
    for i, (x, y, w, h) in zip(ids, boxes):
        canvas[y:y + h, x:x + w] = labels[y:y + h, x:x + w] == i
        out.append(mask_utils.encode(canvas))
        canvas[y:y + h, x:x + w] = 0
    return out


def prob_to_rles(prob, p: InstanceParams):
    labels, ids, stats, boxes = components(upsample(prob), p.t_lo, p.close_px, p.min_area, p.t_hi)
    keep = keep_mask(stats, p)
    return encode_components(labels, ids[keep], boxes[keep])
