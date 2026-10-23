"""Competition data access: annotations, annotator records, physical images and grouped folds."""
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from . import H, W

ANN_NAME = "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


def load_gray(path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def ann_rle(ann):
    """One filament's polygon set → a single compressed RLE at 2048²."""
    polys = [p for p in ann["segmentation"] if len(p) >= 6]
    return mask_utils.merge(mask_utils.frPyObjects(polys, H, W)) if polys else None


class FilamentData:
    """Train/test index.

    * ``records``  — one row per annotator record (``record_id = "<annotator>-<stem>"``); the metric scores each separately.
    * ``physical`` — one row per physical image with its record ids, a medoid record and a date-grouped fold.
    """

    def __init__(self, root, folds=5, seed=42):
        root = Path(root)
        self.ann_path = next(root.rglob(ANN_NAME))
        self.train_dir = next(p for p in root.rglob("train_images") if p.is_dir())
        self.test_dir = next(p for p in root.rglob("test_images") if p.is_dir())
        self.test_files = sorted(p for p in self.test_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)

        coco = json.load(open(self.ann_path))
        self.anns = defaultdict(list)
        for a in coco["annotations"]:
            self.anns[a["image_id"]].append(a)
        self.records = pd.DataFrame(coco["images"])[["id", "file_name"]].rename(columns={"id": "record_id"})
        self.records["stem"] = self.records.file_name.str.rsplit(".", n=1).str[0]

        physical = self.records.groupby("stem").agg(file_name=("file_name", "first"),
                                                    record_ids=("record_id", list)).reset_index()
        physical["date"] = physical.stem.str[:8]
        physical["train_record"] = [self.medoid_record(r) for r in physical.record_ids]
        # Group by observation date so near-duplicate frames from different GONG sites never straddle a split.
        dates = np.array(sorted(physical.date.unique()))
        perm = np.random.default_rng(seed).permutation(dates)
        physical["fold"] = physical.date.map(dict(zip(perm, np.arange(len(dates)) % folds)))
        self.physical = physical

    @lru_cache(maxsize=None)
    def record_rles(self, record_id):
        return tuple(r for r in (ann_rle(a) for a in self.anns[record_id]) if r is not None)

    def medoid_record(self, record_ids):
        """Record whose union mask agrees best (mean IoU) with the other annotators of the same image."""
        if len(record_ids) == 1:
            return record_ids[0]
        unions = [mask_utils.merge(list(self.record_rles(r))) if self.record_rles(r) else None for r in record_ids]
        valid = [i for i, u in enumerate(unions) if u is not None]
        if len(valid) < 2:
            return record_ids[valid[0]] if valid else record_ids[0]
        iou = mask_utils.iou([unions[i] for i in valid], [unions[i] for i in valid], [0] * len(valid))
        return record_ids[valid[int(np.argmax(iou.sum(1)))]]

    def soft_target(self, record_ids, size):
        """Mean over annotators of each record's union mask, area-resampled to ``size``² (values in [0, 1])."""
        acc = np.zeros((size, size), np.float32)
        for rid in record_ids:
            rles = self.record_rles(rid)
            if rles:
                union = mask_utils.decode(mask_utils.merge(list(rles))).astype(np.float32)
                acc += cv2.resize(union, (size, size), interpolation=cv2.INTER_AREA)
        return acc / len(record_ids)

    def train_path(self, file_name):
        return self.train_dir / file_name
