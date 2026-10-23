# %% [markdown]
# # Solar Filament Segmentation 2026 — YOLO11-seg + PQ-tuned post-processing
#
# End-to-end starter: **train → grouped hold-out → official Panoptic Quality → threshold sweep → test inference → `submission.csv`**.
#
# **Why instance segmentation:** since 7 Aug 2026 the metric is Panoptic Quality,
# `PQ = Σ IoU(TP) / (|TP| + ½|FP| + ½|FN|)` with a TP only when `IoU > 0.5`.
# Every missed, extra, split or merged filament costs score, so we predict instances directly.
#
# **Data facts that drive the design**
# - 707 physical H-alpha images (2048²), 1,154 annotator records (`image_id = "<annotator>-<stem>"`), ~7 filaments/record.
# - The metric scores each annotator record separately → we validate the same way.
# - For training we keep one **medoid** record per physical image (the annotation that agrees most with the other annotators).
# - Hold-out is grouped by **observation date** so near-duplicate frames from different GONG sites never leak across the split.
#
# **Run settings:** GPU (L4 / L40S / T4), Internet **on** (pip + pretrained weights). Not a code competition — submit the CSV.

# %%
import importlib.util, subprocess, sys
if not all(importlib.util.find_spec(m) for m in ("ultralytics", "pycocotools")):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.3.0", "pycocotools"], check=False)

# %%
import json, os, random, shutil, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from pycocotools import mask as mask_utils
from tqdm.auto import tqdm
from ultralytics import YOLO
from ultralytics.data.converter import merge_multi_segment

# %% [markdown]
# ## Config

# %%
DEBUG = os.environ.get("DEBUG", "0") == "1"   # True → tiny smoke run (few images, 2 epochs)
SEED = 42
FOLDS, VAL_FOLD = 5, 0           # date-grouped hold-out used for threshold tuning
FULL_FIT = os.environ.get("FULL_FIT", "1") == "1"   # retrain on all images with the tuned thresholds

GPU_MEM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
MODEL = "yolo11l-seg.pt" if GPU_MEM_GB > 40 else "yolo11m-seg.pt"   # L40S → large, L4/T4 → medium
IMGSZ = 1024                     # images are 2048²; 1024 keeps thin filaments visible
EPOCHS = 2 if DEBUG else int(os.environ.get("EPOCHS", 80))
BATCH = 16 if GPU_MEM_GB > 40 else (8 if GPU_MEM_GB > 20 else 4)
MASK_RATIO = 2                   # finer mask head than the default 4 — filaments are thin

CONF_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]
MIN_AREA_GRID = [0, 100, 300, 600, 1000]   # pixels at 2048²
PRED_CONF = 0.05                 # low floor; the tuned threshold is applied afterwards

WORK = Path(os.environ.get("WORK_DIR", "/kaggle/working" if Path("/kaggle/working").exists() else "work"))
WORK.mkdir(parents=True, exist_ok=True)
YOLO_DIR = WORK / "yolo"
H = W = 2048

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"GPU={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} "
      f"mem={GPU_MEM_GB:.0f}GB model={MODEL} batch={BATCH}")

# %% [markdown]
# ## Locate data

# %%
INPUT = Path(os.environ.get("DATA_DIR", "/kaggle/input" if Path("/kaggle/input").exists() else "data"))
ANN_PATH = next(INPUT.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
TRAIN_IMG_DIR = next(p for p in INPUT.rglob("train_images") if p.is_dir())
TEST_IMG_DIR = next(p for p in INPUT.rglob("test_images") if p.is_dir())
test_files = sorted(p for p in TEST_IMG_DIR.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"})

coco = json.load(open(ANN_PATH))
anns_by_record = defaultdict(list)
for a in coco["annotations"]:
    anns_by_record[a["image_id"]].append(a)
records = pd.DataFrame(coco["images"])[["id", "file_name"]].rename(columns={"id": "record_id"})
records["stem"] = records["file_name"].str.rsplit(".", n=1).str[0]
records["date"] = records["stem"].str[:8]
records["n_ann"] = records["record_id"].map(lambda r: len(anns_by_record[r]))
print(f"{len(records)} annotator records | {records.stem.nunique()} physical images | "
      f"{len(coco['annotations'])} filaments | {len(test_files)} test images")

# %% [markdown]
# ## Masks, medoid annotator, grouped split

# %%
def ann_rle(ann):
    """One filament polygon set → single compressed RLE at 2048²."""
    polys = [p for p in ann["segmentation"] if len(p) >= 6]
    return mask_utils.merge(mask_utils.frPyObjects(polys, H, W))


def record_rles(record_id):
    return [ann_rle(a) for a in anns_by_record[record_id] if any(len(p) >= 6 for p in a["segmentation"])]


def medoid_record(record_ids):
    """Record whose union mask has the highest mean IoU with the other annotators."""
    if len(record_ids) == 1:
        return record_ids[0]
    unions = [mask_utils.merge(record_rles(r)) if record_rles(r) else None for r in record_ids]
    valid = [i for i, u in enumerate(unions) if u is not None]
    iou = mask_utils.iou([unions[i] for i in valid], [unions[i] for i in valid], [0] * len(valid))
    best = valid[int(np.argmax((iou.sum(1) - 1) / max(len(valid) - 1, 1)))]
    return record_ids[best]


physical = records.groupby("stem").agg(file_name=("file_name", "first"), date=("date", "first"),
                                       record_ids=("record_id", list)).reset_index()
physical["train_record"] = [medoid_record(r) for r in tqdm(physical.record_ids, desc="medoid")]

dates = np.array(sorted(physical.date.unique()))
rng = np.random.default_rng(SEED)
date_fold = dict(zip(rng.permutation(dates), np.arange(len(dates)) % FOLDS))
physical["fold"] = physical.date.map(date_fold)
if DEBUG:
    physical = physical.groupby("fold").head(8).reset_index(drop=True)
print(physical.fold.value_counts().sort_index().to_dict())

# %% [markdown]
# ## YOLO-seg dataset (images downscaled to 1024, polygons normalised)

# %%
def yolo_lines(record_id):
    lines = []
    for a in anns_by_record[record_id]:
        polys = [p for p in a["segmentation"] if len(p) >= 6]
        if not polys:
            continue
        seg = np.concatenate(merge_multi_segment(polys), 0) if len(polys) > 1 else np.array(polys[0]).reshape(-1, 2)
        seg = np.clip(seg / np.array([W, H]), 0, 1).reshape(-1)
        lines.append("0 " + " ".join(f"{v:.5f}" for v in seg))
    return lines


def load_resized(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.cvtColor(cv2.resize(img, (IMGSZ, IMGSZ), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)


def build_dataset(train_df, val_df, name):
    root = YOLO_DIR / name
    for split, df in (("train", train_df), ("val", val_df)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for row in df.itertuples():
            img_out = root / "images" / split / f"{row.stem}.jpg"
            if not img_out.exists():
                cv2.imwrite(str(img_out), load_resized(TRAIN_IMG_DIR / row.file_name), [cv2.IMWRITE_JPEG_QUALITY, 95])
            (root / "labels" / split / f"{row.stem}.txt").write_text("\n".join(yolo_lines(row.train_record)))
    cfg = {"path": str(root.resolve()), "train": "images/train", "val": "images/val", "names": {0: "filament"}}
    (root / "data.yaml").write_text(yaml.safe_dump(cfg))
    return root / "data.yaml"


val_df = physical[physical.fold == VAL_FOLD].reset_index(drop=True)
trn_df = physical[physical.fold != VAL_FOLD].reset_index(drop=True)
data_yaml = build_dataset(trn_df, val_df, f"fold{VAL_FOLD}")
print(f"train={len(trn_df)} val={len(val_df)} → {data_yaml}")

# %% [markdown]
# ## Train
# Full-disk solar images are rotation/flip invariant, so geometric augmentation is aggressive; colour jitter is off (single-channel H-alpha).

# %%
TRAIN_ARGS = dict(
    imgsz=IMGSZ, epochs=EPOCHS, batch=BATCH, device=0, seed=SEED, workers=4,
    optimizer="AdamW", lr0=1e-3, cos_lr=True, patience=25, close_mosaic=10,
    degrees=180, flipud=0.5, fliplr=0.5, scale=0.3, translate=0.1, mosaic=0.5, mixup=0.0,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
    mask_ratio=MASK_RATIO, overlap_mask=True, max_det=100, amp=True, plots=True,
    project=str(WORK / "runs"), exist_ok=True,
)


def train(data, name):
    model = YOLO(MODEL)
    model.train(data=str(data), name=name, **TRAIN_ARGS)
    return Path(model.trainer.best)


t0 = time.time()
best_fold = train(data_yaml, f"fold{VAL_FOLD}")
print(f"trained in {(time.time() - t0) / 60:.1f} min → {best_fold}")

# %% [markdown]
# ## Inference + post-processing
# Predicted masks are upsampled to 2048², overlaps are resolved greedily by confidence (higher-confidence instance keeps the pixels),
# and each instance is RLE-encoded. Conf / min-area thresholds are applied afterwards so one prediction pass serves the whole sweep.

# %%
def predict_instances(model, paths, batch=8):
    """→ {stem: [(rle, conf, area), ...]} after de-overlap, sorted by conf desc."""
    out = {}
    for i in tqdm(range(0, len(paths), batch), desc="predict"):
        chunk = paths[i:i + batch]
        imgs = [load_resized(p) for p in chunk]
        results = model.predict(imgs, imgsz=IMGSZ, conf=PRED_CONF, iou=0.5, retina_masks=True,
                                max_det=100, verbose=False, half=True)
        for path, r in zip(chunk, results):
            out[Path(path).stem] = postprocess(r)
    return out


def postprocess(r):
    if r.masks is None or len(r.masks) == 0:
        return []
    probs = r.masks.data.float()                                     # (N, 1024, 1024)
    probs = torch.nn.functional.interpolate(probs[None], size=(H, W), mode="bilinear")[0]
    masks = (probs > 0.5).cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    claimed = np.zeros((H, W), bool)
    inst = []
    for k in np.argsort(-confs):
        m = masks[k] & ~claimed
        area = int(m.sum())
        if area == 0 or area < 0.5 * masks[k].sum():                 # mostly swallowed by a stronger instance
            continue
        claimed |= m
        inst.append((mask_utils.encode(np.asfortranarray(m.astype(np.uint8))), float(confs[k]), area))
    return inst


def select(inst, conf_thr, min_area):
    return [rle for rle, c, a in inst if c >= conf_thr and a >= min_area]

# %% [markdown]
# ## Official-style Panoptic Quality
# Mirrors the organisers' self-evaluation notebook: each annotator record is scored against the predictions for its image,
# TP ⇔ IoU > 0.5, PQ pooled over all records.

# %%
gt_cache = {}


def pq_score(pred_by_stem, record_ids):
    tp_iou, fp, fn = [], 0, 0
    for rid in record_ids:
        if rid not in gt_cache:
            gt_cache[rid] = record_rles(rid)
        gt = gt_cache[rid]
        pred = pred_by_stem.get(rid.split("-", 1)[1], [])
        if not gt:
            fp += len(pred); continue
        if not pred:
            fn += len(gt); continue
        iou =np.asarray(mask_utils.iou(gt, pred, [0] * len(pred)))
        hit = iou > 0.5
        tp_iou.extend(iou[hit].tolist())
        fp += int((hit.sum(0) == 0).sum())
        fn += int((hit.sum(1) == 0).sum())
    tp = len(tp_iou)
    denom = tp + 0.5 * fp + 0.5 * fn
    return (sum(tp_iou) / denom if denom else 0.0), dict(TP=tp, FP=fp, FN=fn, SQ=np.mean(tp_iou) if tp else 0)


val_records = [r for ids in val_df.record_ids for r in ids]

# Reference point: how well do annotators agree with each other? (medoid record as "prediction")
human_pred = {row.stem: record_rles(row.train_record) for row in val_df.itertuples()}
others = [r for row in val_df.itertuples() for r in row.record_ids if r != row.train_record]
if others:
    print("inter-annotator PQ (medoid vs other annotators): %.4f %s" % pq_score(human_pred, others))

# %% [markdown]
# ## Tune conf / min-area on the hold-out

# %%
val_model = YOLO(best_fold)
val_inst = predict_instances(val_model, [TRAIN_IMG_DIR / f for f in val_df.file_name])

sweep = []
for c in CONF_GRID:
    for a in MIN_AREA_GRID:
        pq, st = pq_score({s: select(v, c, a) for s, v in val_inst.items()}, val_records)
        sweep.append(dict(conf=c, min_area=a, PQ=pq, **st))
sweep = pd.DataFrame(sweep).sort_values("PQ", ascending=False)
print(sweep.head(10).to_string(index=False))
sweep.to_csv(WORK / "threshold_sweep.csv", index=False)
BEST_CONF, BEST_MIN_AREA = float(sweep.iloc[0].conf), int(sweep.iloc[0].min_area)
print(f"hold-out PQ={sweep.iloc[0].PQ:.4f} @ conf={BEST_CONF} min_area={BEST_MIN_AREA}")

# %% [markdown]
# ## Visual check

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, row in zip(axes, val_df.sample(3, random_state=SEED).itertuples()):
    img = cv2.imread(str(TRAIN_IMG_DIR / row.file_name), cv2.IMREAD_GRAYSCALE)
    over = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for rle in record_rles(row.train_record):
        over[mask_utils.decode(rle).astype(bool)] = (0, 255, 0)
    for rle in select(val_inst[row.stem], BEST_CONF, BEST_MIN_AREA):
        m = mask_utils.decode(rle).astype(bool)
        over[m] = (over[m] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)
    ax.imshow(over); ax.set_title(f"{row.stem}  green=GT  red=pred"); ax.axis("off")
plt.tight_layout(); plt.savefig(WORK / "val_overlay.png", dpi=80); plt.show()

# %% [markdown]
# ## Final model → test predictions
# With `FULL_FIT`, retrain on every physical image (same recipe/epochs) and reuse the hold-out thresholds.

# %%
if FULL_FIT and not DEBUG:
    full_yaml = build_dataset(physical, val_df, "full")      # val split only feeds Ultralytics' own logging
    final_weights = train(full_yaml, "full")
else:
    final_weights = best_fold
print("final weights:", final_weights)

test_inst = predict_instances(YOLO(final_weights), test_files)

rows = []
for stem in sorted(test_inst):
    for k, rle in enumerate(select(test_inst[stem], BEST_CONF, BEST_MIN_AREA), 1):
        rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": rle["counts"].decode("ascii")})
sub = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])

# sanity: unique ids, known images, non-empty 2048² masks
assert sub.filament_id.is_unique
assert set(sub.filament_id.str.rsplit("_", n=1).str[0]) <= {p.stem for p in test_files}
assert all(mask_utils.area({"size": [H, W], "counts": c.encode()}) > 0 for c in sub.segmentation_rle)
sub.to_csv(WORK / "submission.csv", index=False)
shutil.copy(final_weights, WORK / "final_weights.pt")
print(f"submission.csv: {len(sub)} filaments over {sub.filament_id.str.rsplit('_', n=1).str[0].nunique()}/{len(test_files)} images")
sub.head()
