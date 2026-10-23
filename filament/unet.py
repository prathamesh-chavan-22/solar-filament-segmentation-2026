"""Semantic U-Net: disk-normalised input, random-crop training on soft annotator-mean targets, D4 TTA inference."""
import math
import os
import time

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def preprocess(gray, size):
    """2048² uint8 → size² float32, normalised by the mean/std of the solar disk (off-disk set to 0)."""
    img = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    disk = img > 0.1 * img.max()
    mean, std = img[disk].mean(), img[disk].std() + 1e-6
    out = (img - mean) / std
    out[~disk] = 0
    return out


def d4(x, k):
    """k ∈ [0, 8): rotation by 90°·(k % 4), then horizontal flip if k ≥ 4. Works on (..., H, W)."""
    x = torch.rot90(x, k % 4, dims=(-2, -1))
    return torch.flip(x, dims=(-1,)) if k >= 4 else x


def d4_inverse(x, k):
    if k >= 4:
        x = torch.flip(x, dims=(-1,))
    return torch.rot90(x, -(k % 4), dims=(-2, -1))


class CropDataset(Dataset):
    """Random ``crop``² windows with random D4 transform; length = number of samples per training run."""

    def __init__(self, images, targets, crop, length, seed=0):
        self.images, self.targets, self.crop, self.length = images, targets, crop, length
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        i = self.rng.integers(len(self.images))
        size = self.images[i].shape[0]
        y, x = self.rng.integers(0, size - self.crop + 1, 2)
        img = torch.from_numpy(self.images[i][y:y + self.crop, x:x + self.crop].astype(np.float32))[None]
        tgt = torch.from_numpy(self.targets[i][y:y + self.crop, x:x + self.crop].astype(np.float32))[None]
        k = int(self.rng.integers(8))
        return d4(img, k).contiguous(), d4(tgt, k).contiguous()


def build_model(encoder="resnet34", pretrained=True):
    return smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None, in_channels=1, classes=1)


def loss_fn(logits, target):
    """BCE on soft targets + 0.5 · soft Dice (per batch)."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * target).sum() + 1) / (p.sum() + target.sum() + 1)
    return bce + 0.5 * dice


def speedups(model, device, channels_last=True, compile_model=False):
    """cuDNN autotune + TF32, NHWC layout (tensor-core friendly convs), optional torch.compile (Triton kernels).

    Returns the module to call in the training loop; ``model`` itself keeps the uncompiled weights for saving.
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    if channels_last:
        model.to(memory_format=torch.channels_last)
    return torch.compile(model) if compile_model and device.type == "cuda" else model


def train(model, images, targets, *, steps, batch, crop, lr, device, log_every=200, seed=0,
          channels_last=False, compile_model=False, amp_dtype=torch.float16):
    ds = CropDataset(images, targets, crop, steps * batch, seed)
    # Forked workers inherit identical copies of ds.rng; reseed each so they don't sample the same crops.
    dl = DataLoader(ds, batch_size=batch, num_workers=0 if os.name == "nt" else 4,
                    worker_init_fn=lambda w: setattr(torch.utils.data.get_worker_info().dataset, "rng",
                                                     np.random.default_rng(seed + 1 + w)),
                    persistent_workers=os.name != "nt")
    model.to(device).train()
    net = speedups(model, device, channels_last, compile_model) if (channels_last or compile_model) else model
    fmt = torch.channels_last if channels_last else torch.contiguous_format
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, fused=device.type == "cuda")
    warmup = max(steps // 20, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / (steps - warmup))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda" and amp_dtype == torch.float16)
    t0, running = time.time(), 0.0
    for step, (x, y) in enumerate(dl, 1):
        x = x.to(device, non_blocking=True).to(memory_format=fmt)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            loss = loss_fn(net(x).float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        running += loss.item()
        if step % log_every == 0 or step == steps:
            print(f"step {step}/{steps} loss {running / log_every:.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            running = 0.0
    # Inference runs uncompiled; eager NHWC depthwise convs are ~9× slower than NCHW on consumer GPUs.
    return model.to(memory_format=torch.contiguous_format).eval()


@torch.no_grad()
def predict_prob(model, image, device, tta=8, chunk=8):
    """image: preprocessed (S, S) float32 → mean sigmoid probability over ``tta`` D4 views, (S, S) float32.

    ``chunk`` views go through the network at once (lower it on small GPUs).
    """
    x = torch.from_numpy(image)[None, None].to(device)
    acc = torch.zeros_like(x)
    for k0 in range(0, tta, chunk):
        ks = range(k0, min(k0 + chunk, tta))
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            probs = torch.sigmoid(model(torch.cat([d4(x, k) for k in ks])).float())
        for j, k in enumerate(ks):
            acc += d4_inverse(probs[j:j + 1], k)
    return (acc / tta)[0, 0].cpu().numpy()
