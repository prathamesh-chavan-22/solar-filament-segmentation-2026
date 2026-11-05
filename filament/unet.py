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


class GpuCrops:
    """The whole training set resident on the GPU; random ``crop``² windows + D4 drawn on the device.

    Replaces the DataLoader (no CPU slicing/copies per step). Both are stored as uint8 so 566 native 2048² images take
    ~4.7 GB: disk-normalised images as round(16·x) + 128 (step σ/16, clipped at ±8σ; about the raw 8-bit precision),
    soft targets as round(255·t) (error ≤ 0.002).
    """

    def __init__(self, images, targets, crop, device, seed=0, aug=False):
        n, size = len(images), images[0].shape[0]
        self.images = torch.empty((n, size, size), dtype=torch.uint8, device=device)
        self.targets = torch.empty((n, size, size), dtype=torch.uint8, device=device)
        for i, (im, t) in enumerate(zip(images, targets)):
            self.images[i] = torch.from_numpy(np.clip(np.rint(np.asarray(im, np.float32) * 16) + 128, 0, 255)
                                              .astype(np.uint8))
            self.targets[i] = torch.from_numpy(np.rint(np.asarray(t, np.float32) * 255).astype(np.uint8))
        self.crop, self.size, self.aug = crop, size, aug
        self.rng = np.random.default_rng(seed)

    def batch(self, b):
        if self.aug:
            return self._augmented(b)
        c, xs, ys = self.crop, [], []
        for i, y, x, k in zip(self.rng.integers(len(self.images), size=b), self.rng.integers(0, self.size - c + 1, b),
                              self.rng.integers(0, self.size - c + 1, b), self.rng.integers(8, size=b)):
            xs.append(d4(self.images[i, y:y + c, x:x + c], k))
            ys.append(d4(self.targets[i, y:y + c, x:x + c], k))
        return (torch.stack(xs)[:, None].float() - 128) / 16, torch.stack(ys)[:, None].float() / 255

    def _augmented(self, b):
        """D4 + any-angle rotation, scale 0.85–1.15 (bilinear resampling of a larger window), contrast / brightness
        jitter, and on 30 % of crops each a Gaussian blur (seeing) and additive noise."""
        c, r = self.crop, self.rng
        w = min(int(np.ceil(c * 1.15 * np.sqrt(2))) + 2, self.size)  # window that covers the rotated, scaled crop
        dev = self.images.device
        xs, ys = [], []
        for i, y, x, k in zip(r.integers(len(self.images), size=b), r.integers(0, self.size - w + 1, b),
                              r.integers(0, self.size - w + 1, b), r.integers(8, size=b)):
            xs.append(d4(self.images[i, y:y + w, x:x + w], k))
            ys.append(d4(self.targets[i, y:y + w, x:x + w], k))
        x = (torch.stack(xs)[:, None].float() - 128) / 16
        t = torch.stack(ys)[:, None].float() / 255
        ang = torch.from_numpy(r.uniform(-np.pi, np.pi, b)).float()
        scale = torch.from_numpy(r.uniform(0.85, 1.15, b)).float() * (c / w)  # output c px ← c·scale input px
        cos, sin = torch.cos(ang) * scale, torch.sin(ang) * scale
        theta = torch.stack([torch.stack([cos, -sin, torch.zeros(b)], 1), torch.stack([sin, cos, torch.zeros(b)], 1)], 1)
        grid = F.affine_grid(theta.to(dev), (b, 1, c, c), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", align_corners=False)
        t = F.grid_sample(t, grid, mode="bilinear", align_corners=False)
        on_disk = (x != 0).float()  # keep off-disk pixels at exactly 0, as in preprocess
        gain = torch.from_numpy(r.uniform(0.8, 1.25, (b, 1, 1, 1))).float().to(dev)
        bias = torch.from_numpy(r.uniform(-0.2, 0.2, (b, 1, 1, 1))).float().to(dev)
        x = x * gain + bias
        for j in np.flatnonzero(r.random(b) < 0.3):
            sigma = r.uniform(0.5, 1.5)
            k1 = torch.exp(-0.5 * (torch.arange(-3, 4, device=dev, dtype=torch.float32) / sigma) ** 2)
            k1 = k1 / k1.sum()
            xj = F.conv2d(F.pad(x[j:j + 1], (3, 3, 0, 0), mode="reflect"), k1.view(1, 1, 1, 7))
            x[j:j + 1] = F.conv2d(F.pad(xj, (0, 0, 3, 3), mode="reflect"), k1.view(1, 1, 7, 1))
        noisy = torch.from_numpy(r.random((b, 1, 1, 1)) < 0.3).float().to(dev)
        x = x + noisy * torch.randn_like(x) * float(r.uniform(0.02, 0.1))
        return (x * on_disk).contiguous(), t.contiguous()


def build_model(encoder="resnet34", pretrained=True):
    """smp U-Net for CNN encoders; ``dinov2-{s,b,l,g}`` gives the LoRA-adapted ViT U-Net (filament.vit)."""
    if encoder.startswith("dinov2"):
        from .vit import ViTUNet
        return ViTUNet(encoder, pretrained=pretrained, lora_rank=int(os.environ.get("LORA_RANK", 16)))
    model = smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None, in_channels=1, classes=1)
    if os.environ.get("GRAD_CKPT", "0") == "1":  # recompute encoder activations in backward: ~half the memory, ~30 % slower
        model.encoder.model.set_grad_checkpointing(True)
    return model


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
          channels_last=False, compile_model=False, amp_dtype=torch.float16, gpu_data=False, aug=False):
    if gpu_data:
        crops = GpuCrops(images, targets, crop, device, seed, aug)
        dl = (crops.batch(batch) for _ in range(steps))
    else:
        ds = CropDataset(images, targets, crop, steps * batch, seed)
        # Forked workers inherit identical copies of ds.rng; reseed each so they don't sample the same crops.
        dl = DataLoader(ds, batch_size=batch, num_workers=0 if os.name == "nt" else 4,
                        worker_init_fn=lambda w: setattr(torch.utils.data.get_worker_info().dataset, "rng",
                                                         np.random.default_rng(seed + 1 + w)),
                        persistent_workers=os.name != "nt")
    model.to(device).train()
    net = speedups(model, device, channels_last, compile_model) if (channels_last or compile_model) else model
    fmt = torch.channels_last if channels_last else torch.contiguous_format
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4,
                            fused=device.type == "cuda")
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
