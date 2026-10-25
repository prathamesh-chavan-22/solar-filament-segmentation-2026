"""Smoke benchmark on one GPU: training throughput / peak memory per input-pipeline × layout × compile configuration,
then full-image inference time. Synthetic data of the real shape (speed does not depend on pixel values).

Env: ENCODER (efficientnet-b4; comma list to compare encoders), SIZE (2048), CROP (1024), BATCH (4), N_IMAGES (566),
     WARM (25), STEPS (50),
     CONFIGS ("loader-nhwc-compile,gpu-nhwc-compile,gpu-nchw-compile,gpu-nchw-eager"), PRED_CHUNKS ("1,2,4").
"""
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from filament import unet  # noqa: E402

env = os.environ.get
SIZE, CROP, BATCH = int(env("SIZE", 2048)), int(env("CROP", 1024)), int(env("BATCH", 4))
N, WARM, STEPS = int(env("N_IMAGES", 566)), int(env("WARM", 25)), int(env("STEPS", 50))
CONFIGS = env("CONFIGS", "loader-nhwc-compile,gpu-nhwc-compile,gpu-nchw-compile,gpu-nchw-eager").split(",")
dev = torch.device("cuda")
print(torch.cuda.get_device_name(0), torch.__version__, f"{os.cpu_count()} CPUs", flush=True)

rng = np.random.default_rng(0)
image = rng.standard_normal((SIZE, SIZE)).astype(np.float16)
target = (rng.random((SIZE, SIZE)) < 0.05).astype(np.float16)
images, targets = [image] * N, [target] * N


def batches(source, seed=0):
    if source == "gpu":
        crops = unet.GpuCrops(images, targets, CROP, dev, seed)
        while True:
            yield crops.batch(BATCH)
    dl = DataLoader(unet.CropDataset(images, targets, CROP, 10 ** 6, seed), batch_size=BATCH, num_workers=4,
                    worker_init_fn=lambda w: setattr(torch.utils.data.get_worker_info().dataset, "rng",
                                                     np.random.default_rng(seed + 1 + w)))
    yield from dl


def bench_train(cfg, encoder):
    source, layout, mode = cfg.split("-")
    nhwc = layout == "nhwc"
    torch.cuda.empty_cache(), torch.cuda.reset_peak_memory_stats()
    model = unet.build_model(encoder).to(dev).train()
    params = sum(p.numel() for p in model.parameters()) / 1e6
    net = unet.speedups(model, dev, channels_last=nhwc, compile_model=mode == "compile")
    fmt = torch.channels_last if nhwc else torch.contiguous_format
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    scaler = torch.amp.GradScaler()
    it, wait = batches(source), 0.0
    for step in range(WARM + STEPS):
        if step == WARM:
            torch.cuda.synchronize()
            t0, wait = time.time(), 0.0
        t = time.time()
        x, y = next(it)
        x, y = x.to(dev, non_blocking=True).to(memory_format=fmt), y.to(dev, non_blocking=True)
        wait += time.time() - t
        with torch.autocast("cuda", dtype=torch.float16):
            loss = unet.loss_fn(net(x).float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    sec = (time.time() - t0) / STEPS
    print(f"train {encoder} ({params:.0f}M) {cfg:22s} {sec:.3f} s/step  {BATCH / sec:5.1f} crops/s  data wait {wait / STEPS:.3f} s/step  "
          f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB", flush=True)
    del it, net, model, opt
    return sec


def bench_predict(encoder):
    model = unet.build_model(encoder).to(dev).eval()
    x = rng.standard_normal((SIZE, SIZE)).astype(np.float32)
    for layout, chunk in itertools.product(("nchw", "nhwc"), map(int, env("PRED_CHUNKS", "1,2,4").split(","))):
        model.to(memory_format=torch.channels_last if layout == "nhwc" else torch.contiguous_format)
        torch.cuda.empty_cache(), torch.cuda.reset_peak_memory_stats()
        try:
            unet.predict_prob(model, x, dev, 8, chunk)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(3):
                unet.predict_prob(model, x, dev, 8, chunk)
            torch.cuda.synchronize()
            print(f"predict {encoder} {layout} chunk {chunk}: {(time.time() - t0) / 3:.2f} s/image (8 TTA views)  "
                  f"peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB", flush=True)
        except torch.OutOfMemoryError:
            print(f"predict {encoder} {layout} chunk {chunk}: out of memory", flush=True)


if __name__ == "__main__":
    for encoder in env("ENCODER", "efficientnet-b4").split(","):
        for cfg in CONFIGS:
            try:
                bench_train(cfg, encoder)
            except torch.OutOfMemoryError:
                print(f"train {encoder} {cfg}: out of memory", flush=True)
            except Exception as e:  # unsupported encoder etc.: report and go on with the next one
                print(f"train {encoder} {cfg}: {type(e).__name__}: {e}", flush=True)
            torch._dynamo.reset()
        try:
            bench_predict(encoder)
        except Exception as e:
            print(f"predict {encoder}: {type(e).__name__}: {e}", flush=True)
