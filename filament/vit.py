"""DINOv2 ViT encoder (frozen, LoRA-adapted) + light CNN detail path → U-Net-style decoder.

A ViT sees 14×14 patches, but filaments can be a few pixels wide, so semantic features from four ViT depths
(stride 14) are fused, upsampled and merged with a small trainable CNN branch at strides 4 and 2 before the
full-resolution head.  Only LoRA adapters, the CNN branch and the decoder are trained.
"""
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

BACKBONES = {
    "dinov2-s": "vit_small_patch14_dinov2.lvd142m",
    "dinov2-b": "vit_base_patch14_dinov2.lvd142m",
    "dinov2-l": "vit_large_patch14_dinov2.lvd142m",
    "dinov2-g": "vit_giant_patch14_dinov2.lvd142m",
}


class LoRALinear(nn.Module):
    """y = W x + (α/r) · B A x with W frozen; B starts at zero so training begins from the pretrained function."""

    def __init__(self, base: nn.Linear, rank=16, alpha=16):
        super().__init__()
        self.base, self.scale = base, alpha / rank
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def forward(self, x):
        return self.base(x) + self.b(self.a(x)) * self.scale


def add_lora(vit, rank):
    for blk in vit.blocks:
        blk.attn.qkv = LoRALinear(blk.attn.qkv, rank)
        blk.attn.proj = LoRALinear(blk.attn.proj, rank)
        blk.mlp.fc1 = LoRALinear(blk.mlp.fc1, rank)
        blk.mlp.fc2 = LoRALinear(blk.mlp.fc2, rank)


def conv_bn(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ViTUNet(nn.Module):
    def __init__(self, name="dinov2-b", pretrained=True, lora_rank=16, dec=128):
        super().__init__()
        self.vit = timm.create_model(BACKBONES[name], pretrained=pretrained, num_classes=0,
                                     dynamic_img_size=True, dynamic_img_pad=True)
        self.vit.requires_grad_(False)
        add_lora(self.vit, lora_rank)
        self.vit.set_grad_checkpointing(True)
        depth, dim = len(self.vit.blocks), self.vit.embed_dim
        self.taps = [depth // 4 - 1, depth // 2 - 1, 3 * depth // 4 - 1, depth - 1]
        self.fuse = nn.Sequential(nn.Conv2d(4 * dim, 2 * dec, 1, bias=False), nn.BatchNorm2d(2 * dec), nn.ReLU(True))
        self.stem2 = nn.Sequential(conv_bn(1, 32, 2), conv_bn(32, 32))        # stride 2
        self.stem4 = nn.Sequential(conv_bn(32, 64, 2), conv_bn(64, 64))       # stride 4
        self.dec4 = nn.Sequential(conv_bn(2 * dec + 64, dec), conv_bn(dec, dec))
        self.dec2 = nn.Sequential(conv_bn(dec + 32, 64), conv_bn(64, 64))
        self.head = nn.Sequential(conv_bn(64 + 1, 32), nn.Conv2d(32, 1, 1))

    def forward(self, x):
        h, w = x.shape[-2:]
        feats = self.vit.forward_intermediates(x.expand(-1, 3, -1, -1), indices=self.taps, intermediates_only=True)
        sem = self.fuse(torch.cat(feats, 1))                                   # (B, 2·dec, ⌈h/14⌉, ⌈w/14⌉)
        s2 = self.stem2(x)
        s4 = self.stem4(s2)
        y = self.dec4(torch.cat([F.interpolate(sem, size=s4.shape[-2:], mode="bilinear", align_corners=False), s4], 1))
        y = self.dec2(torch.cat([F.interpolate(y, size=s2.shape[-2:], mode="bilinear", align_corners=False), s2], 1))
        y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
        return self.head(torch.cat([y, x], 1))
