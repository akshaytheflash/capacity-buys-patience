"""Model factory: width/depth grids for the capacity sweep."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.short = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch)
            )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        o = F.relu(self.bn1(self.conv1(x)))
        o = self.drop(o)
        o = self.bn2(self.conv2(o))
        return F.relu(o + self.short(x))


class ResNetN(nn.Module):
    """Height-configurable ResNet: depth in {10, 18, 34}, width multiplier in {0.25..2}."""

    def __init__(self, depth: int = 18, width: float = 1.0, num_classes: int = 10,
                 dropout: float = 0.0):
        super().__init__()
        # depth is counted in conv layers (10 -> blocks [1,1,1,1], 18 -> [2,2,2,2])
        n_blocks = {10: [1, 1, 1, 1], 18: [2, 2, 2, 2], 34: [3, 4, 6, 3]}[depth]
        base = int(64 * width)
        self.base = base
        self.in_ch = base
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 3, 1, 1, bias=False), nn.BatchNorm2d(base), nn.ReLU(inplace=True)
        )
        self.layer1 = self._make(n_blocks[0], base, 1, dropout)
        self.layer2 = self._make(n_blocks[1], base * 2, 2, dropout)
        self.layer3 = self._make(n_blocks[2], base * 4, 2, dropout)
        self.layer4 = self._make(n_blocks[3], base * 8, 2, dropout)
        self.head = nn.Linear(base * 8, num_classes)

    def _make(self, n, out_ch, stride, dropout):
        blocks = [BasicBlock(self.in_ch, out_ch, stride, dropout)]
        self.in_ch = out_ch
        for _ in range(n - 1):
            blocks.append(BasicBlock(out_ch, out_ch, 1, dropout))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.head(x)


class Patch(nn.Module):
    def __init__(self, patch: int, dim: int, embed: int):
        super().__init__()
        self.p = patch
        self.proj = nn.Conv2d(dim, embed, kernel_size=patch, stride=patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class ViTTiny(nn.Module):
    """Minimal ViT for architecture-generality checks. Width scales with `width`."""

    def __init__(self, width: float = 1.0, num_classes: int = 10, img_size: int = 32,
                 patch: int = 4, dropout: float = 0.0):
        super().__init__()
        embed = max(64, int(192 * width))
        depth = 4
        heads = 4 if embed % 4 == 0 else 2
        self.patch = Patch(patch, 3, embed)
        n_tokens = (img_size // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, embed))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens + 1, embed))
        layer = nn.TransformerEncoderLayer(embed, heads, embed * 4, dropout,
                                          batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(embed)
        self.head = nn.Linear(embed, num_classes)

    def forward(self, x):
        x = self.patch(x)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1) + self.pos
        x = self.norm(self.blocks(x))
        return self.head(x[:, 0])


def build_model(cfg: dict) -> nn.Module:
    arch = cfg.get("arch", "resnet")
    if arch == "vit_tiny":
        return ViTTiny(width=cfg.get("width", 1.0), num_classes=cfg.get("num_classes", 10),
                       dropout=cfg.get("dropout", 0.0))
    return ResNetN(depth=cfg.get("depth", 18), width=cfg.get("width", 1.0),
                   num_classes=cfg.get("num_classes", 10), dropout=cfg.get("dropout", 0.0))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def approx_flops_per_example(model: nn.Module, img_size: int = 32) -> float:
    """Rough FLOP estimate (2 x MACs) for the ResNet family. Used for FLOP matching only."""
    total = 0.0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            total += math.prod(m.weight.shape) * (img_size ** 2) * 2
        elif isinstance(m, nn.Linear):
            total += math.prod(m.weight.shape) * 2
    return total
