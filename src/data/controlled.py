"""Controlled shortcut datasets.

We generate a *controllable* vision problem in which we know the ground-truth
shortcut, its strength, and its correlation with the label. This is what lets us
measure "when does the network abandon the shortcut" without guessing.

Task:
    Input  : 32x32x3 image
    Signal : a 4x4 "shape" patch rendered at a random interior location.
             The shape identity is the true label y in {0..9}.
    Shortcut: a coloured 8x8 patch in a fixed corner whose *colour* encodes y.
    Correlation rho: with probability rho the colour is the correct code for y,
             otherwise it is a random other code. rho=0.99 is very strong evidence,
             rho=0.85 is weaker.

Probe sets produced alongside training data:
    train             : used for training (contains both signal and shortcut)
    test              : both signal and shortcut consistent  -> in-distribution accuracy
    shortcut_probe    : SHORTCUT ONLY (no shape), used to measure reliance
    shortcut_only_test: shortcut only, held out   -> "would shortcut alone suffice?"
    signal_only_test  : shape only (no shortcut), no colour -> counterfactual accuracy
    conflict_test     : shape says A, shortcut says B. What does the model pick?

The headline metric SD_t (shortcut dependence) is computed on `shortcut_probe`:
    SD = mean(log p(y | shortcut-only image)) - log(1/K)
i.e. how confidently the model decodes the label from a shortcut-only input.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


SHAPES = 10
PALETTE = np.array(
    [
        [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255],
        [0, 255, 255], [255, 128, 0], [128, 0, 255], [0, 128, 255], [128, 255, 0],
    ],
    dtype=np.uint8,
)

# Each shape is a 4x4 binary template, one per class.
TEMPLATES = _t = np.zeros((SHAPES, 4, 4), dtype=np.uint8)  # placeholder replaced below


def _make_templates() -> np.ndarray:
    t = np.zeros((SHAPES, 4, 4), dtype=np.uint8)
    t[0, :, 0] = 1                    # vertical bar left
    t[1, :, 3] = 1                    # vertical bar right
    t[2, 0, :] = 1                    # horizontal bar top
    t[3, 3, :] = 1                    # horizontal bar bottom
    t[4, np.arange(4), np.arange(4)] = 1   # diagonal
    t[5, np.arange(4), np.arange(4)[::-1]] = 1  # anti-diagonal
    t[6, 0, :] = 1; t[6, 3, :] = 1    # two horizontal bars
    t[7, :, 0] = 1; t[7, :, 3] = 1    # two vertical bars
    t[8, :, :] = 1                    # solid block
    t[9, 0, 0] = 1                    # single dot
    return t


TEMPLATES = _make_templates()


@dataclass
class ControlledConfig:
    seed: int = 0
    image_size: int = 32
    num_classes: int = 10
    rho: float = 0.95
    shortcut_strength: float = 1.0
    noise: float = 0.08
    shape_size: int = 4
    shortcut_size: int = 8


def _render_batch(cfg: ControlledConfig, n: int, rng: np.random.Generator,
                  correlation_override: float | None = None,
                  conflict: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render n images. Returns (images uint8 [n,32,32,3], labels, shortcut_codes)."""
    H = W = cfg.image_size
    imgs = np.zeros((n, H, W, 3), dtype=np.float32)
    imgs += rng.normal(0, cfg.noise * 255, size=imgs.shape).clip(-255, 255)

    y = rng.integers(0, cfg.num_classes, size=n)
    rho = cfg.rho if correlation_override is None else correlation_override

    # shortcut colour code: usually correct, sometimes random other class
    use_correct = rng.random(n) < rho
    codes = y.copy()
    wrong = rng.integers(0, cfg.num_classes, size=n)
    codes[~use_correct] = wrong[~use_correct]

    # in the conflict set, keep the shortcut colour consistent with y but
    # invert the shape, so the two signals disagree
    shape_ids = y.copy()

    ss = cfg.shape_size
    for i in range(n):
        # --- true signal: the shape, random interior location
        y0 = int(rng.integers(2, H - ss - 2))
        x0 = int(rng.integers(2, W - ss - 2))
        tpl = TEMPLATES[shape_ids[i]] * 255
        for c in range(3):
            patch = imgs[i, y0:y0 + ss, x0:x0 + ss, c]
            imgs[i, y0:y0 + ss, x0:x0 + ss, c] = np.maximum(patch, tpl)
        # --- shortcut: colour block in fixed corner, strength alpha
        col = PALETTE[codes[i]].astype(np.float32) * cfg.shortcut_strength
        imgs[i, 0:cfg.shortcut_size, 0:cfg.shortcut_size, :] = col

    imgs = imgs.clip(0, 255).astype(np.uint8)
    if conflict:
        # labels stay as the true shape-code mapping; conflict is expressed by
        # flipped shape vs. shortcut colour already handled by caller
        pass
    return imgs, shape_ids.astype(np.int64), codes.astype(np.int64)


def _render_shortcut_only(cfg: ControlledConfig, n: int, rng: np.random.Generator,
                          labels: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Images containing ONLY the shortcut colour block and noise: no shape."""
    H = W = cfg.image_size
    imgs = rng.normal(0, cfg.noise * 255, size=(n, H, W, 3)).astype(np.float32)
    if labels is None:
        labels = rng.integers(0, cfg.num_classes, size=n)
    for i in range(n):
        imgs[i, 0:cfg.shortcut_size, 0:cfg.shortcut_size, :] = (
            PALETTE[labels[i]].astype(np.float32) * cfg.shortcut_strength
        )
    return imgs.clip(0, 255).astype(np.uint8), labels.astype(np.int64)


def _render_signal_only(cfg: ControlledConfig, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Images with the shape but the shortcut block *erased to background*."""
    imgs, y, _ = _render_batch(cfg, n, rng)
    imgs[:, 0:cfg.shortcut_size, 0:cfg.shortcut_size, :] = 0
    return imgs, y


def _render_conflict(cfg: ControlledConfig, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shape says y, shortcut colour says a different class. Returns (imgs, y_shape, y_shortcut)."""
    imgs, y, codes = _render_batch(cfg, n, rng, correlation_override=1.0)
    # rewrite the colour block so it encodes a *different* class
    wrong = (y + rng.integers(1, cfg.num_classes, size=n)) % cfg.num_classes
    for i in range(n):
        imgs[i, 0:cfg.shortcut_size, 0:cfg.shortcut_size, :] = (
            PALETTE[wrong[i]].astype(np.float32) * cfg.shortcut_strength
        )
    return imgs, y, wrong


class TensorDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        self.y = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]


def build_controlled(cfg: ControlledConfig, num_train: int, num_test: int,
                     num_shortcut_probe: int, num_shortcut_only_test: int,
                     num_workers: int = 2, batch_size: int = 128):
    """Build every split of the controlled task and return loaders + probe tensors."""
    rng = np.random.default_rng(cfg.seed)
    tr_x, tr_y, _ = _render_batch(cfg, num_train, rng)
    te_x, te_y, _ = _render_batch(cfg, num_test, rng)
    pr_x, _ = _render_shortcut_only(cfg, num_shortcut_probe, rng)
    so_x, so_y = _render_shortcut_only(cfg, num_shortcut_only_test, rng)
    # signal_only is a *counterfactual* of the SAME test images, with the
    # shortcut erased. Rendering it from a fresh RNG draw (as an earlier version
    # did) makes it an identical copy of `test`, which silently inflates acc_signal.
    sig_x = te_x.copy()
    sig_x[:, 0:cfg.shortcut_size, 0:cfg.shortcut_size, :] = 0
    sig_y = te_y
    cf_x, cf_y_sig, cf_y_short = _render_conflict(cfg, num_test, rng)

    probes = {
        "signal_only": TensorDataset(sig_x, sig_y),
        "shortcut_only": TensorDataset(so_x, so_y),
        "conflict": TensorDataset(cf_x, cf_y_sig),
        "conflict_shortcut_labels": cf_y_short,
    }
    loaders = {
        # shuffle=True draws from torch's *global* RNG, which train.py seeds per epoch.
        # That keeps batch order reproducible across a checkpoint/resume without needing
        # a per-loader generator (a fresh generator would replay epoch 0 every epoch).
        "train": DataLoader(TensorDataset(tr_x, tr_y), batch_size=batch_size, shuffle=True,
                            num_workers=0, drop_last=True),
        "test": DataLoader(TensorDataset(te_x, te_y), batch_size=256, shuffle=False,
                           num_workers=num_workers),
    }
    return loaders, probes


def build_cifar_watermark(root: str, seed: int, rho: float, num_workers: int = 2,
                          batch_size: int = 128, download: bool = True):
    """CIFAR-10 with a synthetic colourful watermark whose colour encodes the label.

    Kept deliberately simple: correlation rho as in the controlled task.
    """
    from torchvision import datasets, transforms

    class _WM(Dataset):
        def __init__(self, base, rng, rho):
            self.base = base
            self.rng = rng
            self.rho = rho

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            x, y = self.base[i]
            x = x.numpy()
            code = y if self.rng.random() < self.rho else int(self.rng.integers(0, 10))
            x[:, :6, :6] = torch.from_numpy(PALETTE[code].astype(np.float32) / 255.0)[:, None, None]
            return torch.from_numpy(x), y

    tf = transforms.Compose([transforms.ToTensor()])
    train_base = datasets.CIFAR10(root, train=True, download=download, transform=tf)
    test_base = datasets.CIFAR10(root, train=False, download=download, transform=tf)
    rng = np.random.default_rng(seed)
    train = _WM(train_base, rng, rho)
    test = _WM(test_base, rng, rho)
    loaders = {
        "train": DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            drop_last=True),
        "test": DataLoader(test, batch_size=256, shuffle=False, num_workers=num_workers),
    }
    # No clean counterfactual splits here: reported as a sanity check only.
    probes = {}
    return loaders, probes
