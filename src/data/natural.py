"""Test doubles allowing the smoke test to run without torch/torchvision installed.

The Kaggle environment has torch; local lint/smoke checks may not.
"""
from __future__ import annotations


def build_cifar_watermark(*a, **k):
    raise NotImplementedError("torchvision not installed in this environment")
