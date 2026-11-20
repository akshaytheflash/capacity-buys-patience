"""Single-run training entry point.

Writes one JSON object per epoch to results/raw/<run_id>.jsonl, plus a final
<run_id>.done marker so the sweep can resume safely after a Kaggle disconnect.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data.controlled import ControlledConfig, build_controlled, build_cifar_watermark
from .diagnose import evaluate_epoch
from .models.factory import build_model, count_params, approx_flops_per_example


def _build_data(cfg: Dict[str, Any], seed: int):
    d = cfg["data"]
    if d["name"] == "controlled":
        dc = ControlledConfig(
            seed=seed, image_size=d["image_size"], num_classes=d["num_classes"],
            rho=float(d["shortcut_correlation"]), shortcut_strength=float(d["shortcut_strength"]),
        )
        return build_controlled(
            dc, num_train=d["num_train"], num_test=d["num_test"],
            num_shortcut_probe=d["num_shortcut_probe"],
            num_shortcut_only_test=d["num_shortcut_only_test"],
            num_workers=d["num_workers"], batch_size=cfg["train"]["batch_size"],
        )
    if d["name"] == "cifar10_watermark":
        return build_cifar_watermark(d["root"], seed, float(d["shortcut_correlation"]),
                                     d["num_workers"], cfg["train"]["batch_size"], d["download"])
    raise ValueError(f"unknown dataset {d['name']}")


def _lr_lambda(scheduler: str, warmup: int, total: int, base_lr: float):
    def fn(epoch: int) -> float:
        if warmup and epoch < warmup:
            return (epoch + 1) / warmup
        if scheduler == "cosine":
            t = (epoch - warmup) / max(1, total - warmup)
            return 0.5 * (1 + math.cos(math.pi * t))
        if scheduler == "step":
            return 0.1 ** (epoch // max(1, total // 3))
        return 1.0
    return fn


def run(cfg: Dict[str, Any], run_id: str, out_dir: str | Path) -> Dict[str, Any]:
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    device = cfg.get("device", "auto")
    device = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else
                          ("cpu" if device == "auto" else device))

    loaders, probes = _build_data(cfg, seed)
    mcfg = dict(cfg["model"]); mcfg["num_classes"] = cfg["data"]["num_classes"]
    model = build_model(mcfg).to(device)

    tcfg = cfg["train"]
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]),
                            weight_decay=float(tcfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _lr_lambda(tcfg["scheduler"], int(tcfg["warmup_epochs"]), int(tcfg["epochs"]),
                        float(tcfg["lr"]))
    )

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / f"{run_id}.jsonl"
    done = out_dir / f"{run_id}.done"

    meta = {
        "run_id": run_id, "seed": seed, "device": str(device),
        "params": count_params(model),
        "flops_per_example_est": approx_flops_per_example(model, cfg["data"]["image_size"]),
        "config": cfg,
    }
    with open(jsonl, "w") as f:
        f.write(json.dumps({"event": "start", **meta}) + "\n")

        seen_examples = 0
        max_examples = tcfg.get("max_examples")
        for epoch in range(int(tcfg["epochs"])):
            model.train()
            ep_loss = 0.0; n = 0
            for x, y in loaders["train"]:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=float(tcfg["label_smoothing"]))
                loss.backward()
                opt.step()
                ep_loss += loss.item() * y.numel(); n += y.numel()
                seen_examples += y.numel()
            sched.step()

            rec = evaluate_epoch(model, loaders, probes, device, cfg["data"]["num_classes"])
            rec.update({"event": "epoch", "epoch": epoch, "train_loss": ep_loss / max(n, 1),
                        "lr": opt.param_groups[0]["lr"], "examples_seen": seen_examples,
                        "ts": time.time()})
            f.write(json.dumps(rec) + "\n"); f.flush()

            if max_examples is not None and seen_examples >= int(max_examples):
                break

        f.write(json.dumps({"event": "done", **{k: v for k, v in meta.items() if k != "config"}}) + "\n")
    done.write_text("ok")
    return meta
