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
    out_dir_p = Path(out_dir)
    state_path = out_dir_p / f"{run_id}.state.pt"
    done_path = out_dir_p / f"{run_id}.done"
    canonical_path = out_dir_p / f"{run_id}.canonical.json"

    # The canonical schedule length lives with the run, not with whichever config file
    # launched it. A 2-epoch probe and the final 4-epoch run are the *same* run if they
    # share a run_id; the LR curve must therefore stay 4 epochs long throughout.
    out_dir_p.mkdir(parents=True, exist_ok=True)
    requested = int(tcfg["epochs"])
    prev_epochs = -1
    if canonical_path.exists():
        prev_epochs = int(json.loads(canonical_path.read_text())["epochs"])
    # Monotone rule: the canonical length is the longest this run has ever been asked
    # to train for. Shorter runs stop early via `stop_at_epoch` (see below) and must
    # NEVER shorten the LR curve, because the curve depends on the total, not the plan.
    epochs_for_sched = max(requested, prev_epochs)
    if prev_epochs != epochs_for_sched:
        canonical_path.write_text(json.dumps({"epochs": epochs_for_sched, "run_id": run_id}))
    # Planned stop: train only up to this epoch, keeping the full-length schedule.
    stop_at = tcfg.get("stop_at_epoch")
    planned_end = epochs_for_sched if stop_at is None else min(int(stop_at), epochs_for_sched)

    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _lr_lambda(tcfg["scheduler"], int(tcfg["warmup_epochs"]), epochs_for_sched,
                        float(tcfg["lr"]))
    )

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / f"{run_id}.jsonl"
    done = done_path
    # atomic state file: the single source of truth for resuming this run

    meta = {
        "run_id": run_id, "seed": seed, "device": str(device),
        "params": count_params(model),
        "flops_per_example_est": approx_flops_per_example(model, cfg["data"]["image_size"]),
        "config": cfg,
    }

    # The canonical schedule length lives with the run, not with whatever config file
    # happened to launch it. (Maintained above; nothing to do here.)

    # ---- resume support -------------------------------------------------
    start_epoch = 0
    seen_examples = 0
    if state_path.exists() and not done.exists():
        ck = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        # Restore only the scheduler *counter*, never its lambda: the lambda must come
        # from epochs_for_sched, or a shortened predecessor config compresses the
        # remaining LR curve and silently changes results.
        sched.last_epoch = int(ck.get("sched_last_epoch", ck["epoch"] + 1))
        # Setting last_epoch does NOT refresh the optimizer's LR (the scheduler only
        # writes it inside step()). Set it explicitly so the first resumed epoch trains
        # at the same rate an uninterrupted run would have used.
        _lam = _lr_lambda(tcfg["scheduler"], int(tcfg["warmup_epochs"]),
                          epochs_for_sched, float(tcfg["lr"]))
        for pg in opt.param_groups:
            pg["lr"] = float(pg.get("initial_lr", tcfg["lr"])) * _lam(sched.last_epoch)
        start_epoch = int(ck["epoch"]) + 1
        seen_examples = int(ck.get("examples_seen", 0))
        print(f"[resume] {run_id}: continuing at epoch {start_epoch} "
              f"(schedule total={epochs_for_sched})")

    # Deterministic batch order: reseed and skip the batches already consumed so that
    # a resumed run sees exactly the sequence an uninterrupted run would have seen.
    # NOTE: we re-seed *per epoch* (seed + epoch) and only skip within the epoch that
    # was interrupted, so the skip count is bounded by one epoch of batches.
    batches_per_epoch = len(loaders["train"])

    def _epoch_batches(epoch: int, skip_within_epoch: int = 0):
        torch.manual_seed(seed + epoch)
        it = iter(loaders["train"])
        for _ in range(skip_within_epoch):
            try:
                next(it)
            except StopIteration:
                break
        return it

    with open(jsonl, "a") as f:
        if start_epoch == 0 and jsonl.stat().st_size == 0:
            f.write(json.dumps({"event": "start", **meta}) + "\n")

        max_examples = tcfg.get("max_examples")
        for epoch in range(start_epoch, planned_end):
            model.train()
            ep_loss = 0.0; n = 0
            epoch_lr = opt.param_groups[0]["lr"]   # LR actually used this epoch
            for x, y in _epoch_batches(epoch):
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
                        "lr": epoch_lr, "examples_seen": seen_examples,
                        "ts": time.time()})
            f.write(json.dumps(rec) + "\n"); f.flush()

            # ---- durable checkpoint: write to tmp then atomically replace ----
            tmp = state_path.with_suffix(".pt.tmp")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "sched_last_epoch": sched.last_epoch,
                        "epoch": epoch,
                        "examples_seen": seen_examples, "config": cfg, "run_id": run_id}, tmp)
            os.replace(tmp, state_path)   # atomic on POSIX: never a half-written file
            _maybe_drive_sync(state_path)

            if max_examples is not None and seen_examples >= int(max_examples):
                break

        finalised = planned_end >= epochs_for_sched
        if finalised:
            f.write(json.dumps({"event": "done", **{k: v for k, v in meta.items() if k != "config"}}) + "\n")
    if finalised:
        done.write_text("ok")
    return meta


def _maybe_drive_sync(path: Path) -> None:
    """On Colab, flush checkpoints to Drive so a VM reset cannot lose progress."""
    drive = os.environ.get("CBPP_DRIVE_DIR")
    if not drive:
        return
    try:
        import shutil
        dst = Path(drive); dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst / path.name)
        jsonl = path.with_suffix(".jsonl")
        if jsonl.exists():
            shutil.copy2(jsonl, dst / jsonl.name)
    except Exception as e:  # never let syncing kill training
        print(f"[drive-sync] skipped: {e}")
