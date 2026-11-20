"""Per-epoch shortcut diagnostics.

Primary metric SD_t (shortcut dependence), evaluated on a shortcut-only probe set:

    SD_t = mean_i log p_theta(y_i | x_shortcut_only_i) - log(1/K)

SD_t > 0 means the model decodes the label from the shortcut better than chance.
SD_t -> 0 means the model has (mostly) stopped relying on it.

Secondary metrics recorded each epoch:
    acc_id            in-distribution test accuracy
    acc_shortcut      accuracy on shortcut-only probe
    acc_signal        accuracy when the shortcut is erased (counterfactual)
    acc_conflict_follow_shortcut  fraction of conflict images where the model picks
                                  the shortcut's class rather than the shape's class
    mean_conf         mean max-softmax confidence on test
    entropy           mean predictive entropy on test
    grad_norm         gradient L2 norm at the evaluation point
    param_norm        L2 norm of all parameters
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@torch.no_grad()
def _eval_loader(model: nn.Module, loader: DataLoader, device) -> Dict[str, float]:
    model.eval()
    correct = total = 0
    conf_sum = ent_sum = 0.0
    logp_sum = 0.0
    probs_all, labels_all = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        p = F.softmax(logits, dim=1)
        logp = F.log_softmax(logits, dim=1)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
        conf_sum += p.max(1).values.sum().item()
        ent_sum += (-(p * logp).sum(1)).sum().item()
        logp_sum += logp.gather(1, y[:, None]).sum().item()
        probs_all.append(p.cpu())
        labels_all.append(pred.cpu())
    out = {
        "acc": correct / max(total, 1),
        "logp_true": logp_sum / max(total, 1),
        "conf": conf_sum / max(total, 1),
        "entropy": ent_sum / max(total, 1),
    }
    if total:
        out["_probs"] = torch.cat(probs_all)
    return out


def evaluate_epoch(model: nn.Module, loaders: Dict[str, DataLoader], probes: Dict[str, Dataset],
                   device, num_classes: int = 10) -> Dict[str, float]:
    rec: Dict[str, float] = {}

    m = _eval_loader(model, loaders["test"], device)
    rec["acc_id"] = m["acc"]
    rec["conf"] = m["conf"]
    rec["entropy"] = m["entropy"]
    m.pop("_probs", None)

    if "shortcut_only" in probes:
        sl = DataLoader(probes["shortcut_only"], batch_size=256, shuffle=False, num_workers=0)
        s = _eval_loader(model, sl, device)
        rec["acc_shortcut"] = s["acc"]
        rec["logp_shortcut"] = s["logp_true"]
        rec["sd"] = s["logp_true"] - math.log(1.0 / num_classes)

        # early-warning features: is there a *confident but wrong* signal immediately?
        probs = s.get("_probs")
        if probs is not None:
            rec["shortcut_max_prob_mean"] = probs.max(1).values.mean().item()

    if "signal_only" in probes:
        sl = DataLoader(probes["signal_only"], batch_size=256, shuffle=False, num_workers=0)
        rec["acc_signal"] = _eval_loader(model, sl, device)["acc"]

    if "conflict" in probes:
        sl = DataLoader(probes["conflict"], batch_size=256, shuffle=False, num_workers=0)
        short_labels = probes["conflict_shortcut_labels"]
        model.eval()
        follow = tot = 0
        with torch.no_grad():
            for (x, _), sl_lbl in zip(sl, torch.split(torch.as_tensor(short_labels), 256)):
                x = x.to(device)
                pred = model(x).argmax(1).cpu()
                follow += (pred == sl_lbl[: len(pred)]).sum().item()
                tot += len(pred)
        rec["acc_conflict_follow_shortcut"] = follow / max(tot, 1)

    # gradient norm at the current point (a stability proxy)
    model.eval()
    try:
        xs, ys = next(iter(loaders["test"]))
        xs, ys = xs.to(device).requires_grad_(False), ys.to(device)
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xs), ys)
        loss.backward()
        rec["grad_norm"] = math.sqrt(sum((p.grad.float() ** 2).sum().item()
                                         for p in model.parameters() if p.grad is not None))
        model.zero_grad(set_to_none=True)
    except StopIteration:
        rec["grad_norm"] = float("nan")

    rec["param_norm"] = math.sqrt(sum(p.float().pow(2).sum().item() for p in model.parameters()))
    return rec
