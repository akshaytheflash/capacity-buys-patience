"""Figures for the paper, generated only from results/processed.

Every figure reads the per-epoch logs in results/processed/runs/ (or the summary CSV),
so nothing here is illustrative or schematic. If the data are missing, the figure is
skipped rather than drawn empty.

Usage:  python -m src.paper_figures
Output: results/figures/fig1_trajectories.pdf
        results/figures/fig2_settled_vs_width.pdf
        results/figures/fig3_span_vs_width.pdf
"""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = Path("results/processed/runs")
OUT = Path("results/figures")

WIDTH_STYLE = {0.25: ("C0", "o"), 0.5: ("C1", "s"), 1.0: ("C2", "^"), 2.0: ("C3", "D")}


def load_runs():
    """Return {rho: {width: [epoch dicts]}} from the per-epoch JSONL logs."""
    data = defaultdict(lambda: defaultdict(list))
    for f in sorted(RUNS.glob("*.jsonl")):
        base = f.name.rstrip(".jsonl")
        rho = float(re.search(r"correlation=([\d.]+)", base).group(1))
        w = float(re.search(r"width=([\d.]+)", base).group(1))
        eps = [json.loads(l) for l in f.read_text().splitlines() if '"epoch"' in l]
        if eps:
            data[rho][w] = eps
    return data


def fig1_trajectories(data, out: Path) -> str | None:
    """Shortcut dependence over training, one panel per correlation level."""
    if not data:
        return None
    rhos = sorted(data)
    fig, axes = plt.subplots(1, len(rhos), figsize=(3.1 * len(rhos), 2.9), sharey=True)
    if len(rhos) == 1:
        axes = [axes]
    for ax, rho in zip(axes, rhos):
        for w, eps in sorted(data[rho].items()):
            sd = [e["sd"] for e in eps]
            color, _ = WIDTH_STYLE.get(w, ("k", "o"))
            ax.plot(range(len(sd)), sd, color=color, lw=1.3, label=f"w={w}")
        ax.axhline(0, color="0.55", ls=":", lw=0.8)
        ax.set_title(f"$\\rho$ = {rho}", fontsize=9)
        ax.set_xlabel("epoch", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel(r"shortcut dependence  $SD_t$", fontsize=8)
    axes[-1].legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    p = out / "fig1_trajectories.png"; fig.savefig(p, dpi=300); plt.close(fig)
    return p.name


def _summary(data):
    rows = []
    for rho, by_w in data.items():
        for w, eps in by_w.items():
            sd = [e["sd"] for e in eps]
            rows.append({"rho": rho, "w": w, "sd_late": float(np.mean(sd[-5:])),
                         "span": float(max(sd) - min(sd)), "n": len(sd)})
    return rows


def fig2_settled(data, out: Path) -> str | None:
    """Settled shortcut dependence vs width: the null result."""
    rows = [r for r in _summary(data) if r["n"] >= 25]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    for rho in sorted({r["rho"] for r in rows}):
        g = sorted([r for r in rows if r["rho"] == rho], key=lambda r: r["w"])
        ax.plot([r["w"] for r in g], [r["sd_late"] for r in g], marker="o",
                lw=1.3, label=f"$\\rho$={rho}")
    ax.axhline(0, color="0.55", ls=":", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("width multiplier", fontsize=8)
    ax.set_ylabel(r"settled $SD$ (mean of last 5 epochs)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    p = out / "fig2_settled_vs_width.png"; fig.savefig(p, dpi=300); plt.close(fig)
    return p.name


def fig3_span(data, out: Path) -> str | None:
    """The finding: transient excursion grows with width."""
    rows = [r for r in _summary(data) if r["n"] >= 25]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    for rho in sorted({r["rho"] for r in rows}):
        g = sorted([r for r in rows if r["rho"] == rho], key=lambda r: r["w"])
        ax.plot([r["w"] for r in g], [r["span"] for r in g], marker="s",
                lw=1.3, label=f"$\\rho$={rho}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("width multiplier", fontsize=8)
    ax.set_ylabel(r"$SD$ span  (max $-$ min)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    p = out / "fig3_span_vs_width.png"; fig.savefig(p, dpi=300); plt.close(fig)
    return p.name


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_runs()
    made = [f for f in (fig1_trajectories(data, OUT), fig2_settled(data, OUT),
                        fig3_span(data, OUT)) if f]
    print(f"[paper_figures] {len(made)} figures from {len(glob.glob(str(RUNS / '*.jsonl')))} run logs -> {OUT}")
    for m in made:
        print("  ", m)


if __name__ == "__main__":
    main()
