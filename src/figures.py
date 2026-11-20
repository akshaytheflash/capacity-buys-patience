"""Paper figures generated entirely from results/processed.

Every figure recognises missing data and renders an explicit placeholder rather than
a misleading empty plot: the manuscript is built before the experiments finish.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PENDING = "[PENDING EXPERIMENT]"


def _placeholder(ax, msg=PENDING):
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11, color="0.45",
            transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])


def fig_release_curve(raw_dir: Path, out: Path) -> None:
    from .analyse.transition import load_run
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    runs = sorted(raw_dir.glob("*.jsonl")) if raw_dir.exists() else []
    plotted = False
    if runs:
        d = load_run(runs[0])
        for p in runs:
            r = load_run(p)
            if r["rho"] != d["rho"]:
                continue
            df = r["epochs"]
            if df.empty:
                continue
            ax.plot(df["epoch"], df["sd"], label=f"w={r['width']}", lw=1.2)
            plotted = True
    if plotted:
        ax.axhline(0, color="0.6", ls="--", lw=0.8)
        ax.set_xlabel("epoch"); ax.set_ylabel("shortcut dependence  $SD_t$")
        ax.legend(fontsize=7, frameon=False)
    else:
        _placeholder(ax)
        ax.set_title("Shortcut dependence over training", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "fig_release_curve.pdf"); plt.close(fig)


def fig_tau_vs_capacity(csv: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    if csv.exists():
        df = pd.read_csv(csv)
        if not df.empty and "tau" in df and df["tau"].notna().any():
            for rho, g in df.groupby("rho"):
                m = g.groupby("width")["tau"].agg(["mean", "std"]).reset_index()
                ax.errorbar(m["width"], m["mean"], yerr=m["std"], marker="o", capsize=3,
                            label=f"ρ={rho}")
            ax.set_xscale("log", base=2)
            ax.set_xlabel("width multiplier"); ax.set_ylabel(r"release epoch $\tau$")
            ax.legend(fontsize=7, frameon=False)
        else:
            _placeholder(ax)
    else:
        _placeholder(ax)
    ax.set_title("Release epoch vs. capacity", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "fig_tau_vs_capacity.pdf"); plt.close(fig)


def fig_mediation(csv: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    cells = Path(str(csv).replace(".csv", "_cells.csv"))
    if cells.exists():
        df = pd.read_csv(cells)
        if not df.empty:
            ax.scatter(df["tau_gap"], df["acc_gap"], s=28)
            ax.set_xlabel(r"$\tau_{small}-\tau_{large}$")
            ax.set_ylabel("accuracy gap (large − small)")
        else:
            _placeholder(ax)
    else:
        _placeholder(ax)
    ax.set_title("Release-timing gap vs. accuracy gap", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "fig_mediation.pdf"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="results/processed")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--raw", default="results/raw/capacity")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    proc = Path(a.processed)
    fig_release_curve(Path(a.raw), out)
    fig_tau_vs_capacity(proc / "transitions.csv", out)
    fig_mediation(proc / "mediation.csv", out)
    print(f"[figures] wrote 3 figures -> {out} (placeholders where data is missing)")


if __name__ == "__main__":
    main()
