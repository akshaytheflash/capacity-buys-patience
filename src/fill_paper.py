"""Fill paper/paper.md result placeholders from results/processed.

The manuscript is the source of truth, and its numbers must never be typed by hand.
This script replaces each `[PENDING EXPERIMENT]` block, and the matching
`[RESULT: <key>]` markers, with generated tables.

Usage:  python -m src.fill_paper --processed results/processed --paper paper/paper.md
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def fmt(x, nd=3):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "n/a"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)


def tau_table(df: pd.DataFrame) -> str:
    """Table 1: release epoch tau by width and correlation level."""
    if df.empty:
        return "*No runs available.*"
    lines = ["| width | ρ | τ (mean) | τ (sd) | final acc | n |",
             "|---|---|---|---|---|---|"]
    for (w, rho), g in df.groupby(["width", "rho"], dropna=False):
        taus = g["tau"].dropna()
        lines.append(
            f"| {w} | {rho} | {fmt(taus.mean(), 2)} | {fmt(taus.std(), 2)} | "
            f"{fmt(g['acc_id_final'].mean(), 4)} | {len(g)} |"
        )
    return "\n".join(lines)


def tau_sensitivity(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c.startswith("tau_d")]
    if not cols:
        return "*No sensitivity columns.*"
    lines = ["| width | " + " | ".join(c.replace("tau_d", "δ=").replace("_p", ", k=") for c in cols) + " |",
             "|---|" + "---|" * len(cols)]
    for w, g in df.groupby("width"):
        vals = " | ".join(fmt(g[c].mean(), 2) for c in cols)
        lines.append(f"| {w} | {vals} |")
    return "\n".join(lines)


def mediation_table(cells: Path) -> str:
    if not cells.exists():
        return "*Mediation analysis has not run yet.*"
    c = pd.read_csv(cells)
    if c.empty:
        return "*No small/large pairs available.*"
    lines = ["| ρ | seed | τ small − large | accuracy gap (large − small) |",
             "|---|---|---|---|"]
    for _, r in c.iterrows():
        lines.append(f"| {r['rho']} | {int(r['seed'])} | {fmt(r['tau_gap'], 2)} | {fmt(r['acc_gap'], 4)} |")
    return "\n".join(lines)


def summary_line(df: pd.DataFrame) -> str:
    if df.empty:
        return "No completed runs."
    return (f"{len(df)} runs across {df['width'].nunique()} widths and "
            f"{df['rho'].nunique()} correlation levels; mean final in-distribution "
            f"accuracy {fmt(df['acc_id_final'].mean(), 4)}.")

def earlywarning_text(path: Path) -> str:
    if not path.exists():
        return "*Early-warning probe has not run yet.*"
    e = pd.read_csv(path)
    if e.empty or "mean_auc" not in e:
        return "*Insufficient runs for the early-warning probe.*"
    r = e.iloc[0]
    if pd.isna(r.get("mean_auc")):
        return f"Not estimable ({r.get('note','')})."
    return (f"Grouped cross-validated AUC = {fmt(r['mean_auc'], 3)} over "
            f"{int(r['n_runs'])} runs and {int(r['n_folds'])} folds "
            f"({int(r['n_features'])} early-training features).")


def fill(paper: Path, processed: Path, out: Path) -> None:
    text = paper.read_text()
    t_csv = processed / "transitions.csv"
    df = pd.read_csv(t_csv) if t_csv.exists() else pd.DataFrame()

    repl = {
        "[RESULT:summary]": summary_line(df),
        "[RESULT:tau_table]": tau_table(df),
        "[RESULT:tau_sensitivity]": tau_sensitivity(df),
        "[RESULT:mediation_table]": mediation_table(processed / "mediation_cells.csv"),
        "[RESULT:earlywarning]": earlywarning_text(processed / "earlywarning.csv"),
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    out.write_text(text)
    print(f"[fill_paper] wrote {out} ({sum(k in paper.read_text() for k in repl)} markers found)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="results/processed")
    ap.add_argument("--paper", default="paper/paper.md")
    ap.add_argument("--out", default="paper/paper_filled.md")
    a = ap.parse_args()
    fill(Path(a.paper), Path(a.processed), Path(a.out))
