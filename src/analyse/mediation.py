"""H2: is the small/large accuracy gap mediated by the release epoch tau?

For every (rho, seed) cell we compare the small width to the large width and ask:
    total gap        = acc_large - acc_small
    tau difference   = tau_small - tau_large
Correlation between them is the first evidence. The intervention arm provides the
causal half: if interventions that move tau also move accuracy in the same direction,
the mediation story holds.

This module deliberately reports the raw effect sizes and a bootstrap CI rather than a
fitted causal model: with a few dozen runs, SEM software would over-claim.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SMALL, LARGE = 0.25, 2.0


def _boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def analyse(transitions_csv: str, out: str) -> pd.DataFrame:
    df = pd.read_csv(transitions_csv)
    if df.empty:
        res = pd.DataFrame([{"note": "no transitions yet"}])
        res.to_csv(out, index=False); return res

    rows = []
    for rho, g in df.groupby("rho"):
        for seed, h in g.groupby("seed"):
            s = h[np.isclose(h["width"], SMALL)]
            l = h[np.isclose(h["width"], LARGE)]
            if s.empty or l.empty:
                continue
            rows.append({
                "rho": rho, "seed": seed,
                "acc_gap": float(l["acc_id_final"].mean() - s["acc_id_final"].mean()),
                "tau_gap": float(s["tau"].mean() - l["tau"].mean()),
                "tau_small": float(s["tau"].mean()), "tau_large": float(l["tau"].mean()),
            })
    cell = pd.DataFrame(rows)
    if cell.empty:
        res = pd.DataFrame([{"note": "small/large pairs missing"}])
        res.to_csv(out, index=False); return res

    corr = float("nan")
    if len(cell) > 2 and cell["tau_gap"].std() > 0 and cell["acc_gap"].std() > 0:
        corr = float(cell["tau_gap"].corr(cell["acc_gap"]))

    lo, hi = _boot_ci(cell["acc_gap"].to_numpy())
    res = pd.DataFrame([{
        "n_cells": len(cell),
        "mean_acc_gap_large_minus_small": float(cell["acc_gap"].mean()),
        "acc_gap_ci95_lo": lo, "acc_gap_ci95_hi": hi,
        "mean_tau_small_minus_large": float(cell["tau_gap"].mean()),
        "corr_tau_gap_acc_gap": corr,
        "note": "mediation is supported if corr>0 AND interventions move both; see paper",
    }])
    cell.to_csv(str(out).replace(".csv", "_cells.csv"), index=False)
    res.to_csv(out, index=False)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", dest="raw", default=None)
    ap.add_argument("--transitions", default="results/processed/transitions.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = analyse(a.transitions, a.out)
    print(res.to_string(index=False))
