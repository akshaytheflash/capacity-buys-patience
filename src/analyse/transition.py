"""Estimate the shortcut-release epoch tau from per-epoch diagnostics.

SD_t is a noisy curve. We define tau as the first epoch at which SD_t emerges from the
"early maximum" and stays down for `patience` consecutive epochs:

    early_max = max SD_t over the first third of training
    tau       = min { t : SD_s < (1 - delta) * early_max  for all s in [t, t+patience) }

Sensitivity over delta and patience is reported rather than picking a single setting.
The same function is used for the in-run delta/patience and for the robustness table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def load_run(path: Path) -> Dict:
    rows, meta = [], {}
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if rec["event"] == "start":
            meta = rec
        elif rec["event"] == "epoch":
            rows.append(rec)
    cfg = meta.get("config", {})
    return {"run_id": meta.get("run_id", path.stem), "meta": meta,
            "params": meta.get("params"), "epochs": pd.DataFrame(rows),
            "width": cfg.get("model", {}).get("width"),
            "depth": cfg.get("model", {}).get("depth"),
            "rho": cfg.get("data", {}).get("shortcut_correlation"),
            "seed": meta.get("seed")}


def estimate_tau(sd: np.ndarray, delta: float = 0.2, patience: int = 3) -> float:
    if len(sd) == 0 or np.all(np.isnan(sd)):
        return float("nan")
    early = sd[: max(3, len(sd) // 3)]
    early_max = float(np.nanmax(early))
    if not np.isfinite(early_max) or early_max <= 0:
        return float("nan")
    thr = (1.0 - delta) * early_max
    below = sd < thr
    for t in range(len(sd) - patience):
        if below[t: t + patience].all():
            return float(t)
    return float("nan")  # never released within the run


def summarise(raw_dir: str | Path, out: str | Path) -> pd.DataFrame:
    recs: List[Dict] = []
    for p in sorted(Path(raw_dir).glob("*.jsonl")):
        r = load_run(p)
        df = r["epochs"]
        if df.empty:
            continue
        rec = {"run_id": r["run_id"], "params": r["params"], "width": r["width"],
               "depth": r["depth"], "rho": r["rho"], "seed": r["seed"],
               "n_epochs": len(df),
               "sd_early_max": float(df["sd"].iloc[: max(3, len(df) // 3)].max()),
               "sd_final": float(df["sd"].iloc[-1]),
               "acc_id_final": float(df["acc_id"].iloc[-1]),
               "acc_shortcut_final": float(df["acc_shortcut"].iloc[-1])
               if "acc_shortcut" in df else np.nan,
               "acc_signal_final": float(df["acc_signal"].iloc[-1])
               if "acc_signal" in df else np.nan,
               "follow_shortcut_final": float(df["acc_conflict_follow_shortcut"].iloc[-1])
               if "acc_conflict_follow_shortcut" in df else np.nan}
        for delta in (0.1, 0.2, 0.3):
            for patience in (2, 3, 5):
                rec[f"tau_d{delta}_p{patience}"] = estimate_tau(df["sd"].to_numpy(), delta, patience)
        rec["tau"] = rec["tau_d0.2_p3"]
        recs.append(rec)
    out_df = pd.DataFrame(recs)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    return out_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = summarise(a.raw, a.out)
    print(f"[transition] wrote {len(df)} rows -> {a.out}")
    if not df.empty:
        print(df.groupby(["width", "rho"], dropna=False)[["tau", "acc_id_final"]].mean().round(3))
