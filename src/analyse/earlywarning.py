"""H3: is the release epoch predictable from early-training signals?

Features come only from epochs 0..early_k-1 (default 5). Target: the model releases the
shortcut at all (binary) and tau (regression). Evaluation uses GroupKFold over seeds so
that a model is never tested on a seed it has seen: the question is generalisation,
not memorisation.

This is deliberately a *shallow* probe (logistic regression / small gradient boosting).
If a linear model on epochs 1-5 predicts tau, the early-warning claim is strong; if it
needs a deep net, the claim is weak.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .transition import load_run

FEATURES = ["sd", "acc_id", "conf", "entropy", "grad_norm", "param_norm",
            "logp_shortcut", "acc_shortcut"]


def build_features(raw_dir: Path, early_k: int = 5) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(raw_dir).glob("*.jsonl")):
        r = load_run(p)
        df = r["epochs"]
        if df.empty or len(df) < 2 * early_k:
            continue
        early = df.iloc[:early_k]
        rec = {"run_id": r["run_id"], "seed": r["seed"], "width": r["width"],
               "rho": r["rho"]}
        for f in FEATURES:
            if f not in early:
                continue
            v = early[f].to_numpy(dtype=float)
            rec[f"{f}_mean"] = float(np.nanmean(v))
            rec[f"{f}_slope"] = float(np.polyfit(np.arange(len(v)), v, 1)[0]) if len(v) > 1 else 0.0
        # target: did the model release (SD drops by >20% from early max) at all?
        sd = df["sd"].to_numpy(dtype=float)
        early_max = float(np.nanmax(sd[: max(3, len(sd) // 3)]))
        released = sd[-1] < 0.8 * early_max if np.isfinite(early_max) and early_max > 0 else False
        rec["released"] = int(released)
        rows.append(rec)
    return pd.DataFrame(rows)


def evaluate(feat: pd.DataFrame, out: Path) -> pd.DataFrame:
    cols = [c for c in feat.columns if c.endswith(("_mean", "_slope"))]
    if len(feat) < 10 or len(cols) == 0:
        res = pd.DataFrame([{"note": "insufficient runs for the early-warning probe"}])
        res.to_csv(out, index=False)
        return res
    X = feat[cols].fillna(0.0).to_numpy()
    y = feat["released"].to_numpy()
    groups = feat["seed"].to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    aucs = []
    if len(np.unique(y)) > 1 and n_splits >= 2:
        for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            clf.fit(X[tr], y[tr])
            if len(np.unique(y[te])) > 1:
                aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    res = pd.DataFrame([{"n_runs": len(feat), "n_features": len(cols),
                         "mean_auc": float(np.mean(aucs)) if aucs else np.nan,
                         "n_folds": len(aucs),
                         "note": "GroupKFold over seeds; AUC~0.5 means no early signal"}])
    res.to_csv(out, index=False)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    feat = build_features(Path(a.raw))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res = evaluate(feat, Path(a.out))
    print(res.to_string(index=False))
