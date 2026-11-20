"""Grid launcher: expands a config into runs and skips anything already finished."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .train import run


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and ("inherits" in v):
            out[k] = _deep_update(out.get(k, {}) if isinstance(out.get(k), dict) else {}, load_config_dict(v))
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config_dict(cfg: dict, here: Path | None = None) -> dict:
    """Resolve a config dict that may inherit from a parent file in the same directory."""
    cfg = copy.deepcopy(cfg)
    parent = cfg.pop("inherits", None)
    if parent and here is not None:
        with open(here / parent) as f:
            base = yaml.safe_load(f)
        cfg = _deep_update(base, cfg)
    return cfg


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return load_config_dict(cfg, Path(path).parent)


def expand(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    grid = cfg.get("grid", {})
    if not grid:
        return [({}, cfg)]
    keys = list(grid.keys())
    runs = []
    for values in itertools.product(*[grid[k] for k in keys]):
        cell = dict(zip(keys, values))
        run_cfg = copy.deepcopy(cfg)
        # every grid key k maps onto a dotted config path by convention:
        #   width/depth/dropout -> model.*   shortcut_correlation -> data.*
        #   seeds -> top-level seed          everything else -> top-level
        keymap = {"seeds": "seed", "width": "model.width", "depth": "model.depth",
                  "arch": "model.arch", "shortcut_correlation": "data.shortcut_correlation",
                  "epochs": "train.epochs"}
        for k, v in cell.items():
            path = keymap.get(k, k)
            node = run_cfg
            parts = path.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = v
        # per-intervention overrides
        if "intervention" in cell:
            overrides = cfg.get("interventions", {}).get(cell["intervention"], {})
            for k, v in overrides.items():
                run_cfg.setdefault("train", {})[k] = v
        runs.append((cell, run_cfg))
    return runs


def run_id_for(cell: Dict[str, Any]) -> str:
    return "-".join(f"{k}={v}" for k, v in sorted(cell.items())).replace("/", "_").replace(" ", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="run only the first N cells")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg.get("out_dir", "results/raw")
    runs = expand(cfg)
    if args.limit:
        runs = runs[: args.limit]
    print(f"[sweep] {len(runs)} runs -> {out_dir}")

    for i, (cell, run_cfg) in enumerate(runs, 1):
        rid = run_id_for(cell)
        if (Path(out_dir) / f"{rid}.done").exists():
            print(f"[sweep] {i}/{len(runs)} skip {rid}"); continue
        print(f"[sweep] {i}/{len(runs)} start {rid}")
        meta = run(run_cfg, rid, out_dir)
        print(f"[sweep] {i}/{len(runs)} done  {rid} params={meta['params']:,}")


if __name__ == "__main__":
    main()
