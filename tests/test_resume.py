"""Regression test: checkpoint/resume must be *exact*.

This test exists because three separate bugs were found while writing it:
  1. the JSONL was opened in "w" mode, truncating earlier epochs on restart;
  2. a shortened config compressed the LR schedule, changing results on resume;
  3. setting sched.last_epoch does not refresh the optimizer LR, so the first
     resumed epoch trained at the wrong rate.

It is deliberately a *fast* end-to-end test (~40 s on CPU) rather than a unit test of a
helper, because all three bugs lived in the interaction between the loop and the file
layout, which unit tests would have mocked away.

Run:  python -m tests.test_resume
Exit code 0 means resume is bit-identical.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "results" / "_resume_test"

BASE = """
inherits: base.yaml
out_dir: {out}
device: cpu
train:
  epochs: {epochs}
  batch_size: 64
data:
  num_train: 1024
  num_test: 256
  num_shortcut_probe: 256
  num_shortcut_only_test: 256
  num_workers: 0
grid:
  seeds: [0]
  width: [0.25]
  depth: [10]
model:
  width: 0.25
  depth: 10
"""


def _write(name: str, out: str, epochs: int, stop: int | None = None) -> Path:
    p = ROOT / "configs" / name
    body = BASE.format(out=out, epochs=epochs)
    if stop is not None:
        body = body.replace("train:\n", f"train:\n  stop_at_epoch: {stop}\n")
    p.write_text(body)
    return p


def _epochs(out: Path) -> list[dict]:
    f = sorted(out.glob("*.jsonl"))[0]
    return [json.loads(x) for x in f.read_text().splitlines() if json.loads(x)["event"] == "epoch"]


def _run(cfg: Path) -> None:
    subprocess.run([sys.executable, "-m", "src.sweep", "--config", str(cfg)],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


def main() -> int:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    full = WORK / "full"
    part = WORK / "part"

    _run(_write("_rt_full.yaml", str(full), epochs=3))
    # same canonical length, but stop after epoch 1, then resume
    _run(_write("_rt_part.yaml", str(part), epochs=3, stop=1))
    assert not (part / f"{sorted(part.glob('*.jsonl'))[0].stem}.done").exists(), \
        "partial run must not be marked done"
    _run(_write("_rt_resume.yaml", str(part), epochs=3))

    a, b = _epochs(full), _epochs(part)
    if len(a) != len(b):
        print(f"FAIL: epoch count {len(a)} vs {len(b)}"); return 1
    for r1, r2 in zip(a, b):
        for k in ("sd", "train_loss", "lr"):
            if abs(r1[k] - r2[k]) > 1e-9:
                print(f"FAIL: epoch {r1['epoch']} field {k}: {r1[k]} vs {r2[k]}"); return 1
    # no duplicated start/done events, no truncated prefix
    f = sorted(part.glob("*.jsonl"))[0]
    rows = [json.loads(x) for x in f.read_text().splitlines()]
    if sum(r["event"] == "start" for r in rows) != 1:
        print("FAIL: start event duplicated or missing"); return 1
    if [r["epoch"] for r in rows if r["event"] == "epoch"] != list(range(len(a))):
        print("FAIL: epoch sequence is not contiguous 0..N-1"); return 1

    print(f"PASS: resume is bit-identical across {len(a)} epochs")
    shutil.rmtree(WORK, ignore_errors=True)
    for n in ("_rt_full.yaml", "_rt_part.yaml", "_rt_resume.yaml"):
        (ROOT / "configs" / n).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
