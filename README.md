# Does Capacity Buy Patience?

**Working title:** *Capacity buys patience: model size, not inductive bias, controls how long a network stays trapped on a shortcut*

A low-compute empirical study of how model **capacity** interacts with the **timing** of
shortcut acquisition and release. Designed to run end-to-end on Kaggle free GPUs
(T4 / P100, ~30 h/week).

---

## Why this project exists

Recent work (arXiv 2603.07323, *Norm-Hierarchy Transitions*) explains **when and why** a
network abandons a shortcut: weight decay slowly traverses a hierarchy of parameter
norms. That work does *not* vary capacity. A very recent preprint
(arXiv 2610.12397) shows OOD degradation is *predictable* from early training dynamics,
but again does not manipulate capacity.

The open question this project asks:

> When several models of different capacity are trained on the same shortcut-laden data
> with the same optimiser budget, does capacity change **when** the network abandons the
> shortcut — and is the capacity gap in final accuracy caused by that timing?

We deliberately frame this as a **capacity × resources** question. Compute, dataset size
and training length are collinear with capacity in practice, so we sweep them explicitly
rather than assuming capacity is the operative variable.

---

## Hypotheses

| ID | Hypothesis | Design | Falsified if |
|----|-----------|--------|--------------|
| **H1** | Release epoch $\tau$ (the epoch at which the shortcut is abandoned) is a non-monotonic or monotonic function of capacity; larger models commit **later** and release **later**. | Capacity sweep at fixed resources | $\tau$ flat in capacity |
| **H2** | The final-accuracy gap between small and large models is **mediated** by $\tau$: intervening to delay commitment in small models closes part of the gap. | Intervention sweep | interventions move $\tau$ but not accuracy |
| **H3** | $\tau$ is predictable from early-training signals (epochs 1–5) with an **early-warning score**. | Logistic regression / shallow probe on early features | AUC ≈ 0.5 |
| **H4** | Part of the apparent "capacity effect" is actually a **compute / dataset-size** effect (confound test). | Resource-matched controls | effect survives full resource matching |

H1–H4 are preregistered here **before** any results exist. Deviations will be listed in
`paper/DEVIATIONS.md`.

---

## Repository layout

```
capacity-buys-patience/
├── README.md                  ← you are here
├── requirements.txt
├── Makefile
├── configs/                   ← one YAML per experiment arm
│   ├── base.yaml
│   ├── capacity.yaml
│   ├── resources.yaml
│   └── interventions.yaml
├── src/
│   ├── data/                  ← controlled shortcut dataset + natural datasets
│   │   ├── controlled.py
│   │   └── natural.py
│   ├── models/
│   │   └── factory.py         ← ResNet / ViT-tiny width & depth grids
│   ├── train.py               ← single run entry point (Kaggle-friendly)
│   ├── diagnose.py            ← shortcut-dependence metrics
│   ├── sweep.py               ← grid launcher (reads configs/*.yaml)
│   ├── analyse/
│   │   ├── transition.py      ← τ estimation
│   │   ├── mediation.py       ← H2 mediation analysis
│   │   └── earlywarning.py    ← H3 early-warning probe
│   └── figures.py             ← paper figures from processed results
├── scripts/
│   ├── kaggle_run.sh          ← one-command Kaggle session
│   ├── build_paper.sh         ← markdown → docx + pdf
│   └── check_no_placeholders.sh
├── results/
│   ├── raw/                   ← per-run JSONL (never edited by hand)
│   ├── processed/             ← tidy CSVs used by the paper
│   └── figures/
└── paper/
    ├── paper.md               ← source of truth for the manuscript
    ├── refs.bib
    ├── DEVIATIONS.md
    ├── capacity-buys-patience.docx
    └── capacity-buys-patience.pdf
```

---

## Quick start

```bash
pip install -r requirements.txt

# smoke test: 2 epochs, tiny model, synthetic data (CPU, ~1 min)
make smoke

# full capacity sweep (GPU recommended)
make capacity

# analysis + figures
make analyse
make figures

# rebuild the paper
make paper
```

### On Kaggle

```bash
# in a Kaggle notebook cell
!git clone <your-repo> && cd capacity-buys-patience && bash scripts/kaggle_run.sh capacity
```

`scripts/kaggle_run.sh` resumes from whatever is already in `results/raw/`, so a
disconnected session never loses completed runs.

---

## Experiment design (summary)

**Data.**
1. `ControlledShortcut` — generated, fully controllable: a colour patch, a texture patch
   and a background patch compete with the true shape signal. Correlation level
   $\rho \in \{0.85, 0.95, 0.99\}$ is a knob, so we know the ground-truth shortcut.
2. `CIFAR-10` with a synthetic textured watermark (secondary, sanity check).
3. Optionally Waterbirds for a natural-distribution check (no new dataset creation).

**Models.** ResNet-style CNNs with width multiplier
$\{0.25, 0.5, 1.0, 2.0\}$ and depth $\{10, 18, 34\}$; one ViT-tiny for architecture
generality. All models see the same number of *examples*, not just epochs, in the
resource-matched arm.

**Primary measurement.** Shortcut dependence per epoch:
$$\mathrm{SD}_t = \mathbb{E}_{x\sim D_{\text{shortcut-only}}}\,\big[\log \hat{p}(y_{\text{shortcut}} \mid x)\big] - \log(1/|\mathcal{Y}|)$$
i.e. how much the model relies on the shortcut-only probe set. $\tau$ is the first epoch
where $\mathrm{SD}_t$ drops below $(1-\delta)$ of its early maximum, sustained for $k$
epochs. $\delta$ and $k$ are sensitivity-tested.

**Interventions (H2).** warm-up length, weight decay, learning-rate schedule, label
smoothing, and early-checkpoint self-distillation — each applied to the smallest model
only, then compared to the largest model's final accuracy.

**Early-warning (H3).** Features from epochs 1–5: loss trajectory, SD$_t$ slope, gradient
norm, parameter norm, entropy. Model: logistic regression with grouped CV over seeds.
Report AUC and calibration; no rehearsal of the test seeds.

**Confound arm (H4).** Small and large models matched on *examples seen*,
*wall-clock compute* (approximate, via FLOPs), and *update count*, separately.

**Statistics.** ≥ 5 seeds per cell; report mean ± 95% bootstrap CI; paired comparisons
where seeds are shared. Pre-registered primary contrasts to avoid p-hacking.

---

## Compute budget

| Arm | Runs | Est. T4-hours |
|---|---|---|
| Smoke + debugging | ~20 | 1 |
| Controlled, capacity sweep (3 ρ × 4 widths × 3 seeds) | 36 | 12 |
| Resources arm | 30 | 8 |
| Interventions (5 × 3 seeds, small model only) | 15 | 4 |
| CIFAR watermark sanity | 12 | 4 |
| Early-warning (reuses capacity runs) | 0 | 0 |
| **Total** | | **≈ 29 h** |

Fits comfortably in one to two weeks of Kaggle quota.

---

## Status

- [x] Project scaffolding, configs, paper skeleton
- [ ] Smoke test green on CPU
- [ ] Controlled sweep run
- [ ] Analysis + figures
- [ ] Results pasted into the paper (no manual editing of numbers)

> **Current state of the manuscript:** the paper contains the full argument, method,
> design and analysis plan, with results sections deliberately left as
> `[PENDING EXPERIMENT]`. No numbers are invented anywhere. See
> `paper/DEVIATIONS.md` for what is intentionally missing.

---

## Licence

MIT for code; manuscript text is the author's.

---

## Build notes (this machine)

- pandoc 2.5 has no `--citeproc`; the draft keeps `[@key]` citation markers and the
  `.bib` as the source of record. For a submission build use pandoc ≥ 2.11 or a Zotero
  export. Recorded in `paper/DEVIATIONS.md`.
- Smoke test verified on CPU: `python -m src.sweep --config configs/smoke.yaml`
  (308K-param model, 2 epochs, ~6 s).
