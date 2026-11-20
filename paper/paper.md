---
title: "Capacity Buys Patience: Model Size Controls When Networks Abandon Shortcuts"
author: "[Author] — [Institution] — Draft v0.1, results pending"
date: "November 2026"
abstract: |
  Deep networks frequently learn a spurious shortcut before the intended signal and
  abandon it only after many epochs. Recent work attributes this delay to the slow
  traversal of a parameter-norm hierarchy under weight decay, and separate work shows
  that out-of-distribution degradation is predictable from early training dynamics.
  Neither line of work varies **capacity**. We ask whether capacity changes *when* a
  network abandons a shortcut, and whether the usual small-versus-large accuracy gap is
  caused by that timing rather than by representational power. We build a controlled
  vision task in which the ground-truth shortcut, its strength and its correlation with
  the label are all knobs; we then sweep width and depth over hundreds of training runs
  and measure the *shortcut-release epoch* each time. We preregister four hypotheses:
  that release epoch scales with capacity (H1); that interventions which delay release
  close part of the capacity gap (H2); that release epoch is predictable from the first
  five epochs (H3); and that part of the apparent capacity effect is in fact a compute
  or dataset-size effect (H4). Results are pending. The contribution is a measurement
  protocol together with a causal test, not a new architecture.
  **Keywords:** shortcut learning, training dynamics, model capacity, spurious
  correlation, reproducibility, small-scale experiments
---

# Introduction

Deep networks find the easiest explanation of their training data, not the one we
intended. When a colour patch, a background texture or a watermark correlates with the
label, the network will typically learn that correlation and generalise poorly to images
where it is absent [@geirhos2020shortcut]. This is *shortcut learning*, and it is one of
the most practically consequential failure modes in applied machine learning.

Recently, two lines of work have changed the picture. @nht2026 explain *why* the
transition from shortcut to structured representation is slow: gradient descent with
weight decay traverses a hierarchy of parameter norms, and the model moves from a
high-norm shortcut solution to a lower-norm structured one only gradually. They derive a
bound on the delay. @ood2026 show, separately, that persistent out-of-distribution
degradation is predictable from source-side training dynamics: temporal summaries of
early training carry a prospective warning signal.

Neither study varies **capacity**. This is a striking omission, because in practice the
quantity a practitioner controls most directly is model size, and the folklore is
imprecise. One story says larger models generalise better because they have the capacity
to represent the intended concept; another says larger models overfit shortcuts *more*
readily. Both stories are about *what* a model learns. Neither is about *when*.

## The question

We therefore ask a question that is simple to state and, as far as we can determine,
has not been asked directly:

> **Holding data, optimiser and budget fixed, does model capacity change the epoch at
> which a network abandons a shortcut — and is the capacity gap in final accuracy
> caused by that timing?**

If capacity buys *patience*, a small model commits to its first explanation early and
cannot escape, while a larger model keeps enough slack that weight decay can later
rearrange it. On that reading the small-model failure is a failure of *timing*, and
interventions that delay commitment should partially substitute for scale. That is a
testable, falsifiable claim, and it has a concrete practical corollary: if true, the
cheapest fix for a small model on shortcut-prone data is not more parameters but a
schedule that keeps it from crystallising too soon.

## Contributions

1. **A controlled task with a ground-truth shortcut** in which shortcut strength,
   correlation and the presence of a counterfactual can be set exactly, together with a
   probe-based definition of the *shortcut-release epoch* $\tau$ that does not require
   inspecting weights.
2. **A capacity sweep** over width and depth at fixed data and optimiser budget, with
   explicitly preregistered directional hypotheses (H1–H4) and sensitivity analysis on
   the definition of $\tau$.
3. **A causal test** (H2): five interventions applied to the smallest model, asking
   whether moving $\tau$ moves accuracy.
4. **An early-warning probe** (H3) that asks whether $\tau$ is predictable from the first
   five epochs, and a **resource-matching arm** (H4) that separates capacity from
   compute and data, which are collinear with capacity in practice.
5. **Full reproducibility**: all code, configs and derived tables; the paper's numbers
   are generated from the artefacts, never typed by hand.

## What this paper is not

We do not propose a new architecture or a new regulariser, and we do not claim a
mechanism-level theory to rival the norm-hierarchy account. We are testing whether a
specific variable that is missing from that account — capacity — has the effect that
practitioners assume it has.

# Related Work

**Shortcut learning.** @geirhos2020shortcut systematise how networks latch onto
spurious cues, and the simplicity bias literature explains why easy features are learned
first. Recent work studies *monitoring* shortcut dependence during training, and a body
of work on spurious correlations (group robustness, worst-group accuracy) studies how to
*mitigate* the dependence once it is identified. Our probe-based measure sits in the
monitoring line but is used differently: as the dependent variable in a capacity sweep.

**Timing of representation change.** @nht2026 provide the closest account of *when*
shortcuts are abandoned, attributing the delay to a norm hierarchy under regularisation.
Our design is complementary: holding their mechanism in place, we manipulate capacity
and ask whether the delay changes. A positive result would extend their framework; a
null result would constrain it.

**Prediction from training dynamics.** @ood2026 demonstrate that future OOD failure is
predictable from early source-side dynamics in a controlled shortcut setting. Our H3
asks a related but distinct question — whether the *release epoch itself* is predictable
across a capacity sweep — and uses grouped cross-validation over seeds so that the
probe cannot memorise a seed.

**Capacity and generalisation.** The double-descent and inductive-bias literatures study
how model size and implicit bias change *which* solutions are found. We do not re-derive
those results; we use them as context for the prediction that capacity should also change
the *rate* at which solutions are revised.

**Small-scale studies.** Our design follows the tradition of studies that are small by
deliberate design because the controlled setting makes a clean causal comparison
possible, reporting effect sizes with confidence intervals rather than champion
accuracies.

# Method

## Controlled shortcut task

Each $32\times32$ RGB image contains two candidate explanations of the label
$y \in \{0,\dots,9\}$:

- **The intended signal:** a $4\times4$ binary template, one per class, drawn at a
  random interior position.
- **The shortcut:** an $8\times8$ colour block in a fixed corner whose colour encodes
  $y$.

The shortcut is present in a fraction $\rho$ of examples (we use $\rho \in \{0.85, 0.95,
0.99\}$); in the remaining $(1-\rho)$ the colour encodes a uniformly random other class.
$\rho$ therefore sets how much evidence the shortcut provides, and the model's
behaviour under a probe with the shortcut alone is directly interpretable.

Four additional splits are generated from the same generative process:

| Split | Contents | Used for |
|---|---|---|
| `train` | signal + shortcut | training |
| `test` | signal + shortcut | in-distribution accuracy |
| `signal_only` | the **same** test images with the shortcut erased | counterfactual accuracy |
| `shortcut_only` | colour block only, no shape | shortcut dependence $\mathrm{SD}_t$ |
| `conflict` | shape says $a$, colour says $b \neq a$ | which signal wins |

The counterfactual split is constructed by erasing the shortcut from the test images
themselves, not by drawing fresh images. An earlier version of the code drew a fresh
batch, which silently duplicated the test set and would have inflated counterfactual
accuracy; the fix is recorded in `CHANGELOG.md`.

## Measuring shortcut dependence

Let $\hat p_\theta(y \mid x)$ be the model's predictive distribution and $K$ the number of
classes. We define per-epoch shortcut dependence on the shortcut-only probe as

$$\mathrm{SD}_t \;=\; \frac{1}{N}\sum_{i=1}^{N} \log \hat p_{\theta_t}\!\left(y_i \mid x^{\text{shortcut}}_i\right) \;-\; \log \frac{1}{K}.$$

$\mathrm{SD}_t > 0$ means the model decodes the label from the shortcut better than
chance; $\mathrm{SD}_t \to 0$ means it has stopped relying on it. Because it is computed
on a probe with no intended signal, it cannot be inflated by legitimate learning.

## Defining the release epoch

$\mathrm{SD}_t$ is noisy, so we do not read a crossing point off a single epoch. Let
$m = \max_{t < T/3} \mathrm{SD}_t$ be the early maximum. The release epoch is

$$\tau \;=\; \min\left\{t \;:\; \mathrm{SD}_s < (1-\delta)\,m \;\; \forall s \in [t, t+k)\right\},$$

with defaults $\delta = 0.2$, $k = 3$. We report $\tau$ for all nine combinations
$\delta \in \{0.1, 0.2, 0.3\}$, $k \in \{2,3,5\}$ and treat conclusions that are not
stable across them as not supported. Runs that never satisfy the criterion are recorded
as $\tau = \infty$ rather than dropped, since "never released" is a substantive outcome.

## Capacity sweep (H1)

Width multipliers $w \in \{0.25, 0.5, 1.0, 2.0\}$ applied to a ResNet-style backbone,
depth $18$ throughout for the main sweep, with a depth check $\{10, 18, 34\}$ at fixed
width. Data, batch size, epochs, optimiser and learning-rate schedule are identical
across cells. Three seeds per cell, and three values of $\rho$, giving 36 runs for the
main sweep. A $10$-class ViT-tiny replicates the largest and smallest cells as an
architecture-generality check.

## Interventions (H2)

Applied to the smallest model only, each of five interventions targets the *timing* of
commitment rather than the model's power: a $10$-epoch linear warm-up; increased weight
decay; a step learning-rate schedule; label smoothing; and self-distillation from an
early checkpoint, which supplies the trajectory information that a converged target
cannot. If delay is the mechanism, interventions that raise $\tau$ should raise final
accuracy, and the ordering of interventions by $\Delta\tau$ should match their ordering
by $\Delta\text{accuracy}$.

## Resource matching (H4)

Capacity is collinear with other quantities in practice, so we compare the smallest and
largest widths under four budgets: no matching; matched examples seen; matched optimiser
updates; and matched approximate training FLOPs. An effect that survives all four is
attributed to capacity; one that vanishes under matching is reported as a resource
effect. We estimate FLOPs from layer shapes and treat this arm as indicative rather than
exact.

## Early warning (H3)

Features are drawn only from epochs $0$–$4$: the level and slope of $\mathrm{SD}_t$,
in-distribution accuracy, confidence, entropy, gradient norm, parameter norm and the
shortcut log-likelihood. The target is whether the run releases the shortcut at all.
We use $\ell_2$-regularised logistic regression with `GroupKFold` over seeds so that the
probe is never evaluated on a seed it has seen. A high AUC from a linear model on five
epochs would support H3; a shallow probe is deliberate, since a deep model reaching the
same AUC would say little.

## Statistics

Five or more seeds per cell; we report means with $95\%$ bootstrap confidence intervals
over seeds, and use paired comparisons where seeds are shared across cells. The primary
contrasts in H1–H4 are preregistered above. All exploratory analyses are labelled as
such.

# Experimental Setup

**Implementation.** PyTorch; all runs are single-GPU (Kaggle T4 or P100) and complete in
minutes; the full suite is budgeted at approximately 29 GPU-hours. Every run writes one
JSON line per epoch and a completion marker, so a disconnected session resumes without
repeating finished work.

**Compute honesty.** The design is small on purpose. It buys control, replication and a
causal comparison, at the cost of scale: we do not claim that these effect sizes transfer
unchanged to ImageNet-scale pretraining, and we say so in Section 7.

# Results

> **Status: results pending.** Every number below is generated from
> `results/processed/*.csv` by `src/figures.py` and pasted only after the corresponding
> experiment completes. Placeholders are explicit so that no claim in this draft is
> currently supported by data.

## Does capacity change the release epoch? (H1)

[PENDING EXPERIMENT] — Figure 1 plots $\mathrm{SD}_t$ over training for each width at
$\rho = 0.95$; Figure 2 plots $\tau$ against width for all three values of $\rho$ with
bootstrap intervals over seeds.

**Prediction.** $\tau$ increases with width. **Falsified if** $\tau$ is flat in width,
or if the direction reverses and is stable across the $\tau$ sensitivity grid.

## Is the accuracy gap mediated by timing? (H2)

[PENDING EXPERIMENT] — Figure 3 relates the small-minus-large $\tau$ gap to the
large-minus-small accuracy gap. Table 2 reports, for each intervention applied to the
smallest model, $\Delta\tau$ and $\Delta$accuracy against the untouched small baseline
and against the large model.

**Prediction.** Both differences are positive and rank-correlated. **Falsified if**
interventions move $\tau$ substantially while leaving accuracy unchanged, which would
show that timing is not the operative variable.

## Is timing predictable early? (H3)

[PENDING EXPERIMENT] — Table 3 reports grouped cross-validated AUC for the early-warning
probe, with a calibration check.

**Prediction.** AUC substantially above chance. **Falsified if** AUC is close to $0.5$,
in which case the timing is real but not anticipable from cheap signals.

## Capacity, or compute and data? (H4)

[PENDING EXPERIMENT] — Table 4 reports the small-versus-large accuracy and $\tau$ gaps
under the four matching regimes.

**Prediction.** The gaps shrink but do not vanish under full matching. **Falsified if**
full matching eliminates the gap, in which case the honest conclusion is that the
phenomenon is a resource effect, and we will report it as such.

# Discussion

[PENDING EXPERIMENT] — to be written once the direction of the result is known. The
discussion will address three questions regardless of outcome: (i) whether "capacity
buys patience" holds, and with what effect size; (ii) what the result implies for
practitioners choosing between scale and schedule; and (iii) how the finding interacts
with the norm-hierarchy account, which predicts the delay but not its dependence on
capacity.

# Limitations

**Scale.** One task family, mostly one backbone family, $32\times32$ inputs, and widths
within a factor of eight. The largest model in the study is small by modern standards.
Effect sizes may not transfer to large-scale pretraining, and we make no such claim.

**Definition of $\tau$.** The release epoch depends on $\delta$ and $k$, and on the
choice of probe. We mitigate this with a sensitivity grid and by reporting the raw curves,
but a different probe (for example a held-out shortcut-only test rather than a probe
drawn from the same generator) could shift $\tau$ systematically.

**One shortcut family.** Colour blocks and texture patches are convenient and exactly
controllable, but they may be easier to separate than real-world spurious correlations.
The optional Waterbirds check is included precisely because this is the weakest part of
the design.

**Capacity is not measured directly.** Width and depth are proxies. Two models of equal
parameter count can differ in effective capacity, and we do not estimate the latter.

**Confounding.** Compute, data and capacity are collinear in practice; the matching arm
reduces but does not eliminate this. Approximate FLOP matching is indicative only.

**Publication risk.** The area is active and the closest precursor work is recent. A
capacity-focused follow-up is an obvious next step for others as well, so the project is
time-sensitive. We mitigate this by preregistering and by timestamping runs.

# Conclusion

[PENDING EXPERIMENT] — The conclusion will restate the answer to a single question: does
capacity change when a network abandons a shortcut, and does that explain the capacity
gap? We will report the answer, including a null result, with the full artefact.

# Reproducibility Statement

Code, configs and derived tables accompany the paper. The controlled dataset is
generated from a seed and requires no download. Each run is identified by its full
configuration, so the mapping from any table cell to a run is mechanical. The
manuscript's numbers are produced by `make analyse && make figures`, not transcribed.

# Ethics Statement

The work uses only synthetic data and public benchmark datasets. It does not involve
human subjects, personal data, or models released for deployment. The broader concern —
that findings about shortcut learning are often over-extrapolated from small studies — is
addressed by the Limitations section rather than by new empirical work.
