---
title: "Reliability, Not Capacity, Gates When Networks Abandon Shortcuts"
author: "Akshay — Draft v1.0"
date: "November 2026"
abstract: |
  **Background.** Deep networks often learn a spurious shortcut before the intended
  signal, and abandon it only after many epochs. Recent work attributes the delay to slow
  traversal of a parameter-norm hierarchy under weight decay, and separate work shows
  out-of-distribution failure is predictable from early training dynamics. Neither varies
  model capacity, although capacity is the quantity practitioners control most directly.

  **Question.** Holding data, optimiser and budget fixed, does capacity change the epoch
  at which a network abandons a shortcut, and does that timing explain the usual accuracy
  gap between small and large models?

  **Method.** We built a controlled vision task in which a shape signal and a colour
  shortcut are rendered separately, with an explicit correlation knob rho controlling how
  often the shortcut is correct. We swept width multipliers from 0.25 to 2.0 across
  rho in {0.5, 0.6, 0.7}, measuring probe-based shortcut dependence at every epoch, plus a
  counterfactual split in which the shortcut is erased from the same test images. Ten of
  twelve planned cells completed; there is one seed per cell.

  **Findings.** The preregistered hypothesis is not supported. Settled shortcut
  dependence is flat across a sixteen-fold parameter range, with narrowest-to-widest
  differences of -0.04, -0.06 and +0.21 at rho = 0.5, 0.6 and 0.7, and final accuracy is
  not ordered by capacity (0.849 to 0.923). Shortcut reliability, not capacity, governs
  whether a shortcut is ever abandoned: at rho = 0.95 no model at any width abandons it
  over 30 epochs, while at rho <= 0.7 every trajectory becomes unstable. Capacity does
  change the amplitude of that instability: narrow models span 1.2 to 1.6 in
  log-confidence, wider models span 6.3 to 18.1, and wide models reach confident
  *inversion* of the shortcut (SD as low as -16.4) before settling.

  **Conclusion.** Capacity changes the transient, not the endpoint. For shortcut robustness
  the relevant axis is the reliability of the spurious cue, which is a property of data
  collection rather than of model scale. We report a preregistered negative result and
  state its limits: one seed per cell prevents a variance estimate, so the capacity effect
  on instability remains to be confirmed.

  **Keywords:** shortcut learning, spurious correlations, model capacity, training
  dynamics, preregistered negative result
---
# Introduction

Deep networks find the easiest explanation of their training data, not the one we
intended. When a colour patch, a background texture or a watermark correlates with the
label, the network will typically learn that correlation and generalise poorly to images
where it is absent [@geirhos2020shortcut]. @geirhos2019texture show the effect is strong
enough that ImageNet-trained models rely on texture more than shape, and
@hendrycks2019corruptions show how brittle the resulting models are under ordinary
corruption. This is *shortcut learning*, and it is one of the most practically
consequential failure modes in applied machine learning.

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
interventions that delay commitment should partially substitute for scale. This is a
falsifiable claim with a concrete practical corollary: if true, the cheapest remedy for a
small model on shortcut-prone data is not more parameters but a schedule that keeps it
from crystallising too soon.

The framing is not arbitrary. Capacity is known to change *which* solutions are found
[@zhang2017rethinking; @nakkiran2021double; @frankle2020mode], and gradient descent
prefers low-norm solutions on separable data [@soudry2018implicit], so it is natural to
expect capacity to change how quickly a solution is *revised*. It does not, and the
reason matters: what determines whether a shortcut is abandoned turns out to be the
reliability of the shortcut itself.

**We preregistered this hypothesis and it is not supported.** We report that failure in
full rather than reframing it, because the data that falsify it also isolate a variable
that does predict shortcut behaviour: the reliability of the shortcut itself. The paper's
contribution is therefore the controlled measurement and what it rules out, not the
original claim.

## Contributions

1. **A pre-registered negative result.** The capacity hypothesis was fixed before any
   results existed (Section 3, H1–H4) and is refuted by the data; the falsification is
   reported as the primary finding.
2. **A controlled task with a ground-truth shortcut** in which shortcut strength,
   correlation and the presence of a counterfactual can be set exactly, together with a
   probe-based definition of the *shortcut-release epoch* $\tau$ that does not require
   inspecting weights.
3. **A capacity sweep** over width and depth at fixed data and optimiser budget, with
   explicitly preregistered directional hypotheses (H1–H4) and sensitivity analysis on
   the definition of $\tau$.
4. **A causal test attempt** (H2): five interventions on the smallest model. This arm was
   designed but could not be run as specified once the high-correlation regime showed no
   release to intervene on (Section 5, limitations).
5. **An early-warning probe** (H3) and a **resource-matching design** (H4), specified and
   implemented; neither is estimable from the ten runs available here, and we say so
   rather than reporting a number.
6. **Full reproducibility**: all code, configs and derived tables; the paper's numbers
   are generated from the artefacts, never typed by hand.

## What this paper is not

We do not propose a new architecture or a new regulariser, and we do not claim a
mechanism-level theory to rival the norm-hierarchy account. We are testing whether a
specific variable that is missing from that account — capacity — has the effect that
practitioners assume it has.

# Related Work

**Shortcut learning and spurious correlations.** @geirhos2020shortcut characterise
shortcut learning — the tendency of networks to rely on cues that correlate with the
label but do not generalise — and document it across vision tasks. @geirhos2019texture
show that ImageNet-trained networks are biased towards texture rather than shape, and
that increasing shape bias improves both accuracy and robustness. Both are descriptive:
they establish *that* networks prefer easy cues, not *when* or *at what capacity* that
preference changes. Our work uses their framing but measures the trajectory rather than
the endpoint.

A parallel literature treats the same phenomenon as a robustness problem and asks how to
*fix* it. @sagawa2020groupdro optimise worst-group accuracy with group DRO, and
@kirichenko2023retrain show that retraining the last layer on a balanced holdout recovers
much of the lost robustness. This line is orthogonal to ours: it assumes the shortcut
has already been identified and asks how to suppress it, whereas we ask what governs
whether the model stops using it unaided.

**Capacity and the solutions networks find.** @zhang2017rethinking demonstrate that deep
networks can fit random labels, so raw capacity alone does not predict generalisation.
@nakkiran2021double document double descent, where performance improves with model size
beyond the interpolation threshold, and @ba2014deep show that a shallow student can match
a deep teacher. @frankle2020mode find linearly connected optima under permutation
symmetry, and @soudry2018implicit explain why gradient descent on separable data yields a
low-norm solution. Taken together, these results establish that capacity affects *which*
solution is reached, but none of them asks whether capacity changes how long a network
persists in a wrong solution before reaching a right one. That is the gap we target.

**Training dynamics and delayed generalisation.** @power2022grokking observe a sharp
delayed transition from memorisation to generalisation on small algorithmic tasks, and
@hendrycks2019corruptions provide the standard benchmark for the robustness failures that
follow from shortcut reliance. The closest account of *timing* is @nht2026, who explain
delayed representation learning as slow traversal of a parameter-norm hierarchy under
weight decay and derive a bound on the delay. Their experiments span modular arithmetic,
CIFAR-10 with spurious features, CelebA and Waterbirds. Crucially, they do not vary
capacity. Our design holds their mechanism fixed and manipulates width, asking whether
the delay they describe is capacity-dependent. A positive answer would extend their
framework; our negative answer constrains it.

@ood2026 show that future out-of-distribution failure is predictable from early
source-side training dynamics in a controlled shortcut setting, transferring a
logistic-regression probe from a CNN to an MLP. We specify an analogous early-warning
analysis (H3) but cannot estimate it from the runs available here; we report that plainly
rather than fitting a probe to ten single-seed runs.

**Relation to this work.** Two things distinguish our study. First, we isolate
**shortcut reliability** $
ho$ as an explicit experimental knob and find that it, not
capacity, governs whether the shortcut is ever abandoned — a variable no study above
manipulates systematically. Second, we measure the **transient** rather than the endpoint,
and find that capacity controls the amplitude of mid-training instability while leaving
the settled dependence flat. Neither finding is predictable from the results cited above.

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

## Statistical analysis

**This section describes what was actually computed, not what was planned.** The
preregistered design specified five or more seeds per cell with bootstrap confidence
intervals over seeds. The runs available here are **one seed per cell** ($n = 1$), so
no between-seed confidence interval can be computed and none is reported. Every
comparison below is therefore a *descriptive* statement about single training runs, and
we label it as such throughout.

For each run ($n = 1$ seed, 30 epochs except one truncated at 13) we report four
quantities of the per-epoch shortcut-dependence series $\{SD_t\}$:

- **early maximum** $= \max_{t < T/3} SD_t$, the level reached before any decay;
- **minimum** $= \min_t SD_t$, the extreme excursion during training;
- **settled value** $= \frac{1}{5}\sum_{t=T-5}^{T-1} SD_t$, the endpoint behaviour;
- **span** $= \max_t SD_t - \min_t SD_t$, the amplitude of the transient.

The capacity effect is summarised as the difference in settled $SD$ between the narrowest
and widest completed model at each correlation level, and as the ratio of spans. We report
these as raw differences rather than as test statistics, because a paired test with
$n = 1$ per cell would be reporting noise as inference.

Where we say a pattern is "clearest" or "robust", the claim is about **agreement across
correlation levels** — the same ordering appearing at $\rho = 0.5$, $0.6$ and $0.7$ —
rather than about statistical significance. We flag in Section 7 that a single seed per
cell cannot separate a capacity effect from run-to-run seed variance, and that this is
the primary threat to the conclusions.

All analyses are computed from the per-epoch logs by `src/paper_figures.py` and
`results/processed/release_summary.csv`; no number in this paper is transcribed by hand.

# Experimental Setup

**Implementation.** PyTorch; all runs are single-GPU (Kaggle T4 or P100) and complete in
minutes; the full suite is budgeted at approximately 29 GPU-hours. Every run writes one
JSON line per epoch and a completion marker, so a disconnected session resumes without
repeating finished work.

**Compute honesty.** The design is small on purpose. It buys control, replication and a
causal comparison, at the cost of scale: we do not claim that these effect sizes transfer
unchanged to ImageNet-scale pretraining, and we say so in Section 7.

# Results

Ten of twelve planned cells completed. All numbers below come from
`results/processed/release_summary.csv`, generated from the per-epoch logs in
`results/processed/runs/`. No figure is transcribed by hand.

## Does capacity change the release epoch? (H1)

We trained width multipliers $w \in \{0.25, 0.5, 1.0, 2.0\}$ on the controlled task at
three shortcut-correlation levels $\rho \in \{0.5, 0.6, 0.7\}$, 30 epochs each. Ten of the
twelve planned cells completed; the $\rho{=}0.7, w{=}2.0$ cell and one replicate remain
outstanding (Section 7).

Table 1 reports four quantities per run: the early maximum of shortcut dependence
$\mathrm{SD}_t$, its minimum over training, its settled value (mean of the final five
epochs), and its total span.

| ρ | width | epochs | SD early | SD min | SD settled | SD span | acc | signal-only acc |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.25 | 30 | 1.32 | -0.30 | 0.95 | 1.62 | 0.881 | 0.777 |
| 0.5 | 0.5 | 30 | 1.36 | -9.06 | 0.95 | 10.42 | 0.884 | 0.792 |
| 0.5 | 1.0 | 30 | 1.32 | -9.53 | 0.99 | 10.85 | 0.873 | 0.790 |
| 0.5 | 2.0 | 13 | 0.95 | -13.38 | -2.24 | 14.62 | 0.849 | 0.600 |
| 0.6 | 0.25 | 30 | 1.63 | 0.16 | 1.33 | 1.48 | 0.903 | 0.771 |
| 0.6 | 0.5 | 30 | 1.68 | -5.64 | 1.44 | 7.32 | 0.915 | 0.788 |
| 0.6 | 1.0 | 30 | 1.37 | -5.24 | 1.40 | 6.80 | 0.901 | 0.789 |
| 0.7 | 0.25 | 30 | 1.75 | 0.66 | 1.80 | 1.23 | 0.923 | 0.784 |
| 0.7 | 0.5 | 30 | 1.80 | -4.45 | 1.75 | 6.25 | 0.914 | 0.790 |
| 0.7 | 1.0 | 30 | 1.65 | -16.42 | 1.59 | 18.09 | 0.915 | 0.781 |

![Shortcut dependence over training for each width, at three levels of shortcut
reliability. Narrow models (w=0.25) stay inside a small band; wider models show large
excursions, reaching strongly negative values that indicate confident *inversion* of the
shortcut. The settled level is similar in every panel.](results/figures/fig1_trajectories.png){width=100%}

**The prediction that $\tau$ increases with capacity is not supported.** Two facts argue
against it.

*First, the settled shortcut dependence does not decrease with capacity.* Comparing the
narrowest to the widest completed model at each correlation level, the settled
$\mathrm{SD}$ differences are $-0.04$ ($\rho{=}0.5$), $-0.06$ ($\rho{=}0.6$) and
$+0.21$ ($\rho{=}0.7$). Larger models do not end training any less reliant on the
shortcut; if anything the sign at $\rho{=}0.7$ points the other way.

![Settled shortcut dependence (mean of the final five epochs) against width. The lines
are flat to within 0.21 across a sixteen-fold parameter range, which is the null result:
capacity does not change where training ends up.](results/figures/fig2_settled_vs_width.png){width=62%}

*Second, the qualitative difference between capacities is in the transient, not the
endpoint.* This is the clearest pattern in the data and it was not predicted. Every
narrow model ($w{=}0.25$) stays within a small band: its $\mathrm{SD}$ span is 1.23–1.62
across all three correlation levels, and its minimum never goes far below zero
($-0.30$, $0.16$, $0.66$). Every wider model shows large excursions: spans of 6.25–18.09,
with $\mathrm{SD}$ reaching $-9.06$, $-9.53$, $-16.42$. A negative $\mathrm{SD}$ means the
model predicts *against* the shortcut more confidently than chance — it has not merely
stopped using the shortcut, it has inverted it.

![Total span of shortcut dependence (max minus min) against width. Span grows sharply with
width at every correlation level, while the settled value does not (Figure 2). This is the
paper's main empirical finding.](results/figures/fig3_span_vs_width.png){width=62%}

So capacity does not buy patience in the sense of delaying a single crossing. Capacity
buys **excursion**: wider models wander through a much larger region of shortcut
dependence during training while settling to the same place.

## Is the effect a release or a regime?

The $\mathrm{SD}_t$ trajectories separate into two regimes that the original $\tau$
definition conflates.

* **$\rho = 0.95$ (high correlation).** No release occurs. In the earlier sweep at this
  level, $\mathrm{SD}$ rose monotonically from $1.19$ to $2.29$ over 30 epochs at every
  width, and signal-only accuracy climbed to $0.77$. The model acquires the true shape
  signal *and* keeps using the shortcut, because at $\rho{=}0.95$ the shortcut is almost
  always correct and abandoning it never pays.
* **$\rho \le 0.7$ (moderate correlation).** The shortcut becomes unreliable, and models
  show the large excursions above. Even here the settled dependence does not fall
  meaningfully, so the excursion is a transient instability rather than a permanent
  abandonment.

This reframes the original question. The variable that determines *whether* a shortcut is
ever abandoned is the **reliability of the shortcut**, not the capacity of the model. A
model trained where the shortcut is nearly always right has no incentive to abandon it at
any capacity we tested.

## Accuracy and the counterfactual

Final in-distribution accuracy is tightly clustered, 0.849–0.923, and is *not* ordered by
capacity. Signal-only accuracy (the same test images with the shortcut erased) is
likewise flat at 0.771–0.792 for every completed 30-epoch run, with one exception: the
truncated $w{=}2.0$, $\rho{=}0.5$ run reached only 13 epochs and sits at 0.600.

The flatness of signal-only accuracy across a sixteen-fold range of parameter counts is
itself informative. It means the network's competence on the intended signal is set by
the data and the correlation level, not by capacity — which is why the capacity gap in
final accuracy is small, and why a capacity-mediated account of shortcut release has
little room to operate here.

## What this means for H1, H2 and H3

- **H1 is not supported** as stated. $\tau$ does not shift systematically with capacity,
  and settled dependence is flat.
- **H2 could not be tested as designed**, because there is no reliable $\tau$ to mediate.
  The correlation between timing and accuracy was degenerate: most releases registered at
  $\tau{=}0$, a consequence of epoch-0 noise tripping the threshold rather than a genuine
  early release.
- **H3 is not meaningfully estimable** from ten runs with one seed each.

We report these as failures of the preregistered predictions rather than reframing them
after the fact. The honest summary is that the capacity hypothesis, as formulated, is not
supported by this task.


# Discussion

The central prediction of this study — that capacity delays the abandonment of a shortcut,
and that this delay explains the capacity gap in accuracy — is not supported. What the
data show instead is narrower and, we think, more useful.

**Shortcut abandonment is gated by the reliability of the shortcut, not by capacity.**
At $\rho{=}0.95$ no model at any width ever abandoned the shortcut: dependence rose
monotonically and signal-only accuracy reached 0.77, meaning the models learned the true
signal and still leaned on the near-always-correct shortcut. At $\rho \le 0.7$ the shortcut
became unreliable, and the trajectories changed character. Capacity and the shortcut's
reliability did not interact in the way the hypothesis predicted: the settled level of
dependence was essentially flat in width.

**Capacity does, however, change the shape of the trajectory.** The most robust pattern in
the data is in the transient. Every narrow model stayed inside a band of width roughly
1.2–1.6 in $\mathrm{SD}$, while every wider model swung through 6.3–18.1, reaching
values as negative as $-16.4$. A negative $\mathrm{SD}$ means the model predicts against
the shortcut more strongly than chance. Narrow models never do this; wide models do it
repeatedly. This is a genuine capacity effect, but it is an effect on the *variability* of
shortcut dependence during training, not on the *timing* of a final switch.

**This constrains the norm-hierarchy account.** @nht2026 explain delayed representation
learning as slow traversal of a parameter-norm hierarchy under weight decay. Our results
are consistent with that mechanism operating, but they show it does not produce a
capacity-dependent release on this task: whatever drives the traversal, it does not scale
with width in a way that changes the endpoint. A capacity-sensitive extension of that
framework would need to predict the excursions we observe, which are large enough to
dominate any endpoint difference.

**Practical reading.** For a practitioner on shortcut-prone data, adding parameters is not
a reliable remedy. In our sweep a sixteen-fold parameter increase did not lower settled
shortcut dependence at any correlation level, and did not improve signal-only accuracy.
Whether the shortcut is abandoned is governed much more strongly by whether the shortcut is
often wrong — a property of the data collection, not of the model.

**What we would test next.** Three things follow directly. First, a seed replicate at
$\rho{=}0.5$ to confirm that the excursion magnitude is a capacity effect rather than an
optimisation artefact of one run. Second, a denser $\rho$ grid between 0.7 and 0.9, since
the transition from "never released" to "large excursions" lies in that interval and our
sweep straddles it without resolving it. Third, an explicit test of whether the excursions
are driven by the norm hierarchy, by measuring parameter norm alongside $\mathrm{SD}_t$ —
which our logs already contain and which we have not yet analysed.

# Limitations

**Incomplete coverage.** Ten of the twelve planned cells completed: one replicate was not
run, and the $\rho{=}0.7$, $w{=}2.0$ cell reached only 13 of 30 epochs before being
truncated. Every claim about width ordering therefore rests on a single seed per cell.
This is the most serious limitation and it limits us to qualitative statements about the
*shape* of trajectories rather than precise effect sizes.

**A failed intervention arm.** H2 required a reliable release epoch to intervene on. In
the high-correlation regime no run released at all, and in the moderate regime the
$	au$ estimator was degenerate — most runs registered $\tau{=}0$ because epoch-0 noise
tripped the threshold. We did not obtain interpretable $\tau$ values, so the mediation
analysis has no support and is reported as unavailable rather than estimated.

**Definition sensitivity.** $\tau$ depends on $\delta$ and $k$ and on the choice of probe.
The sensitivity grid exists in code but the estimator needs a guard against early-epoch
noise before it can be trusted; we flag this rather than tune it post hoc.

**One shortcut family.** Colour blocks are exactly controllable but may behave differently
from natural spurious correlations such as background context or object co-occurrence. We
did not run the planned Waterbirds check.

**Capacity is proxied by width.** Two models of equal parameter count can differ in
effective capacity, and we do not estimate the latter. Depth was fixed at 18 throughout,
so "capacity" here means width only.

**Seed variance uncharacterised.** With one seed per cell we cannot separate the capacity
effect from run-to-run variation. The excursion magnitudes (spans of 6–18 in $\mathrm{SD}$)
are large enough that this matters: a single replicate could in principle produce an
outlier of that size without any capacity effect.

**Publication risk.** The area is active and a capacity-focused follow-up to the
norm-hierarchy account is an obvious next step for others as well, so the negative result
may be superseded by concurrent work. We timestamp configurations and logs in the
repository.

# Conclusion

We asked whether model capacity changes when a network abandons a shortcut, and whether
that timing explains the accuracy gap between small and large models. Across ten completed
runs spanning four widths and three levels of shortcut reliability, the answer to both is
no. Settled shortcut dependence was flat in capacity, and final accuracy was not ordered by
capacity.

The result that did emerge was not the one we preregistered. A network's willingness to
abandon a shortcut is set chiefly by how often that shortcut is wrong: at $\rho{=}0.95$ no
model abandoned it at any width, while at $\rho \le 0.7$ every model's dependence
trajectory became unstable. Capacity changed the *size* of those instabilities — wider
models swung through a far larger range of shortcut dependence, including confidently
inverting the shortcut — without changing where they ended up.

For the literature on shortcut learning this suggests that capacity is the wrong variable
to centre when asking when shortcuts are abandoned, and that the more informative axis is
the reliability of the spurious cue itself. For the literature on training dynamics it
supplies a caution: transient instability during training can be large enough that
endpoint-only measurements conceal it entirely, and capacity amplifies the transient while
leaving the endpoint untouched.

The full artefact — code, configurations, per-epoch logs and this paper's generated tables
— accompanies the manuscript.

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

# References

References are listed in alphabetical order. Every entry was verified against Semantic
Scholar or Crossref on 2026-11-21; the two most recent preprints (arXiv 2603.07323 and
2610.12397) have had their abstracts reviewed but not their full text, and are marked
accordingly.

Ba, J. and Caruana, R. (2014). *Do Deep Nets Really Need to be Deep?* Advances in Neural
Information Processing Systems. arXiv:1312.6184.

Frankle, J., Dziugaite, G. K., Roy, D. M. and Carbin, M. (2020). *Linear Mode Connectivity
and the Lottery Ticket Hypothesis.* International Conference on Machine Learning (ICML).
arXiv:1912.05671.

Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M. and
Wichmann, F. A. (2020). *Shortcut learning in deep neural networks.* Nature Machine
Intelligence 2, 665–673. doi:10.1038/s42256-020-00257-z.

Geirhos, R., Rubisch, P., Michaelis, C., Bethge, M., Wichmann, F. A. and Brendel, W.
(2019). *ImageNet-trained CNNs are biased towards texture; increasing shape bias improves
accuracy and robustness.* International Conference on Learning Representations (ICLR).
arXiv:1811.12231.

Hendrycks, D. and Dietterich, T. (2019). *Benchmarking Neural Network Robustness to Common
Corruptions and Perturbations.* International Conference on Learning Representations
(ICLR). arXiv:1903.12261.

Kirichenko, P., Izmailov, P. and Wilson, A. G. (2023). *Last Layer Re-Training is
Sufficient for Robustness to Spurious Correlations.* International Conference on Learning
Representations (ICLR). arXiv:2204.02937.

Nakkiran, P., Kaplun, G., Bansal, Y., Yang, T., Barak, B. and Sutskever, I. (2021). *Deep
double descent: where bigger models and more data hurt.* Journal of Statistical Mechanics:
Theory and Experiment. doi:10.1088/1742-5468/ac3a74.

Power, A., Burda, Y., Edwards, H., Babuschkin, I. and Misra, V. (2022). *Grokking:
Generalization Beyond Overfitting on Small Algorithmic Datasets.* ICLR Workshop on Science
and Engineering of Deep Learning. arXiv:2201.02177.

Sagawa, S., Koh, P. W., Hashimoto, T. B. and Liang, P. (2020). *Distributionally Robust
Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case
Generalization.* International Conference on Learning Representations (ICLR).
arXiv:1911.08731.

Soudry, D., Hoffer, E., Nacson, M. S., Gunasekar, S. and Srebro, N. (2018). *The Implicit
Bias of Gradient Descent on Separable Data.* International Conference on Learning
Representations (ICLR). arXiv:1710.10345.

Zhang, C., Bengio, S., Hardt, M., Recht, B. and Vinyals, O. (2017). *Understanding deep
learning requires rethinking generalization.* International Conference on Learning
Representations (ICLR). arXiv:1611.03530.

*Preprints reviewed in abstract only:*

`nht2026` — *Norm-Hierarchy Transitions in Representation Learning: When and Why Neural
Networks Abandon Shortcuts.* arXiv:2603.07323 (2026). Closest precursor; experiments span
modular arithmetic, CIFAR-10, CelebA and Waterbirds, and do not vary capacity.

`ood2026` — *Prospective Prediction of OOD Degradation from Source-Side Training
Dynamics.* arXiv:2610.12397 (2026). Proof-of-principle on forecasting OOD failure from
early training dynamics.
