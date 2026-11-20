# Deviations and open items

This file records **everything** that is missing, unverified, or changed relative to the
preregistered plan. It exists so that a reader (or a future you) never has to guess which
parts of the paper are supported.

## Deliberately empty at v0.1

| Item | Status | What fills it |
|---|---|---|
| Section 5 Results, all four subsections | `[PENDING EXPERIMENT]` | `make capacity && make analyse && make figures` |
| Section 6 Discussion | `[PENDING EXPERIMENT]` | written after the result direction is known |
| Section 7 Conclusion | `[PENDING EXPERIMENT]` | written last |
| Figures 1–3 | placeholders rendered by `src/figures.py` | real data |
| Table 2 (interventions) | not yet included | `results/processed/mediation*.csv` |
| Table 3 (early warning) | not yet included | `results/processed/earlywarning.csv` |
| Table 4 (resource matching) | not yet included | `results/raw/resources` |
| Author names, affiliation | `[Author]` / `[Institution]` | you |
| Venue | none chosen | depends on result strength |

## Bibliographic items still to verify

1. **@nht2026 (arXiv 2603.07323)** — abstracts-only review so far. Read in full and
   confirm it does not already vary capacity; this is the single biggest novelty risk.
2. **@ood2026 (arXiv 2610.12397)** — confirm what "controlled shortcut-learning setting"
   means there, and whether their probe uses capacity at all.
3. **@asd2026 (arXiv 2608.11789)** — this uses early checkpoints as a *shortcut negative*,
   which overlaps with our self-distillation intervention (H2, `early_distill`). We must
   cite it explicitly and state that our use is as an *intervention on timing for a small
   model*, not as a distillation objective.
4. **@geirhos2020shortcut** — already verified as real.

## Known design weaknesses (stated in the paper, repeated here)

- Only one main task family and largely one backbone family.
- $\tau$ is definition-dependent; the sensitivity grid mitigates but does not remove this.
- Width/depth are proxies for capacity; effective capacity is not estimated.
- FLOP matching is approximate.
- Time-sensitive area; the closest precursor is ~9 months old, so a competing follow-up
  may appear mid-project.

## Code changes worth knowing about

- **2026-11-20, `signal_only` split bug.** The `signal_only` probe originally called
  `_render_signal_only`, which drew fresh images from the RNG and therefore duplicated
  `test`. It is now built by erasing the shortcut from the actual test images. Any run
  produced before this fix has an inflated `acc_signal` and must be discarded.

## Decisions made without your input (change these if wrong)

- Task name "`Capacity Buys Patience`" — a working title, deliberately a claim that the
  experiments may falsify.
- Width grid $\{0.25, 0.5, 1.0, 2.0\}$, depth $18$, 40 epochs, AdamW, cosine schedule.
- Five interventions; the fifth (`early_distill`) is the most implementation-heavy and is
  the first to cut if time is short.
- ResNet-style backbone rather than a modern architecture, for speed and because width
  scaling is clean there.
