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

## Checkpointing and resume (added after the first Kaggle attempt failed)

Three bugs were found and fixed in the checkpoint path. All three would have silently
corrupted results, so they are recorded here rather than only in git history.

1. **Truncation.** The per-epoch JSONL was opened in `"w"` mode, so restarting a run
   erased the epochs already completed. It is now `"a"` (append).
2. **Compressed LR schedule.** The cosine schedule is a function of the *total* number of
   epochs, so a run started as a 2-epoch probe and later resumed to 4 epochs trained its
   first epochs on a 2-epoch curve. The schedule length is now stored in
   `<run_id>.canonical.json` and is monotone (a shorter run can never shrink it). Short
   runs now stop early via `stop_at_epoch` instead of changing `epochs`.
3. **Stale optimizer LR.** Assigning `sched.last_epoch` on resume does not refresh the
   optimizer's learning rate; the scheduler only writes it inside `step()`. The first
   resumed epoch therefore trained at the wrong LR. The LR is now set explicitly.

Verified by `tests/test_resume.py`, which asserts that a run interrupted after epoch 1
and resumed produces bit-identical `sd`, `train_loss` and `lr` for every epoch compared
with an uninterrupted run. Current status: **PASS**.

Durability properties now guaranteed:
- One `{"event":"epoch",...}` line, flushed, per completed epoch.
- `<run_id>.state.pt` written every epoch via `os.replace` (atomic; never half-written).
- `<run_id>.done` written only when the run reaches its final epoch.
- On Colab, set `CBPP_DRIVE_DIR` to copy each checkpoint to Drive after every epoch.
