# Changelog

## v0.1 — 2026-11-20
- Project scaffolding: configs, data generators, model factory, training loop, sweep
  launcher, analysis modules, figure generation, paper build.
- Manuscript drafted with the full argument, method, preregistered hypotheses and
  analysis plan. Results, discussion and conclusion are explicit placeholders.
- **Fix:** the `signal_only` counterfactual split is now built by erasing the shortcut
  from the real test images instead of re-rendering from the RNG (which duplicated the
  test set). Runs from before this fix must be discarded.
- Smoke test passes on CPU (2 epochs, 308K-parameter model).
