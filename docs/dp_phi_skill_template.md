---
name: dp-phi-training
description: Use this skill whenever the user asks to check PHI training status, run eval for PHI, compare multiple PHI runs, analyze tail fitting (large-r behavior), or tune PHI hyperparameters. Always run the bundled scripts first instead of writing one-off analysis snippets.
---

# dp-phi-training

This skill standardizes PHI train/eval analysis with reusable scripts.

## Scripts

- `python experiments/check_phi.py <run_dir>`
  - Summarize single-run training/eval metrics
  - Handles swapped `residual_std`/`residual_p99_abs` columns

- `python experiments/compare_phi.py <run1> <run2> ... [--base <run>]`
  - Compare eval residual stats, radial errors, and tail-region `phi_mae`

## Expected workflow

1. If user asks "检查结果 / check status":
   - Run `check_phi.py` for target run.
2. If user asks "对比 runs / compare runs":
   - Run `compare_phi.py` with provided run list.
3. If user asks "eval":
   - Run `python -m experiments.eval_phi ...` first,
   - then run `compare_phi.py` or `check_phi.py`.
4. Report:
   - Core metrics (`residual_std/p99/p999/max`)
   - `phi_mae` in `r[3,10]` tail region
   - Actionable next hyperparameter suggestions.

## Output style

- Always provide a concise summary table first.
- Then state one clear conclusion:
  - best run overall
  - whether tail fitting improved
  - next recommended run setup.
