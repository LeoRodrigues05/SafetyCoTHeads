# Tracker Exp 6: CoT Trajectory Analysis

No runnable YAML config exists yet for this tracker experiment.

Implementation hooks currently live in:

- `src/safety_cot_heads/analysis/trajectory.py`
- `src/safety_cot_heads/judging/`
- `src/safety_cot_heads/generation/`

The future config should generate normal and CoT-prompted completions for the
baseline plus ablation conditions, then judge cumulative sentence prefixes.
