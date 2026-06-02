# Experiment Documentation

These docs follow the research-experiment numbering in
`docs/general/ExperimentTracker.md`. The YAML files under `configs/experiments/` are
run configs grouped by the tracker experiment they support.

## Organization

| Tracker exp | Documentation | Config folder | Status |
| --- | --- | --- | --- |
| 1 | `exp01_reproduce_ships_sahara.md` | `configs/experiments/exp01_reproduce_ships_sahara/` | runnable |
| 2 | `exp02_judge_pipeline.md` | `configs/experiments/exp02_judge_pipeline/` | runnable |
| 3 | `exp03_safety_vs_random_ablation.md` | `configs/experiments/exp03_safety_vs_random_ablation/` | runnable |
| 4 | `exp04_coherency_head_discovery.md` | `configs/experiments/exp04_coherency_head_discovery/` | partially wired |
| 5 | `exp05_joint_disentangled_ablation.md` | `configs/experiments/exp05_joint_disentangled_ablation/` | partially wired |
| 6 | `exp06_cot_trajectory_analysis.md` | `configs/experiments/exp06_cot_trajectory_analysis/` | planned |

## Source Of Truth

- `docs/general/ExperimentTracker.md` defines the research plan and experiment names.
- `configs/experiments/exp*/` defines the concrete run configs.
- The `tracker:` block in each YAML maps that run back to the tracker.
- `configs/experiments/README.md` lists each YAML's owning experiment and
  clarifies cross-experiment consumers such as the shared judge config.

Current caveat: `scripts/run_attribution.py` dispatches only `ships` and
`sahara`. Coherency and quality attribution modules exist, but their configs are
marked `planned_unwired` until the CLI is connected to those modules.
