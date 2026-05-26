# Experiment Config Ownership

YAML files are grouped by the `tracker.experiment` value inside each file.
The folder name is the owner of the run config; downstream commands may read
outputs from other experiments without changing that ownership.

| Config | Owner | Purpose |
| --- | --- | --- |
| `exp01_reproduce_ships_sahara/01-ships-discovery.yaml` | Exp 1 | SHIPS safety-head discovery |
| `exp01_reproduce_ships_sahara/02-sahara-discovery.yaml` | Exp 1 | Sahara safety-head discovery |
| `exp02_judge_pipeline/judge.yaml` | Exp 2 | Shared judge configuration used to score completions from later experiments |
| `exp03_safety_vs_random_ablation/03-baseline.yaml` | Exp 3 | Baseline JailbreakBench generation with no ablation |
| `exp03_safety_vs_random_ablation/04-safety-head-ablation.yaml` | Exp 3 | Safety-head ablation generation |
| `exp03_safety_vs_random_ablation/04-safety-head-ablation-top1.yaml` | Exp 3 | Safety-head ablation generation with top-1 SHIPS head |
| `exp03_safety_vs_random_ablation/04-safety-head-ablation-top3.yaml` | Exp 3 | Safety-head ablation generation with top-3 SHIPS heads |
| `exp03_safety_vs_random_ablation/04-safety-head-ablation-top5.yaml` | Exp 3 | Safety-head ablation generation with top-5 SHIPS heads |
| `exp03_safety_vs_random_ablation/04-safety-head-ablation-top8.yaml` | Exp 3 | Safety-head ablation generation with top-8 SHIPS heads |
| `exp03_safety_vs_random_ablation/05-random-head-ablation.yaml` | Exp 3 | Uniform random-head ablation control |
| `exp03_safety_vs_random_ablation/06-layer-matched-random.yaml` | Exp 3 | Layer-matched random-head ablation control |
| `exp03_safety_vs_random_ablation/matrix.yaml` | Exp 3 | Optional matrix expansion for Exp 3 generation cells |
| `exp04_coherency_head_discovery/07-coherency-discovery.yaml` | Exp 4 | Coherency-head discovery |
| `exp04_coherency_head_discovery/07b-coherency-ablation.yaml` | Exp 4 | Coherency-head ablation generation |
| `exp05_joint_disentangled_ablation/00-analysis.yaml` | Exp 5 | Cross-condition judged-output analysis and head-set overlap |
| `exp05_joint_disentangled_ablation/08-quality-discovery.yaml` | Exp 5 | Quality-head discovery |
| `exp05_joint_disentangled_ablation/08b-quality-ablation.yaml` | Exp 5 | Quality-head ablation generation |
| `exp05_joint_disentangled_ablation/09-safety-minus-coherency.yaml` | Exp 5 | Safety heads with coherency overlap removed |
| `exp05_joint_disentangled_ablation/10-overlap-only.yaml` | Exp 5 | Safety/coherency overlap-head ablation |

For example, `exp02_judge_pipeline/judge.yaml` can judge
`runs/03-baseline/completions_baseline.jsonl`, but the baseline generation
config remains `exp03_safety_vs_random_ablation/03-baseline.yaml`.
