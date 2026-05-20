# Experiment 5: Joint / Disentangled Ablation

## Goal

Separate safety effects from broader response-quality effects by comparing
safety heads, coherency heads, quality heads, their overlap, and safety-only
subsets.

## Configs

- `configs/experiments/exp05_joint_disentangled_ablation/08-quality-discovery.yaml`
- `configs/experiments/exp05_joint_disentangled_ablation/08b-quality-ablation.yaml`
- `configs/experiments/exp05_joint_disentangled_ablation/09-safety-minus-coherency.yaml`
- `configs/experiments/exp05_joint_disentangled_ablation/10-overlap-only.yaml`
- `configs/experiments/exp05_joint_disentangled_ablation/00-analysis.yaml`

## Intended Commands

```bash
python -m scripts.run_attribution \
  --config configs/experiments/exp05_joint_disentangled_ablation/08-quality-discovery.yaml

python -m scripts.run_generation \
  --config configs/experiments/exp05_joint_disentangled_ablation/08b-quality-ablation.yaml

python -m scripts.run_analysis \
  --config configs/experiments/exp05_joint_disentangled_ablation/00-analysis.yaml

python -m scripts.run_generation \
  --config configs/experiments/exp05_joint_disentangled_ablation/09-safety-minus-coherency.yaml

python -m scripts.run_generation \
  --config configs/experiments/exp05_joint_disentangled_ablation/10-overlap-only.yaml
```

## Current Status

Partially wired. Quality attribution code exists, but `scripts/run_attribution.py`
does not yet dispatch `method == "quality"`. The quality discovery and quality
ablation configs are blocked until that branch is added.

The explicit overlap configs are structurally runnable, but their placeholder
`heads.heads` lists must be replaced using the overlap report from analysis.

## Code Paths

- Quality attribution: `src/safety_cot_heads/attribution/quality_heads.py`
- Generation and ablation: `scripts/run_generation.py`
- Cross-condition analysis: `scripts/run_analysis.py`
- Overlap utilities: `src/safety_cot_heads/analysis/overlap.py`
- Judge aggregation: `src/safety_cot_heads/judging/`

## Data Flow

1. Quality discovery should load benign prompts, ablate heads one at a time, and
   rank heads by helpfulness/quality drop.
2. Quality-head ablation uses that ranking to generate JailbreakBench
   completions under a Q-head mask.
3. Analysis reads judged outputs plus attribution rankings and writes overlap
   reports.
4. The safety-minus-coherency and overlap-only configs use explicit head lists
   filled from `runs/analysis/overlap_report.json`.
5. Those explicit-head conditions are regenerated and judged.

## Outputs

- `runs/08-quality-discovery/quality_dataset_ranking.json` once wired
- `runs/08b-quality-ablation/completions_quality_head_ablation.jsonl`
- `runs/analysis/summary_by_condition.json`
- `runs/analysis/overlap_report.json`
- `runs/09-safety-minus-coherency/completions_safety_excluding_coherency_overlap.jsonl`
- `runs/10-overlap-only/completions_overlap_heads_only.jsonl`

## Review Checks

- Do not interpret 09/10 until their placeholder heads are replaced.
- Check the overlap report before choosing explicit heads.
- The central comparison is harmful-compliance delta at similar quality loss.
