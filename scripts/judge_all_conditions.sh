#!/usr/bin/env bash
# Run the Qwen2.5-32B (4-bit) safety judge over all four ablation conditions.
# Requires a single GPU with >= 24 GB VRAM (RTX 5000 Ada / A100 / H100).
# Usage:  bash scripts/judge_all_conditions.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CFG=configs/experiments/exp02_judge_pipeline/judge.yaml

declare -A INPUTS=(
  [baseline]="runs/03-baseline/completions_baseline.jsonl"
  [safety_head_ablation]="runs/04-safety-head-ablation/completions_safety_head_ablation.jsonl"
  [random_head_ablation]="runs/05-random-head-ablation/completions_random_head_ablation.jsonl"
  [layer_matched_random_head_ablation]="runs/06-layer-matched-random/completions_layer_matched_random_head_ablation.jsonl"
)

declare -A OUTS=(
  [baseline]="runs/03-baseline/judge_safety.jsonl"
  [safety_head_ablation]="runs/04-safety-head-ablation/judge_safety.jsonl"
  [random_head_ablation]="runs/05-random-head-ablation/judge_safety.jsonl"
  [layer_matched_random_head_ablation]="runs/06-layer-matched-random/judge_safety.jsonl"
)

for cond in baseline safety_head_ablation random_head_ablation layer_matched_random_head_ablation; do
  echo "=== judging ${cond} ==="
  python scripts/run_judge.py \
    --config "${CFG}" \
    --completions "${INPUTS[$cond]}" \
    --out         "${OUTS[$cond]}"
done

echo "=== evaluation report ==="
python scripts/run_evaluation.py \
  --baseline runs/03-baseline/judge_safety.jsonl \
  --compare  runs/04-safety-head-ablation/judge_safety.jsonl \
             runs/05-random-head-ablation/judge_safety.jsonl \
             runs/06-layer-matched-random/judge_safety.jsonl \
  --completions-for-coherence \
             runs/03-baseline/completions_baseline.jsonl \
             runs/04-safety-head-ablation/completions_safety_head_ablation.jsonl \
             runs/05-random-head-ablation/completions_random_head_ablation.jsonl \
             runs/06-layer-matched-random/completions_layer_matched_random_head_ablation.jsonl \
  --coherence \
  --out runs/analysis/evaluation_report.json
