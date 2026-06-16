#!/usr/bin/env bash
set -euo pipefail

BASELINE="runs/03-baseline/completions_baseline.jsonl"

for k in 1 3 5 8; do
  cfg="configs/experiments/exp03_safety_vs_random_ablation/04-safety-head-ablation-top${k}.yaml"
  condition="safety_head_ablation_top${k}"
  completion="runs/04-safety-head-ablation-top${k}/completions_${condition}.jsonl"
  report_dir="runs/analysis/safety_head_pre_post_top${k}"

  if [[ -s "${completion}" ]]; then
    echo "Found ${completion}; skipping generation for top-${k}."
  else
    python -m scripts.run_generation --config "${cfg}"
  fi

  python -m scripts.make_pre_post_report \
    --baseline "${BASELINE}" \
    --ablation "${completion}" \
    --ablation-label "${condition}" \
    --out-dir "${report_dir}" \
    --title "Safety Head Ablation Top-${k}: Pre/Post Responses"
done
