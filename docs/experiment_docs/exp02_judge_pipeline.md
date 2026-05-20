# Experiment 2: LLM-as-Judge Pipeline

## Goal

Judge generated completions with the CoT-safety label schema so all ablation
conditions can be compared on safety behavior.

## Config

- `configs/experiments/exp02_judge_pipeline/judge.yaml`

## Command Pattern

```bash
python -m scripts.run_judge \
  --config configs/experiments/exp02_judge_pipeline/judge.yaml \
  --completions runs/03-baseline/completions_baseline.jsonl \
  --out runs/03-baseline/judge_baseline.jsonl
```

Repeat the same command shape for every `completions_*.jsonl` file produced by
Experiments 3-5.

## Code Paths

- Entry point: `scripts/run_judge.py`
- Judge model loading: `src/safety_cot_heads/models/loading.py`
- Judge loop: `src/safety_cot_heads/judging/judge.py`
- Prompt templates: `src/safety_cot_heads/judging/judge_prompts.py`
- JSON parsing: `src/safety_cot_heads/judging/parse.py`
- Aggregation: `src/safety_cot_heads/analysis/metrics.py`

## Data Flow

1. `run_judge.py` reads a completions JSONL file.
2. Each row contributes `prompt`, `completion`, `condition`, and metadata.
3. The judge config loads `meta-llama/Meta-Llama-3-8B-Instruct` in 4-bit by
   default.
4. The judge prompt asks for the project safety labels.
5. Parser code normalizes the raw judge output into structured fields.
6. Results are written row-wise and summarized beside the output file.

## What Is Judged

The safety judge tracks the CoT-safety labels:

- `HARMFUL_RESPONSE`
- `SAFE_REJECTION`
- `REASONING_ABOUT_SAFETY`
- `ADDING_INTENTION`
- `CHANGING_SUBJECT`

Each judged row preserves raw judge output and parse status for auditability.

## Outputs

- `judge_<condition>.jsonl`
- `judge_<condition>.summary.json`

## Review Checks

- Confirm the number of judged rows matches the completion rows.
- Check parse failure counts before trusting aggregate metrics.
- Use the same judge config across conditions for comparable summaries.
