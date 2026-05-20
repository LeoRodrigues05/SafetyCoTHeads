# Experiment 6: CoT Trajectory Analysis

## Goal

Measure how ablation changes the structure of safety reasoning across the
response trajectory, not only the final answer.

## Config Folder

- `configs/experiments/exp06_cot_trajectory_analysis/`

No runnable YAML exists yet.

## Current Code Hooks

- Sentence splitting and trajectory rows:
  `src/safety_cot_heads/analysis/trajectory.py`
- Judge pipeline:
  `scripts/run_judge.py`
- Generation pipeline:
  `scripts/run_generation.py`

## Intended Data Flow

1. Generate completions in normal and CoT-prompted modes for baseline and
   selected ablation conditions.
2. Split each completion into sentences.
3. Convert each sentence index into a cumulative partial response:
   `response[:sentence_i]`.
4. Judge each cumulative partial response with the same safety-label schema.
5. Compute trajectory metrics such as first harmful flip, first safety-reasoning
   sentence, safety-reasoning fraction, and contradiction between early
   reasoning and final answer.

## Missing Pieces

- A trajectory-specific generation config.
- A CLI that expands completions into cumulative sentence rows.
- A judge command or wrapper for sentence-level trajectory rows.
- Aggregate plots and summaries for the trajectory metrics.

## Review Checks

- Keep this experiment separate from final-answer judging.
- Pair baseline and ablated rows by prompt ID.
- Treat CoT text as behavioral evidence, not direct proof of internal reasoning.
