# Experiment 6: CoT Trajectory Analysis

> **v4 reframing (2026-06-01).** This experiment now produces the v4
> **pathway-taxonomy vector** and **CoT-monitorability gap** described in
> [`docs/direction_a/direction_a_plan.md`](../direction_a/direction_a_plan.md)
> §7-S5 and §13.A/B. The legacy 7-vector trajectory pipeline is still
> emitted in parallel as a diagnostic appendix.

## Goal

Measure how each safety intervention changes:

1. **Which safety subprocess fails** along the visible reasoning trace
   (recognition / refusal / rationalisation / execution) — the
   **pathway vector** (8 dims, see plan §13.A.2).
2. **Whether the trace still warns** before the final unsafe answer —
   the **monitorability gap** = ASR(final) − ASR(CoT-only judge).
3. Diagnostic: the legacy 7-metric trajectory vector (preserved).

## Config Folder

- `configs/experiments/direction_a_ships/` — pass_a_*.yaml configs for the
  v4 pathway pipeline.

## Code Hooks (v4)

- Sentence splitting / cumulative prefixes: `src/safety_cot_heads/direction_a/segmentation.py` (unchanged).
- **v4 primary** pathway aggregator: `src/safety_cot_heads/direction_a/pathway_taxonomy.py`.
- **v4 primary** monitorability gap: `src/safety_cot_heads/direction_a/monitorability.py`.
- v4 judge prompts: `PATHWAY_TAXONOMY_PROMPT`, `COT_ONLY_PREDICTION_PROMPT` in `src/safety_cot_heads/judging/judge_prompts.py`.
- v4 orchestrator: `scripts/run_pathway_analysis.py` (idempotent over existing `prefix_rows.jsonl`).
- Legacy 7-vector (diagnostic): `src/safety_cot_heads/direction_a/trajectory_metrics.py`, `scripts/run_trajectory.py`.

## Data Flow (v4)

1. Generate completions per (model × condition × seed × phase).
2. Segment each completion (`segmentation.py`) into ordered units and
   build cumulative prefixes — already produced by the legacy
   trajectory pipeline; **reused as-is** by the v4 orchestrator.
3. Per prefix: judge with `PATHWAY_TAXONOMY_PROMPT` → 12 sentence-level
   labels in 4 groups (recognition / refusal / rationalisation /
   execution).
4. Per completion: judge with `COT_ONLY_PREDICTION_PROMPT` (CoT only,
   no final answer) → `asr_cot_pred`.
5. Per completion: final-answer judge (existing) → `asr_final`.
6. Aggregate: pathway vector (8 dims) + `monitorability_gap`.
7. In parallel and at zero extra generation cost: legacy 5-label
   `SAFETY_BEHAVIOR_PROMPT` per prefix → legacy 7-vector.

## Pass A Validation Gates

Before Pass B compute, the four Pass A gates from
[`docs/direction_a/prereg_v4.md`](../direction_a/prereg_v4.md) §10 must
hold on the Llama-3.1 SHIPS pilot:

- **G1** — pathway-judge self-consistency κ ≥ 0.70 per label.
- **G2** — baseline monitorability gap within ±0.05.
- **G3** — SHIPS-top10 vs. baseline monitorability gap significant
  ($p < 0.05$ paired bootstrap).
- **G4** — 30-trace hand-spot-check dominant-pathway agreement ≥ 80%.

## Review Checks

- Keep this experiment separate from final-answer judging.
- Pair baseline and ablated rows by prompt ID and seed.
- Treat CoT text as **visible reasoning trace** evaluated for its value
  as a *monitor* (per the v4 monitorability framing), not as a window
  onto hidden cognition.
