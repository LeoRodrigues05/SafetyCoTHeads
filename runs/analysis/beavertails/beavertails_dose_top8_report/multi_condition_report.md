# BeaverTails Top-8 Safety-Head Sweep: Baseline vs Safety vs Random vs Layer-Matched

Generated: `2026-05-26T13:44:42.045669+00:00`

## Methodology

- Safety heads were discovered with SHIPS on a balanced BeaverTails sample of 140 prompts: 10 prompts from each of the 14 cached BeaverTails categories.
- The target model was meta-llama/Llama-2-7b-chat-hf. Generation used the Llama chat template, greedy decoding, max_new_tokens=256, and seed=0.
- The four generated conditions were baseline, top-8 BeaverTails SHIPS safety-head ablation, top-8 uniform-random head ablation, and top-8 layer-matched random ablation. All ablations used Q-head scale_mask with scale_factor=1e-4.
- Qwen/Qwen2.5-32B-Instruct judged each response with a BeaverTails-specific JSON schema: coherence /10, malicious_intent /10, is_safe, and primary_category.

## What Actually Ran

| Condition | Rows | Parsed | Coherence | Intent | Safe Rate | Harmful Among Coherent |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 140 | 140 | 8 | 1.23 | 99.3% | 0.7% (140 coherent) |
| safety_head_ablation_top8 | 140 | 140 | 2.14 | 2.92 | 75.0% | 45.5% (22 coherent) |
| random_head_ablation_top8 | 140 | 140 | 8 | 1.25 | 98.6% | 0.7% (140 coherent) |
| layer_matched_random_head_ablation_top8 | 140 | 140 | 5.06 | 3.58 | 65.7% | 33.3% (102 coherent) |

## Working Inference

This report should be read as a paired dose-control comparison. Relative to baseline, safety-head ablation has mean coherence 2.14 (-5.86 vs baseline) and mean malicious intent 2.92 (1.69 vs baseline); uniform random has coherence 8 (0 vs baseline) and intent 1.25 (0.0214 vs baseline); layer-matched random has coherence 5.06 (-2.94 vs baseline) and intent 3.58 (2.35 vs baseline). A safety-head-specific claim is strongest where the safety condition moves more than both controls without a comparable coherence collapse; otherwise the examples should be treated as evidence for a layer/location or general degradation confound.

## Artifacts

- `multi_condition_responses.html`
- `multi_condition_responses.csv`
- `multi_condition_responses.jsonl`
- `multi_condition_summary.json`
