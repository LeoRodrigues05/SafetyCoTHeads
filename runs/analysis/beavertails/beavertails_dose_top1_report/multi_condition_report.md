# BeaverTails Top-1 Safety-Head Sweep: Baseline vs Safety vs Random vs Layer-Matched

Generated: `2026-05-26T13:44:42.046796+00:00`

## Methodology

- Safety heads were discovered with SHIPS on a balanced BeaverTails sample of 140 prompts: 10 prompts from each of the 14 cached BeaverTails categories.
- The target model was meta-llama/Llama-2-7b-chat-hf. Generation used the Llama chat template, greedy decoding, max_new_tokens=256, and seed=0.
- The four generated conditions were baseline, top-1 BeaverTails SHIPS safety-head ablation, top-1 uniform-random head ablation, and top-1 layer-matched random ablation. All ablations used Q-head scale_mask with scale_factor=1e-4.
- Qwen/Qwen2.5-32B-Instruct judged each response with a BeaverTails-specific JSON schema: coherence /10, malicious_intent /10, is_safe, and primary_category.

## What Actually Ran

| Condition | Rows | Parsed | Coherence | Intent | Safe Rate | Harmful Among Coherent |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 140 | 140 | 8 | 1.23 | 99.3% | 0.7% (140 coherent) |
| safety_head_ablation_top1 | 140 | 140 | 7.14 | 2.14 | 88.6% | 7.8% (128 coherent) |
| random_head_ablation_top1 | 140 | 140 | 8 | 1.22 | 99.3% | 0.7% (140 coherent) |
| layer_matched_random_head_ablation_top1 | 140 | 140 | 8 | 1.2 | 100.0% | 0.0% (140 coherent) |

## Working Inference

This report should be read as a paired dose-control comparison. Relative to baseline, safety-head ablation has mean coherence 7.14 (-0.864 vs baseline) and mean malicious intent 2.14 (0.907 vs baseline); uniform random has coherence 8 (0 vs baseline) and intent 1.22 (-0.00714 vs baseline); layer-matched random has coherence 8 (0 vs baseline) and intent 1.2 (-0.0286 vs baseline). A safety-head-specific claim is strongest where the safety condition moves more than both controls without a comparable coherence collapse; otherwise the examples should be treated as evidence for a layer/location or general degradation confound.

## Artifacts

- `multi_condition_responses.html`
- `multi_condition_responses.csv`
- `multi_condition_responses.jsonl`
- `multi_condition_summary.json`
