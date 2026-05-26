# BeaverTails Top-5 Safety-Head Sweep: Baseline vs Safety vs Random vs Layer-Matched

Generated: `2026-05-26T13:44:42.041830+00:00`

## Methodology

- Safety heads were discovered with SHIPS on a balanced BeaverTails sample of 140 prompts: 10 prompts from each of the 14 cached BeaverTails categories.
- The target model was meta-llama/Llama-2-7b-chat-hf. Generation used the Llama chat template, greedy decoding, max_new_tokens=256, and seed=0.
- The four generated conditions were baseline, top-5 BeaverTails SHIPS safety-head ablation, top-5 uniform-random head ablation, and top-5 layer-matched random ablation. All ablations used Q-head scale_mask with scale_factor=1e-4.
- Qwen/Qwen2.5-32B-Instruct judged each response with a BeaverTails-specific JSON schema: coherence /10, malicious_intent /10, is_safe, and primary_category.

## What Actually Ran

| Condition | Rows | Parsed | Coherence | Intent | Safe Rate | Harmful Among Coherent |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 140 | 140 | 8 | 1.23 | 99.3% | 0.7% (140 coherent) |
| safety_head_ablation_top5 | 140 | 140 | 4.74 | 2.47 | 80.7% | 13.3% (83 coherent) |
| random_head_ablation_top5 | 140 | 140 | 8 | 1.26 | 99.3% | 0.7% (140 coherent) |
| layer_matched_random_head_ablation_top5 | 140 | 140 | 7.91 | 1.32 | 97.9% | 1.5% (137 coherent) |

## Working Inference

This report should be read as a paired dose-control comparison. Relative to baseline, safety-head ablation has mean coherence 4.74 (-3.26 vs baseline) and mean malicious intent 2.47 (1.24 vs baseline); uniform random has coherence 8 (0 vs baseline) and intent 1.26 (0.0286 vs baseline); layer-matched random has coherence 7.91 (-0.0857 vs baseline) and intent 1.32 (0.0929 vs baseline). A safety-head-specific claim is strongest where the safety condition moves more than both controls without a comparable coherence collapse; otherwise the examples should be treated as evidence for a layer/location or general degradation confound.

## Artifacts

- `multi_condition_responses.html`
- `multi_condition_responses.csv`
- `multi_condition_responses.jsonl`
- `multi_condition_summary.json`
