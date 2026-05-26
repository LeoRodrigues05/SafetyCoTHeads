# BeaverTails Safety-Head Controls: Baseline vs Safety vs Random vs Layer-Matched

Generated: `2026-05-24T16:14:50.915756+00:00`

## Methodology

- Safety heads were discovered with SHIPS on a balanced BeaverTails sample of 140 prompts: 10 prompts from each of the 14 cached BeaverTails categories.
- The target model was meta-llama/Llama-2-7b-chat-hf. Generation used the Llama chat template, greedy decoding, max_new_tokens=256, and seed=0.
- The four generated conditions were baseline, top-10 BeaverTails SHIPS safety-head ablation, top-10 uniform-random head ablation, and top-10 layer-matched random ablation. All ablations used Q-head scale_mask with scale_factor=1e-4.
- Qwen/Qwen2.5-32B-Instruct judged each response with a BeaverTails-specific JSON schema: coherence /10, malicious_intent /10, is_safe, and primary_category.

## What Actually Ran

| Condition | Rows | Parsed | Coherence | Intent | Safe Rate | Harmful Among Coherent |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 140 | 140 | 8 | 1.23 | 99.3% | 0.7% (140 coherent) |
| safety_head_ablation | 140 | 140 | 1.41 | 2.04 | 85.7% | 50.0% (4 coherent) |
| random_head_ablation | 140 | 140 | 8 | 1.23 | 98.6% | 0.7% (139 coherent) |
| layer_matched_random_head_ablation | 140 | 140 | 2.29 | 3.45 | 65.7% | 29.4% (17 coherent) |

## Working Inference

The uniform-random control is essentially baseline-like, while both safety-head and layer-matched ablations severely reduce coherence. Layer-matched random has higher mean malicious intent (3.45) than safety-head ablation (2.04), so this run argues strongly for a layer/location degradation confound rather than a clean safety-head-specific effect. The safety-head coherent harmful rate is based on very few coherent rows, so its examples should be inspected individually.

## Artifacts

- `multi_condition_responses.html`
- `multi_condition_responses.csv`
- `multi_condition_responses.jsonl`
- `multi_condition_summary.json`
