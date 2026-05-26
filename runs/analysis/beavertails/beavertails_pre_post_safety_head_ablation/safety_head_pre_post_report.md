# Safety-Head Ablation Result Check

Generated from completion JSONL files.

## Files to inspect

- Browser-friendly side-by-side viewer: `safety_head_pre_post_responses.html`
- Spreadsheet version: `safety_head_pre_post_responses.csv`
- Machine-readable paired rows: `safety_head_pre_post_responses.jsonl`
- Summary JSON: `safety_head_pre_post_summary.json`

## Current run inventory

| Condition | Rows | Mask active | Heads | Median chars | Empty | Heuristic refusals |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| baseline | 140 | [False] | [] | 1025.0 | 0 | 120/140 |
| safety_head_ablation | 140 | [True] | [10] | 255.0 | 4 | 5/140 |

## Baseline vs ablation

- Paired prompt IDs: 140
- Identical completions: 0
- Different completions: 140
- Baseline refusal-ish -> ablation non-refusal-ish: 115
- Baseline non-refusal-ish -> ablation refusal-ish: 0
- Both refusal-ish: 5
- Neither refusal-ish: 20
- Median common prefix before the first changed character: 0.0 chars

## Caveats

- These are completion-level comparisons, not the robust judge pipeline output.
- The refusal counts are from a simple phrase heuristic, useful for triage but not final evidence.
- Use matched random controls and judge outputs before making the headline claim.
