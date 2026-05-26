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
| random_head_ablation | 140 | [True] | [10] | 1042.0 | 0 | 121/140 |

## Baseline vs ablation

- Paired prompt IDs: 140
- Identical completions: 1
- Different completions: 139
- Baseline refusal-ish -> ablation non-refusal-ish: 1
- Baseline non-refusal-ish -> ablation refusal-ish: 2
- Both refusal-ish: 119
- Neither refusal-ish: 18
- Median common prefix before the first changed character: 117.0 chars

## Caveats

- These are completion-level comparisons, not the robust judge pipeline output.
- The refusal counts are from a simple phrase heuristic, useful for triage but not final evidence.
- Use matched random controls and judge outputs before making the headline claim.
