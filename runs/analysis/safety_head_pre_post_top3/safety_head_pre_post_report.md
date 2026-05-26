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
| baseline | 100 | [False] | [] | 1123.5 | 0 | 94/100 |
| safety_head_ablation_top3 | 100 | [True] | [3] | 1102.0 | 0 | 66/100 |

## Baseline vs ablation

- Paired prompt IDs: 100
- Identical completions: 0
- Different completions: 100
- Baseline refusal-ish -> ablation non-refusal-ish: 29
- Baseline non-refusal-ish -> ablation refusal-ish: 1
- Both refusal-ish: 65
- Neither refusal-ish: 5
- Median common prefix before the first changed character: 3.0 chars

## Caveats

- These are completion-level comparisons, not the robust judge pipeline output.
- The refusal counts are from a simple phrase heuristic, useful for triage but not final evidence.
- Use matched random controls and judge outputs before making the headline claim.
