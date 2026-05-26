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
| safety_head_ablation_top1 | 100 | [True] | [1] | 1059.0 | 2 | 48/100 |

## Baseline vs ablation

- Paired prompt IDs: 100
- Identical completions: 0
- Different completions: 100
- Baseline refusal-ish -> ablation non-refusal-ish: 46
- Baseline non-refusal-ish -> ablation refusal-ish: 0
- Both refusal-ish: 48
- Neither refusal-ish: 6
- Median common prefix before the first changed character: 0.0 chars

## Caveats

- These are completion-level comparisons, not the robust judge pipeline output.
- The refusal counts are from a simple phrase heuristic, useful for triage but not final evidence.
- Use matched random controls and judge outputs before making the headline claim.
