# Changelog — v6 reconciliation (2026-07-23)

Source of truth: `runs/direction_a_v6/reports/` (generated 2026-07-23 10:47 UTC;
`answer_source=v5`, paired covert-failure S, 2000 bootstrap reps).
Edits are minimal; structure, wording, and argument preserved.

## Numerical changes (required by completed v6)

| # | File / location | Before → After | Basis |
|---|---|---|---|
| 1 | `04afindings.tex` §4.2 Qwen3-8B para | Directional-ablation mean SFS **0.57 → 0.58** | v6 mean SFS(Dir) = 0.5845 (paired-S correction on the two high-over-warning Qwen dir/steer cells) |
| 2 | `04afindings.tex` §4.2 Qwen3-8B para | strongest steering dose SFS **0.60 → 0.61** | v6 Qwen `steering_a1.5` mean SFS = 0.6133 |
| 3 | `09appendix.tex` Tab. `appendix-model-family`, Qwen3-8B row | Dir **.57 → .58** (bold) | same as #1 |

No other numeric value in the paper differs from the completed v6 bundle at the
paper's printed precision. Confirmed unchanged (v6 rounds identically): Qwen
steering 0.49, SHIPS 0.13, neuron 0.12; OLMo-3-Think Dir .25 / neuron .14 /
SHIPS .27; per-model steering-dose Potency ladders (Qwen 0.05/0.13/0.23,
OLMo-3-Think 0.05/0.33/0.77); Qwen grid-average 0.28.

## Terminology / metric-description changes (accuracy)

| # | File / location | Change |
|---|---|---|
| 4 | `03prelim.tex` §3.2 (S axis def.) | Added: because `S = 1 − clip(U_c − U_b)` penalizes only covert failures U and is insensitive to over-warnings O, it is a **threat-weighted** monitorability measure — equivalently a **Covert-Failure Retention** score — not a symmetric trace–answer agreement score. |
| 5 | `03prelim.tex` §3.3 (eq. 3 prose) | **Corrected an error:** old text said "S rewards keeping the *gap* near the baseline's." S depends only on U, not the gap. Now: "because S = 1 − clip(U_c − U_b) depends only on U_c, it rewards keeping the covert-failure rate near the baseline's." Also added: the signed gap `g = U − O = P(answer harmful) − P(trace predicts unsafe)` is a **descriptive** net under-/over-warning measure on the common paired prompt set and is **not a prompt-level agreement metric** (U and O can cancel); we also report A and trace FNR where relevant. Added FNR to the list of reported quantities. |
| 6 | `04bfindings.tex` §4.2.1 (gap def.) | Added: the signed gap is a descriptive net measure, **not** a prompt-level agreement score (U and O can cancel); we also inspect paired agreement A and trace FNR per cell. |

## Non-rendering flag added for coauthors

| # | File / location | Change |
|---|---|---|
| 7 | `09appendix.tex` top comment | Expanded the existing "IMPORTANT FOR FINAL REAGGREGATION" LaTeX comment with the concrete v6 status (see the unresolved-items list). Comment only — no rendered output changes. |

## Explicitly checked and left unchanged (already consistent with completed v6)

- Judge-validation tables (`03prelim.tex` Tab. `judge-val`; `09appendix.tex`
  Tab. `appendix-validation`): all rows match `validation_report.json` exactly —
  Answer harmfulness F1 .782 / κ .575; Trace predicts unsafe F1 .790 / κ .600;
  Trace contains safety reasoning F1 .851 / κ .650 (= `has_safety_reasoning`,
  n=20 unique); Pathway transfer-domain F1 .976 / κ .963 (= `eval_sample_n180`
  n=180 row). No change.
- Axis equations (P, Q, S, SFS, U, O, A, FNR) already match the v6 definitions.

## Deliverables

- Revised sources: `papers/ARR_Aug_SafetyIntervention/` (edited in place).
- Repackaged: `papers/ARR_Aug_SafetyIntervention_v6revised.zip`.
- Original preserved: `papers/ARR_Aug_SafetyIntervention.orig.bak.zip`.
