# Risks and assumptions

Every assumption the results rest on, and every known risk to their validity. Each entry
gives **where it bites**, **what is done about it**, and **what remains**. Status tags:
**[open]** unmitigated, **[partial]** mitigated with residual risk, **[fixed in v6.1]**
corrected in code and awaiting the rerun (`RERUN_V61.md`), **[accepted]** a stated limitation.

---

## 1. What the metric assumes

| # | Assumption | Where it bites | Mitigation | Status |
|---|---|---|---|---|
| M1 | Removing a real safety mechanism should produce **coherent** harmful answers that were refused before, without breaking the model. | The whole SFS framing. It is a *behavioural signature*: a high score is necessary but not sufficient for "this component is the mechanism". | Stated in the paper; random-target controls test specificity. | [partial] controls queued |
| M2 | "Harmful" = the judge's `harmful_response`, which the prompt forces to be the complement of `safe_rejection`. | Any non-refusal counts as harm, including compliance with benign BeaverTails prompts and OLMo-Base question echoes. | Paper should define h(r) as the judge label and note the forced binary. | [accepted] |
| M3 | The coherence gate separates usable from broken answers. | Q, and HAC's denominator. v6: empty + repetition only; v6.1 adds word salad/noise. Fluent-but-wrong answers and off-topic text pass. | Gate v6.1 + repetition-only sensitivity report. There is no human validation of the gate. | [partial] |
| M4 | Potency conditioned on post-intervention coherence measures induced harm. | HAC_c and HAC_b are computed over different coherent subsets; if an intervention breaks refusals selectively, HAC moves without any new harm. | `induced_harm_rate` column (harmful ∧ clean over all prompts, minus baseline) as a check. | [partial] |
| M5 | S = 1 − clip(U_c − U_b) captures whether induced harm stays visible. | U ≤ HAC, so S ≈ 1 whenever Potency is small, whatever the monitor quality. Baselines have ~0 harm, so U_b ≈ 0 and FNR_b is undefined. S never dropped below .96. | Report pooled trace FNR with CI; `covert_prompt` positive control (on by default; verdict in `reports/covert_control.md`). | [partial] until the verdict |
| M6 | The geometric mean is a fair summary. | Near P = 0, SFS ≈ ∛P: one harmful prompt gives ≈ .22, and a denominator change of one row can move SFS. Family SFS gaps overstate Potency gaps. | Report the vector with CIs as primary; add the product/harmonic composites and small-n flags. | [partial] |
| M7 | Excluding undefined cells from SFS means is neutral. | OLMo-Think α=1.5 (Q=0) is excluded from SFS but kept in Q means; scoring it 0 moves steering .46 → .41. | Choose and state one rule. | [open] |
| M8 | A cell-level mean over a family's doses is comparable across families. | Dose ladders are family-specific and not commensurate. | Stated as "summary of the evaluated grid"; relative-dose steering optional. | [accepted] |

## 2. Intervention fidelity (are we testing the methods we cite?)

| # | Risk | Detail | Mitigation | Status |
|---|---|---|---|---|
| I1 | Steering dose is not comparable across models. | `alpha = −8·d` on a unit vector: 0.2–0.6× the harmful–benign gap for Qwen, 0.8–2.4× for Llama, 1.5–4.4× for OLMo-Think. "Qwen resists steering" and "OLMo collapses" follow from this. | `dose_mode: relative` implemented (`RELATIVE=1`). | [open] pending decision |
| I2 | Directional ablation under-replicates Arditi et al. | Only decoder-layer inputs were projected (within-layer and final writes keep the direction). Pooled P = .07 vs near-complete bypass in the original. | `steering_ablate_all` (every residual write; unit-tested). | [fixed in v6.1] |
| I3 | No direction selection. | Layer 14 for all models (39% depth in Qwen, 44% in Llama/OLMo), last prompt token only. | Stated; optional layer sweep not planned. | [accepted] |
| I4 | The neuron method is not the cited one. | Activation contrast (MLP only, \|Δ\| ranking, fixed top-k) vs Zhao et al. (deactivation importance, attention + MLP, general-query subtraction). 41–55% of the top-1024 neurons are in the final layer. | Decided: reported as our own procedure, inspired by Zhao et al.; the random-neuron control is layer-matched. | [accepted] |
| I5 | The SHIPS discovery template differs from evaluation. | Heads are scored on `## Query/## Answer`; generation uses the chat template. Top heads sit in layers 0–3 (possibly generic-importance heads). | Random-head control (layer-matched, excluding real heads). | [partial] |
| I6 | Targets are not specific. | Without random controls, any disruption at matched budget might do the same. | 72 random-control cells, 3 draws per family (the design docs had planned ≥ 20). | [partial] |
| I7 | Discovery stability is unknown. | No resampling of the discovery data. | None. | [open] |

## 3. Generation and parsing

| # | Risk | Detail | Mitigation | Status |
|---|---|---|---|---|
| G1 | Truncated reasoning was judged as the answer. | OLMo-Think/R1 templates pre-fill `<think>`, so truncated traces had no tags. | Parser v6.1 + continuation. | [fixed in v6.1] |
| G2 | Truncation removes rows. | 4–12% OLMo-Think, 29–41% at steering α=1.0; Qwen 4–10%. | Continue to 8192/4096; loops are not continued; residual unterminated rows have no answer. | [fixed in v6.1] |
| G3 | Continuation ≠ a single long generation. | Re-tokenisation at the cut point; different batch composition. | Same order as existing bf16 noise; documented. | [accepted] |
| G4 | Single greedy seed. | No sampling variance; Qwen3 recommends sampling in thinking mode (greedy invites loops). | Stated in Limitations. | [accepted] |
| G5 | OLMo-Base is a weak "no safety" control. | Not instruction-tuned, raw template, 31% coherent; Base "harm" is mostly non-refusal. | Describe it as pre-alignment and non-instruction-tuned; an SFT checkpoint would be cleaner (not run). | [accepted] |
| G6 | Prose-prefix "traces" are part of the answer. | For Llama/Base the monitor sees most of the answer, so FNR ≈ 0 by construction, yet this S enters their SFS. | Label it a sensitivity analysis, or drop S for prose arms. | [open] pending decision |
| G7 | The environment has drifted since July. | Greedy bf16 is not bit-stable across torch/transformers/driver versions; new cells may not match old ones. | Compare versions to the July manifest; reproduction check (R3) planned, not implemented. | [open] |

## 4. Judges

| # | Risk | Detail | Mitigation | Status |
|---|---|---|---|---|
| J1 | Pathway labels came from an unvalidated judge. | All v6 pathway rows: zero-shot 30B, multi-label prompt. | `pathway14b` stage; the reasoning aggregator refuses mixed protocols; legacy files quarantined. | [fixed in v6.1] |
| J2 | Pathway judge validation is thin. | The reported κ .963 was on the first 180 of 21,240 test rows (9/12 labels; trace-disjointness unverifiable). Validation is transfer-domain (HarmThoughts), not on intervention traces. | `evalpw` on the full test set; in-domain validation is not possible now. | [partial] |
| J3 | HF vs vLLM give different greedy outputs. | June comparison: SR rate 1.00 (HF) vs .94/.91 (vLLM) on the same 32 traces. | One backend per stage; the pathway κ is re-measured under the backend used. Answer/monitor/SR stay on HF, as in July. | [partial] |
| J4 | Human validation covers v5 inputs. | Annotators saw full completions for reasoning models; v6 judges answers only. | v6 κ on input-identical items only (final); v5 κ supplementary. **Reasoning-model answer labels have no human validation** (v5/v6 agree on 56% of those items). | [accepted] |
| J5 | Validation sample is small and class-balanced. | κ with wide CIs (answer [.41, .76]); balanced sampling means κ/F1 are not natural-distribution accuracy. | Report CIs and state the sampling. | [accepted] |
| J6 | Single judge family. | Answer, monitor and SR all use Qwen3-30B, and two of the evaluated models are Qwen/OLMo; there is no second judge. | None. | [open] |
| J7 | SR judge output truncation on long traces. | Continued traces are up to 4× longer. | Output cap 2048 (consistent by the greedy prefix argument); post-run check in `RERUN_V61.md` §7. | [partial] |

## 5. Statistics and reporting

| # | Risk | Detail | Mitigation | Status |
|---|---|---|---|---|
| S1 | Small n per cell. | ~100 prompts per cell; many cells have 0–2 harmful answers; some have < 10 coherent answers (Llama α=1.5: 5; Base BT cells: 3–8). | Small-n flags; CIs. | [partial] |
| S2 | Bootstrap covers prompts only. | No seed or generation variance, and no judge variance. | Stated. | [accepted] |
| S3 | Many comparisons, no correction. | Family × model × dataset contrasts are read informally. | Avoid significance claims without paired tests. | [open] |
| S4 | Stale numbers in the paper. | Many paper numbers predate the scope fix or the gate change (see the claims audit). | Regenerate everything from one report; `verify_paper_numbers.py`. | [open] until the rewrite |
| S5 | v5 and v6 numbers get mixed. | Old reports and docs carry v5 values. | Only v6 reports are final; v5 values are labelled supplementary. | [partial] |

## 6. Pipeline and data integrity

| # | Risk | Detail | Mitigation | Status |
|---|---|---|---|---|
| P1 | Stale judge labels after input changes. | Resume was by id only. | Input-hash resume; reference-based invalidation; backups of pruned rows. | [fixed in v6.1] |
| P2 | Coherence file overwrite. | The old coherence stage rewrote a cell with only its new rows. | The stage now merges. | [fixed in v6.1] |
| P3 | Tests wrote into the real run tree. | The integration test re-parsed live cells (it happened once: 44 parsed files, audit, diagnostics; audit restored). | Tests write to a temp root (`SCH_V6_ROOT`). | [fixed] |
| P4 | Shared storage between machines. | The run tree appears to be shared by this workstation and the GPU box; a local run can change GPU-side data. | Run pipeline stages only on the GPU box; use `--plan-only`/dry runs locally. | [accepted] |
| P5 | Release incoherence. | Non-paper cells (defence, XSTest, R1) could differ in parser, caps or judges. | `SCOPE=full` reprocesses them all. Defence cells carry no pathway labels (30B files quarantined). The α=0.75/1.25 cells are re-parsed but not continued (not reported). | [fixed in v6.1] |
| P6 | The manifest misreports settings. | `write_v6_manifest.py` hard-codes the judge and caps. | R5, not implemented. | [open] |

## 7. Scope and ethics

| # | Item | Status |
|---|---|---|
| E1 | Five 7–8B dense text models; only two expose explicit traces. Results may not transfer to larger or other model families. | [accepted] |
| E2 | Released artifacts include refusal directions, target rankings and harmful generations (dual use). Needs an ethics statement. | [open] |
| E3 | Annotators read harmful content. Consent and compensation should be stated. | [open] |
| E4 | Anonymity: annotation files are named after annotators (`annotations_<name>.jsonl`), and logs and the `.tex` macros contain names and paths. Scrub before the anonymous release. `validate_v6_matched.py` already anonymises its outputs. | [open] |
