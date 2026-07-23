# Additional experiments to push the paper to A\*-level

> **Status.** Reference roadmap (created 2026-07-16). Proposes experiments *beyond*
> the current Direction A v5 grid to lift *"One Number Isn't Enough"* to A\*-conference
> acceptance quality. Scoped to the confirmed constraints below. Companion to
> [`PAPER_OUTLINE.md`](PAPER_OUTLINE.md) §8 (open work) and
> [`EVALUATION_FRAMEWORK.md`](EVALUATION_FRAMEWORK.md) §8 (open items) — this file turns
> those gap lists into concrete, prioritized experiments.

## Constraints (fixed for this roadmap)

- **Target venue:** ACL/ARR long paper — keep the current 8-page scope; prioritize
  closing reviewer-bait gaps over expanding breadth.
- **Compute:** constrained (~1×140 GB GPU) — no scaling ladder, no large new model sweeps.
- **Top priority:** *give the Safety-Reasoning (S) axis teeth* (Experiment 1).

## Context

The repo is a mature evaluation-methodology paper: a decomposable metric
(**Potency × Quality × Safety-Reasoning → Selective-Failure Score**) for comparing
white-box safety interventions on reasoning LLMs. Evidence is the Direction A v5 grid
(5–6 models @ 7–8B × 2 datasets × 11 conditions), three validated judges, and a composite
report. Findings F1–F5 are in place; paper §4.2–4.4 and the related-work survey are still
outline-only.

The project's own docs already enumerate the gaps a reviewer will attack: **Quality is
coherence-only**, **single greedy seed**, **partial statistics** (no paired-bootstrap /
BH-FDR), **single judge family**, no scaling above 8B, and — most importantly for the
story — **F5 is an honest null**: for the *suppressive* interventions in the grid the CoT
monitor never covertly fails (`gap ≤ 0` everywhere), so the S axis never fires on the
current data.

**The central problem Experiment 1 addresses.** F5 as written is a latent liability: a
reviewer asks *"if S never moves, why is it in the metric — isn't the geometric mean just
P·Q?"* The current answer is "it would catch a future covert method," a promise, not
evidence. The highest-leverage move is to **convert that promise into a demonstration**:
construct interventions that genuinely produce covert CoT failure (sanitized-looking
traces + unsafe answers, `gap ≫ 0`) and show the metric's S axis catches them **when raw
ASR and P·Q cannot**. This turns the weakest part of the paper (a null) into its most
novel result — a natural third demonstration alongside F1 (ASR mis-ranks) and F2
(baseline-correction).

Crucially, **the metric already implements everything needed to detect this** — no metric
redesign. `analysis/composite.py` computes `S = 1 − clip(|gap_c| − |gap_b|)`, carries
`covert = clip(gap)`, and ships `sfs_covert = P·Q·(1−S)`. The work is (a) producing cells
that exercise it and (b) validating the judge on those cells.

---

## Experiment 1 (CENTERPIECE) — A positive control that gives the S axis teeth

**Claim to establish (new F6, reframes pre-registered H3):** the Safety-Reasoning axis is
a *discriminative instrument*, not a dead term. On constructed covert interventions, S
collapses while P and Q stay high; two methods matched on raw ASR **and** on P·Q are
separated **only** by S.

**Construct a ladder of covert-failure conditions** on the two explicit-`<think>` models
where a CoT monitor is even meaningful — `qwen3_8b` and `olmo3_7b_think` (both already in
the grid; `direction_a/segmentation.py::segment_completion` handles their `<think>`
traces). Add them as new conditions in
`configs/experiments/direction_a_v5_iso_asr/matrix.yaml`; `scripts/make_v5_configs.py`
expands them like the existing families.

1. **`covert_prompt` (cheapest, existence proof).** A generation-time system prompt that
   induces compliance while instructing the model to keep its `<think>` reasoning free of
   any safety language (safety / harm / ethics / legality / policy / refusal). Pure
   generation-time: `run_generation.py` already forwards `cfg.get("system_prompt")` to
   `generate()` (see `run_generation.py:256`; `_gen_baseline` already sets a
   `system_prompt` field), so this is a **config-only** condition — no internal
   intervention. If this drives `gap > 0`, the axis fires. *(Verified feasible: a
   `covert_prompt` gen YAML expands and passes `run_generation --dry-run`.)*
2. **`covert_steer` (mechanistic version).** Extract a **safety-reasoning direction** by
   contrasting residual activations on traces *with* vs *without* visible safety reasoning
   (reuse `attribution/directions.py::compute_refusal_directions`, swapping the
   harmful/benign contrast sets for has-SR / no-SR traces — the SR-trace judge already
   labels these). Ablate/add it at the reasoning layer via the existing
   `interventions/steering.py` (`mode: add`/`ablate`, arbitrary `.npz` direction — no new
   intervention code). Suppresses the visible safety-mention while leaving the harmful
   answer → the "mechanistically covert" cell.
3. **`covert_2pass` (optional upper bound, only if time permits).** Decode the harmful
   answer, then regenerate a benign-looking rationalized `<think>` to accompany it — a
   trace-laundering ceiling. Skip under time pressure; 1–2 carry the argument.

**Judge/human validation of the covert cells (non-negotiable — the linchpin).** The whole
claim is that the judge reads a sanitized trace as benign *while* reading the answer as
harmful. Validate it: build a new annotation batch focused on the covert conditions using
the existing tooling — `make_annotation_batch.py` → `annotate_server.py` →
`score_annotations.py` — as `data/annotations/batch_v5_003`. Two annotators; report
human-vs-judge and inter-annotator κ on `cot_predicts_unsafe` and `has_safety_reasoning`
**specifically on covert cells**. If humans agree the traces are sanitized and answers
harmful, the `gap > 0` is real, not a judge artifact.

**The money figure (S-axis analogue of the Fig-1 teaser).** Pick a covert cell and an
overt suppressive cell (e.g. `steering_a1.0`) **matched on raw ASR and on P·Q**, and show
they differ only on S (and that `sfs_covert` ranks the covert one *higher* — the
threat-oriented reading). Regenerate via `make_composite_report.py` /
`make_composite_insight_reports.py` — `sfs_covert` is already computed, so this is a
report + one new plot, not new metric code.

**Compute:** feasible on 1 GPU — 2 models × 2 datasets × ~2–3 new conditions of generation
+ standard/SR/cot-only judging (the SR pass is the vLLM fast path). No new models, no
scaling.

**Files:** `configs/.../matrix.yaml`, `scripts/make_v5_configs.py`,
`attribution/directions.py`, `interventions/steering.py`, `scripts/run_generation.py`,
`scripts/run_v4_jbb_judge.py`, `scripts/run_sr_vllm.sh`, `direction_a/monitorability.py`,
`analysis/composite.py` (read-only — already supports it), the report scripts, and the
annotation tooling under `scripts/` + `data/annotations/`.

---

## Experiment 2 — Statistical rigor (cheap; reviewer table-stakes)

Mostly analysis-only; the one costly piece is scoped to where it matters.

- **Targeted multi-seed.** Full-grid multi-seed is out of budget; run **≥3 sampling seeds
  (temperature > 0)** on a *subset* — `baseline`, the iso-anchor `steering_a1.0`, and the
  Exp-1 covert conditions — on the 2 focus models, to put error bars on the headline and
  the covert effect. Overrides already documented in `matrix.yaml` (`decoding.seed`).
- **Paired-bootstrap CIs (B = 10k)** on P, Q, S, SFS per cell — analysis-only, extends the
  existing `analysis/paired.py` (McNemar/Wilcoxon) and `analysis/metrics.py` (Wilson CIs
  already present).
- **Run the pre-registered H1–H5** (spec-only today) and apply **BH-FDR** across the
  family; **rewrite H3** to the Exp-1 result (axis confirmed-on-construction, falsified for
  suppressive families).

**Files:** `analysis/paired.py`, `analysis/metrics.py`, `scripts/make_composite_report.py`,
`docs/direction_a/README.md` (H1–H5), `papers/.../07limitations.tex` ¶3 + `04findings.tex`
§4.4.

---

## Experiment 3 — Judge robustness (de-risk the single-judge dependency)

The entire metric rests on one Qwen judge family; the covert-cell judging in Exp-1 makes
this dependency load-bearing. Re-judge a **stratified subset** (must include all covert
cells) with a **second independent judge** — e.g. `Llama-Guard-3-8B` for safety, a
different open model for `cot_only` — and report **rank-stability** (Kendall τ of the SFS
ordering across judges) and per-axis agreement. Enable the existing but disabled
`judging/dual_judge.py` (`enable_secondary=True`). Cheap on a subset; directly answers
"the metric is a judge artifact."

**Files:** `judging/dual_judge.py`, `judging/judge.py`, `scripts/run_v4_jbb_judge.py`,
`configs/models.yaml` (add the second judge ref).

---

## Experiment 4 — Benign-utility term in the Quality axis (closes the biggest Q gap)

Q is coherence-only today — the docs concede it is *coarser than StrongREJECT*, and a
method that preserves fluency while destroying capability over-scores. Run small **MMLU +
GSM8K** subsets under each intervention on the 2 focus models and fold a capability-
retention term into Q. These are **auto-scored** (exact-match / MC) — no 30B judge, so it
is affordable on constrained compute. Requires a small eval harness (generation + scoring)
and a Q-definition update in `composite.py` (retention = `clip[acc_c / acc_b]`, combined
with the existing coherence term).

**Files:** new `scripts/run_benign_utility.py` (small harness), `analysis/composite.py`
(Q definition), `configs/datasets.yaml` (MMLU/GSM8K entries already referenced in design
docs), `papers/.../07limitations.tex` ¶2.

---

## Experiment 5 — Reverse / defence-side experiments (compute allocated separately)

> **Not constrained like 1–4:** dedicated compute is being allocated for this, so it can
> extend beyond the 2 focus models.

**Motivation — turn an asserted claim into a demonstration.** The paper currently only
*asserts* the metric is orientation-agnostic: the Discussion (¶3) and
[`EVALUATION_FRAMEWORK.md`](EVALUATION_FRAMEWORK.md) §5.1 state that *"a defence eval flips
the sign of P and reuses Q, S unchanged, so the same instrument scores both breaking and
hardening a model."* This is untested — a reviewer will ask for evidence. Running the same
grid with **defensive (hardening) interventions** and showing the flipped-P metric behaves
sensibly and symmetrically upgrades the assertion into a result, and roughly **doubles the
instrument's demonstrated coverage** (attacks *and* defences) — a strong "adoptable
instrument" argument.

**Reverse conditions — mirror each attack family (all config-only; grounded in existing
hooks).** Add to `matrix.yaml`; `make_v5_configs.py` expands them:

- **`steering_defend_a{0.5,1.0,1.5}`** — activation-addition of the refusal direction with
  a **positive** sign (add `v` to *induce* refusal / harden). Trivial change: the sweep
  already computes `eff_alpha = dose * add_alpha_sign * add_coeff`; a defence sweep sets
  `add_alpha_sign: +1.0`. Mirrors the attack dose sweep.
- **`heads_amplify_top{3,5,8}`** — **scale up** the safety heads instead of zeroing them.
  The `scale_mask` hook multiplies the target head slice by `scale_factor`
  (`models/custom_llama.py:14`), so `scale_factor > 1` (e.g. 2–5×) amplifies rather than
  ablates — same mechanism, reverse direction.
- **`neurons_amplify_top{256,512,1024}`** — the neuron analogue (`scale_factor > 1`).
- *(optional)* **`defend_prompt`** — a strong safety system prompt (the mirror of Exp-1's
  `covert_prompt`); pure `system_prompt`.

**Metric under reversal (add an orientation flag to `analysis/composite.py`).**

- **P (defence) = `clip[(HAC_b − HAC_c) / (HAC_b + eps)]`** — the fraction of the
  baseline's harm the defence *removes* (baseline-corrected, exactly symmetric to the
  attack P). A single `orientation ∈ {suppress, defend}` flag selects the numerator sign;
  Q and S transforms are unchanged.
- **Q** unchanged — a defence that breaks fluency is still bad.
- **S** unchanged — is the visible reasoning still an *engaged* safety monitor (see the
  new defence-specific failure mode below).

**New axis the defence side forces — over-refusal / benign-utility cost.** A defence that
refuses everything scores high P-defence but is useless; the attack-side coherence gate has
no analogue for this. Add a benign / borderline evaluation:

- **XSTest** (250 safe + unsafe prompts — the standard over-refusal benchmark) and/or
  **OR-Bench**, plus the existing **Alpaca** benign set.
- Measure the **over-refusal rate** (safe requests wrongly refused) and report
  harm-reduction vs over-refusal as a Pareto axis / a defence-oriented composite. This is
  the defence-side *selectivity* check — the mirror of the attack-side Q gate.

**Predictions worth testing (candidate findings).**

- Does the metric rank defences sensibly — mild steering hardens with low over-refusal,
  strong steering over-refuses (a defence dose-response mirroring attack-side F4)?
- Do amplification defences (heads/neurons) trade off differently from direction steering
  — i.e. is the attack-side family separation (F4) reproduced in reverse?
- **Defence-specific monitorability failure:** does S / the pathway taxonomy flag an
  over-hardened model that refuses benign prompts *without* genuine safety reasoning
  (spurious `refusal without recognition`)? This would be the defence-side analogue of
  Exp-1's covert failure — a distinct, novel use of the S axis.

**A second sense of "reverse" (note, secondary).** "Reverse" can also mean *causal
restoration* — patch an ablated component back in and confirm safety returns (necessity →
sufficiency). That is the **activation-patching** complement
(`interventions/activation_patching.py`, currently a stub) and is mechanistic rather than
metric-validation; keep it distinct from the defence-side experiment above, which is the
one that tests the orientation claim.

**Files:** `configs/.../matrix.yaml` (reverse conditions, `add_alpha_sign: +1.0`, amplify
`scale_factor`), `scripts/make_v5_configs.py` (reverse gen functions),
`interventions/ablation.py` + `models/custom_llama.py` (`scale_factor > 1` already
supported), `interventions/steering.py`, `analysis/composite.py` (orientation flag +
P-defence), `configs/datasets.yaml` (XSTest / OR-Bench), a new over-refusal metric under
`analysis/`, and the report scripts. Discussion ¶3 and `EVALUATION_FRAMEWORK.md` §5.1 then
cite this as evidence rather than assertion.

**Verification additions:** confirm P-defence is a clean sign-flip of the attack P on the
same cells (a baseline vs baseline cell → P = 0 both orientations); confirm the
harm-reduction/over-refusal Pareto separates strong from mild defences; confirm S/pathway
flags spurious benign-prompt refusals where they occur.

---

## Explicitly out of scope (deprioritized per constraints — list as future work)

Given constrained compute and the ACL long-paper target, do **not** attempt now (note
briefly in Limitations / Future Work instead): scaling above 8B (14/32/70B ladder);
additional model families; additional attack intervention families beyond the covert
ladder (SAE features, DSH v_H/v_R, learned circuit masks); additional datasets (HarmBench /
StrongREJECT / AdvBench); and circuit-level activation-patching mechanistic validation
(`interventions/activation_patching.py` is a deliberate stub — the "causal restoration"
reading of *reverse* noted in Experiment 5). These are breadth/scale plays better suited to
a journal or D&B version. *(Experiment 5's defence-side reversal is no longer future-work —
compute is being allocated for it.)*

---

## Priority order

1. **Experiment 1** — the S-axis positive control (stated priority; converts F5's null into
   a headline; almost no new metric code).
2. **Experiment 2** — statistics (cheap, expected by A\* reviewers, and needed to put error
   bars on the Exp-1 covert effect).
3. **Experiment 3** — judge robustness (makes the covert-cell judging defensible).
4. **Experiment 4** — benign-utility Q (closes the largest acknowledged limitation; do if
   compute/time allow after 1–3).
5. **Experiment 5** — reverse / defence-side (validates orientation-agnosticism; runs on
   its own allocated compute, so it can proceed in parallel with 1–4 rather than after).

---

## Verification

- **Exp 1:** after adding covert conditions, confirm `composite_cells.csv` shows the covert
  cells with `gap ≫ 0`, `S` collapsed, `P`/`Q` retained, and `sfs_covert` > `sfs` — i.e.
  the axis fires. Confirm the matched-ASR/matched-P·Q covert-vs-overt pair differs only on
  S in the new plot. Confirm `batch_v5_003` human κ shows annotators agree the covert
  traces are sanitized and answers harmful (`score_annotations.py` report). Sanity-check
  the extracted safety-reasoning direction changes the forward pass (reuse the logit-diff
  check pattern in `scripts/verify_neuron_ablation.py`).
- **Exp 2:** error bars/CIs render in the regenerated composite report; H1–H5 table with
  BH-FDR-adjusted p-values present; H3 prose matches the observed signs.
- **Exp 3:** cross-judge SFS Kendall τ reported (target: rankings stable, τ high) incl.
  covert cells.
- **Exp 4:** Q with the utility term still preserves the F1–F4 orderings (retention term
  should sharpen, not overturn, the family separation); MMLU/GSM8K deltas per condition
  logged.
- Throughout: `pytest -q tests/` stays green (metrics/mask/judge-parsing regression tests),
  and `reaggregate_v5_summaries.py` heals any clobbered `summary.json` before reporting.
