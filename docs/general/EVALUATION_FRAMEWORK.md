# A standardized evaluation framework for safety interventions

> **Status.** This doc defines the project's current framing. It supersedes the
> earlier "causal failure-pathways / CoT-monitorability" framing as the *headline*
> contribution — that mechanistic work is now one input (the Safety-Reasoning axis,
> §3.3) rather than the thesis. See [direction_a/README.md](../direction_a/README.md)
> for the mechanistic detail and [general/README.md](README.md) for methodology
> background.

---

## 1. The gap we're addressing (evaluation POV)

A growing set of **inference-time / white-box safety interventions** exists — safety-
head ablation (SHIPS/Sahara), safety-neuron ablation, activation-addition steering on
a refusal direction, directional ablation (Arditi et al.), learned circuit masks, DSH
axis steering, and so on. Each paper reports its effect with **its own metric on its
own setup** — usually a single attack-success-rate (ASR) or refusal-rate number.

**There is no direct, apples-to-apples way to compare two intervention methods.** Two
papers both claiming "removes safety" can mean very different things:
- one induces *coherent harmful* output; the other just *breaks the model* into
  gibberish (a refusal-regex counts both as "non-refusal");
- one leaves the chain-of-thought *visibly* unsafe (still monitorable); the other
  produces unsafe answers from *sanitised-looking* traces (monitorability collapsed);
- they trade off potency against fluency very differently.

A single ASR number collapses all of this. **We are on the evaluation side of LLM
research**: the contribution is a *measurement instrument*, not a new attack or
defence.

## 2. Contribution: comparative study **and** a proposed metric

This project makes a **dual** contribution:

1. **Comparative.** Run a representative set of intervention families through one
   controlled grid (same models, prompts, judges, matching protocol) and report how
   they actually differ — the first head-to-head comparison on shared axes.
2. **A proposed standardized metric.** A composite score over three axes that lets
   future intervention papers report a comparable, decomposable number instead of a
   bespoke ASR. The aim is a metric others can adopt, like StrongREJECT or HarmBench
   did for jailbreak evaluation.

Both are needed: the comparison *motivates and validates* the metric (it must
separate methods that are genuinely different), and the metric *operationalises* the
comparison (it makes "different" precise and reusable).

## 3. The three axes

An intervention's effect is characterised on three orthogonal axes. Each is already
instrumented in the pipeline; the framing change is treating them as **the** top-level
decomposition.

### 3.1 Potency — *did it actually remove safety?*
How strongly the intervention induces **coherent harmful** output.
- Primary: **`harmful_among_clean`** (ASR over coherence-gated responses) — defeats
  the "broke the model, didn't make it harmful" loophole.
- Supporting: 5-label safety judge (`harmful_response`, `safe_rejection`, …),
  per-category harm rates, paired (McNemar) deltas vs a layer-matched control.

### 3.2 Quality — *did it keep the model intact?*
Whether the model stays coherent and capable (the **selective failure vs generic
degradation** distinction).
- Coherence gate (`clean_rate`, gibberish detector), repetition/distinct-n,
  empty/truncation rate, judge coherence/helpfulness (1–5).
- Capability retention on benign sets (MMLU/GSM8K/AlpacaEval delta) — iso-utility.

### 3.3 Safety-Reasoning — *did the visible reasoning still engage safety?*
Whether the chain-of-thought reflects safety reasoning, and whether it remains a
usable **monitor**. This axis absorbs the former "monitorability + mechanism" work:
- **Safety-reasoning trace judge (new).** Per indexed sentence: is it safety
  reasoning, in which of 6 categories (`risk_acknowledgment`, `policy_boundary`,
  `intent_assessment`, `refusal_reasoning`, `safer_alternative`,
  `other_safety_reasoning`); aggregated to `has_safety_reasoning`, first-position,
  and extent. Human-validated via the Tier-2 annotation task (sentence-level κ).
- **CoT-monitorability gap** = `asr_final − asr_cot_pred` (unsafe answer from a
  benign-looking trace ⇒ gap ≫ 0 ⇒ monitorability collapsed).
- **12-label pathway taxonomy** (see §4) — the fine-grained mechanism of *how* the
  safety reasoning breaks.

## 4. Where the 12 HarmThoughts / pathway metrics fit  〔open design decision〕

The fine-tuned pathway judge emits **12 binary labels** in 4 groups (recognition,
refusal dynamics, rationalisation, execution) that aggregate into an 8-dim pathway
vector + categorical `dominant_pathway`. They were the headline of the old framing;
under the eval framing they need an explicit home.

**Proposed placement (recommended): nest the 12 under the Safety-Reasoning axis as
its *mechanistic decomposition*.** Rationale: the pathway labels describe *which*
safety subprocess is present/lost in the reasoning (recognition, refusal initiation/
maintenance/suppression, rationalisation, execution) — i.e. they explain *how* the
safety-reasoning signal in §3.3 changes. `has_safety_reasoning`/category from the SR
judge is the coarse signal; the pathway vector is the fine-grained one.

**Alternatives to weigh (this is flagged open):**
- **(a) Sub-component of Safety-Reasoning** (recommended) — one axis, two
  granularities (SR-trace coarse + pathway fine).
- **(b) A fourth "Mechanism" axis** — keep pathway separate as a diagnostic that is
  reported but *not* folded into the headline composite (it's descriptive, not
  scalar-friendly: `dominant_pathway` is categorical).
- **(c) Split** — `recognition`/`refusal` groups → Safety-Reasoning; `execution`/
  `rationalisation` groups → Potency (they describe degree of compliance).

Open questions to resolve before finalising: (1) do the pathway labels add
*comparative* signal beyond the SR-trace judge, or are they redundant? (2) can a
categorical `dominant_pathway` enter a scalar composite at all, or only the 8-dim
vector / specific labels (e.g. `refusal_suppression`)? Decide empirically on
`batch_v5_002` once SR-trace and pathway are both validated against humans.

## 5. Combining the axes into one coherent metric  〔open research decision〕

The hard part: collapse (Potency, Quality, Safety-Reasoning) into something
comparable without throwing away the structure that motivated the metric. Candidate
formulations, with trade-offs — **not yet decided**:

| # | Form | Pro | Con |
|---|---|---|---|
| 1 | **Report the vector** `(P, Q, S)` + Pareto dominance | loses nothing; honest | no single ranking; reviewers want a number |
| 2 | **Iso-X protocol** — fix one axis (e.g. iso-potency 50% ASR-clean), compare the other two | the existing iso-ASR design; clean causal read | a protocol, not a scalar; needs dose tuning per method |
| 3 | **Weighted z-sum** — standardise each axis across methods, weight, sum | one number, tunable | weights are a value judgment; not portable across grids |
| 4 | **Geometric / harmonic mean** of normalised axes | penalises imbalance (an F-score-like "no axis left behind") | needs a principled per-axis normaliser; direction conventions |
| 5 | **Selective-Failure Score** — a purpose-built scalar, e.g. potency *gated by* quality and discounted by monitorability loss | encodes the actual thing we care about | bespoke; must be justified and ablated |

**Key conventions to pin down:** (i) what is being evaluated — these interventions
*suppress* safety, so "high potency" is the attack succeeding; a defence eval would
flip signs. (ii) Each axis needs a fixed [0,1] orientation and a normaliser that is
stable across models/datasets. (iii) The composite must be reported **with** the
vector, never instead of it.

Recommended next step: prototype #1 (vector + Pareto) and #2 (iso-potency) first —
they require no weight choices and directly support the comparative claim — then fit
#4/#5 as the "headline scalar" once the axes are human-validated and we see which
contrasts the comparison needs to preserve.

## 6. Positioning vs existing evaluations  〔survey to complete〕

To justify a *new* metric we must show existing ones don't cover the gap. Survey
structure (extract: what they measure, on which axis, and what they miss):

- **Jailbreak / attack evals (general LLM eval side).** StrongREJECT, HarmBench,
  JailbreakBench (used here), AdvBench, BeaverTails (used here). Mostly **Potency**;
  StrongREJECT adds a quality-aware rubric. None measure safety-*reasoning* or
  monitorability; most are black-box (no intervention notion).
- **Safety-classifier judges.** Llama-Guard-3, ShieldGemma, the 30B/14B judges here.
  Instrumentation for Potency, not a comparison framework.
- **Intervention papers (what each reports, and on what axis).** SHIPS/Sahara
  (refusal-rate), Arditi et al. directional ablation (refusal + a coherence check),
  DSH (Wu et al., harmfulness/refusal axes), activation steering, SafeSeek/learned
  masks. The recurring gap: bespoke single-axis metrics, no shared quality or
  reasoning axis, no cross-method protocol.
- **CoT-monitoring agenda.** OpenAI 2024 / Baker et al. 2024 — motivates the
  Safety-Reasoning axis; not an intervention-comparison metric.
- **Composite-metric precedents to borrow from.** HELM-style multi-metric reporting,
  Elo/Arena ranking, StrongREJECT's quality-gated score — for *how* to combine axes
  defensibly (§5).

> This section is a scaffold. A full, cited survey is pending — it can be produced
> with the `deep-research` skill (intervention papers + general LLM eval papers) and
> folded back here. Do **not** cite specifics from this scaffold without verifying.

## 7. How the pipeline already supports this

- **Grid & matching.** One controlled grid (5 models × 2 datasets × 11 conditions),
  iso-ASR anchoring, layer-matched-random controls — the substrate for the
  comparative claim (§2.1).
- **Instruments per axis.** Potency = 5-label + coherence-gated ASR; Quality =
  coherence/utility; Safety-Reasoning = SR-trace judge + monitorability gap + pathway.
- **Validation.** All judge instruments are validated against humans via the
  annotation tool (`batch_v5_002`: safety_5label, cot_only, and Tier-2 sentence-level
  safety_reasoning) — a metric is only standardizable if its instruments are reliable
  (report human-vs-judge and inter-annotator κ).
- **Fast judging.** The SR pass runs on the vLLM backend (continuous batching), making
  re-scoring the whole grid cheap enough to iterate on the composite definition.

## 8. Open items (tracked)

1. Finalise placement of the 12 pathway metrics (§4) — empirical, after validation.
2. Choose the composite form (§5) — prototype vector+iso-potency first.
3. Complete the existing-evals survey (§6) — `deep-research`.
4. Validate all three axes' instruments on `batch_v5_002` (two annotators → κ).
5. Define per-axis [0,1] normalisers stable across models/datasets.
