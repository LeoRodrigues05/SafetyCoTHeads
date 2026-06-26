# Evaluation methodology & project background

Consolidated from the former `ExperimentTracker.md`, `richer_evaluation.md`, and
`safety_ablation_validation.md`. Captures *why* the evaluation is built the way it
is — the metrics, controls, and the "selective failure vs. generic degradation"
question that motivates the current pipeline. (The reference paper is at
[../papers/On_the_Role_of_Attention_Heads_LLM_Safety.pdf](../papers/On_the_Role_of_Attention_Heads_LLM_Safety.pdf).)

> **Headline framing** lives in
> [EVALUATION_FRAMEWORK.md](EVALUATION_FRAMEWORK.md): the project is an *evaluation*
> contribution — a **comparative** study of intervention methods plus a **proposed
> standardized metric** over three axes (Potency, Quality, Safety-Reasoning). This
> doc is the methodology background that grounds those axes; the metric catalogue
> below is tagged by axis.

---

## 1. North star

> Intervention methods can't currently be compared head-to-head (each reports its own
> ASR). Provide that comparison and a reusable composite metric. The metric's core
> discriminator is the **Potency × Quality** distinction: separate **selective safety
> failure** (safety removed, model still capable and coherent) from **generic model
> degradation** (the model just broke) — and add a **Safety-Reasoning** axis for
> whether the visible trace still engages safety / stays monitorable. CoT traces are
> **behavioral signals to be causally validated**, not direct evidence of internal
> reasoning.

## 2. Why naïve metrics mislead (the load-bearing finding)

A collaborator reported that ablating SHIPS safety heads on Llama-2-7b-chat doesn't
produce *harmful* output — it just breaks the model. Inspecting flipped prompts:
ablated outputs often start with an emoji, loop on a prompt phrase, flip polarity
mid-sentence, or emit nothing. These are **broken**, not **harmful**.

Two confounds the headline number must survive:
1. **Refusal-regex ≠ harm.** Refusal rate collapsing 96%→7% conflates "safety
   removed" with "model damaged" — the regex misses incoherent non-refusals.
2. **Location ≠ circuit.** A *layer-matched random* control already drops refusal to
   ~21% — ablating *any* early-layer head disrupts the model. The real contrast is
   `safety_head − layer_matched_random`, not `safety_head − baseline`.

**Conclusion:** the only defensible potency metric is **harm rate among coherent
outputs** (`harmful_among_clean`), measured with a paired design against a
layer-matched control.

## 3. Metric catalogue  (tagged by framework axis)

**Potency** — harmful-compliance rate, refusal rate, reasoning-about-safety,
adding-intention, changing-subject (5-label judge); category-level harm. *Headline
rate:* `harmful_among_clean` (a.k.a. ASR-clean).
**Quality** — coherence gate: gibberish detector (`madhurjindal/autonlp-Gibberish-Detector`,
label `clean`), n-gram repeat / distinct-n, compression ratio, empty/truncation
rate, benign NLL, MMLU/GSM8K accuracy delta, judge coherence/helpfulness 1–5.
**Safety-Reasoning** — CoT-monitorability gap (`asr_final − asr_cot_pred`), the
sentence-level safety-reasoning trace judge (`has_safety_reasoning`, 6 categories,
position/extent), and the 12-label pathway taxonomy (mechanism).
**Cross-axis selectivity:** `Δ harmful_compliance / Δ coherence_loss` (selective
failure iff ≫ 1) — a Potency-vs-Quality ratio; the composite that combines all three
axes is an open design choice ([EVALUATION_FRAMEWORK.md §5](EVALUATION_FRAMEWORK.md)).

## 4. Controls & statistics (the evaluation suite)

In rough cost/benefit order:

- **Tier 1 (must-have):** `harmful_rate_among_coherent`; **paired McNemar** on same
  prompt ids (tests flip asymmetry, far stronger than two independent rates);
  **layer-matched random** as the *primary* control (report the delta).
- **Tier 2:** coherence/helpfulness 1–5 → harm-vs-coherence curve; per-category
  breakdown (JBB 10 cats, BeaverTails 14); repetition/degeneracy diagnostics;
  over-refusal damage on a benign set.
- **Tier 3 (independent verification):** second judge (Llama-Guard-3 / ShieldGemma);
  StrongREJECT rubric; AdvBench/HarmBench expansion for tighter CIs; **random-head
  bootstrap** (≥20 random + 20 layer-matched draws — the SHIPS estimate is only
  meaningful if it sits outside that distribution).
- **Tier 4 (circuit-level):** ablation-strength sweep (`scale_factor`, top-k — a real
  circuit shifts monotonically); complement patch-back (necessity/sufficiency);
  activation patching instead of weight scaling (cleaner causal read).

## 5. Judge & validation

Primary judge `Qwen2.5/Qwen3-30B-Instruct` (strong JSON adherence); robust parsing
(strict → smart-quote/trailing-comma fix → per-field regex → `parse_status`), retry
≤3, persist raw output + model name. **Validate the judge against humans:** manually
label ~100–200 (prompt, response) pairs across the 5 labels, report per-label F1 and
Cohen's κ, flag labels where ≥2 judges disagree >20%. (This is now operationalised by
the annotation tool — see [ANNOTATION_SETUP.md](ANNOTATION_SETUP.md).) The committed
`batch_v5_002` validates all three judge instruments: `safety_5label` (Potency),
`cot_only` (monitorability), and the **Tier-2 sentence-level `safety_reasoning`** task
for the SR-trace judge (sentence-level + per-category κ).

## 6. Project lineage

The early plan was a 6-experiment SHIPS study — (1) reproduce SHIPS+Sahara,
(2) build the LLM-judge pipeline, (3) safety-head vs random-head ablation,
(4) coherency-head discovery, (5) **joint/disentangled ablation at iso-quality**
(the central claim), (6) CoT-trajectory analysis. That work generalised into
**Direction A** (causal failure pathways + CoT monitorability) — see
[../direction_a/README.md](../direction_a/README.md) — whose v5 iso-ASR sweep is the
current pipeline. Open future threads from the early plan: activation-patching /
refusal-direction rescue, neuron-level attribution cross-comparison, an OLMo
checkpoint sweep (*when do safety heads emerge in training?*), and CoT-faithfulness.
