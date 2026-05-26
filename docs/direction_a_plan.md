# Direction A — Failure-Mode Atlas: Mechanism-Resolved CoT Trajectory Analysis of Safety Interventions

**Status:** plan v2, post subagent-review.
**Target venue:** NeurIPS / ICLR main track.
**Primary author:** Leo Rodrigues.
**Last updated:** 2026-05-26.

---

## 1. TL;DR

Four widely-used safety-localisation methods — attention heads (SHIPS/Sahara), MLP neurons (Chen et al., NeurIPS 2025), differentiable head∪neuron circuits (SafeSeek), and DSH steering directions (Wu et al., Recognition axis $v_H$ vs. Execution axis $v_R$) — are compared at **both iso-ASR and iso-perturbation-magnitude** on Llama-3.1-8B-Instruct and DeepSeek-R1-Distill-Llama-8B. We measure not only final-answer attack-success but a 7-dimensional **CoT-trajectory fingerprint**. The central claim is operationalised as an *a-priori classifier-AUC test*: if a held-out classifier can identify the intervention type from its trajectory fingerprint above a pre-registered threshold, the failure-mode-distinctness claim is supported. R1-Distill trajectories are analysed with model-specific metrics and reported in a parallel panel — never pooled with Llama-3.1.

---

## 2. Central claim and contributions

**Central claim.** At matched attack-success rate, the four intervention families produce *mechanism-identifying* CoT-trajectory fingerprints. Final-answer benchmarks (ASR, refusal rate) systematically under-measure these differences. The activation-geometry dissociation between $v_H$ and $v_R$ predicted by DSH appears as a generation-level double dissociation, and this geometry-to-behaviour mapping predicts which steering layer is most effective — a finding ASR alone cannot surface.

**Contributions.**
1. A pre-registered, multi-method failure-mode atlas across four safety-localisation families and two model regimes (instruction-tuned, reasoning).
2. A reusable trajectory-metric suite + dual-judge validation protocol with human-gold reliability.
3. A falsifiability framework (classifier-AUC on trajectory vectors with permutation null) for "do safety methods truly behave differently?" questions.
4. The first generation-level test of the DSH Recognition/Execution dissociation.

---

## 3. Background and positioning

- **SHIPS / Sahara (Zhou et al., NeurIPS 2024)** — the base paper. Per-head KL on last-token softmax + greedy SVD for circuits. Gives the "heads" family of this study.
- **Safety Neurons (Chen et al., NeurIPS 2025)** — inference-time activation contrasting on harmful-vs-benign pairs to rank MLP-down neurons.
- **SafeSeek (open code)** — differentiable mask over heads ∪ neurons with sparsity penalty.
- **DSH (Wu et al.)** — decomposes safety into a Recognition axis ($v_H$, "does the model know this is harmful?") and an Execution axis ($v_R$, "does the model refuse?"). Demonstrates dissociation at activation-geometry level.

**Gap.** No prior work compares all four families on a common axis with mechanism-resolved CoT trajectory analysis, and the DSH dissociation has never been tested at generation-trajectory level.

---

## 4. Falsifiability framework

The headline claim is operationalised in three pre-registered tests:

### 4.1 Classifier-AUC test (primary)
Per (prompt $p$, condition $c$) we collect a 7-dimensional trajectory vector $\mathbf{x}_{c,p} \in \mathbb{R}^7$. We split *prompts* into train/test, fit a multi-class classifier (random forest baseline + logistic regression) to predict intervention identity from $\mathbf{x}_{c,p}$, and report held-out macro-AUC against a permutation null.

- **Pre-registered threshold:** macro-AUC $\geq 0.75$ on held-out prompts.
- **Permutation null:** shuffle intervention labels, refit, repeat $B=1000$. Report 95% CI of the null and the empirical $p$-value.
- **Reporting:** AUC per pairwise comparison (heads vs. neurons, heads vs. steering, …) plus the global macro-AUC.

### 4.2 DSH dissociation test
Pre-registered contrast between $\Delta$(safety-reasoning rate) and $\Delta$(refusal-verbalisation rate) under $v_H$ vs. $v_R$ steering. Tested with a paired mixed-effects model. A positive double dissociation requires both $v_H$ and $v_R$ to selectively affect their predicted metric.

### 4.3 Geometry → behaviour mapping test
Across layers, the magnitude of activation-geometry separation between harmful and benign at layer $\ell$ predicts the trajectory-level effect of steering at $\ell$, with rank correlation $\rho \geq 0.5$ pre-registered.

---

## 5. Methods

### 5.1 Models
| Role | Model | Notes |
|---|---|---|
| Primary instruction-tuned | Llama-3.1-8B-Instruct | All metrics. |
| Reasoning (primary) | DeepSeek-R1-Distill-Llama-8B | R1-adapted metrics only. |
| Reasoning (secondary, optional) | DeepSeek-R1-Distill-Qwen-7B | Adds n=2 for cross-LRM interaction test. |
| Reproduction parity | Llama-2-7b-chat | SHIPS replication only. |

### 5.2 Datasets
| Use | Dataset | n |
|---|---|---|
| Head/neuron discovery | MaliciousInstruct | ~100 |
| Eval (jailbreak) | JailbreakBench | 100 |
| Eval (categorical) | BeaverTails | 140 (10 × 14 cats) |
| Benign quality | AlpacaEval | 200 |
| Disentanglement probe | AmbiguityBench (DSH) — fallback: held-out JBB split | TBD by P0 |
| Capability controls | MMLU, GSM8K | per-cell subsample |

### 5.3 Interventions
1. **Heads** — SHIPS top-$k$ + Sahara circuits via [`HeadMaskController`](../src/safety_cot_heads/models/custom_llama.py). GQA-aware zero ablation.
2. **Neurons** — Reimplementation of Chen et al.: harmful-vs-benign activation contrast on MLP-down outputs, rank by absolute contrast, ablate top-$k$ by zeroing. New file: `src/safety_cot_heads/attribution/safety_neurons.py`.
3. **Circuits (SafeSeek)** — differentiable binary mask over heads ∪ neurons, optimised on harmful split with $\ell_1$ sparsity penalty and benign-quality penalty. Held-out split for overfitting check. New file: `src/safety_cot_heads/attribution/safeseek_circuit.py`.
4. **Steering ($v_H$, $v_R$)** — DSH-style. $v_H$ = mean-diff of activations between harmful and benign prompts at last template position. $v_R$ = mean-diff between **length-matched, content-controlled** refused and complied completions (paired curation, length within ±20% and topic-controlled). Layer chosen on AmbiguityBench (or JBB val fallback). Stability of $v_R$ bootstrapped over completion subsets. New file: `src/safety_cot_heads/attribution/steering_vectors.py`.

### 5.4 Matching protocols
Each intervention is run at:
- **Iso-ASR:** strengths tuned on a calibration split so ASR hits {≈50%, ≈85%} ±5pp.
- **Iso-magnitude:** strengths tuned so perturbation norm matches across methods at three matched levels (norm of suppressed activations for heads/neurons; equivalent steering norm for $v_H, v_R$; mask $\ell_0$ for SafeSeek).

Fingerprints are reported under *both* protocols. Disagreement between protocols is itself a finding.

### 5.5 Trajectory metrics
**Llama-3.1 variant (prose CoT).** Per response, per sentence:
1. Reasoning fraction = (# CoT sentences) / (# total sentences).
2. First-safety-reasoning-sentence index (∞ if absent).
3. Safety-reasoning-sentence rate.
4. Intention-invention rate (CoT introduces a benign frame absent from the prompt).
5. Self-contradiction rate (CoT refuses, answer complies — or vice versa).
6. Refusal-verbalisation rate.
7. Repetition / degeneration score.

**R1-Distill variant (`<think>` block).** Per response:
1. `<think>` block length (tokens).
2. Presence of explicit refusal keyword in `<think>`.
3. Safety-reasoning sentence count within `<think>` (segmenter validated, see §6.3).
4. Intention-invention within `<think>`.
5. Cross-block contradiction (think vs. answer).
6. Answer-level refusal verbalisation.
7. Repetition / degeneration score.

R1 and Llama metrics are **never pooled**. Cross-model contrasts use only final-answer ASR + steering dissociation.

### 5.6 Template-anchoring diagnostic
$\rho_\text{tpl}$ per head = fraction of attention mass placed on template-region keys, averaged over harmful prompts. Neuron analogue: gradient attribution mass on template-region input positions. Reported as a SHIPS-and-neurons robustness check; we report two head/neuron rankings (raw vs. residualised on $\rho_\text{tpl}$ via OLS) and check whether the trajectory fingerprint qualitatively changes.

### 5.7 Judging
- **Sentence-level dual judge:** Qwen-2.5-32B (primary) + a non-safety-trained second judge (Mistral-Large or annotation-finetuned Qwen-base) for robustness to judge safety-bias.
- **Human gold:** ≥500 sentences stratified by (model, condition, metric). 3 annotators. Report inter-human $\kappa$, then judge-vs-human $\kappa$ per metric. **Drop any metric with judge-vs-human $\kappa < 0.70.**
- Per-response judge as in current pipeline ([`judging/judge_prompts.py`](../src/safety_cot_heads/judging/judge_prompts.py)) with 5-label CoT-safety schema.

### 5.8 Statistics
- 5 generation seeds per (prompt, condition); **same seeds reused per prompt across conditions** (paired design) with `(1|seed)` random effect.
- Mixed-effects logistic regression per binary outcome, **fit separately per model**:
  $$\text{outcome} \sim \text{intervention} \times \text{asr\_level} + (1|\text{prompt}) + (1|\text{category}) + (1|\text{seed})$$
- Paired bootstrap 95% CIs for rate differences ($B=10{,}000$).
- BH-FDR across the full contrast family.
- A-priori power calculation per primary contrast (target power 0.80, effect size $\Delta p = 0.10$).

---

## 6. Verification

1. **Pre-registration freeze** — commit + tag prereg document before Phase 1 runs.
2. **Judge sanity** — judge-vs-human $\kappa$ per metric in the main paper; metrics with $\kappa<0.70$ excluded.
3. **Iso-ASR tolerance** — within ±5pp per cell; report deviations.
4. **Iso-magnitude robustness** — fingerprint conclusions stable across both matching protocols.
5. **Random-head and layer-matched-random controls** — distinct from all four real interventions at each band.
6. **Benign-quality control** — AlpacaEval coherence, MMLU, GSM8K reported per intervention for selectivity.
7. **Falsifiability gate** — classifier AUC > 0.75 (held-out, prompt-disjoint) for the main claim to hold.
8. **SafeSeek overfitting check** — train-vs-held-out gap reported; if large, SafeSeek demoted to case study only.
9. **$v_R$ stability** — bootstrap variance across completion subsets; instability halts publication of DSH dissociation claim.
10. **Post-hoc power** — reported for any null findings.

---

## 7. Phased execution

### Phase 0 — Gates and pre-registration (hard blockers)
- **P0.1** Confirm AmbiguityBench availability; size; license. Hard gate. Fallback: held-out JBB split.
- **P0.2** Confirm DSH reference code reproduces $v_H, v_R$ on Llama-3.1.
- **P0.3** Confirm SafeSeek code released and runnable.
- **P0.4** Write and freeze sentence-level annotation schema (7 metrics, ≥3 positive and negative examples per label).
- **P0.5** Pre-register all primary tests, thresholds (classifier-AUC ≥ 0.75, $\rho \geq 0.5$, $\kappa \geq 0.70$), iso-ASR bands, prompt train/val/test partition.
- **P0.6** A-priori power calc; lock seed count and grid.

### Phase 1 — Foundations (parallel after P0)
- **P1.1** Port SHIPS/Sahara to Llama-3.1; baseline runs on both models on JBB + BT + benign sets. Reuse [`HeadMaskController`](../src/safety_cot_heads/models/custom_llama.py) and `ships_legacy/ships.py`.
- **P1.2** Extend [`judging/judge_prompts.py`](../src/safety_cot_heads/judging/judge_prompts.py) with per-sentence schema; build dual-judge driver.
- **P1.3** Collect 500-sentence human gold (3 annotators). Compute $\kappa$.
- **P1.4** R1-Distill trajectory adapter on `<think>` blocks; validate segmenter on ≥100 manually-segmented blocks.
- **P1.5** Template-anchoring diagnostic for heads + neuron analogue.

### Phase 2 — New intervention implementations (parallel after P1)
- **P2.1** Safety-neuron discovery (Chen et al.). ~150 LOC. Output: ranked neuron lists per (model, layer).
- **P2.2** SafeSeek mask training. Held-out split. Overfitting analysis.
- **P2.3** DSH steering: curate length-matched completion pairs for $v_R$; compute $v_H$; layer selection on AmbiguityBench (or fallback); bootstrap stability.

### Phase 3 — Calibration and main sweep (depends on P2)
- **P3.1** Iso-ASR calibration: tune strengths on calibration split for {≈50%, ≈85%} ± 5pp.
- **P3.2** Iso-magnitude calibration: three matched magnitude bands.
- **P3.3** Main grid: 2 models × (4 mechanisms + 2 random controls) × (2 ASR bands ∪ 3 magnitude bands) × 5 seeds × (JBB + BT + AlpacaEval).

### Phase 4 — Analysis and writing (depends on P3)
- **P4.1** Mixed-effects regression per model; paired bootstrap; BH-FDR.
- **P4.2** Classifier-AUC falsifiability test (RF + LR; permutation null $B=1000$).
- **P4.3** DSH dissociation test and geometry-to-behaviour mapping test.
- **P4.4** Iso-ASR vs. iso-magnitude robustness panel.
- **P4.5** All five headline figures.
- **P4.6** Write-up; pre-registration deviation log; appendix with all judge-vs-human $\kappa$ tables.

---

## 8. Headline figures

1. **The atlas.** Heatmap (4 interventions × 7 trajectory metrics) at fixed ASR, per model.
2. **Fingerprints.** Radar plots, one per intervention, seven metrics, with bootstrap CI bands.
3. **DSH dissociation.** Scatter of $\Delta$(safety-reasoning rate) vs. $\Delta$(refusal verbalisation) under $v_H$ vs. $v_R$.
4. **Template anchoring.** SHIPS score vs. $\rho_\text{tpl}$, coloured by trajectory effect; raw vs. residualised rankings.
5. **Reasoning-model panel.** Same-intervention fingerprints, R1-Distill vs. Llama-3.1, side-by-side.

Supplementary: classifier-AUC bar chart per pairwise comparison; iso-magnitude robustness panel; benign-quality table.

---

## 9. Decisions and scope

- **D1.** SafeSeek is a *case study*, not a fourth head-to-head iso-ASR competitor: its training-time optimisation creates an unavoidable selection bias against post-hoc methods. Three-way iso-ASR comparison is heads vs. neurons vs. steering.
- **D2.** R1-Distill results in a parallel panel; pooled cross-model tests only for final-answer ASR + DSH dissociation, never for sentence-level trajectory metrics.
- **D3.** 2 ASR bands × 5 seeds (not 3 × 3) — power over breadth.
- **D4.** Template-anchoring diagnostic scoped to heads and neurons; circuits and steering get a different, methodology-appropriate robustness check (mask-sparsity vs. prompt-length sensitivity for SafeSeek; direction-stability bootstrap for $v_H, v_R$).
- **D5.** No vision-language; no MoE; no defence/mitigation methods; no human red-team beyond manual spot-check; no fine-tuning interventions; only 7–8B scale.

---

## 10. Open questions for author

1. **Reasoning-model n.** Add DeepSeek-R1-Distill-Qwen-7B as a second LRM for a formal `model × intervention` interaction test, or weaken claim to qualitative case study? Recommendation: **add** the second LRM (modest compute uplift).
2. **Judge stack.** Dual judge throughout (best, most expensive), dual judge for validation subset only (cheaper), or single Qwen with explicit caveat (cheapest, weakest)? Recommendation: **dual judge on validation subset + sensitivity analysis**.
3. **Post-hoc fourth method.** Drop SafeSeek and replace with a post-hoc probing-classifier-derived head set (Marks & Tegmark line), to keep four interventions on the same evaluation footing? Recommendation: keep SafeSeek as case study (Decision D1) **and** add a post-hoc probing-derived method only if compute allows.

---

## 11. Reusable artefacts produced

- `src/safety_cot_heads/attribution/safety_neurons.py`
- `src/safety_cot_heads/attribution/safeseek_circuit.py`
- `src/safety_cot_heads/attribution/steering_vectors.py`
- Extended [`src/safety_cot_heads/models/custom_llama.py`](../src/safety_cot_heads/models/custom_llama.py) with neuron-ablation, mask, and pre-residual steering hooks.
- Extended [`src/safety_cot_heads/analysis/trajectory.py`](../src/safety_cot_heads/analysis/trajectory.py) with the 7-metric Llama suite and the 7-metric R1 adapter.
- Extended [`src/safety_cot_heads/analysis/metrics.py`](../src/safety_cot_heads/analysis/metrics.py) with paired bootstrap, mixed-effects wrapper, classifier-AUC falsifiability test.
- Extended [`src/safety_cot_heads/judging/judge_prompts.py`](../src/safety_cot_heads/judging/judge_prompts.py) with per-sentence schema and dual-judge driver.
- `data/annotations/sentence_gold/` — 500-sentence human gold set.
- Pre-registration document (committed before Phase 1).
