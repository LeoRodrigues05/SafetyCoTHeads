# Experiment Tracker — "Where do you Lie?"

> Causal study of attention heads in CoT safety reasoning.
> **North star:** distinguish *selective safety failure* from *generic model degradation*,
> using safety-head ablation, matched controls, coherency heads, and CoT trajectory analysis.
> CoT traces are used as **behavioral signals** to be causally validated, not as direct
> evidence of internal reasoning.

---

## 0. Setup

### Models under test
| Role | Model | Notes |
|---|---|---|
| Primary | `meta-llama/Llama-3.1-8B-Instruct` | Already used in CoT-safety repo; CoT-friendly. |
| Secondary | `mistralai/Mistral-7B-Instruct-v0.2` | Original SHIPS wrapper targets it (needs GQA-aware re-impl). |
| Reproduction parity | `meta-llama/Llama-2-7b-chat-hf` | Matches the SHIPS/Sahara paper. |
| Stretch (ask Thomas) | `allenai/OLMo-2-1124-7B-Instruct` + intermediate SFT/DPO checkpoints | Lets us track when safety heads emerge across training. |

### LLM-as-judge models (larger, separate from model under test)
- Primary: `Qwen/Qwen2.5-32B-Instruct` (open weights, strong JSON adherence).
- Stretch: `meta-llama/Llama-3.3-70B-Instruct` (4-bit if VRAM-limited).
- API fallback (optional, credentials-gated): GPT-4o / Claude 3.5 Sonnet.

### Datasets
**Harmful / safety:**
- **MaliciousInstruct** (ships with SHIPS repo) — *safety-head discovery split*.
- **JailbreakBench** harmful-behaviors — *held-out safety evaluation* post-ablation; never used for discovery.
- **BeaverTails** (14 PKU-Alignment categories, already cached from CoT-safety repo) — category-level analysis + judge eval.
- **AdvBench** — optional sanity check.

**Benign / quality controls:**
- **Alpaca-Eval** or **Dolly-15k** — instruction-following helpfulness.
- **WikiText-103** continuations — fluency / perplexity for coherency-head discovery.
- **MMLU** + **TriviaQA** (small samples) — task competence.
- **GSM8K** (small sample) — reasoning preservation.

**Judge validation:** 100–200 manually labeled (prompt, response) pairs spanning all 5 behavior labels.

### Metrics
**Safety**
- Harmful compliance rate
- Refusal / rejection rate
- Reasoning-about-safety rate (CoT)
- Adding-intention rate (CoT)
- Changing-subject rate (CoT)
- Category-level harmfulness
- Manual spot-check agreement

**Quality**
- Helpfulness score (judge)
- Coherence score (judge)
- Repetition (n-gram repeat rate, self-BLEU)
- Malformed-output rate, truncation rate
- Benign NLL / perplexity
- MMLU / GSM8K accuracy delta

**Headline ratio** — `Δ harmful_compliance / Δ coherence_loss`. Selective safety failure iff ≫ 1.

---

## Implementation Map

Detailed per-experiment implementation notes live in `docs/experiment_docs/`.
Run configs are grouped by the same tracker experiment numbering:

| Tracker experiment | Config folder | Implementation status |
|---|---|---|
| Exp 1 — Reproduce SHIPS + Sahara | `configs/experiments/exp01_reproduce_ships_sahara/` | runnable |
| Exp 2 — LLM-as-Judge Pipeline | `configs/experiments/exp02_judge_pipeline/` | runnable |
| Exp 3 — Safety-Head vs Random-Head Ablation | `configs/experiments/exp03_safety_vs_random_ablation/` | runnable |
| Exp 4 — Coherency-Head Discovery | `configs/experiments/exp04_coherency_head_discovery/` | partially wired |
| Exp 5 — Joint / Disentangled Ablation | `configs/experiments/exp05_joint_disentangled_ablation/` | partially wired |
| Exp 6 — CoT Trajectory Analysis | `configs/experiments/exp06_cot_trajectory_analysis/` | planned |

Each YAML has a `tracker:` block with `experiment`, `run`, and `status`.

---

## Experiment 1 — Reproduce SHIPS + Sahara

**Goal.** Confirm the migrated code reproduces ranked safety heads matching the original paper qualitatively (top heads in similar layers/clusters) on Llama-2-7b-chat and Llama-3.1-8B-Instruct.

**Setup.**
- Mask config: `mask_qkv=['q']`, `mask_type='scale_mask'`, `scale_factor=1e-4` (SHIPS) and `1e-5` (Sahara). Matches repo defaults.
- SHIPS: per-prompt on MaliciousInstruct, KL on last-token softmax, rank all (layer, head) pairs.
- Sahara: greedy search with `search_step=8`, principal-angle subspace shift on last hidden states (top-1 left singular vector, per original).
- Save ranked-heads JSONL with full metadata (model, dataset, prompt template, `mask_qkv`, `mask_type`, `scale_factor`, seed, timestamp).

**Success criteria.**
- Top safety heads concentrated in mid-to-late layers.
- Non-trivial SHIPS↔Sahara overlap (≥30% in top-16).
- Ablating top-1 Sahara head increases harmful compliance on MaliciousInstruct (sanity).

---

## Experiment 2 — LLM-as-Judge Pipeline

**Goal.** Build a robust, reproducible behavior-judging pipeline reusable by all later experiments.

**Setup.**
- Reuse the 5-label CoT-safety judge prompt verbatim:
  `HARMFUL_RESPONSE`, `SAFE_REJECTION`, `REASONING_ABOUT_SAFETY`, `ADDING_INTENTION`, `CHANGING_SUBJECT`,
  with TRUE/FALSE + 0–1 certainty per label.
- Extend with `coherence_score` (1–5), `helpfulness_score` (1–5), `refusal_strength` (1–5).
- **Robust JSON parsing:** strict parse → smart-quote / trailing-comma fix → regex fallback per field → mark `parse_status`.
- Batched generation, configurable judge model (HF or API), retry on malformed output (≤3 attempts), `max_new_tokens ≥ 512`.
- Persist `judge_raw_output`, `judge_model_name`, `parse_status` for every row.

**Validation.**
- Manually label 150 (prompt, response) pairs across categories.
- Report per-label F1 vs human, Cohen's κ.
- Sensitivity: ≥2 judge models; flag labels where judges disagree >20%.

**Success criteria.** ≥0.7 macro-F1 vs human on `HARMFUL_RESPONSE` and `SAFE_REJECTION`; <5% unparseable outputs.

---

## Experiment 3 — Safety-Head vs Random-Head Ablation

**Goal.** Show that safety-head ablation drives selective harmful compliance beyond what random head removal can explain.

**Conditions** (k ∈ {1, 2, 4, 8, 16}, ≥5 seeds for random controls):
1. `baseline` — no ablation.
2. `safety_head_ablation` — top-k Sahara heads.
3. `random_head_ablation` — uniformly random k heads.
4. `layer_matched_random_head_ablation` — random heads drawn from the same layers as the safety heads.
5. *(stretch)* `activation_magnitude_matched_random_ablation` — random heads with matched activation magnitude.

**Eval datasets.** JailbreakBench (held-out) + BeaverTails-balanced subset (≤200/category).

**Outputs.** Dose-response curves (k vs harmful compliance), refusal-rate curves, per-category heatmaps, paired bootstrap CIs for the safety-vs-random gap.

---

## Experiment 4 — Coherency-Head Discovery

**Goal.** Identify attention heads that, when ablated, strongly damage benign / coherent generation — the necessary control for distinguishing safety failure from degradation.

**Method.** Re-use the SHIPS masking primitive on benign prompts; score each head by ablation-induced damage to benign output. Three scoring modes (configurable; hybrid weighted sum supported):
- **NLL mode** — Δ benign NLL / perplexity on WikiText-103 + Alpaca-Eval prompts.
- **Judge-coherence mode** — drop in LLM-judge `coherence_score` on benign generation.
- **Pathology mode** — rule-based score combining
  (a) repetition (4-gram repeat rate, self-BLEU),
  (b) truncation (EOS before min length),
  (c) malformed output (length, encoding, gibberish-detector),
  (d) off-topic rate (sentence-BERT similarity to prompt).

**Outputs.** Ranked coherency heads per model in the same JSONL schema as safety heads.

**Analysis.** Overlap with Sahara safety-head set (Jaccard @ k=1, 4, 16); layer distribution comparison.

---

## Experiment 5 — Joint / Disentangled Ablation

**Goal.** Disentangle "safety" from "quality" by manipulating overlap between the two head sets. **This is the key headline experiment.**

**Conditions.**
1. `safety_head_ablation` — reference (from Exp 3).
2. `coherency_head_ablation` — top-k coherency heads only.
3. `quality_head_ablation` — broader response-quality heads (judge-coherence-defined).
4. `safety_heads_excluding_coherency_overlap` — safety heads with overlap removed.
5. `overlap_heads_only` — heads ranked high in both sets.
6. `matched_degradation` — heads chosen to **match the benign-quality loss** of `safety_head_ablation`, then compare harmful compliance at iso-quality.

**Outputs.** Per-condition table (harmful compliance, refusal, coherence, helpfulness); iso-quality scatter (quality-loss vs harmful-compliance).

---

## Experiment 6 — CoT Trajectory Analysis

**Goal.** Characterize how ablation changes the *structure* of safety reasoning, not just the final answer.

**Setup.**
- Two prompt modes: `normal` and `cot` ("Let's think step by step before responding.").
- Run baseline + each condition from Exp 3 and Exp 5 on JailbreakBench + BeaverTails.

**Sentence-level judging.** Split responses (spaCy / regex fallback); judge each sentence for: safety reasoning, harmful compliance, benign-intent invention, refusal planning, subject change.

**Trajectory metrics.**
- First-safety-reasoning sentence index (or `None`).
- Number / fraction of safety-reasoning sentences.
- "Final answer contradicts earlier reasoning" flag.
- "Safety reasoning disappears after ablation" indicator (paired baseline vs ablated).
- "Invented benign intent" rate.

**Outputs.** Trajectory plots (sentence-index × behavior), per-category trajectory heatmaps, paired baseline-vs-ablated diffs.

---

## Engineering / Reproducibility
- Config-driven CLI (`scripts/run_*.py` + YAML; OmegaConf).
- Deterministic seeds; no hardcoded paths or CUDA ids; `utils/device.py` resolves from env.
- JSONL for row-wise outputs; CSV/JSON summaries; matplotlib for plots.
- Dry-run mode (2–5 examples) on every script.
- Smoke test using a small ungated model (e.g. `HuggingFaceTB/SmolLM2-135M-Instruct`).

## Priority order
1. Exp 1 (reproduction unblocks everything).
2. Exp 2 (judge unblocks all downstream eval).
3. Exp 3 (first headline: selective vs random).
4. Exp 4 (coherency heads — new contribution).
5. Exp 5 (matched-degradation — **the** central claim).
6. Exp 6 (CoT trajectory — primary novelty over the SHIPS paper).

## TODO / future phases
- Activation patching + refusal-direction (Arditi et al.) rescue experiments (proposal RQ4).
- Neuron-level safety attribution (Zhao et al.) cross-comparison (RQ3).
- OLMo checkpoint sweep — when do safety heads emerge during training?
- CoT-faithfulness: does ablation-induced CoT change reflect internal computation?
