# A richer evaluation suite for pre/post ablation

The current pipeline measures one thing well (LLM-judge `harmful_response`) and
one thing crudely (regex refusal). For the kind of question we care about —
"did this intervention make the model less safe, or did it just break it?" —
we need more axes. Concrete suggestions, roughly in order of cost/benefit:

## Tier 1 — must-have to make any claim at all

1. **`harmful_rate_among_coherent`**
   Run the gibberish detector
   (`madhurjindal/autonlp-Gibberish-Detector-492513457`, label `clean`), drop the
   non-clean rows, then compute `harmful_response` rate over the remainder.
   This is the single metric that defeats the "broke the model" loophole.
   Implementation: `safety_cot_heads.analysis.coherence.classify_gibberish`.
2. **Paired comparison (McNemar on same prompts)**
   For each prompt id, you have a baseline label and an ablated label; test
   whether the off-diagonal cells (`flip_to_harmful`, `flip_to_safe`) are
   asymmetric. Far more powerful than two independent rates.
3. **Layer-matched random as the primary control**
   Report `safety_head − layer_matched_random` for every metric, with the
   uniform random as a secondary control. Without this delta you cannot
   distinguish "circuit" from "location".

## Tier 2 — strengthens the picture

4. **Coherence / helpfulness 1–5** with the existing coherence judge prompt,
   on the same outputs. Lets you plot harm-rate as a function of coherence
   bin — the "useful-harm" curve.
5. **Per-category breakdown** on the 10 JailbreakBench harm categories. SHIPS
   was discovered on MaliciousInstruct; categories like *self-harm* may move
   differently from *privacy*. The original notebooks did this per category.
6. **Repetition / degeneracy diagnostics**
   - distinct-n (n=1,2,3) and total n-gram repeat fraction
   - compression ratio (zlib size / raw size) — low = repetitive
   - fraction of empty completions
   - fraction starting with non-alphabetic characters (emojis, punctuation)
7. **Over-refusal damage on benign prompts**
   Run all four conditions on Alpaca-200 or MT-Bench and report the coherence
   score. If safety_head and layer_matched_random both crater benign
   helpfulness, the "safety" effect is collateral.

## Tier 3 — independent verification

8. **Second judge**: Llama-Guard-3-8B and/or ShieldGemma-9B on the same
   completions. The CoT-safety judge is one opinion; a guard-trained
   classifier removes single-judge bias.
9. **StrongREJECT** (Souly et al. 2024): a published jailbreak scoring rubric
   that returns a refusal score and a "useful response" score — exactly the
   axes we already care about, with a published baseline.
10. **AdvBench / HarmBench expansion**. JailbreakBench-100 is small. The
    AdvBench harmful-behaviors set (520) is a low-cost extension that lets
    you compute confidence intervals that don't have ±10pp error bars.
11. **Random-head bootstrap** (rather than a single sample). Sample 20
    different `top-10` uniform-random selections and 20 layer-matched ones;
    plot the distribution of `harmful_rate_among_coherent`. The SHIPS point
    estimate is only meaningful if it sits outside this distribution.

## Tier 4 — circuit-level claims

12. **Ablation strength sweep**. Vary `scale_factor ∈ {1e-3, 1e-4, 1e-5}`
    and `k ∈ {1,3,5,10,20}`. A real safety circuit should produce a
    monotone shift; a "model damage" effect tends to saturate or be erratic.
13. **Targeted intervention vs. complement**. Patch back in the *other* heads
    at the same layers and check that harm rate falls — i.e. the
    necessary/sufficient sense of "safety head".
14. **Activation patching** instead of weight scaling. Replace head outputs
    from baseline forward pass on the same prompt — much cleaner causal
    interpretation, doesn't perturb capability.

## Suggested first wave

Run, in this order:

1. The 30B-scale judge over all 4 condition × 100 prompts.
2. Gibberish detector over the same 400 rows.
3. Report:
   - `harmful_rate`, `safe_rejection_rate` per condition (current numbers)
   - `harmful_rate_among_coherent` per condition (new)
   - paired McNemar `safety_head` vs `baseline` and vs `layer_matched`
   - per-category harm-rate table
4. Run benign-set coherence judge on the same 4 conditions × 100 Alpaca prompts.

That gets us a defensible answer to the collaborator's claim with a single
additional run plus two analysis passes.
