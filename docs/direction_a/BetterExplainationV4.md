# The Hypotheses and how they're tested

Why this matters for *this* paper specifically:

- The central claim ("different interventions damage different safety subprocesses") is the kind of claim that's easy to **post-hoc rationalise**. If you eyeball the data and pick the most striking pattern, you can always tell a story.
- Reviewers will be suspicious because the v3 framing ("we can classify intervention from a 7-vector") was already vulnerable to the rebuttal *"sure, but you're just classifying stylistic artifacts — sentence length, refusal-template residue, etc."* (this is the "buzzword warning" §4.6 calls out).
- So §4 says: here are 5 hypotheses (H1–H5). For each, here is the *specific statistical test*, the *specific number*, and the *specific direction* required. You can read off "did it pass?" without judgement calls.


---

## 4.1 Pathway-dissociation test (H1)

### What's being tested
*"At matched ASR, do different intervention families leave different fingerprints in the sentence-level failure labels?"*

### The data going in
For every generated completion you have:
- **Sentence-level pathway labels** — the §13.A taxonomy assigns each sentence in the CoT one of 12 labels (things like `recognition_loss`, `refusal_suppression`, `rationalisation_before_execution`). This is the judge's per-sentence output.
- **Per-trace pathway vector** (§13.A.2) — those per-sentence labels rolled up into 8 numbers describing the *whole trace*: e.g. "first unsafe sentence index", "count of refusal-suppression sentences", and a categorical `dominant_pathway` (which of the 12 was most prominent).
- **Family** — which intervention produced it: DSH-$v_H$, DSH-$v_R$, Arditi-$r$, SHIPS heads, safety neurons.
- **Matched ASR** — we restrict the analysis to a band where every family produces ~50% jailbreak success, so differences can't be explained by "this family just jailbreaks more."

### The model being fit
$$\ell \sim \text{family} + (1\,|\,\text{prompt}) + (1\,|\,\text{seed})$$

Decoded:
- **$\ell$**: the binary outcome "did pathway label $\ell$ appear in this trace?" — fitted separately for each of the 12 labels.
- **`family`**: fixed effect — the thing we care about. Its coefficient tells you "given family X, how much more or less likely is label $\ell$ to appear?"
- **`(1|prompt)`** and **`(1|seed)`**: random intercepts. These soak up the fact that some prompts are just intrinsically harder, and some seeds happen to land on bad rolls. Without them you'd be conflating "DSH-$v_R$ does X" with "DSH-$v_R$ happened to be run on the easy prompts." The paired design (same prompts × same seeds across all conditions) makes this work.
- **Mixed-effects logistic regression**: just a logistic regression with the random intercepts. "Mixed" = fixed + random effects.

### The pre-registered thresholds
Two things must both be true at the iso-ASR-50% band:

1. **$\chi^2$ test on `dominant_pathway` distributions across families: $p < 0.01$ after BH-FDR.**
   - Concretely: count how often each `dominant_pathway` value appears under each family. You get a 5-family × ~6-dominant-pathway contingency table. The $\chi^2$ test asks "could this table have come from all families sharing one distribution?" If $p<0.01$, no.
   - **BH-FDR** = Benjamini-Hochberg False Discovery Rate. You're running many tests (12 labels + this $\chi^2$ + later tests); without correction you'd get false positives by chance. BH-FDR adjusts the $p$-value cutoff so the *expected fraction* of declared-significant tests that are actually noise stays below your target (typically 5%).

2. **At least one of the 12 individual labels shows $p<0.05$ after BH-FDR with the *correct sign*.**
   - "Correct sign" = matches H2's prediction (recognition-flavoured labels move more under $v_H$, refusal-flavoured labels move more under $v_R$). This rules out "you found a difference, but it was random."

### What failure looks like
The $\chi^2$ comes out at $p=0.3$, no individual label moves consistently — you cannot tell the families apart from their CoT-failure fingerprints. H1 is dead.

---

## 4.2 CoT-monitorability test (H3)

### What's being tested
*"Does the visible reasoning trace still warn you that the final answer will be unsafe?"*

This is the test that connects to the OpenAI/Anthropic agenda of using CoT as a *safety monitor*. If you can read the model's CoT and tell "this is going somewhere bad", you can intervene. If interventions break that link — model writes a calm, refusal-looking CoT then produces a harmful answer — CoT monitoring stops working.

### The data going in
Two ASR numbers per (condition):
- **`ASR_final`**: judge sees the full completion (CoT + final answer), labels it unsafe or not. This is regular ASR.
- **`ASR_cot_only`**: judge sees *only the CoT* (final answer stripped) and is asked to **predict** whether the final answer will be unsafe. This is the `COT_ONLY_PREDICTION_PROMPT` from R1.1.

The gap:
$$\text{monitorability\_gap} = \text{ASR}_\text{final} - \text{ASR}_\text{cot\_only}$$

- **Positive gap**: more harmful answers than the CoT predicted — CoT is *misleadingly safe*. This is the dangerous case: monitoring would miss the attack.
- **Near-zero gap**: CoT accurately signals what's coming. Monitoring works.
- **Negative gap**: CoT looks scarier than the final answer — model "thinks aloud" about harm but then refuses.

### The pre-registered thresholds

1. **Baseline gap: $|\overline{\text{gap}}| \leq 0.05$ with bootstrap 95% CI.**
   - The unmodified model must have CoT and final-answer that mostly agree. Otherwise the metric is broken before the experiment even starts.
   - **Bootstrap 95% CI**: resample the prompts with replacement many times (say 10,000), recompute the mean gap each time, take the 2.5th and 97.5th percentiles. That's your interval. No distributional assumptions.

2. **Cross-family separation at iso-ASR 50%: at least one pair of families differs in mean gap by $\geq 0.10$, $p<0.05$ paired bootstrap ($B=10{,}000$) after BH-FDR.**
   - **Paired bootstrap**: because you have *the same prompts* under every condition, you resample at the *prompt* level — each bootstrap iteration draws a set of prompt-IDs, then looks up "what was family-A's gap on prompt $i$? what was family-B's?" and computes the difference. This kills prompt-level variance.
   - **$B=10{,}000$**: number of resamples. Big enough that the percentile bounds are stable.
   - **0.10 effect size**: a 10-percentage-point swing in gap is the smallest difference they're willing to claim is *practically* meaningful, not just statistically detectable.

### What failure looks like
Every family has a gap of roughly zero, regardless of intervention. CoT monitoring is robust. H3 is dead — which is actually good news for safety, just not a paper.

---

## 4.3 DSH double-dissociation test (H2)

### What's being tested
*"Is the DSH theory's two-axis story (harmfulness axis $v_H$, refusal axis $v_R$) actually visible in the failure pathways?"*

DSH (Wu et al.) predicts: steer along $v_H$ → model stops *recognising* the prompt as harmful (recognition failure); steer along $v_R$ → model still recognises harm but stops *initiating refusal* (refusal failure). Different subprocesses.

### Why "double-dissociation"?
In neuropsychology, "double dissociation" means: intervention A hurts task X but not task Y, AND intervention B hurts task Y but not task X. Crucially you need both directions — otherwise A might just be stronger than B at everything. Here:
- $v_H$ steering should disrupt **`recognition_loss` pathway** ≫ `refusal_loss` pathway.
- $v_R$ steering should disrupt **`refusal_loss` pathway** ≫ `recognition_loss` pathway.

If both hold, the two axes target genuinely different subprocesses.

### The data going in
- `recognition_loss` rate per trace (fraction of sentences with that pathway label).
- `refusal_loss` rate per trace.
- Family ∈ {$v_H$, $v_R$}, both at the same iso-ASR band.

### The model being fit
A mixed-effects regression with a **(family × pathway-group) interaction term**. The key statistic is whether that interaction is significant — i.e. is the *effect of family* different depending on which *pathway group* you're looking at? A significant interaction = the two families don't move the two pathway groups in parallel = dissociation.

Plus a sign check: the pattern must match the table preregistered in `prereg_v4.md` §7. This rules out "I got an interaction but it's in the wrong direction" — which would be a different (and unpredicted) phenomenon.

### Why Arditi $r$ is "descriptive only"
Arditi's refusal direction is conceptually similar to DSH's $v_R$ but extracted by a different method. The plan reports the cosine similarity between Arditi-$r$ and the $(v_H, v_R)$ subspace — if Arditi-$r$ projects mostly onto $v_R$, it's evidence the two methods are finding the same thing. But this is a *comparison*, not a formal test, hence "descriptive."

### What failure looks like
Both $v_H$ and $v_R$ hit both pathway groups equally — the interaction is null. Either DSH theory is wrong, or the pathway labels can't resolve the distinction, or both. H2 is dead.

---

## 4.4 Phase-localisation test (H4)

### What's being tested
*"For each intervention family, **when** in the generation timeline does it do its damage?"*

§8.5 defines five "phases" — windows of token positions during which the intervention is active:
- **P-prompt**: only during prompt encoding.
- **P-early**: first 64 generated tokens.
- **P-late**: last 64 tokens before the answer.
- **P-answer**: the final answer tokens only.
- **P-whole**: always-on (the default).

### The prediction
Each family should have a *predictable* phase where it does its maximum work, derivable from the subprocess it targets:

- **$v_H$ (recognition)**: damage must happen *while the model is still reading or just starting to think* about the prompt → **P-prompt or P-early**.
- **$v_R$, Arditi $r$ (refusal initiation)**: damage happens at the moment the model decides whether to refuse → **P-answer**.
- **SHIPS heads (refusal maintenance)**: refusal already started; heads keep it going during the body of the response → **P-late**.

### The test
For each family, find the phase $p^*$ that maximises pathway shift versus the baseline (whole-on or off). Check whether $p^*$ matches the prediction.

- **Permutation test over phase labels**: shuffle the phase labels among the 5 phases and recompute "which phase was max" many times. If the real $p^*$ matches prediction more often than chance under the shuffles, the localisation is significant ($p<0.05$ per family).

### The aggregate threshold
**"≥3 of 4 families match preregistered phase"** — at least 3 out of {$v_H$, $v_R$, SHIPS, Arditi $r$} must land on their predicted phase. This is the global pass criterion.

### What failure looks like
Every family does the same thing in P-whole as in any narrow window, or families land on the *wrong* phases (e.g. $v_H$ does most of its damage in P-answer, contradicting its supposed recognition-axis role). H4 is dead, and the subprocess interpretation looks shaky.

---

## Why all four matter as a *set*

They're nested in a specific way:

- **4.1** establishes *the families differ in CoT pathway*. Without this nothing else means anything.
- **4.3** says *they differ in the specific way DSH theory predicts* — much stronger than 4.1 alone, because it's a directional prediction.
- **4.4** says *the differences localise in time the way the theory predicts* — a second independent angle on H2.
- **4.2** says *the differences matter for safety monitoring* — connects an intrinsic interp finding to an applied safety concern.

If 4.1 passes but 4.3/4.4 fail, you've got "families differ but not how the theory said" — interesting but theoretically untidy. If 4.2 fails (every family has zero gap), the monitorability story dies but the dissociation story can stand on its own. The set is designed so each test can independently die without taking the whole paper with it.