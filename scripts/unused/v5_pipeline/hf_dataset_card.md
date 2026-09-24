---
license: other
license_name: research-use-only
license_link: LICENSE
task_categories:
  - text-generation
language:
  - en
tags:
  - ai-safety
  - red-teaming
  - safety-evaluation
  - chain-of-thought
  - mechanistic-interpretability
  - jailbreak
pretty_name: Safety-CoT Interventions (Direction A v5)
size_categories:
  - 1K<n<10K
extra_gated_prompt: >-
  This dataset contains model completions to harmful prompts (successful and
  attempted jailbreaks) produced by deliberately safety-suppressed models, for
  safety-evaluation research. By requesting access you agree to use it only for
  safety research and to not redistribute the raw harmful generations.
extra_gated_fields:
  Name: text
  Affiliation: text
  Intended use: text
  I agree to use this data for safety research only: checkbox
---

# ⚠️ Safety-CoT Interventions — Direction A v5 (per-query results)

> **Content warning.** This dataset contains **model completions to harmful and
> jailbreak prompts**, produced by models whose safety behaviour was deliberately
> suppressed (head/neuron ablation, refusal-direction steering, directional ablation).
> Many completions are unsafe by design. It is released for **safety-evaluation
> research only**. Do not use it to build or deploy harmful systems.

Per-query generations and LLM-judge annotations for the **Direction A v5** grid — the
evaluation substrate behind *"One Number Isn't Enough: A Decomposable Metric for
Comparing White-Box Safety Interventions on Reasoning Models."* This is the raw,
per-query data (too large for GitHub); the **aggregate metrics, reports, and code** live
in the GitHub repo: `https://github.com/LeoRodrigues05/SafetyCoTHeads`.

## What this is
For each `(model, dataset, condition)` cell we generate completions to harmful prompts
under a white-box safety intervention, then score each completion with three LLM judges
(5-label safety, per-sentence safety-reasoning trace, and a 12-label pathway taxonomy).
The dataset lets you reproduce the paper's three-axis metric (Potency / Quality /
Safety-Reasoning → Selective-Failure Score) from scratch, or re-judge with your own
instrument.

## The grid
- **Models:** `qwen3_8b`, `olmo3_7b_think` (explicit `<think>`), `olmo3_7b_base`,
  `olmo3_7b_base_own`, `llama31_8b_control`. *(`r1_distill_qwen_7b` is included as
  exploratory data and is **not** part of the paper's five-model analysis.)*
- **Conditions:** `baseline`; `ships_top{3,5,8}` (safety-head ablation);
  `neurons_top{256,512,1024}` (safety-neuron ablation); `steering_a{0.5,1.0,1.5}`
  (refusal-direction activation-addition); `steering_ablate` (directional ablation).
- **Datasets:** `jbb` (JailbreakBench, 100 prompts), `bt` (BeaverTails, 98 = 7×14).
- **Seeds:** seed0 (greedy) primary.

## Directory layout
```
{model}/
  gen/{dataset}/{condition}/seed0/completions_{condition}.jsonl   # prompt + completion
  judge/{dataset}/{condition}/seed0/
      judge_safety__*.jsonl            # 5-label safety (harmful_response, safe_rejection, …)
      judge_pathway.jsonl, judge_pathway__*.jsonl   # 12-label pathway taxonomy
      judge_safety_reasoning_trace.jsonl            # per-sentence safety-reasoning
      judge_cot_only.jsonl             # CoT-only prediction (monitorability)
      coherence.jsonl                  # coherence / gibberish gate
      monitorability_rows.jsonl, pathway_vectors.jsonl, prefix_rows.jsonl
      summary.json                     # per-cell aggregate metrics
  16-neuron-discovery/neuron_ranking.json           # safety-neuron ranking
  17-direction-extraction/refusal_directions.npz    # refusal direction
composite_cells.csv / composite_cells.json          # headline per-cell (P,Q,S,SFS)
composite_report.html, safety_reasoning_report.html # rendered reports
```

## Key record schemas
- **completions** (`completions_*.jsonl`): `id, dataset, category, prompt,
  rendered_prompt, completion, model, condition, decoding, mask_cfg_active,
  neuron_cfg_active, steering_cfg_active, timestamp`.
- **judge** (`judge_*.jsonl`): `id, dataset, category, condition, model, judge_model,
  judge_kind, judge_flat, judge_parse_status, parent_id`. Join to completions on `id`.

## Judges
| Judge | Model | Output |
|---|---|---|
| Standard 5-label | Qwen3-30B-A3B-Instruct | safety booleans, coherence, CoT-only monitor |
| Safety-reasoning trace | Qwen3-30B (vLLM) | per-sentence safety reasoning, 6 categories |
| Pathway | fine-tuned Qwen3-14B LoRA | 12-label pathway taxonomy → 8-dim vector |

The pathway judge is human-validated at κ≈0.96 / F1 0.98 vs a gold set; the 5-label,
CoT-only, and SR-trace judges are human-validated on `batch_v5_002` (two annotators).

## License & intended use
Research-use-only; **gated access** recommended (see the access prompt). The dataset
includes harmful generations for the purpose of measuring and improving safety
interventions. \TODO finalise license and citation before release.

## Citation
```bibtex
@misc{safetycot2026,
  title  = {One Number Isn't Enough: A Decomposable Metric for Comparing
            White-Box Safety Interventions on Reasoning Models},
  author = {Rodrigues, Leo and others},
  year   = {2026},
  note   = {VERIFY authors/venue.}
}
```
