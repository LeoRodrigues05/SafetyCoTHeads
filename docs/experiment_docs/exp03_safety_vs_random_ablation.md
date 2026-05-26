# Experiment 3: Safety-Head vs Random-Head Ablation

## Goal

Test whether ablating discovered safety heads changes held-out harmful behavior
more than ablating arbitrary or layer-matched random heads.

## Configs

- `configs/experiments/exp03_safety_vs_random_ablation/03-baseline.yaml`
- `configs/experiments/exp03_safety_vs_random_ablation/04-safety-head-ablation.yaml`
- `configs/experiments/exp03_safety_vs_random_ablation/04-safety-head-ablation-top1.yaml`
- `configs/experiments/exp03_safety_vs_random_ablation/04-safety-head-ablation-top3.yaml`
- `configs/experiments/exp03_safety_vs_random_ablation/04-safety-head-ablation-top5.yaml`
- `configs/experiments/exp03_safety_vs_random_ablation/04-safety-head-ablation-top8.yaml`
- `configs/experiments/exp03_safety_vs_random_ablation/05-random-head-ablation.yaml`
- `configs/experiments/exp03_safety_vs_random_ablation/06-layer-matched-random.yaml`
- Optional sweep: `configs/experiments/exp03_safety_vs_random_ablation/matrix.yaml`

## Commands

```bash
python -m scripts.run_generation \
  --config configs/experiments/exp03_safety_vs_random_ablation/03-baseline.yaml

python -m scripts.run_generation \
  --config configs/experiments/exp03_safety_vs_random_ablation/04-safety-head-ablation.yaml

python -m scripts.run_generation \
  --config configs/experiments/exp03_safety_vs_random_ablation/05-random-head-ablation.yaml

python -m scripts.run_generation \
  --config configs/experiments/exp03_safety_vs_random_ablation/06-layer-matched-random.yaml
```

### Safety-head dose sweep

The top-10 safety-head run can over-degrade generation. To inspect smaller
interventions, run the top-1, top-3, top-5, and top-8 safety-head configs:

```bash
python -m scripts.run_generation --config configs/experiments/exp03_safety_vs_random_ablation/04-safety-head-ablation-top1.yaml
python -m scripts.run_generation --config configs/experiments/exp03_safety_vs_random_ablation/04-safety-head-ablation-top3.yaml
python -m scripts.run_generation --config configs/experiments/exp03_safety_vs_random_ablation/04-safety-head-ablation-top5.yaml
python -m scripts.run_generation --config configs/experiments/exp03_safety_vs_random_ablation/04-safety-head-ablation-top8.yaml
```

To run all four and generate the same side-by-side response reports used for
the top-10 check:

```bash
bash scripts/run_safety_head_dose_sweep.sh
```

The report script can also be run manually for a single condition:

```bash
python -m scripts.make_pre_post_report \
  --baseline runs/03-baseline/completions_baseline.jsonl \
  --ablation runs/04-safety-head-ablation-top1/completions_safety_head_ablation_top1.jsonl \
  --ablation-label safety_head_ablation_top1 \
  --out-dir runs/analysis/safety_head_pre_post_top1 \
  --title "Safety Head Ablation Top-1: Pre/Post Responses"
```

## Code Paths

- Entry point: `scripts/run_generation.py`
- Dataset loader: `src/safety_cot_heads/data/jailbreakbench.py`
- Prompt rendering: `src/safety_cot_heads/generation/prompts.py`
- Generation loop: `src/safety_cot_heads/generation/generate.py`
- Mask builder: `src/safety_cot_heads/interventions/ablation.py`
- Random controls: `src/safety_cot_heads/attribution/random_heads.py`
- Runtime mask hooks: `src/safety_cot_heads/models/custom_llama.py`

## Data Flow

1. All runs load `data/raw/sha/jailbreakbench.csv`, column `input`.
2. The first `dataset.n=100` prompts become rows with IDs like `jbb-00000`.
3. Prompts are rendered through the tokenizer chat template when available.
4. Generation uses greedy decoding with `max_new_tokens=256`.

## Conditions

`baseline` uses no mask.

`safety_head_ablation` reads the top 10 SHIPS heads from
`runs/01-ships-discovery/ships_dataset_ranking.json`, builds one mask config,
and keeps it active for the whole `model.generate(...)` call.

`safety_head_ablation_top1`, `top3`, `top5`, and `top8` read the same SHIPS
ranking but use smaller `heads.top_k` values. New generation rows record
`n_ablated_heads` and `ablated_heads` so the exact intervention is visible in
the output artifact.

`random_head_ablation` samples 10 heads uniformly over the full layer/head grid
with seed 0.

`layer_matched_random_head_ablation` reads the same SHIPS top-10 reference set,
matches its per-layer counts, and samples random head indices within those
layers. The sampler does not explicitly exclude the original safety heads.

All ablation runs use `mask_qkv: [q]`, `mask_type: scale_mask`, and
`scale_factor: 1e-4`.

## Outputs

- `runs/03-baseline/completions_baseline.jsonl`
- `runs/04-safety-head-ablation/completions_safety_head_ablation.jsonl`
- `runs/04-safety-head-ablation-top1/completions_safety_head_ablation_top1.jsonl`
- `runs/04-safety-head-ablation-top3/completions_safety_head_ablation_top3.jsonl`
- `runs/04-safety-head-ablation-top5/completions_safety_head_ablation_top5.jsonl`
- `runs/04-safety-head-ablation-top8/completions_safety_head_ablation_top8.jsonl`
- `runs/05-random-head-ablation/completions_random_head_ablation.jsonl`
- `runs/06-layer-matched-random/completions_layer_matched_random_head_ablation.jsonl`
- `runs/analysis/safety_head_pre_post_top*/` for dose-sweep side-by-side
  reports after `scripts/make_pre_post_report.py` runs.

## Review Checks

- Judge all four completion files with Experiment 2.
- Compare rows by shared prompt ID across conditions.
- The main evidence is the safety-head delta versus the random and
  layer-matched controls.
