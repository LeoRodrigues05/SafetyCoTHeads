# Experiment 1: Reproduce SHIPS + Sahara

## Goal

Discover safety-relevant attention heads on MaliciousInstruct with the two
paper-aligned attribution methods: SHIPS and Sahara. These ranked heads feed the
later ablation experiments.

## Configs

- `configs/experiments/exp01_reproduce_ships_sahara/01-ships-discovery.yaml`
- `configs/experiments/exp01_reproduce_ships_sahara/02-sahara-discovery.yaml`

## Commands

```bash
python -m scripts.run_attribution \
  --config configs/experiments/exp01_reproduce_ships_sahara/01-ships-discovery.yaml

python -m scripts.run_attribution \
  --config configs/experiments/exp01_reproduce_ships_sahara/02-sahara-discovery.yaml
```

## Code Paths

- Entry point: `scripts/run_attribution.py`
- Dataset loader: `src/safety_cot_heads/data/maliciousinstruct.py`
- SHIPS: `src/safety_cot_heads/ships_legacy/ships.py`
- Sahara: `src/safety_cot_heads/sahara_legacy/sahara.py`
- Mask hooks: `src/safety_cot_heads/models/custom_llama.py`
- Mask config helpers: `src/safety_cot_heads/models/masks.py`

## Data Flow

1. The config loads `dataset.name=maliciousinstruct`.
2. `load_maliciousinstruct(...)` reads `data/raw/sha/maliciousinstruct.csv`,
   column `input`, and keeps the first `dataset.n=100` prompts.
3. Each prompt is rendered as:

```text
## Query:{q}
## Answer:
```

4. `load_model(...)` loads `meta-llama/Llama-2-7b-chat-hf` and attaches the
   `HeadMaskController`.

## What Happens

SHIPS runs one unmasked forward pass per prompt, then ablates each head
individually and scores the KL shift in the final-token softmax distribution.
It mean-aggregates per-prompt scores into a dataset ranking.

Sahara runs the unmasked prompt set, collects final-token last hidden states,
then greedily selects heads whose Q-head ablation most shifts the hidden-state
subspace. The default search runs `search_step=8`.

Both methods use runtime activation hooks, not weight edits. The default mask
targets Q only: `mask_qkv: [q]`.

## Outputs

- `runs/01-ships-discovery/ships.jsonl`
- `runs/01-ships-discovery/ships_dataset_ranking.json`
- `runs/02-sahara-discovery/sahara.json`
- `runs/02-sahara-discovery/sahara_ranked.jsonl`

## Review Checks

- SHIPS `all_scores` should have one entry per model head per prompt.
- Sahara `selected_heads` should contain 8 heads for the full run.
- Top heads should be valid `(layer, head)` indices and should show non-random
  concentration before downstream causal claims are made.
