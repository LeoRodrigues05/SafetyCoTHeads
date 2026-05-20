# Experiment 4: Coherency-Head Discovery

## Goal

Identify heads whose ablation damages benign coherence, then use them as a
quality-degradation control against safety-head ablation.

## Configs

- `configs/experiments/exp04_coherency_head_discovery/07-coherency-discovery.yaml`
- `configs/experiments/exp04_coherency_head_discovery/07b-coherency-ablation.yaml`

## Intended Commands

```bash
python -m scripts.run_attribution \
  --config configs/experiments/exp04_coherency_head_discovery/07-coherency-discovery.yaml

python -m scripts.run_generation \
  --config configs/experiments/exp04_coherency_head_discovery/07b-coherency-ablation.yaml
```

## Current Status

Partially wired. The core coherency attribution module exists, but
`scripts/run_attribution.py` currently dispatches only `ships` and `sahara`.
The discovery command will need a `method == "coherency"` branch before this
experiment is runnable end to end.

The ablation config depends on the discovery output:

```text
runs/07-coherency-discovery/coherency_dataset_ranking.json
```

## Code Paths

- Planned entry point: `scripts/run_attribution.py`
- Coherency scoring: `src/safety_cot_heads/attribution/coherency.py`
- Dataset loaders: `src/safety_cot_heads/data/benign.py`
- Generation ablation: `scripts/run_generation.py`
- Mask builder: `src/safety_cot_heads/interventions/ablation.py`

## Intended Data Flow

1. Load benign prompts from `dataset.name=alpaca`, `dataset.n=100`.
2. Load the target model and attach the mask controller.
3. Run baseline benign scoring.
4. For each head, apply a Q-head ablation and recompute the coherency score.
5. Rank heads by ablation-induced degradation.
6. Ablate the top ranked coherency heads on JailbreakBench using the same
   generation pipeline as Experiment 3.

## Scoring Mode

The current config uses:

```yaml
method_args:
  mode: nll
```

So the implemented scoring target is increased benign negative log likelihood.
The module also sketches judge-coherence, pathology, and hybrid modes.

## Outputs

Planned discovery output:

- `runs/07-coherency-discovery/coherency_dataset_ranking.json`

Planned ablation output:

- `runs/07b-coherency-ablation/completions_coherency_head_ablation.jsonl`

## Review Checks

- First verify the CLI dispatch is wired.
- Confirm the coherency ranking contains valid `(layer, head)` entries.
- Compare this condition against safety-head ablation to separate safety
  effects from generic coherence loss.
