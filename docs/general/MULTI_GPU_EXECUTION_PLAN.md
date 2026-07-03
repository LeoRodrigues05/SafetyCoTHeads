# Multi-GPU execution plan — finish the v5 grid + add one thinking model

> **Purpose.** Finish the remaining Direction A v5 work and add one new thinking
> model, parallelised across N GPUs on the single node. Written 2026-07-03,
> branch `judge-validation-batch-v5_002`. Companion to
> [`COMPOSITE_METRIC_CONTINUATION.md`](COMPOSITE_METRIC_CONTINUATION.md) (which
> covers the *what/why*); this file is the *how, on K cards*.

Hardware assumed: node with **G × B200 (183 GB)**, shared `HF_HOME`, torch
2.11+cu128, vLLM 0.23. All cards share one filesystem and one HF cache.

---

## 0. At-a-glance

| Config | Wall-clock target | Notes |
|---|---|---|
| 1 GPU (serial) | **~38–42 h** | today's baseline |
| 2 GPUs | **~22–26 h** | llama lane ∥ new-model lane |
| 4 GPUs | **~13–16 h** | + shard generation & judging |
| 8 GPUs | **~7–9 h** | + shard the new-model pathway pass wide |

The critical path is the **new model**: its 22-cell generation (~8–14 GPU-h) and
its 14B **pathway** pass (~12–13 GPU-h) dominate. Everything else (the pathway
re-judge already running, and the whole Llama lane) fits in the slack.

---

## 1. State at time of writing

- **RUNNING (GPU 0, token-free):** pathway re-judge of the only cells missing the
  14B pathway pass — `qwen3_8b` 8 steering cells + `olmo3_7b_think` `jbb/steering_ablate`.
  Local judge, `HF_HUB_OFFLINE=1`. ETA ~5–6 h. This is unit **U0**.
- **Complete:** `olmo3_7b_base`, `olmo3_7b_base_own` (22/22 all layers);
  `qwen3_8b` and `olmo3_7b_think` (complete once U0 lands).
- **Missing — Llama lane (needs HF token, gated Llama):** all 8 `llama31_8b_control`
  steering cells (gen + all judge layers) and its `refusal_directions.npz`.
- **New — one thinking model** (see §6), full pipeline from discovery.

---

## 2. Prerequisites (do once, before fan-out)

```bash
cd /work/Work/SafetyCoTHeads && source .venv/bin/activate

# (a) HF auth — needed for gated Llama; harmless for the open models.
hf auth login --token hf_XXXX      # writes ~/.cache/huggingface/token (all procs read it)
hf auth whoami                     # verify
#   ...and click "accept license" once on the Llama-3.1-8B-Instruct model page.

# (b) Regenerate configs so the new model gets its per-cell YAMLs (after §6 edit).
python -m scripts.make_v5_configs --matrix configs/experiments/direction_a_v5_iso_asr/matrix.yaml

# (c) PRE-WARM the shared cache so G parallel jobs don't each re-download the same
#     weights (races + wasted bandwidth). Do this serially, once:
hf download Qwen/Qwen3-30B-A3B-Instruct-2507   # standard + SR judge (~60 GB)
hf download meta-llama/Llama-3.1-8B-Instruct   # llama lane (gated)
hf download <NEW_MODEL_REPO>                    # e.g. deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# (pathway 14B judge is already local: models/pathway_judge_14b_merged)
```

Per-shell env every lane wants:
```bash
export PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=warning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## 3. Work units, dependencies, serial cost

`∥` = the three judge layers (standard 30B / pathway 14B / SR-trace vLLM) are
**independent** given completions — put them on different cards.

| Unit | What | Depends on | 1-GPU cost |
|---|---|---|---|
| **U0** | pathway re-judge (qwen 8 + olmo-think 1) | — | 5–6 h *(running)* |
| **L1** | Llama direction extraction → `refusal_directions.npz` | HF token | ~0.3 h |
| **L2** | Llama gen × 8 steering cells (512-tok, fast) | L1 | ~2 h |
| **L3a** | Llama standard judge × 8 (30B) | L2 | ~3.5 h |
| **L3b** | Llama pathway judge × 8 (14B, short CoT) | L2 | ~1.5 h |
| **L3c** | Llama SR-trace × 8 (vLLM) | L2 | ~0.3 h |
| **M0** | New-model discovery: SHIPS ∥ neurons ∥ direction | token | 1–3 h |
| **M1** | New-model gen × 22 (thinking, 1024-tok) | M0 | 8–14 h |
| **M2a** | New-model standard judge × 22 (30B) | M1 | 8–13 h |
| **M2b** | New-model pathway judge × 22 (14B, long CoT) | M1 | 12–13 h |
| **M2c** | New-model SR-trace (vLLM, whole model) | M1 | ~0.7 h |
| **F** | reaggregate → composite report → plots (CPU) | all | ~0.2 h |

Serial total ≈ **~40 GPU-h** (≈ sum of the longest judge layer per model).

---

## 4. How sharding works in this repo

Three mechanisms, all already supported:

1. **Pin a lane to a card:** `CUDA_VISIBLE_DEVICES=k <command>`. The existing
   `scripts/run_gpu0_olmo3_judging.sh` / `run_gpu1_llm_judging.sh` are examples of
   pinning whole models to cards.
2. **Shard cells across cards (generation & standard/pathway judging).** These
   passes iterate cells serially inside one process, so to parallelise you launch
   **disjoint cell subsets on different cards**. Generation is one `run_generation`
   per cell YAML; judging takes repeatable `--condition` specs you can split.
   Helper (round-robins every gen cell across `G` cards):
   ```bash
   # fan_gen.sh  — usage: bash fan_gen.sh <model_key> <G>
   MODEL=$1; G=$2; i=0
   for cfg in configs/experiments/direction_a_v5_iso_asr/$MODEL/gen/*/*.yaml; do
     gpu=$(( i % G ))
     CUDA_VISIBLE_DEVICES=$gpu nohup python -m scripts.run_generation --config "$cfg" \
       > logs/gen_${MODEL}_$(basename $(dirname $cfg))_$(basename $cfg .yaml).log 2>&1 &
     i=$((i+1)); [ $(( i % G )) -eq 0 ] && wait   # throttle to G in flight
   done; wait
   ```
   (Same shape works for the pathway pass by launching `run_v4_jbb_judge
   --config judge_14b.yaml --skip-safety --skip-coherence --skip-cot-only` with a
   card's share of `--condition` specs — see the runner in
   `/scratchpad/run_pathway_missing.sh` for the exact spec format.)
3. **Built-in SR sharding:** `run_sr_vllm.sh --num-shards G --shard-id k`, one per
   card. It splits the (model,dataset,condition) task list evenly and is resume-safe.

All resume-safe: re-running skips finished cells/rows. HF cache is shared, so
pre-warm (§2c) before fanning out.

---

## 5. Concrete schedules

### 5a. 2 GPUs (~22–26 h)
- **GPU 0:** U0 (running) → Llama lane `L1 → L2 → L3a` → then join GPU 1 on the
  new-model judge.
- **GPU 1:** New-model `M0 → M1` (generation is the long pole) → `M2a/M2b`.
- Fold `L3b/L3c` and `M2c` onto whichever card is idle (they're short).

### 5b. 4 GPUs (~13–16 h)
Phase-based; each phase ends with a short barrier.

| Phase | GPU0 | GPU1 | GPU2 | GPU3 |
|---|---|---|---|---|
| P0 (0–6 h) | U0 (pathway) | L1→L2 (llama gen) | M0 SHIPS discovery | M0 neurons+direction |
| P1 (gen) | M1 gen shard ¼ | M1 gen shard ¼ | M1 gen shard ¼ | M1 gen shard ¼ |
| P2 (judge) | M2b pathway ½ | M2b pathway ½ | M2a std ½ + L3a | M2a std ½ + L3b/c + M2c |
| F | reaggregate + composite + plots (CPU, any card) | | | |

`fan_gen.sh <NEW_MODEL_KEY> 4` drives P1; split the 22 conditions across cards for
P2 (pathway on 2 cards since it's the longest layer, standard on the other 2).

### 5c. 8 GPUs (~7–9 h)
- P0: GPU0=U0, GPU1=Llama `L1→L2→L3*`, GPU2–4 = `M0` three discovery passes in
  parallel, GPU5–7 pre-idle / start pre-warm verification.
- P1: `fan_gen.sh <NEW_MODEL_KEY> 8` — 22 cells over 8 cards ≈ 3 waves (~2–4 h).
- P2: pathway (M2b, the 12–13 GPU-h pole) across **5 cards**, standard (M2a) across
  **3 cards**; SR-trace (M2c) + Llama leftovers slot into the first card that frees.
- F: reaggregate + composite + plots.

Rule of thumb: **give the new-model pathway pass the most cards** — it's the
longest single layer.

---

## 6. The new thinking model (pick + wiring)

Requirement: dense **~7–9 B**, emits explicit `<think>`/reasoning traces (so the
Safety-Reasoning axis applies), open weights, transformers-supported. Shortlist
and recommendation are in the chat reply; **worked example = DeepSeek-R1-Distill-Qwen-7B**
(distinct R1 reasoning lineage, Qwen-math backbone, ungated).

**Wiring (two edits, then regenerate configs):**

1. `configs/models.yaml` — add:
   ```yaml
   r1_distill_qwen_7b:
     name: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
     dtype: bfloat16
     attn_implementation: eager
     load_in_4bit: false
     device_map: null
     trust_remote_code: false
   ```
   (`deepseek_r1_distill_llama_8b` already exists if you prefer the Llama-backbone
   distill — but its backbone overlaps the Llama control, so the Qwen distill adds
   more range.)

2. `configs/experiments/direction_a_v5_iso_asr/matrix.yaml` → `models:` — add:
   ```yaml
   r1_distill_qwen_7b:
     model_ref: r1_distill_qwen_7b
     max_new_tokens: 1024      # room for <think> + answer
     batch_size: 8
     # full own-model discovery (no reuse_discovery_*): it's a fresh lineage
   ```

3. `python -m scripts.make_v5_configs --matrix .../matrix.yaml` → generates its
   discovery + 22 gen + judge configs. Then run `M0→M1→M2*` as in §5.

> If the chosen model needs a newer arch than the pinned transformers, bump it in
> the venv first and re-run `scripts/setup_vm.sh` smoke (as was done for OLMo-3).

---

## 7. Finalise (unit F, CPU, minutes)

```bash
python -m scripts.reaggregate_v5_summaries
python -m scripts.make_composite_report \
  --out runs/direction_a_v5/composite_report.html \
  --csv-out runs/direction_a_v5/composite_cells.csv \
  --json-out runs/direction_a_v5/composite_cells.json
.venv/bin/python -m scripts.make_v5_plots
```

**Acceptance gates** (from the continuation runbook — must still hold):
- Llama steering now **varies** across α (was flat pre-fix); verify with the
  `steering_mode == add`, `alpha ∈ {−4,−8,−12}`, `layers == [14]` check.
- `olmo3_7b_base/jbb/neurons_top512`: `P ≈ 0.00`, `raw_hac ≈ 0.65`.
- `llama31_8b_control/jbb/ships_top8`: `Q ≈ 0.46`.
- New model appears as a full 22-cell block in `composite_cells.csv`.

---

## 8. Monitoring & resume

- Every stage is **resume-safe** — a killed job re-run skips finished work.
- Watch: `tail -f logs/*.log`; GPU: `watch -n5 nvidia-smi`.
- Cell coverage sanity check (per model, all judge layers):
  ```bash
  for d in runs/direction_a_v5/<model>/judge/*/*/seed0; do
    for f in summary.json coherence.jsonl judge_pathway.jsonl \
             judge_safety_reasoning_trace.jsonl; do
      [ -e "$d/$f" ] || echo "MISSING $d/$f"; done; done
  ```
