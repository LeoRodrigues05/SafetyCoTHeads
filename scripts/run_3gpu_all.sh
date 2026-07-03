#!/usr/bin/env bash
# =============================================================================
# run_3gpu_all.sh — finish the Direction A v5 grid on a 3×GPU node.
#
# Distributes ALL remaining work discussed in docs/general/RUN_ON_NEW_MACHINE.md
# across 3 GPUs via a resume-safe work-pool: 3 workers (one pinned per card) pull
# ready tasks from a dependency-gated DAG.
#
#   discovery/extraction ─▶ per-cell generation ─▶ per-layer judging ─▶ finalise
#
# Everything is resume-safe: "done" is judged from on-disk artifacts, so a killed
# or re-run invocation skips finished work and only fills what's missing.
#
# PREREQS (run once, BEFORE this script — see the block the assistant printed):
#   - source .venv/bin/activate
#   - hf auth login --token hf_XXXX      (gated Llama-3.1 needs it)
#   - weights pre-warmed (Qwen3-30B judge, Llama-3.1-8B, R1-Distill-Qwen-7B)
#
# Usage:
#   bash scripts/run_3gpu_all.sh                # run the whole DAG on GPUs 0,1,2
#   NGPU=2 bash scripts/run_3gpu_all.sh         # use fewer cards
#   bash scripts/run_3gpu_all.sh --plan         # print the task DAG and exit
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# ---- config -----------------------------------------------------------------
NGPU="${NGPU:-3}"
CFG="configs/experiments/direction_a_v5_iso_asr"
RUN="runs/direction_a_v5"
STATE="$RUN/_orch_state"            # persistent done-markers (discovery/extract)
CLAIM="$(mktemp -d)"               # per-run ephemeral claim/attempt dir
LOGS="logs"; mkdir -p "$LOGS" "$STATE"
DATASETS=(jbb bt)
MAX_ATTEMPTS=2                      # per-run retries before giving up on a task

export PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=warning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Call the venv interpreter by ABSOLUTE PATH — never rely on PATH/activation, which
# a tmux/login shell can silently strip (VIRTUAL_ENV set but .venv/bin off PATH).
PY="$ROOT/.venv/bin/python"
[[ -x "$PY" ]] || PY="python"
# Also (re)activate so child scripts that rely on PATH (run_local_pipeline.sh) are covered.
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate 2>/dev/null || true; fi

# ---- model key <-> short name ----------------------------------------------
mkey() { case "$1" in
  r1) echo r1_distill_qwen_7b ;; llama) echo llama31_8b_control ;;
  qwen) echo qwen3_8b ;; olmothink) echo olmo3_7b_think ;;
  *) echo "$1" ;; esac; }

san() { echo "$1" | tr ':.' '__'; }          # id -> filesystem-safe token

# ---- disk-truth predicates (what counts as "done") --------------------------
# Exact per-cell checks (never counts, so archived/_stale dirs can't inflate).
gens_ready() { # <model> <dset> : every EXPECTED gen cell has a completions file
  local m=$1 d=$2 mk cond any=0; mk=$(mkey $m)
  for gcfg in "$CFG/$mk/gen/$d"/*.yaml; do
    any=1; cond=$(basename "$gcfg" .yaml)
    ls "$RUN/$mk/gen/$d/$cond/seed0/completions_"*.jsonl >/dev/null 2>&1 || return 1
  done; [[ $any -eq 1 ]]
}
layer_done() { # <model> <dset> <sentinel-glob> : every expected cell has the judge sentinel
  local m=$1 d=$2 pat=$3 mk cond any=0; mk=$(mkey $m)
  for gcfg in "$CFG/$mk/gen/$d"/*.yaml; do
    any=1; cond=$(basename "$gcfg" .yaml)
    ls "$RUN/$mk/judge/$d/$cond/seed0/"$pat >/dev/null 2>&1 || return 1
  done; [[ $any -eq 1 ]]
}

is_done() {  # id -> 0 if the task's artifacts already exist
  local id="$1"; IFS=':' read -r kind a b c <<<"$id"
  case "$kind" in
    disc) [[ -f "$STATE/disc_$a.done" ]] ;;
    gen)  ls "$RUN/$(mkey $a)/gen/$b/$c/seed0/completions_"*.jsonl >/dev/null 2>&1 ;;
    jstd) layer_done "$a" "$b" "summary.json" ;;
    jpath) layer_done "$a" "$b" "judge_pathway.jsonl" ;;
    sr)   layer_done "$a" jbb "judge_safety_reasoning_trace.jsonl" \
          && layer_done "$a" bt "judge_safety_reasoning_trace.jsonl" ;;
  esac
}

is_ready() {  # id -> 0 if deps satisfied
  local id="$1"; IFS=':' read -r kind a b c <<<"$id"
  case "$kind" in
    disc) return 0 ;;
    gen)  case "$a" in r1) [[ -f "$STATE/disc_r1.done" ]] ;; llama) [[ -f "$STATE/disc_llama.done" ]] ;; *) return 0 ;; esac ;;
    jstd|jpath) gens_ready "$a" "$b" ;;
    sr)   gens_ready "$a" jbb && gens_ready "$a" bt ;;
  esac
}

# ---- task runners -----------------------------------------------------------
# Standard 30B / pathway 14B judge for one (model,dataset). Mirrors run_local_pipeline do_judge.
judge_one() { # <layer:std|pathway> <model> <dset> <gpu>
  local layer=$1 m=$2 d=$3 gpu=$4 mk cfg outb; mk=$(mkey $m)
  local -a flags cond overrides env
  outb="$RUN/$mk/judge/$d"
  for gcfg in "$CFG/$mk/gen/$d"/*.yaml; do
    local cond_name comp; cond_name="$(basename "$gcfg" .yaml)"
    comp="$(ls "$RUN/$mk/gen/$d/$cond_name/seed0/"completions_*.jsonl 2>/dev/null | head -1)"
    [[ -n "$comp" && -f "$comp" ]] && cond+=(--condition "tag=$cond_name,cond=$cond_name,completions=$comp")
  done
  [[ ${#cond[@]} -eq 0 ]] && { echo "no completions for $mk/$d"; return 1; }
  if [[ "$layer" == pathway ]]; then
    cfg="judge_14b.yaml"; flags=(--skip-safety --skip-coherence --skip-cot-only)
    overrides=(--overrides model.load_in_4bit=false batch_size=128); env=(HF_HUB_OFFLINE=1)
  else
    cfg="judge.yaml"; flags=(--skip-pathway); overrides=(--overrides model.load_in_4bit=false); env=()
  fi
  env CUDA_VISIBLE_DEVICES=$gpu "${env[@]}" "$PY" -m scripts.run_v4_jbb_judge \
    --config "$CFG/$mk/$cfg" --out-base "$outb" --seed 0 \
    "${flags[@]}" "${cond[@]}" "${overrides[@]}"
}

run_task() { # <id> <gpu>
  local id="$1" gpu="$2"; IFS=':' read -r kind a b c <<<"$id"
  case "$kind" in
    disc)
      if [[ "$a" == r1 ]]; then
        CUDA_VISIBLE_DEVICES=$gpu bash scripts/run_local_pipeline.sh r1_distill_qwen_7b discover \
          && touch "$STATE/disc_r1.done"
      else
        CUDA_VISIBLE_DEVICES=$gpu "$PY" -m scripts.run_direction_extraction \
          --config "$CFG/llama31_8b_control/17-direction-extraction.yaml" \
          && touch "$STATE/disc_llama.done"
      fi ;;
    gen)
      local gcfg="$CFG/$(mkey $a)/gen/$b/$c.yaml"
      CUDA_VISIBLE_DEVICES=$gpu "$PY" -m scripts.run_generation --config "$gcfg" ;;
    jstd)  judge_one std     "$a" "$b" "$gpu" ;;
    jpath) judge_one pathway "$a" "$b" "$gpu" ;;
    sr)    CUDA_VISIBLE_DEVICES=$gpu "$PY" -m scripts.run_v5_safety_reasoning \
             --models "$(mkey $a)" --datasets jbb bt --backend vllm ;;
  esac
}

# ---- build the task list (topological hint: prereqs first) ------------------
TASKS=()
TASKS+=(disc:r1 disc:llama)
for d in "${DATASETS[@]}"; do
  for cond in $(ls "$CFG/r1_distill_qwen_7b/gen/$d"/*.yaml | xargs -n1 basename | sed 's/.yaml//'); do
    TASKS+=("gen:r1:$d:$cond"); done
  for cond in $(ls "$CFG/llama31_8b_control/gen/$d"/*.yaml | xargs -n1 basename | sed 's/.yaml//'); do
    TASKS+=("gen:llama:$d:$cond"); done
done
# pathway fills (token-free, ready immediately): qwen 8 + olmo-think 2
for d in "${DATASETS[@]}"; do TASKS+=("jpath:qwen:$d" "jpath:olmothink:$d"); done
# llama + r1 judging (after their gen)
for d in "${DATASETS[@]}"; do
  TASKS+=("jstd:llama:$d" "jpath:llama:$d" "jstd:r1:$d" "jpath:r1:$d")
done
TASKS+=("sr:llama" "sr:r1")

if [[ "${1:-}" == "--plan" ]]; then
  printf 'GPUs: %s   Tasks: %s\n' "$NGPU" "${#TASKS[@]}"
  for t in "${TASKS[@]}"; do
    printf '  %-28s done=%s ready=%s\n' "$t" "$(is_done "$t" && echo Y || echo .)" "$(is_ready "$t" && echo Y || echo .)"
  done; exit 0
fi

# ---- GPU worker pool --------------------------------------------------------
inflight() { local n=0; for t in "${TASKS[@]}"; do
    local s; s=$(san "$t"); [[ -d "$CLAIM/$s.run" && ! -f "$CLAIM/$s.ok" ]] && n=$((n+1)); done; echo $n; }

worker() {  # $1 = gpu id
  local gpu="$1"
  while true; do
    local picked="" t s
    for t in "${TASKS[@]}"; do
      s=$(san "$t")
      is_done "$t" && continue
      [[ -f "$CLAIM/$s.giveup" ]] && continue
      is_ready "$t" || continue
      mkdir "$CLAIM/$s.run" 2>/dev/null || continue     # atomic claim
      picked="$t"; break
    done
    if [[ -z "$picked" ]]; then
      [[ "$(inflight)" -eq 0 ]] && break                # drained (or stalled) -> exit
      sleep 10; continue
    fi
    s=$(san "$picked")
    local att; att=$(( $(cat "$CLAIM/$s.att" 2>/dev/null || echo 0) + 1 )); echo "$att" > "$CLAIM/$s.att"
    echo "[gpu$gpu] ($att) START $picked  -> $LOGS/$s.log"
    if run_task "$picked" "$gpu" >>"$LOGS/$s.log" 2>&1; then
      touch "$CLAIM/$s.ok"; echo "[gpu$gpu] OK    $picked"
    else
      echo "[gpu$gpu] FAIL  $picked (attempt $att) — see $LOGS/$s.log"
      [[ "$att" -ge "$MAX_ATTEMPTS" ]] && touch "$CLAIM/$s.giveup"
    fi
    rmdir "$CLAIM/$s.run" 2>/dev/null || rm -rf "$CLAIM/$s.run"
  done
}

echo "=== 3-GPU orchestrator: $NGPU cards, ${#TASKS[@]} tasks, logs in $LOGS/ ==="
pids=()
for ((g=0; g<NGPU; g++)); do worker "$g" & pids+=($!); done
for p in "${pids[@]}"; do wait "$p"; done

# ---- post-run gate + finalise (CPU) ----------------------------------------
echo; echo "=== steering dose gate (llama must vary across alpha) ==="
"$PY" - <<'PY' || true
import json,glob
for a in ["0.5","1.0","1.5"]:
    g=glob.glob(f"runs/direction_a_v5/llama31_8b_control/gen/jbb/steering_a{a}/seed0/completions_*.jsonl")
    if not g: print(f"a{a}: (no completions)"); continue
    m=json.loads(open(g[0]).readline()).get("meta",{})
    print(f"a{a}: mode={m.get('steering_mode')} alpha={m.get('steering_alpha')} layers={m.get('steering_layers')}")
print("EXPECT: mode=add, alpha in {-4,-8,-12}, layers=[14]")
PY

echo; echo "=== finalise: reaggregate -> composite report -> plots ==="
"$PY" -m scripts.reaggregate_v5_summaries
"$PY" -m scripts.make_composite_report \
  --out "$RUN/composite_report.html" \
  --csv-out "$RUN/composite_cells.csv" \
  --json-out "$RUN/composite_cells.json"
"$PY" -m scripts.make_v5_plots || echo "  (plots step failed — reports are intact)"

echo; echo "=== remaining-gap audit ==="
"$PY" - <<'PY'
import os,glob
base="runs/direction_a_v5"
layers={"summary":"summary.json","coherence":"coherence.jsonl","judged5":"judged_*.jsonl",
        "cot_only":"judge_cot_only.jsonl","monitor":"monitorability_rows.jsonl",
        "pathway":"judge_pathway.jsonl","sr_trace":"judge_safety_reasoning_trace.jsonl"}
gaps=0
for m in sorted(os.listdir(base)):
    md=f"{base}/{m}"
    if not os.path.isdir(md) or m.startswith("_"): continue
    for c in sorted(glob.glob(f"{md}/judge/*/*/seed0")):
        miss=[k for k,p in layers.items() if not glob.glob(f"{c}/{p}")]
        if miss: gaps+=1; print("GAP", m, c.split('/judge/')[1], "->", ",".join(miss))
print("clean" if not gaps else f"{gaps} cells still incomplete — re-run this script to fill them")
PY
echo "=== done ==="
