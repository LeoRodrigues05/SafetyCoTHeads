#!/usr/bin/env bash
# =============================================================================
# v6.1 correction + control run (B200 box). Everything is resume-safe: re-run
# with FROM=<stage> to continue. Nothing is deleted; superseded judge rows are
# moved to *.pruned_<tag>.jsonl beside their file.
#
# Stages (rough wall time on 2x B200 / 3x B200, from the Jul-26 logs).
# SCOPE=full (default) reprocesses EVERY cell (paper + defence + XSTest + R1) so
# the released results are uniform; SCOPE=paper restricts continuation and
# re-judging to the paper's cells.
#   preflight  tests + HF auth check                                 ~5 min
#   configs    random controls, steering_ablate_all, covert_prompt    ~1 min
#   gen        86 new cells: 72 random controls (3 safety-trained     ~2 h / ~1.3 h
#              arms x 2 ds x 4 families x 3 draws) + 10 ablate_all
#              + 4 covert_prompt
#   continue   finish truncated traces: OLMo-3-Think 2048->8192,      full: ~4-5 h / ~3 h
#              Qwen3-8B and R1 1024->4096 (~4,500 rows in full scope;  paper: ~1 h
#              ~250 in paper scope; repetition loops are skipped)
#   parse      v6.1 parser (pre-filled <think> fix) + change reports  ~10 min
#   invalidate prune judge rows whose inputs changed                  ~5 min
#   coherence  gate v6.1 (classifier + repetition) on all cells       ~10-20 min
#   answer     answer judge, new + changed rows (HF, July settings)   full: ~2.5 h / ~1.7 h
#   monitor    trace monitor (explicit) + prose-prefix monitor        full: ~1.5 h / ~1 h
#   evalpw     14B pathway judge kappa, full test set, pathway backend ~10-20 min
#   pathway    VALIDATED pathway protocol (14B, single-label), Qwen + ~3.5 h / ~2.5 h
#              OLMo-Think paper cells, vLLM (all v6 pathway labels so
#              far came from the zero-shot 30B)
#   sr         safety-reasoning judge on new/changed traces           ~45 min
#   aggregate  full gate (headline) + repetition-only (sensitivity)   ~15 min
#   report     reasoning table, exports, figures, v6 validation,      ~5 min
#              COVERT-CONTROL TAKEAWAY (last line of the run)
#   TOTAL      full scope ~16 h / ~11 h;  paper scope ~12 h / ~8 h
#
# Usage:
#   NGPU=2 bash scripts/orchestration/run_v61_queue.sh 2>&1 | tee -a runs/direction_a_v6/logs/v61_queue.out
#   FROM=answer NGPU=3 bash scripts/orchestration/run_v61_queue.sh ...
# Env: NGPU (2|3), FROM (stage), PY, PATHWAY_BACKEND (vllm|hf), PATHWAY_BATCH,
#      SCOPE      full (default) | paper
#      COVERT     1 (default) covert_prompt positive control for the monitorability
#                 axis (Qwen + OLMo-Think x 2 datasets); 0 to skip
#      RELATIVE=1 also generate relative-dose steering steering_rel_a{0.5,1,1.5,2}
#                 for the safety-trained arms (+24 cells; dose = d x ||r_14||)
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."
# scripts find src/ and each other via scripts/common/_bootstrap.py; src/ is
# also exported for the test run and any ad-hoc python -c.
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
PY="${PY:-.venv/bin/python}"
NGPU="${NGPU:-2}"
FROM="${FROM:-preflight}"
PATHWAY_BACKEND="${PATHWAY_BACKEND:-vllm}"
PATHWAY_BATCH="${PATHWAY_BATCH:-48}"
LOGDIR="runs/direction_a_v6/logs"; mkdir -p "$LOGDIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CFG=configs/experiments/direction_a_v5_iso_asr
EXPLICIT="qwen3_8b olmo3_7b_think"
PROSE="llama31_8b_control olmo3_7b_base olmo3_7b_base_own"
PRIMARY="$EXPLICIT $PROSE"
SCOPE="${SCOPE:-full}"
COVERT="${COVERT:-1}"
DEFENCE_CONDS="heads_amplify_top3 heads_amplify_top5 heads_amplify_top8 neurons_amplify_top256 \
neurons_amplify_top512 neurons_amplify_top1024 steering_defend_a0.5 steering_defend_a1.0 \
steering_defend_a1.5 defend_prompt"
if [[ "$SCOPE" == full ]]; then
  EXTRA_MODELS="r1_distill_qwen_7b"      # exploratory arm, reprocessed for a uniform release
  DATASETS_ALL="jbb bt xstest"
else
  EXTRA_MODELS=""; DATASETS_ALL="jbb bt"
fi
ALL_MODELS="$PRIMARY $EXTRA_MODELS"
EXPLICIT_ALL="$EXPLICIT $EXTRA_MODELS"
# conditions the paper reports + new controls. Excluded: defence Exp-5 cells and
# the a0.75/a1.25 dose-refinement cells (not reported; decided 2026-09-24).
PAPER_CONDS="baseline ships_top3 ships_top5 ships_top8 neurons_top256 neurons_top512 \
neurons_top1024 steering_a0.5 steering_a1.0 steering_a1.5 steering_ablate \
steering_ablate_all steering_rel_a0.5 steering_rel_a1.0 \
steering_rel_a1.5 steering_rel_a2.0 covert_prompt"
SR_CONDS="$PAPER_CONDS"; [[ "$SCOPE" == full ]] && SR_CONDS="$PAPER_CONDS $DEFENCE_CONDS"

# cells whose pathway labels are (re)judged with the validated 14B protocol
PW_CONDS="baseline ships_top3 ships_top5 ships_top8 neurons_top256 neurons_top512 \
neurons_top1024 steering_a0.5 steering_a1.0 steering_a1.5 steering_ablate steering_ablate_all \
covert_prompt"
[[ "${RELATIVE:-0}" == 1 ]] && PW_CONDS="$PW_CONDS steering_rel_a0.5 steering_rel_a1.0 steering_rel_a1.5 steering_rel_a2.0"

STAGES=(preflight configs gen continue parse invalidate coherence answer monitor evalpw pathway sr aggregate report)
declare -A ORD; i=0; for s in "${STAGES[@]}"; do ORD[$s]=$i; i=$((i+1)); done
run_stage(){ [[ ${ORD[$1]} -ge ${ORD[$FROM]} ]]; }
log(){ echo "[$(date -u +%H:%M:%S)] $*"; }
# vLLM needs the pip CUDA-13 toolkit + spawn workers on this B200 stack (same
# settings as the July vLLM runs). Applied ONLY to the vLLM stages, in a
# subshell, so every HF stage runs in exactly the July HF environment.
with_vllm_env(){ (
  export PYTHONUNBUFFERED=1 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0
  CU13ROOT="$($PY -c 'import os,nvidia;print(os.path.join(os.path.dirname(nvidia.__file__),"cu13"))' 2>/dev/null || true)"
  if [[ -n "$CU13ROOT" && -d "$CU13ROOT" ]]; then
    export CUDA_HOME="$CU13ROOT" PATH="$CU13ROOT/bin:$PATH" LD_LIBRARY_PATH="$CU13ROOT/lib:${LD_LIBRARY_PATH:-}"
  fi
  "$@"
) }
run_pw(){ if [[ "$PATHWAY_BACKEND" == "vllm" ]]; then with_vllm_env "$@"; else "$@"; fi; }

if run_stage preflight; then
  log "=== preflight ==="
  $PY -m pytest -q tests/test_v61_fixes.py tests/test_v6_parsing.py tests/test_v6_aggregate.py \
      tests/test_v6_paired_metrics.py tests/test_neuron_and_steer.py \
      tests/test_v6_immutability_and_integration.py || { log "TESTS FAILED - stop"; exit 1; }
  $PY -c "from huggingface_hub import whoami; print('HF user:', whoami()['name'])" \
    || log "WARNING: no HF auth -> gated Llama-3.1 cells will fail (huggingface-cli login)"
fi

if run_stage configs; then
  log "=== configs ==="
  $PY scripts/configs/make_control_configs.py --random --ablate-all --write | tail -1
  [[ "${RELATIVE:-0}" == 1 ]] && $PY scripts/configs/make_control_configs.py --relative --write | tail -1
  [[ "$COVERT" == 1 ]] && $PY scripts/configs/make_control_configs.py --covert --write | tail -1
fi

gen_model(){  # model gpu  -> all new-condition configs of that model, one model load
  local m="$1" g="$2"; shift 2
  local cfgs=()
  for ds in jbb bt; do
    extra=()
    [[ "${RELATIVE:-0}" == 1 ]] && extra+=($CFG/$m/gen/$ds/steering_rel_a*.yaml)
    [[ "$COVERT" == 1 ]] && extra+=($CFG/$m/gen/$ds/covert_prompt.yaml)
    for f in $CFG/$m/gen/$ds/rand_*.yaml $CFG/$m/gen/$ds/steering_ablate_all.yaml "${extra[@]}"; do
      [[ -f "$f" ]] && cfgs+=("$f")
    done
  done
  [[ ${#cfgs[@]} -eq 0 ]] && return 0
  CUDA_VISIBLE_DEVICES=$g $PY scripts/generation/run_generation_batch.py --configs "${cfgs[@]}" \
      > "$LOGDIR/v61_gen_${m}_$STAMP.log" 2>&1
  log "  gen $m (gpu $g) exit=$?"
}
if run_stage gen; then
  log "=== gen (new cells) on $NGPU GPUs ==="
  if [[ "$NGPU" -ge 3 ]]; then
    gen_model olmo3_7b_think 0 &
    gen_model qwen3_8b 1 &
    { gen_model llama31_8b_control 2; gen_model olmo3_7b_base 2; gen_model olmo3_7b_base_own 2; } &
  else
    gen_model olmo3_7b_think 0 &
    { gen_model qwen3_8b 1; gen_model llama31_8b_control 1; gen_model olmo3_7b_base 1; \
      gen_model olmo3_7b_base_own 1; } &
  fi
  wait
fi

if run_stage continue; then
  log "=== continue truncated traces (scope=$SCOPE) ==="
  RAND="$(ls $CFG/olmo3_7b_think/gen/jbb | sed -n 's/^\(rand_.*\)\.yaml$/\1/p')"
  if [[ "$SCOPE" == full ]]; then
    # every cell except the unreported a0.75/a1.25 dose cells
    SEL=(--datasets $DATASETS_ALL --exclude-prefix steering_a0.75 steering_a1.25)
  else
    SEL=(--datasets jbb bt --conditions $PAPER_CONDS $RAND)
  fi
  cont(){ local g="$1" tag="$2"; shift 2
    CUDA_VISIBLE_DEVICES=$g $PY scripts/generation/continue_truncated_generations.py "$@" "${SEL[@]}" \
        > "$LOGDIR/v61_cont_${tag}_$STAMP.log" 2>&1; }
  # batch 24 for 8k-token OLMo traces, 32 for the 4k-token Qwen / R1 traces
  if [[ "$NGPU" -ge 3 ]]; then
    cont 0 olmo0 --models olmo3_7b_think --shard 0/2 --batch-size 24 &
    cont 1 olmo1 --models olmo3_7b_think --shard 1/2 --batch-size 24 &
    { [[ -n "$EXTRA_MODELS" ]] && cont 2 r1 --models r1_distill_qwen_7b --batch-size 32;
      cont 2 qwen --models qwen3_8b --batch-size 32; } &
  else
    { cont 0 olmo0 --models olmo3_7b_think --shard 0/2 --batch-size 24;
      [[ -n "$EXTRA_MODELS" ]] && cont 0 r1a --models r1_distill_qwen_7b --shard 0/2 --batch-size 32; } &
    { cont 1 olmo1 --models olmo3_7b_think --shard 1/2 --batch-size 24;
      [[ -n "$EXTRA_MODELS" ]] && cont 1 r1b --models r1_distill_qwen_7b --shard 1/2 --batch-size 32;
      cont 1 qwen --models qwen3_8b --batch-size 32; } &
  fi
  wait
  grep -h "^\[continue\]" "$LOGDIR"/v61_cont_*_"$STAMP".log
fi

if run_stage parse; then
  log "=== parse (v6.1) ==="
  # parse, invalidate, gate and re-judge always cover the same models x datasets
  $PY scripts/preprocessing/parse_v6_completions.py --models $ALL_MODELS --datasets $DATASETS_ALL | tail -4
fi
if run_stage invalidate; then
  log "=== invalidate changed rows ==="
  $PY scripts/preprocessing/invalidate_v6_rows.py --models $ALL_MODELS --datasets $DATASETS_ALL --apply | tail -3
fi
if run_stage coherence; then
  log "=== coherence gate v6.1 (all cells) ==="
  CUDA_VISIBLE_DEVICES=0 $PY scripts/judging/run_v6_judge_shard.py --stage coherence --gpu 0 --n-shards 1 \
      --models $ALL_MODELS --datasets $DATASETS_ALL > "$LOGDIR/v61_coherence_$STAMP.log" 2>&1
  log "  coherence exit=$? ($(tail -1 $LOGDIR/v61_coherence_$STAMP.log))"
fi
if run_stage answer; then
  log "=== answer judge (new + changed rows) ==="
  # batch 96 / 384 tokens = the July settings that produced the existing answer labels
  $PY scripts/judging/run_v6_dual_gpu.py --stage answer --gpus "$NGPU" --backend hf --batch-size 96 \
      --max-new-tokens 384 --models $ALL_MODELS --datasets $DATASETS_ALL > "$LOGDIR/v61_answer_$STAMP.log" 2>&1
  log "  answer exit=$?"
fi
if run_stage monitor; then
  log "=== monitor (explicit + prose prefix) ==="
  # batch 96 / 256 tokens = the July monitor settings (explicit and prefix)
  $PY scripts/judging/run_v6_dual_gpu.py --stage monitor --gpus "$NGPU" --backend hf --batch-size 96 \
      --max-new-tokens 256 --models $EXPLICIT_ALL --datasets $DATASETS_ALL > "$LOGDIR/v61_monitor_$STAMP.log" 2>&1
  $PY scripts/judging/run_v6_dual_gpu.py --stage monitor --prose-prefix --gpus "$NGPU" --backend hf \
      --batch-size 96 --max-new-tokens 256 --models $PROSE --datasets $DATASETS_ALL \
      > "$LOGDIR/v61_monitor_prefix_$STAMP.log" 2>&1
  log "  monitor exit=$?"
fi
if run_stage evalpw; then
  log "=== re-validate 14B pathway judge under $PATHWAY_BACKEND ==="
  CUDA_VISIBLE_DEVICES=0 run_pw $PY scripts/training/eval_pathway_judge.py --test-data data/pathway_judge_test.jsonl \
      --finetuned-model models/pathway_judge_14b_merged --backend "$PATHWAY_BACKEND" \
      --out "runs/direction_a_v6/validation/pathway_judge_eval_${PATHWAY_BACKEND}.json" \
      > "$LOGDIR/v61_evalpw_$STAMP.log" 2>&1
  grep -a "Overall fine-tuned" "$LOGDIR/v61_evalpw_$STAMP.log" || log "  WARNING: eval failed - check log"
  log "  (this is the FULL 21,240-row HarmThoughts test set; the paper's .963 was HF on the"
  log "   first 180 rows only, covering 9/12 labels. Review before the pathway stage; if vLLM"
  log "   looks materially worse, re-run evalpw with PATHWAY_BACKEND=hf to compare)"
fi
if run_stage pathway; then
  log "=== pathway: validated 14B single-label protocol ($PATHWAY_BACKEND) ==="
  # paper cells only; traces with no final answer are skipped (--answered-only),
  # so the pathway population is "completions that produced an answer"
  run_pw $PY scripts/judging/run_v6_dual_gpu.py --stage pathway14b --gpus "$NGPU" --backend "$PATHWAY_BACKEND" \
      --batch-size "$PATHWAY_BATCH" --max-new-tokens 96 --max-model-len 16384 \
      --models $EXPLICIT --datasets jbb bt --conditions $PW_CONDS --answered-only \
      > "$LOGDIR/v61_pathway14b_$STAMP.log" 2>&1
  log "  pathway14b exit=$?"
fi
if run_stage sr; then
  log "=== safety-reasoning (new/changed traces) ==="
  # paper cells only (controls need P/Q/S, not SR); only new/changed rows are
  # judged. Same population as the v6 run (every explicit trace), so the new
  # rows are consistent with the existing ones.
  # Output cap 2048 (July: 1024). Continued OLMo traces are up to 4x longer, so
  # the span list can exceed 1024 tokens. Greedy decoding makes this consistent
  # with the July rows: a larger cap never changes an output that already
  # finished under the smaller one, and all 7,404 July rows finished (0 failed).
  $PY scripts/judging/run_v6_dual_gpu.py --stage safety-reasoning --gpus "$NGPU" --backend hf \
      --batch-size 48 --max-new-tokens 2048 --models $EXPLICIT --datasets jbb bt \
      --conditions $SR_CONDS \
      > "$LOGDIR/v61_sr_$STAMP.log" 2>&1
  log "  sr exit=$?"
fi
if run_stage aggregate; then
  log "=== aggregate ==="
  $PY scripts/analysis/aggregate_v6_metrics.py --answer-source v6 --monitor-source v6 --datasets jbb bt \
      --models $ALL_MODELS --coherence-gate full --n-boot 10000 > "$LOGDIR/v61_aggregate_$STAMP.log" 2>&1
  log "  headline (full gate) exit=$?"
  $PY scripts/analysis/aggregate_v6_metrics.py --answer-source v6 --monitor-source v6 --datasets jbb bt \
      --models $ALL_MODELS --coherence-gate repetition-only --n-boot 2000 \
      --reports-dir runs/direction_a_v6/reports_gate_repetition_only \
      >> "$LOGDIR/v61_aggregate_$STAMP.log" 2>&1
  log "  sensitivity (repetition-only gate) exit=$?"
fi
if run_stage report; then
  log "=== report ==="
  # legacy 30B pathway files outside the rerun scope are renamed, not mixed in
  $PY scripts/preprocessing/quarantine_legacy_pathway.py --models $EXPLICIT --datasets jbb bt \
      --keep-conditions $PW_CONDS --apply | tail -2
  $PY scripts/analysis/aggregate_v6_reasoning.py --models $EXPLICIT --datasets jbb bt \
      --conditions $PW_CONDS | tail -2
  $PY scripts/analysis/export_v6_composite_cells.py
  # every paper figure is regenerated into figures/paper/ from the new reports
  $PY scripts/plotting/make_composite_insight_reports.py >/dev/null 2>&1 || log "  composite figures failed"
  $PY scripts/plotting/make_narrative_figs.py >/dev/null 2>&1 || log "  narrative figures failed"
  $PY scripts/plotting/plot_pathway_radar.py --model olmo3_7b_think >/dev/null 2>&1 || log "  pathway radar failed"
  $PY scripts/plotting/plot_validation.py --batch data/annotations/batch_v5_002 >/dev/null 2>&1 || log "  validation plots failed"
  # human validation restricted to items whose shown text == the v6 judge input
  $PY scripts/validation/validate_v6_matched.py --out runs/direction_a_v6/validation | head -14
  $PY scripts/analysis/verify_paper_numbers.py | tail -15
  if [[ "$COVERT" == 1 ]]; then
    $PY scripts/analysis/report_covert_control.py > "$LOGDIR/v61_covert_$STAMP.log" 2>&1
  fi
fi
log "=== v6.1 queue done ==="
# the covert-control takeaway is the last thing printed (full table:
# runs/direction_a_v6/reports/covert_control.md)
[[ "$COVERT" == 1 ]] && grep -h "^TAKEAWAY" "$LOGDIR"/v61_covert_*.log 2>/dev/null | tail -1
