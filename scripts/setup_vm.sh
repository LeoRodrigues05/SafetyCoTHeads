#!/usr/bin/env bash
# =============================================================================
# Direction A v5 — one-shot VM setup for OLMo generation + judging.
#
# Brings a STOCK GPU instance (built/validated for B200 / Blackwell, sm_100)
# from nothing to "ready to generate and judge":
#   1. Python virtualenv
#   2. Blackwell-compatible PyTorch (CUDA 12.8 wheels)
#   3. project package + dependencies
#   4. GPU / torch sanity check
#   5. (optional) Hugging Face login
#   6. model-verification smoke: actually loads BOTH OLMo policy models and the
#      Qwen3-30B-A3B judge and runs a micro gen->judge end-to-end.
#
# OLMo-2 (allenai) and the Qwen3 judge are OPEN models, so HF_TOKEN is optional.
#
# Usage (from repo root):
#   bash scripts/setup_vm.sh
#
# Optional environment:
#   ENV_DIR=.venv               # virtualenv location
#   PYTORCH_CUDA=cu128          # torch CUDA wheel channel (B200 needs >= cu128)
#   HF_TOKEN=hf_xxx             # only for gated models (not needed for OLMo/Qwen)
#   SKIP_SMOKE=1               # skip step 6 (no weight downloads / no GPU run)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"

ENV_DIR="${ENV_DIR:-.venv}"
PYTORCH_CUDA="${PYTORCH_CUDA:-cu128}"

echo "=== [1/6] Python virtualenv ($ENV_DIR) ==="
if [[ ! -d "$ENV_DIR" ]]; then
    python3 -m venv "$ENV_DIR"
fi
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

echo "=== [2/6] PyTorch ($PYTORCH_CUDA — Blackwell/B200 ready) ==="
# B200 (sm_100) needs a CUDA 12.8+ build; torch >= 2.7 ships sm_100 kernels.
# Installed FIRST so the torch>=2.2 pin in requirements.txt is already satisfied
# and pip will not pull a default-index (possibly non-Blackwell) wheel.
pip install --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA}" torch

echo "=== [3/6] Project package + dependencies ==="
pip install -r requirements.txt
pip install -e .

echo "=== [4/6] GPU / torch sanity ==="
python - <<'PY'
import torch
print("torch:", torch.__version__, "| built for CUDA:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    print(f"device: {name} | compute capability: sm_{cc[0]}{cc[1]}")
    if cc[0] < 9:
        print("WARNING: GPU older than Hopper — expect much slower generation.")
else:
    raise SystemExit("ERROR: no CUDA device visible; check the driver / VM image.")

import transformers
print("transformers:", transformers.__version__)
try:
    from transformers import Olmo3ForCausalLM  # noqa: F401
    print("olmo3 arch: supported")
except Exception:
    raise SystemExit(
        "ERROR: this transformers lacks the native `olmo3` architecture "
        "(need >= 4.57). Run: pip install -U 'transformers>=4.57,<4.60'")
PY

echo "=== [5/6] Hugging Face cache + (optional) login ==="
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"
if [[ -n "${HF_TOKEN:-}" ]]; then
    python -c "from huggingface_hub import login; login('${HF_TOKEN}')" || \
        echo "  (hf login failed; OLMo + Qwen judge are open so continuing)"
else
    echo "  HF_TOKEN not set — fine, OLMo-3 and the Qwen3 judge are open models."
fi

if [[ "${SKIP_SMOKE:-0}" == "1" ]]; then
    echo "=== [6/6] smoke SKIPPED (SKIP_SMOKE=1) ==="
    echo "setup complete (no models verified)."
    exit 0
fi

echo "=== [6/6] model-verification smoke (first run downloads ~90 GB of weights) ==="
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=warning
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SMOKE="runs/direction_a_v5/_smoke_vm"
JUDGE_CFG="configs/experiments/direction_a_v5_iso_asr/olmo3_7b_think/judge.yaml"

# Think first (exercises the <think> path + olmo3 arch), then Base (exercises
# the no-chat-template SHIPS fallback). 48 tokens is enough to confirm decode.
for mk in olmo3_7b_think olmo3_7b_base; do
    echo "--- generation smoke: $mk (n=2, 48 tokens) ---"
    python -m scripts.run_generation \
        --config "configs/experiments/direction_a_v5_iso_asr/${mk}/gen/jbb/baseline.yaml" \
        --overrides dataset.n=2 batch_size=2 decoding.max_new_tokens=48 \
            output.dir="${SMOKE}/${mk}/baseline/seed0"
done

echo "--- judge smoke: Qwen3-30B-A3B in bf16 (no 4-bit on Blackwell) ---"
python -m scripts.run_v4_jbb_judge \
    --config "${JUDGE_CFG}" \
    --out-base "${SMOKE}/judge" \
    --seed 0 \
    --condition "tag=olmo3_think,cond=baseline,completions=${SMOKE}/olmo3_7b_think/baseline/seed0/completions_baseline.jsonl" \
    --condition "tag=olmo3_base,cond=baseline,completions=${SMOKE}/olmo3_7b_base/baseline/seed0/completions_baseline.jsonl" \
    --skip-pathway \
    --overrides model.load_in_4bit=false

echo
echo "=============================================================="
echo " SETUP COMPLETE — OLMo-3 gen + Qwen3-30B judge verified on this VM."
echo " Next:  bash scripts/run_local_pipeline.sh olmo3_7b_think all"
echo " (see scripts/run_local_pipeline.sh for staged / parallel runs)"
echo "=============================================================="
