#!/usr/bin/env bash
# =============================================================================
# Safety-reasoning judge via the vLLM backend.
#
# Wraps scripts/run_v5_safety_reasoning.py --backend vllm with the environment
# the vLLM 0.23 + CUDA-13 wheel stack needs on this Blackwell (B200) box:
#
#   * CUDA_HOME / PATH / LD_LIBRARY_PATH -> the pip-installed CUDA-13 toolkit
#     (nvidia/cu13), so libcudart.so.13 and nvcc resolve.
#   * VLLM_WORKER_MULTIPROC_METHOD=spawn -> the parent inits CUDA (seeding)
#     before vLLM forks its engine-core worker; fork+CUDA is illegal.
#   * VLLM_USE_FLASHINFER_SAMPLER=0 -> avoid flashinfer's JIT compile against the
#     version-skewed toolchain (flashinfer-python is uninstalled; vLLM falls back
#     to its native sampler + attention kernels).
#
# One-time setup already applied to .venv: `pip install vllm`,
# `pip install --force-reinstall --no-deps torchvision==0.26.0+cu128
#  torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128`,
# and `pip uninstall flashinfer-python flashinfer-cubin`. If vLLM is ever
# reinstalled, re-run the flashinfer uninstall.
#
# Usage (args are forwarded to the python driver):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_sr_vllm.sh --num-shards 2 --shard-id 0
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0

CU13ROOT=$(python -c 'import os,nvidia;print(os.path.join(os.path.dirname(nvidia.__file__),"cu13"))')
export CUDA_HOME="$CU13ROOT"
export PATH="$CU13ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CU13ROOT/lib:${LD_LIBRARY_PATH:-}"

exec python scripts/run_v5_safety_reasoning.py --backend vllm "$@"
