"""Optional vLLM backend for the LLM-as-judge runner.

The HF ``generate`` path uses *static* batching: a batch runs until its longest
member finishes, so a single long safety-reasoning trace paces the whole batch,
and a 3B-active MoE like Qwen3-30B-A3B is badly under-utilised. vLLM uses
*continuous* batching (paged KV cache + per-step scheduling): finished sequences
are evicted and new ones slotted in immediately, so throughput is bound by total
work, not by the slowest row. For the long-output safety-reasoning pass this is
typically a 5-15x speedup.

This module is import-guarded: it is only imported when ``--backend vllm`` is
requested, so installing vLLM is not required for the default HF path. The
:class:`VLLMJudge` exposes just the surface :func:`judge_rows` needs:
``name``, ``tokenizer``, ``is_vllm`` and :meth:`generate`.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

# Set before vLLM is imported anywhere in-process. On this Blackwell + CUDA-13
# wheel stack vLLM's flashinfer sampler triggers a JIT compile against a
# version-skewed toolchain; the native sampler is used instead. spawn is
# required because the parent process initialises CUDA (seeding) before vLLM
# forks its engine-core worker. (The launcher scripts/run_sr_vllm.sh also exports
# these plus the CUDA-13 lib/nvcc paths, which must be set before process start.)
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


class VLLMJudge:
    """Thin wrapper over a vLLM ``LLM`` engine, drop-in for ``judge_rows``."""

    is_vllm = True

    def __init__(self, model_name: str, *, dtype: str = "bfloat16",
                 max_model_len: int = 8192, gpu_memory_utilization: float = 0.90,
                 tensor_parallel_size: int = 1, trust_remote_code: bool = False,
                 seed: int = 0):
        # Imported lazily so the module is harmless to import without vLLM.
        from vllm import LLM
        from transformers import AutoTokenizer

        self.name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code)
        self._llm = LLM(
            model=model_name,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            seed=seed,
            enforce_eager=False,
        )

    def _format(self, prompt: str, use_chat_template: bool) -> str:
        tok = self.tokenizer
        if not use_chat_template or getattr(tok, "chat_template", None) is None:
            return prompt
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )

    def generate(self, prompts: Sequence[str], max_new_tokens: int,
                 temperature: float, use_chat_template: bool = True) -> list[str]:
        """Greedy (``temperature==0``) or sampled batched generation.

        Returns one decoded continuation per prompt, in input order. vLLM
        schedules all prompts concurrently regardless of list length, so callers
        should pass the entire workload in one call.
        """
        from vllm import SamplingParams

        texts = [self._format(p, use_chat_template) for p in prompts]
        sp = SamplingParams(
            temperature=float(temperature),
            top_p=1.0,
            max_tokens=int(max_new_tokens),
        )
        # vLLM preserves request order in the returned list.
        outs = self._llm.generate(texts, sp, use_tqdm=True)
        return [o.outputs[0].text for o in outs]


def load_vllm_judge(model_name: str, *, dtype: str = "bfloat16",
                    max_model_len: int = 8192,
                    gpu_memory_utilization: float = 0.90,
                    tensor_parallel_size: int = 1,
                    trust_remote_code: bool = False,
                    seed: int = 0) -> VLLMJudge:
    return VLLMJudge(
        model_name, dtype=dtype, max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=trust_remote_code, seed=seed,
    )
