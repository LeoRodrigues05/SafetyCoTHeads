"""Batched generation under an (optional) head-mask config.

Adapted from ``SafetyHeadAttribution/lib/utils/batch_inference.py`` and
``CoT-safety/beaver_experiments2.py`` — the per-prompt loop body is the same
``tokenizer(... return_tensors='pt') → model.generate(...) → batch_decode``
pattern; the only difference is that we (a) optionally activate a
:class:`HeadMaskController` mask_cfg around the ``generate`` call, and (b)
write strict JSONL rows with full metadata for downstream judging.
"""

from __future__ import annotations
from dataclasses import asdict
from typing import Iterable, Mapping, Optional, Sequence

import torch
from tqdm import tqdm

from ..models import LoadedModel
from ..utils import now_iso, set_seed
from .decoding import DecodingConfig
from .prompts import render_chat


def _batched(items: Sequence, batch_size: int) -> Iterable[Sequence]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


@torch.no_grad()
def generate(lm: LoadedModel,
             prompts: Sequence[dict],
             decoding: Optional[DecodingConfig] = None,
             *,
             mask_cfg: Optional[Mapping] = None,
             system_prompt: Optional[str] = None,
             batch_size: int = 4,
             condition_label: str = "baseline",
             extra_meta: Optional[dict] = None) -> list[dict]:
    """Run generation for a list of ``{"id", "prompt", ...}`` rows.

    Returns one row per input with the generated completion, the rendered
    prompt, the decoding kwargs, and a ``condition`` label so the same JSONL
    can hold every experiment in :doc:`ExperimentTracker.md`.
    """
    decoding = decoding or DecodingConfig()
    set_seed(decoding.seed)
    tok = lm.tokenizer
    rows: list[dict] = []
    meta_extra = extra_meta or {}

    with lm.head_mask_controller.active(mask_cfg):
        for batch in tqdm(list(_batched(prompts, batch_size)),
                          desc=f"generate[{condition_label}]"):
            rendered = [
                render_chat(tok, ex["prompt"], system=system_prompt)
                for ex in batch
            ]
            enc = tok(rendered, return_tensors="pt", padding=True,
                      truncation=True).to(lm.device)
            out = lm.model.generate(
                **enc, pad_token_id=tok.pad_token_id,
                **decoding.to_hf_kwargs(),
            )
            # strip the prompt prefix
            gen = out[:, enc["input_ids"].shape[1]:]
            texts = tok.batch_decode(gen, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)
            for ex, prompt_text, completion in zip(batch, rendered, texts):
                rows.append({
                    "id": ex["id"],
                    "dataset": ex.get("dataset"),
                    "category": ex.get("category"),
                    "prompt": ex["prompt"],
                    "rendered_prompt": prompt_text,
                    "completion": completion,
                    "model": lm.name,
                    "condition": condition_label,
                    "decoding": decoding.asdict(),
                    "mask_cfg_active": mask_cfg is not None,
                    "timestamp": now_iso(),
                    **meta_extra,
                })
    return rows
