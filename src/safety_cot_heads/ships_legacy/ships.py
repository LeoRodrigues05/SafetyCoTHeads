"""SHIPS — per-prompt Safety Head ImPortant Score.

Direct adaptation of ``SafetyHeadAttribution/lib/SHIPS/get_ships.py``.

This module reuses the verbatim upstream helpers:

* :func:`safety_cot_heads._legacy.sha.pd_diff.kl_divergence` — the original
  KL implementation, unchanged.
* :func:`safety_cot_heads._legacy.sha.ships_utils.sort_ships_dict` — the
  original top-k aggregator, unchanged.

The only thing we replace is the *head-masking backend*.  Upstream depends on
``CustomLlamaModelForCausalLM`` injecting ``head_mask`` / ``mask_type`` /
``scale_factor`` kwargs into a re-implemented ``LlamaAttention.forward`` — that
forward is pinned to one ``transformers`` version and silently breaks on
GQA architectures (Mistral, Llama-3, Qwen2).  We swap it for the equivalent
hook-based :class:`HeadMaskController`, which produces the same numerical
outputs but works across HF versions and GQA.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import torch
from tqdm import tqdm

from .._legacy.sha.pd_diff import kl_divergence            # verbatim upstream
from .._legacy.sha.ships_utils import sort_ships_dict      # verbatim upstream
from ..models import LoadedModel, num_layers_and_heads
from ..models.masks import add_head, empty_mask_cfg, fmt_head_id
from ..utils import now_iso


@dataclass
class SHIPSConfig:
    """Mirrors the upstream ``mask_config`` dict plus a few I/O knobs."""
    mask_qkv: Sequence[str] = ("q",)
    mask_type: str = "scale_mask"
    scale_factor: float = 1e-4
    layers: Optional[tuple[int, int]] = None
    heads: Optional[tuple[int, int]] = None
    top_k: int = 10
    seed: int = 0
    prompt_template: str = "## Query:{q}\n## Answer:"


class SHIPS:
    """API mirrors upstream ``lib.SHIPS.get_ships.SHIPS``.

    The two changes are:
      * ``__init__`` takes an already-loaded :class:`LoadedModel` (no
        hardcoded ``cuda:0`` / model path) plus a :class:`SHIPSConfig`.
      * Head-masking goes through the hook controller, not a custom forward.
    """

    def __init__(self, lm: LoadedModel, cfg: SHIPSConfig, data=None):
        self.lm = lm
        self.cfg = cfg
        self.tokenizer = lm.tokenizer
        self.model = lm.model
        self.data = data
        n_layers, n_heads, _ = num_layers_and_heads(self.model)
        self.layers = n_layers
        self.heads = n_heads
        self.mask_cfg = empty_mask_cfg(
            mask_qkv=cfg.mask_qkv,
            mask_type=cfg.mask_type,
            scale_factor=cfg.scale_factor,
        )

    # -- upstream API parity ------------------------------------------------
    def one_forward_pass(self, input_text: str, mask_cfg: Mapping | None = None):
        inputs = self.tokenizer(input_text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.lm.device)
        attn_mask = inputs["attention_mask"].to(self.lm.device)
        with torch.no_grad(), self.lm.head_mask_controller.active(mask_cfg):
            output = self.model(input_ids, attention_mask=attn_mask)
        return output.logits

    @staticmethod
    def _get_pd(logits):
        return torch.softmax(logits.float(), dim=-1)

    @staticmethod
    def _get_last_logits(logits):
        return logits[:, -1, :]

    def _get_base_pd(self, input_text: str):
        return self._get_pd(self._get_last_logits(self.one_forward_pass(input_text)))

    def get_ships(self, input_text: str, layers=None, heads=None,
                  ships_type: str = "kl") -> dict[tuple[int, int], torch.Tensor]:
        start_layer, end_layer = layers or (0, self.layers)
        start_head, end_head = heads or (0, self.heads)
        base_pd = self._get_base_pd(input_text)
        out: dict[tuple[int, int], torch.Tensor] = {}
        total = (end_layer - start_layer) * (end_head - start_head)
        with tqdm(total=total, leave=False, desc="SHIPS heads") as bar:
            for layer in range(start_layer, end_layer):
                for head in range(start_head, end_head):
                    cfg = add_head(self.mask_cfg, layer, head)
                    logits = self.one_forward_pass(input_text, cfg)
                    now_pd = self._get_pd(self._get_last_logits(logits))
                    out[(layer, head)] = (kl_divergence(base_pd, now_pd)
                                          if ships_type == "kl" else None)
                    bar.update(1)
        return out

    # -- new clean entry point ---------------------------------------------
    def run(self, prompts: Iterable[tuple[str, str]]) -> list[dict]:
        rows: list[dict] = []
        for example_id, prompt in prompts:
            input_text = self.cfg.prompt_template.format(q=prompt)
            scores = self.get_ships(input_text)
            top = sort_ships_dict(scores, print_flag=False, sorted_item=self.cfg.top_k)
            ranked = [
                {"head_id": hid, "layer": int(hid.split("-")[0]),
                 "head": int(hid.split("-")[1]), "score": float(s)}
                for hid, s in top.items()
            ]
            rows.append({
                "example_id": example_id,
                "prompt": prompt,
                "prompt_template": self.cfg.prompt_template,
                "model": self.lm.name,
                "method": "ships",
                "mask_qkv": list(self.cfg.mask_qkv),
                "mask_type": self.cfg.mask_type,
                "scale_factor": self.cfg.scale_factor,
                "seed": self.cfg.seed,
                "timestamp": now_iso(),
                "ranked_heads": ranked,
                "all_scores": {fmt_head_id(l, h): float(v.item())
                               for (l, h), v in scores.items()},
            })
        return rows


def aggregate_dataset_ranking(rows: list[dict], top_k: int = 16) -> list[dict]:
    """Mean-aggregate per-prompt SHIPS scores into a dataset-level ranking."""
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for r in rows:
        for hid, s in r["all_scores"].items():
            sums[hid] = sums.get(hid, 0.0) + float(s)
            counts[hid] = counts.get(hid, 0) + 1
    items = sorted(
        ((hid, sums[hid] / counts[hid]) for hid in sums),
        key=lambda kv: kv[1], reverse=True,
    )
    out = []
    for hid, m in items[:top_k]:
        l, h = hid.split("-")
        out.append({"head_id": hid, "layer": int(l), "head": int(h), "mean_score": m})
    return out
