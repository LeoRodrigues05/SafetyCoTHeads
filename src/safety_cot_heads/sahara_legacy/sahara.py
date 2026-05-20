"""Sahara — Safety-subspace shift attribution.

Direct adaptation of ``SafetyHeadAttribution/lib/Sahara/attribution.py``.

The principal-angle metric is imported verbatim from the upstream
:mod:`safety_cot_heads._legacy.sha.sahara_svd` module (unchanged).

Changes vs. upstream:

* No hardcoded ``cuda:0`` / model paths.
* Uses :class:`HeadMaskController` for head masking instead of the upstream
  custom-forward ``head_mask`` kwarg path.
* The greedy "search_step" loop is preserved.
"""

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import torch
from tqdm import tqdm

from .._legacy.sha.sahara_svd import compute_subspace_similarity   # verbatim
from ..models import LoadedModel, num_layers_and_heads
from ..models.masks import add_head, empty_mask_cfg, fmt_head_id
from ..utils import now_iso


@dataclass
class SaharaConfig:
    mask_qkv: Sequence[str] = ("q",)
    mask_type: str = "scale_mask"
    scale_factor: float = 1e-5
    search_step: int = 1
    prompt_template: str = "## Query:{q}\n## Answer:"
    seed: int = 0


# ---------------------------------------------------------------------------
# Adapted from lib/Sahara/attribution.py::get_last_hidden_states
# (verbatim algorithm; only the forward call and device handling changed)
# ---------------------------------------------------------------------------
def get_last_hidden_states(lm: LoadedModel,
                            prompts: Iterable[str],
                            mask_cfg: Mapping | None = None) -> torch.Tensor:
    rows = []
    with torch.no_grad(), lm.head_mask_controller.active(mask_cfg):
        for input_text in prompts:
            input_ids = lm.tokenizer.encode(input_text, return_tensors="pt").to(lm.device)
            outputs = lm.model(input_ids, output_hidden_states=True)
            last_hs = outputs.hidden_states[-1]
            rows.append(last_hs[:, -1, :].reshape(-1).float().cpu())
    return torch.stack(rows)


def safety_head_attribution(lm: LoadedModel,
                            prompts: Sequence[str],
                            cfg: SaharaConfig,
                            top_k: int = 16) -> dict:
    """Greedy Sahara search.

    Returns a single JSONL-ready row containing per-(layer,head) shifts for
    every search step and the final selected head set.
    """
    n_layers, n_heads, _ = num_layers_and_heads(lm.model)
    mask_cfg = empty_mask_cfg(
        mask_qkv=cfg.mask_qkv, mask_type=cfg.mask_type, scale_factor=cfg.scale_factor,
    )
    rendered_prompts = [cfg.prompt_template.format(q=p) for p in prompts]
    base_lhs = get_last_hidden_states(lm, rendered_prompts, mask_cfg=None)

    history: list[dict] = []
    selected: list[tuple[int, int]] = []
    for step in range(cfg.search_step):
        shifts: dict[tuple[int, int], float] = {}
        for layer in tqdm(range(n_layers), desc=f"sahara step {step}", leave=False):
            for head in range(n_heads):
                if (layer, head) in selected:
                    continue
                trial_cfg = add_head(mask_cfg, layer, head)
                lhs = get_last_hidden_states(lm, rendered_prompts, mask_cfg=trial_cfg)
                shifts[(layer, head)] = float(compute_subspace_similarity(base_lhs, lhs))
        # greedy pick: the head whose ablation maximally shifts the subspace
        best = max(shifts.items(), key=lambda kv: kv[1])
        selected.append(best[0])
        mask_cfg = add_head(mask_cfg, *best[0])
        history.append({
            "step": step,
            "best_head": fmt_head_id(*best[0]),
            "best_shift_deg": best[1],
            "shifts": {fmt_head_id(l, h): s for (l, h), s in shifts.items()},
        })

    # final ranking: by single-head shift in step 0
    step0 = history[0]["shifts"]
    ranked = sorted(step0.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    ranked_heads = [
        {"head_id": hid, "layer": int(hid.split("-")[0]),
         "head": int(hid.split("-")[1]), "score": s}
        for hid, s in ranked
    ]

    return {
        "model": lm.name,
        "method": "sahara",
        "n_prompts": len(prompts),
        "prompt_template": cfg.prompt_template,
        "mask_qkv": list(cfg.mask_qkv),
        "mask_type": cfg.mask_type,
        "scale_factor": cfg.scale_factor,
        "search_step": cfg.search_step,
        "seed": cfg.seed,
        "timestamp": now_iso(),
        "selected_heads": [fmt_head_id(l, h) for (l, h) in selected],
        "ranked_heads": ranked_heads,
        "history": history,
    }
