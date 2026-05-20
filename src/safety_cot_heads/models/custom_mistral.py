"""Mistral wrapper.

The hook-based :class:`~safety_cot_heads.models.custom_llama.HeadMaskController`
works for Mistral as well (same `q_proj`/`k_proj`/`v_proj`/`o_proj` naming,
and GQA is handled by mapping the head index to a KV-head index).

This module exists for symmetry with the original SafetyHeadAttribution
package layout and to document the Mistral-specific considerations:

* Mistral uses **grouped-query attention** (``num_key_value_heads < num_attention_heads``).
  K and V masks are applied on the KV-head index ``head_idx // num_key_value_groups``.
* Mistral-v0.1/0.2 use **sliding-window attention** (window=4096). Head
  masking is local — it affects the projections only — so SWA is transparent.
* Recent ``transformers`` releases gate SDPA / flash-attention by an env var.
  If you observe NaNs after masking with flash-attn, force eager attention::

      model = load_model(name, attn_implementation="eager")
"""

from __future__ import annotations
from .custom_llama import HeadMaskController, num_layers_and_heads

__all__ = ["HeadMaskController", "num_layers_and_heads"]
