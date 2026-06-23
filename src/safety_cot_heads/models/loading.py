"""Robust model + tokenizer loading.

Adapted from ``SafetyHeadAttribution/lib/utils/get_model.py`` with the
hardcoded ``device_map='auto'`` / ``cuda:0`` removed and proper Llama-2 /
Llama-3 / Mistral / Qwen handling.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.device import resolve_device
from .custom_llama import HeadMaskController
from .neuron_and_steer import NeuronMaskController, SteeringController


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    head_mask_controller: Optional[HeadMaskController]
    neuron_mask_controller: Optional[NeuronMaskController]
    steering_controller: Optional[SteeringController]
    name: str

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device


def load_tokenizer(model_name: str, padding_side: str = "left"):
    tok = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side)
    if tok.pad_token is None:
        # Prefer reusing EOS rather than adding a token (matches CoT-safety repo).
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        else:
            tok.add_special_tokens({"pad_token": "<PAD>"})
    return tok


def load_model(
    model_name: str,
    *,
    device: Optional[str] = None,
    dtype: str = "auto",
    attn_implementation: Optional[str] = None,
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
    device_map: Optional[str] = None,
    attach_controllers: bool = True,
) -> LoadedModel:
    """Load a causal LM and attach the head-mask controller.

    Parameters
    ----------
    device : str | None
        Explicit device (``'cuda:0'`` etc). If ``None``, resolved from env.
    dtype : str
        ``'auto'`` | ``'bf16'`` | ``'fp16'`` | ``'fp32'``.
    attn_implementation : str | None
        Forwarded to HF; pass ``'eager'`` if flash-attn breaks head-masking.
    load_in_4bit : bool
        Use ``bitsandbytes`` 4-bit NF4 (recommended for >13B judge models only).
    device_map : str | None
        Forwarded to HF (e.g. ``'auto'`` for multi-GPU). Mutually exclusive
        with ``device``.
    """
    resolved_dtype = _resolve_dtype(dtype)

    kwargs = {"dtype": resolved_dtype, "trust_remote_code": trust_remote_code}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        # bitsandbytes requires device_map
        kwargs["device_map"] = device_map or "auto"
    elif device_map is not None:
        kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()

    if "device_map" not in kwargs:
        target = resolve_device(device)
        model.to(target)

    tok = load_tokenizer(model_name)
    if tok.pad_token_id is not None and tok.pad_token_id != model.config.pad_token_id:
        model.config.pad_token_id = tok.pad_token_id

    ctrl = HeadMaskController.attach(model) if attach_controllers else None
    nctrl = NeuronMaskController.attach(model) if attach_controllers else None
    sctrl = SteeringController.attach(model) if attach_controllers else None
    return LoadedModel(
        model=model, tokenizer=tok,
        head_mask_controller=ctrl,
        neuron_mask_controller=nctrl,
        steering_controller=sctrl,
        name=model_name,
    )


def _resolve_dtype(dtype: str):
    aliases = {
        "auto": "auto",
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "torch.bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "torch.float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "torch.float32": torch.float32,
    }
    try:
        return aliases[str(dtype).lower()]
    except KeyError as exc:
        raise ValueError(
            f"unknown dtype {dtype!r}; expected one of {sorted(aliases)}"
        ) from exc
