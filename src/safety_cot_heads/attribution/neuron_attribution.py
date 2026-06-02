"""MLP-neuron attribution (Wang et al. 2024, "Finding Safety Neurons in LLMs").

For each (layer, neuron) we score:

    score(l, n) = mean_{harmful} a_l[n]  -  mean_{benign} a_l[n]

where ``a_l ∈ R^{intermediate_size}`` is the **gated activation** that feeds
``down_proj`` at layer ``l`` (i.e. ``act_fn(gate_proj(h)) * up_proj(h)``).
This is captured at the **last prompt token** — the same position used by
SHIPS / Sahara for head attribution, and the most natural choice for
"refusal-relevant" content as established in Arditi et al. (2024).

Output schema (mirrors SHIPS):

    {
      "ranked_neurons": [
        {"neuron_id": "L-N", "layer": L, "neuron": N, "mean_score": float},
        …
      ],
      "n_harmful": int, "n_benign": int,
      "model": str, "datasets": {"harmful": str, "benign": str},
      "default_top_k": 32,
      "citation": "..."
    }

``default_top_k = 32`` follows Wang et al.'s reported "small set of neurons
(~5 % of one layer)" that, when ablated, produce the largest drop in
refusal; for Llama-3.1-8B (intermediate_size = 14336) this is intentionally
conservative — bump it if you want the broader 5 % regime.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
from tqdm import tqdm

from ..generation.prompts import render_chat
from ..models import LoadedModel


@dataclass
class NeuronAttributionConfig:
    batch_size: int = 4
    max_length: int = 512
    capture_last_n: int = 1   # last token only (Wang et al. default)
    top_k_default: int = 32   # for the ranking metadata; selection is downstream
    system_prompt: Optional[str] = None


def _collect_mlp_acts(lm: LoadedModel) -> tuple[list, list]:
    """Attach forward hooks on each layer's MLP that record the input to
    ``down_proj`` (= the gated activation = our neuron vector). Returns
    ``(handles, buffers)`` where ``buffers[layer_idx]`` accumulates tensors.
    """
    inner = getattr(lm.model, "model", lm.model)
    layers = inner.layers
    buffers: list[list[torch.Tensor]] = [[] for _ in layers]
    handles = []
    for layer_idx, layer in enumerate(layers):
        def make_hook(li=layer_idx):
            def pre_hook(_module, inputs):
                x = inputs[0]
                # mean over last_n tokens collected later; here just stash
                buffers[li].append(x.detach().to("cpu", dtype=torch.float32))
            return pre_hook
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(make_hook()))
    return handles, buffers


@torch.no_grad()
def _mean_neuron_activations(
    lm: LoadedModel,
    prompts: Sequence[str],
    cfg: NeuronAttributionConfig,
) -> torch.Tensor:
    """Return tensor of shape ``(num_layers, intermediate_size)`` with the
    mean (over prompts and last_n tokens) of the down_proj input."""
    tok = lm.tokenizer
    inner = getattr(lm.model, "model", lm.model)
    n_layers = len(inner.layers)
    intermediate = int(lm.model.config.intermediate_size)

    sums = torch.zeros(n_layers, intermediate, dtype=torch.float64)
    count = 0

    handles, buffers = _collect_mlp_acts(lm)
    try:
        for i in tqdm(range(0, len(prompts), cfg.batch_size),
                      desc="neuron_attr.acts"):
            batch = prompts[i:i + cfg.batch_size]
            rendered = [render_chat(tok, p, system=cfg.system_prompt) for p in batch]
            enc = tok(rendered, return_tensors="pt", padding=True,
                      truncation=True, max_length=cfg.max_length).to(lm.device)
            for buf in buffers:
                buf.clear()
            _ = lm.model(**enc)
            # Each buffer entry has shape (B, T, intermediate); take last N tokens
            for li, captured in enumerate(buffers):
                assert len(captured) == 1, "expected exactly one fwd per layer"
                act = captured[0]  # (B, T, I)
                last_n = act[:, -cfg.capture_last_n:, :].mean(dim=1)  # (B, I)
                sums[li] += last_n.sum(dim=0).double()
            count += len(batch)
    finally:
        for h in handles:
            h.remove()

    return (sums / max(count, 1)).float()


def neuron_attribution(
    lm: LoadedModel,
    harmful_prompts: Sequence[str],
    benign_prompts: Sequence[str],
    cfg: Optional[NeuronAttributionConfig] = None,
) -> dict:
    """Compute per-neuron contrast score and return a ranked list."""
    cfg = cfg or NeuronAttributionConfig()
    mean_h = _mean_neuron_activations(lm, harmful_prompts, cfg)
    mean_b = _mean_neuron_activations(lm, benign_prompts, cfg)
    diff = mean_h - mean_b  # (L, I)
    flat = diff.flatten()
    order = torch.argsort(flat.abs(), descending=True)
    n_layers, intermediate = diff.shape

    ranked = []
    for idx in order.tolist():
        layer = idx // intermediate
        neuron = idx % intermediate
        ranked.append({
            "neuron_id": f"{layer}-{neuron}",
            "layer": int(layer),
            "neuron": int(neuron),
            "mean_score": float(diff[layer, neuron].item()),
        })
    return {
        "ranked_neurons": ranked,
        "n_harmful": len(harmful_prompts),
        "n_benign": len(benign_prompts),
        "model": lm.name,
        "default_top_k": cfg.top_k_default,
        "citation": (
            "Wang et al. 2024, 'Finding Safety Neurons in LLMs'. "
            "Score = mean(act_harmful) - mean(act_benign) over the last "
            "prompt token, on the gated activation that feeds down_proj."
        ),
    }
