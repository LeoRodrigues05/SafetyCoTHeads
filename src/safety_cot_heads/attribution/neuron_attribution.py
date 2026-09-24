"""MLP-neuron attribution by harmful-vs-benign activation contrast.

What this implements (every ``neurons_top*`` cell in the v5/v6 grid):

    score(l, n) = mean_{x in harmful} a_l(x)[n]  -  mean_{x in benign} a_l(x)[n]

where ``a_l`` is the gated MLP activation feeding ``down_proj``
(``act_fn(gate_proj(h)) * up_proj(h)``) at the **last prompt token**, harmful =
MaliciousInstruct, benign = Alpaca. Neurons are ranked by ``|score|`` over all
layers jointly (no per-layer normalisation) and the top-k are zeroed at every
position during generation (:class:`NeuronMaskController`).

What it is NOT -- keep paper text consistent with this:

* It is not the safety-specific-neuron procedure of Zhao et al. (ICLR 2025),
  which scores *both* attention and FFN neurons by the effect of deactivating
  them on the layer output (FFN: ``|a_n| * ||W_down[:, n]||``), selects neurons
  important on harmful queries, and removes those also important on general
  queries.
* It is not the model-contrast procedure of "Finding Safety Neurons in Large
  Language Models" (2024), which contrasts activations of an aligned model
  against its pre-alignment checkpoint on the same inputs.

Known property: raw activation magnitude grows with depth, so the unnormalised
ranking concentrates the top-k in the last layers (41-55% of the top-1024 sit
in the final layer for the evaluated models). The random-neuron control
(``layer_matched_neurons``) matches this per-layer histogram.

Output schema (mirrors SHIPS):

    {
      "ranked_neurons": [
        {"neuron_id": "L-N", "layer": L, "neuron": N, "mean_score": float},
        ...
      ],
      "n_harmful": int, "n_benign": int,
      "model": str, "datasets": {"harmful": str, "benign": str},
      "default_top_k": 32,
      "citation": "..."
    }
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
            "Activation contrast: score = mean(act_harmful) - mean(act_benign) "
            "at the last prompt token, on the gated activation that feeds "
            "down_proj; ranked by |score| across layers. Not the Zhao et al. "
            "(2025) deactivation-based procedure."
        ),
    }
