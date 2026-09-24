"""Hook-based **neuron** and **direction-steering** controllers for Llama-family models.

Companion to :class:`safety_cot_heads.models.custom_llama.HeadMaskController`.
All three controllers attach `torch` forward (pre-)hooks to existing modules
inside the decoder; none of them mutate weights.

Implemented methods (see citations next to each class):

* :class:`NeuronMaskController` — zero or rescale individual MLP "neurons"
  (= rows of ``down_proj.weight``, equivalently dims of its input); the
  standard neuron-ablation operation — default ablation = scale input dim
  to 0 (full ablation). Which neurons are ablated is decided upstream
  (see attribution/neuron_attribution.py).
* :class:`SteeringController` — add a fixed unit direction $v$ to the residual
  stream at one or more layers, or project $v$ out entirely. The two modes
  cover (a) DSH-style activation addition (Zou et al. 2023; Turner et al.
  2023) and (b) Arditi et al. (2024) "directional ablation".

Composing controllers with :class:`HeadMaskController` is done in
:func:`safety_cot_heads.generation.generate.generate` via ``ExitStack`` —
each controller's ``active(cfg)`` context manager is opened in order, and
the forward hooks fire independently per module.
"""

from __future__ import annotations
from contextlib import contextmanager
from typing import Iterable, Mapping, Optional, Sequence

import torch
from torch import nn


# ===========================================================================
# Neuron-level MLP ablation
# ===========================================================================
class NeuronMaskController:
    """Ablate individual MLP neurons (zero their contribution to the residual
    stream).

    Implementation choice
    ---------------------
    A "neuron" in a SwiGLU MLP (Llama) is the *output channel* of
    ``gate_proj`` / ``up_proj`` and equivalently the *input channel* of
    ``down_proj``. Ablating a neuron means: in the forward pass, force its
    contribution to the residual stream to zero. We register a forward
    **pre-hook** on ``down_proj`` and zero (or rescale) the selected input
    dims of its input tensor. This is exactly the "ablate neuron" operation
    used in the safety-neuron paper (their Eq. 2; their default ``α = 0``).

    Configuration (``cfg`` passed to :meth:`active`)
    ------------------------------------------------
    * ``neuron_mask`` : ``Mapping[(layer:int, neuron:int), str]``
        Marker string is currently ignored (kept for parity with
        ``HeadMaskController.head_mask``'s qkv marker).
    * ``mask_type`` : ``"scale_mask"`` (default) — multiplies by ``scale_factor``.
    * ``scale_factor`` : default ``0.0`` (full ablation).

    Multiple controllers can be active simultaneously; the down_proj
    pre-hook only fires the mask if the current layer has entries.
    """

    def __init__(self, model: nn.Module, mlp_layers: list[nn.Module]):
        self.model = model
        self.mlp_layers = mlp_layers
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._cfg: Mapping | None = None

    # ---- attach / detach ---------------------------------------------------
    @classmethod
    def attach(cls, model: nn.Module) -> "NeuronMaskController":
        mlp_layers = _collect_mlp_layers(model)
        ctrl = cls(model, mlp_layers)
        for layer_idx, mlp in enumerate(mlp_layers):
            ctrl._handles.append(
                mlp.down_proj.register_forward_pre_hook(
                    ctrl._make_down_pre_hook(layer_idx)
                )
            )
        return ctrl

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ---- runtime config ----------------------------------------------------
    @contextmanager
    def active(self, neuron_cfg: Mapping | None):
        prev = self._cfg
        self._cfg = neuron_cfg
        try:
            yield self
        finally:
            self._cfg = prev

    # ---- hook factory ------------------------------------------------------
    def _neurons_for_layer(self, layer_idx: int) -> list[int]:
        if self._cfg is None:
            return []
        mask = self._cfg.get("neuron_mask") or {}
        return [int(n) for (li, n) in mask.keys() if int(li) == layer_idx]

    def _make_down_pre_hook(self, layer_idx: int):
        def pre_hook(_module, inputs):
            if self._cfg is None:
                return None
            neurons = self._neurons_for_layer(layer_idx)
            if not neurons:
                return None
            mask_type = self._cfg.get("mask_type", "scale_mask")
            scale = float(self._cfg.get("scale_factor", 0.0))
            x = inputs[0]
            if mask_type != "scale_mask":
                raise ValueError(
                    f"NeuronMaskController only supports mask_type='scale_mask' "
                    f"(got {mask_type!r}). Mean-masking neurons is not in the original"
                    f" ablation protocol used here."
                )
            new = x.clone()
            idx = torch.tensor(neurons, device=x.device, dtype=torch.long)
            new.index_copy_(
                -1, idx, new.index_select(-1, idx) * scale,
            )
            return (new,) + tuple(inputs[1:])
        return pre_hook


# ===========================================================================
# Direction-steering / directional ablation
# ===========================================================================
class SteeringController:
    """Add a fixed unit direction to the residual stream, or project it out.

    Two modes (selected by ``cfg["mode"]``):

    * ``"add"`` — at every forward pass through the decoder layer(s) listed
      in ``cfg["layers"]``, replace the layer's input hidden state ``h``
      with ``h + alpha * unit(v)``. Default protocol: Turner et al. (2023)
      "Activation Addition" and Zou et al. (2023) "Representation
      Engineering". Choice of layer follows their guidance — usually one
      mid-network layer (≈40 % depth).
    * ``"ablate"`` — for each token, project ``v`` out of the residual stream
      at the *input* of every listed decoder layer:
      ``h <- h - ((h @ v_hat) * v_hat)``. This is what every
      ``steering_ablate`` cell in the v5/v6 grid used. It is weaker than
      Arditi et al. (2024): inside a layer the MLP still reads the ``v``
      component that the same layer's attention just wrote, and the last
      layer's writes are never cleaned before the final norm/unembedding.
    * ``"ablate_all"`` — Arditi et al. (2024) directional ablation proper:
      ``v`` is projected out of *every write* to the residual stream (the
      embedding, via the layer-0 input, plus each layer's attention and MLP
      writes), so no position of the residual stream ever carries ``v``. The
      write points are architecture-specific (see :func:`_residual_writers`):
      pre-norm Llama/Qwen blocks add the raw ``self_attn``/``mlp`` outputs;
      OLMo-2/3 add ``post_attention_layernorm``/``post_feedforward_layernorm``
      outputs, which is where the projection has to happen for them.

    Configuration (``cfg`` passed to :meth:`active`)
    ------------------------------------------------
    * ``mode`` : ``"add"`` | ``"ablate"`` | ``"ablate_all"``
    * ``direction`` : 1-D ``torch.Tensor`` of shape ``(hidden_size,)``;
      it is L2-normalised internally for both ``"add"`` and ``"ablate"``.
    * ``layers`` : ``Sequence[int]`` — which decoder-layer inputs to patch.
      For ``mode="add"`` the original paper picks ONE layer; for
      ``mode="ablate"`` Arditi et al. apply at ALL layers (pass
      ``range(n_layers)``).
    * ``alpha`` : ``float``, only used in ``mode="add"``. Defaults to 1.0.
    """

    def __init__(self, model: nn.Module, decoder_layers: list[nn.Module]):
        self.model = model
        self.decoder_layers = decoder_layers
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._cfg: Mapping | None = None
        self._direction_cache: torch.Tensor | None = None  # normalised, on device

    @classmethod
    def attach(cls, model: nn.Module) -> "SteeringController":
        layers = _collect_decoder_layers(model)
        ctrl = cls(model, layers)
        for layer_idx, layer in enumerate(layers):
            ctrl._handles.append(
                layer.register_forward_pre_hook(
                    ctrl._make_layer_pre_hook(layer_idx),
                    with_kwargs=True,
                )
            )
            # residual-stream writers, used only by mode="ablate_all"
            for writer in _residual_writers(layer):
                ctrl._handles.append(
                    writer.register_forward_hook(ctrl._make_writer_hook(layer_idx))
                )
        return ctrl

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._direction_cache = None

    @contextmanager
    def active(self, steering_cfg: Mapping | None):
        prev = self._cfg
        prev_cache = self._direction_cache
        self._cfg = steering_cfg
        self._direction_cache = None  # forces recompute against current device/dtype
        try:
            yield self
        finally:
            self._cfg = prev
            self._direction_cache = prev_cache

    # ---- internal ----------------------------------------------------------
    def _prepare_direction(self, ref: torch.Tensor) -> torch.Tensor:
        """Get the direction tensor on the same device/dtype as ``ref``."""
        if self._direction_cache is not None:
            return self._direction_cache
        v = self._cfg["direction"]
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(v)
        v = v.to(device=ref.device, dtype=ref.dtype)
        mode = self._cfg.get("mode", "add")
        if mode in ("ablate", "ablate_all", "add"):
            # Unit-normalise the direction. For "ablate" this is required for the
            # projection; for "add" it makes `alpha` the ABSOLUTE perturbation
            # magnitude (h + alpha * v_unit), so the dose is comparable across
            # models — raw ||v|| varies ~4x between models (e.g. 5 vs 20 at L14),
            # which otherwise makes a fixed alpha mean different things per model.
            v = v / (v.norm() + 1e-8)
        self._direction_cache = v
        return v

    def _make_layer_pre_hook(self, layer_idx: int):
        def pre_hook(_module, args, kwargs):
            cfg = self._cfg
            if cfg is None:
                return None
            layers = cfg.get("layers")
            if layers is None or layer_idx not in layers:
                return None
            # Decoder layer's hidden state is either the first positional
            # arg or kwargs["hidden_states"]. Handle both.
            h = None
            from_kwargs = False
            if args and isinstance(args[0], torch.Tensor):
                h = args[0]
            elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
                h = kwargs["hidden_states"]
                from_kwargs = True
            if h is None:
                return None
            v = self._prepare_direction(h)
            mode = cfg.get("mode", "add")
            if mode == "add":
                alpha = float(cfg.get("alpha", 1.0))
                new_h = h + alpha * v
            elif mode in ("ablate", "ablate_all"):
                # h.shape: (batch, seq, hidden); v.shape: (hidden,)
                proj = (h @ v).unsqueeze(-1) * v
                new_h = h - proj
            else:
                raise ValueError(f"SteeringController: unknown mode {mode!r}")
            if from_kwargs:
                new_kwargs = dict(kwargs)
                new_kwargs["hidden_states"] = new_h
                return (args, new_kwargs)
            return ((new_h,) + tuple(args[1:]), kwargs)
        return pre_hook

    def _make_writer_hook(self, layer_idx: int):
        """Project ``v`` out of a residual-writing module's output (ablate_all)."""
        def hook(_module, _inputs, output):
            cfg = self._cfg
            if cfg is None or cfg.get("mode") != "ablate_all":
                return None
            layers = cfg.get("layers")
            if layers is None or layer_idx not in layers:
                return None
            out = output[0] if isinstance(output, tuple) else output
            if not isinstance(out, torch.Tensor):
                return None
            v = self._prepare_direction(out)
            new = out - (out @ v).unsqueeze(-1) * v
            if isinstance(output, tuple):
                return (new,) + tuple(output[1:])
            return new
        return hook


# ===========================================================================
# Module collectors
# ===========================================================================
def _collect_mlp_layers(model: nn.Module) -> list[nn.Module]:
    """Return each decoder layer's MLP module (Llama: ``layer.mlp``)."""
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise ValueError("Could not find `.model.layers`; unsupported decoder.")
    out = []
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not hasattr(mlp, "down_proj"):
            raise ValueError("Decoder layer missing `.mlp.down_proj`; unsupported architecture.")
        out.append(mlp)
    return out


def _residual_writers(layer: nn.Module) -> list[nn.Module]:
    """Modules whose output is added to the residual stream inside ``layer``.

    * OLMo-2/3 (reordered norm): ``h + post_attention_layernorm(attn(h))`` and
      ``h + post_feedforward_layernorm(mlp(h))`` -> the two post-norms.
    * Pre-norm blocks (Llama, Qwen2/3, Mistral): ``h + attn(norm(h))`` and
      ``h + mlp(norm(h))`` -> ``self_attn`` and ``mlp``. (In these models
      ``post_attention_layernorm`` is the MLP's *input* norm, not a writer.)
    """
    if hasattr(layer, "post_feedforward_layernorm") and hasattr(layer, "post_attention_layernorm"):
        return [layer.post_attention_layernorm, layer.post_feedforward_layernorm]
    # (stub layers in unit tests may carry only an mlp; real decoders carry both)
    return [m for m in (getattr(layer, "self_attn", None), getattr(layer, "mlp", None))
            if m is not None]


def _collect_decoder_layers(model: nn.Module) -> list[nn.Module]:
    """Return the decoder-layer modules in order."""
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise ValueError("Could not find `.model.layers`; unsupported decoder.")
    return list(layers)
