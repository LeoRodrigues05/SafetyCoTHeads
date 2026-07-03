"""Smoke tests for NeuronMaskController and SteeringController.

Build minimal Llama-shaped stubs (MLP and decoder layer) and verify that:
  * NeuronMaskController zeroes the specified down_proj input dims.
  * SteeringController('ablate') makes the residual orthogonal to the
    refusal direction at the layer's input.
  * SteeringController('add') shifts the residual by alpha*unit(v).
"""
from __future__ import annotations
import types

import torch
from torch import nn

from safety_cot_heads.models.neuron_and_steer import (
    NeuronMaskController, SteeringController,
)


# -----------------------------------------------------------------------
# Stub builder for a Llama-shaped MLP + decoder layer
# -----------------------------------------------------------------------
def _make_stub(n_layers=2, hidden=8, intermediate=16):

    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
            self.up_proj = nn.Linear(hidden, intermediate, bias=False)
            self.down_proj = nn.Linear(intermediate, hidden, bias=False)
            self.act_fn = nn.SiLU()

        def forward(self, h):
            return self.down_proj(self.act_fn(self.gate_proj(h)) * self.up_proj(h))

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = _MLP()
            # NeuronMaskController.attach also needs the layer; SteeringController
            # registers its pre-hook on _this_ module, so forward(h) must accept h.

        def forward(self, h):
            return h + self.mlp(h)

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_Layer() for _ in range(n_layers)])

        def forward(self, h):
            for layer in self.layers:
                h = layer(h)
            return h

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()
            self.config = types.SimpleNamespace(
                num_hidden_layers=n_layers,
                hidden_size=hidden,
                intermediate_size=intermediate,
            )

        def forward(self, h):
            return self.model(h)

    return _Model().eval(), hidden, intermediate


# -----------------------------------------------------------------------
# NeuronMaskController
# -----------------------------------------------------------------------
def test_neuron_mask_zeros_specified_dims():
    model, hidden, intermediate = _make_stub()
    ctrl = NeuronMaskController.attach(model)

    # Capture down_proj input by registering an additional pre-hook
    captured = {}
    def grab(_m, inputs):
        captured["x"] = inputs[0].detach().clone()
        return None
    h_handle = model.model.layers[0].mlp.down_proj.register_forward_pre_hook(grab)

    h = torch.randn(1, 3, hidden)
    target_neurons = [(0, 1), (0, 5), (0, 7)]
    cfg = {
        "neuron_mask": {(l, n): "n" for (l, n) in target_neurons},
        "mask_type": "scale_mask",
        "scale_factor": 0.0,
    }
    with ctrl.active(cfg):
        _ = model(h)

    h_handle.remove()
    # The captured tensor is the *unmasked* input (our grab hook runs before
    # the NeuronMaskController's pre-hook only if registered earlier — but
    # actually order depends on registration sequence; check that the *final*
    # down_proj sees the masked tensor by re-running with an end-of-chain hook).
    # Simpler: run again with only our grab AFTER ctrl, then ctrl re-runs zeroing.
    # We verify behaviour at the down_proj output level instead:
    model2, hidden2, intermediate2 = _make_stub()
    ctrl2 = NeuronMaskController.attach(model2)
    # capture down_proj input AFTER ctrl's pre-hook fires by registering ours later
    seen = {}
    def grab2(_m, inputs):
        seen["x"] = inputs[0].detach().clone()
    model2.model.layers[0].mlp.down_proj.register_forward_pre_hook(grab2)
    with ctrl2.active(cfg):
        _ = model2(h)
    x = seen["x"]
    for (_, n) in target_neurons:
        assert torch.allclose(x[..., n], torch.zeros_like(x[..., n])), (
            f"neuron {n} not zeroed; got {x[..., n]}"
        )
    # Non-target neurons should be untouched
    untouched = [n for n in range(intermediate2) if n not in {1, 5, 7}]
    assert not torch.allclose(x[..., untouched], torch.zeros_like(x[..., untouched]))


def test_neuron_mask_inactive_no_op():
    model, hidden, _ = _make_stub()
    ctrl = NeuronMaskController.attach(model)
    h = torch.randn(1, 3, hidden)
    out_no = model(h)
    with ctrl.active(None):
        out_inactive = model(h)
    assert torch.allclose(out_no, out_inactive)


# -----------------------------------------------------------------------
# SteeringController
# -----------------------------------------------------------------------
def test_steering_ablate_removes_direction():
    model, hidden, _ = _make_stub()
    ctrl = SteeringController.attach(model)
    v = torch.randn(hidden)

    # capture the input to layer 1 (post-ablation, since ctrl runs first)
    seen = {}
    def grab(_m, args, kwargs):
        h = args[0] if args else kwargs.get("hidden_states")
        seen["x"] = h.detach().clone()
    model.model.layers[1].register_forward_pre_hook(grab, with_kwargs=True)

    h = torch.randn(1, 4, hidden)
    cfg = {"mode": "ablate", "direction": v, "layers": [0, 1], "alpha": 1.0}
    with ctrl.active(cfg):
        _ = model(h)

    x = seen["x"]
    v_hat = v / (v.norm() + 1e-8)
    proj = (x @ v_hat).abs().max().item()
    assert proj < 1e-4, f"residual still has direction component: {proj}"


def test_steering_add_shifts_by_alpha_unit_v():
    model, hidden, _ = _make_stub()
    ctrl = SteeringController.attach(model)
    v = torch.randn(hidden)
    alpha = 0.7

    # baseline residual at layer 0 input
    h = torch.randn(1, 2, hidden)
    seen = {}
    def grab(_m, args, kwargs):
        h0 = args[0] if args else kwargs.get("hidden_states")
        seen["x"] = h0.detach().clone()
    handle = model.model.layers[0].register_forward_pre_hook(grab, with_kwargs=True)

    cfg = {"mode": "add", "direction": v, "layers": [0], "alpha": alpha}
    with ctrl.active(cfg):
        _ = model(h)
    handle.remove()

    expected = h + alpha * (v / (v.norm() + 1e-8))
    assert torch.allclose(seen["x"], expected, atol=1e-5), (
        f"layer-0 input was not shifted by alpha*unit(v)"
    )
