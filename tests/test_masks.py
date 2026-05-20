"""Smoke tests for HeadMaskController without downloading a model.

Build a minimal Llama-shaped stub: ``model.model.layers[i].self_attn`` with
``q_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj`` Linear sub-modules and a
``.config`` carrying the head counts.  Then verify:

* ``scale_factor=1.0`` (no-op): masking does not change ``q_proj`` output.
* ``scale_factor=0.0``: the masked head's slice of ``q_proj`` is zero.
"""
from __future__ import annotations
import types

import torch
from torch import nn

from safety_cot_heads.models import HeadMaskController, num_layers_and_heads
from safety_cot_heads.models.masks import add_head, empty_mask_cfg


def _make_stub(n_layers=2, n_heads=4, head_dim=8, n_kv=None):
    n_kv = n_kv or n_heads
    hidden = n_heads * head_dim
    kv_hidden = n_kv * head_dim

    class _Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(hidden, hidden, bias=False)
            self.k_proj = nn.Linear(hidden, kv_hidden, bias=False)
            self.v_proj = nn.Linear(hidden, kv_hidden, bias=False)
            self.o_proj = nn.Linear(hidden, hidden, bias=False)
            self.num_heads = n_heads
            self.num_key_value_heads = n_kv
            self.head_dim = head_dim

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _Attn()

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_Layer() for _ in range(n_layers)])

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Inner()
            self.config = types.SimpleNamespace(
                num_hidden_layers=n_layers,
                num_attention_heads=n_heads,
                num_key_value_heads=n_kv,
                hidden_size=hidden,
            )

    return _Model().eval(), hidden, head_dim


def test_num_layers_and_heads():
    m, _, _ = _make_stub(n_layers=3, n_heads=8)
    assert num_layers_and_heads(m) == (3, 8, 8)


def test_scale_one_is_noop():
    torch.manual_seed(0)
    m, hidden, _ = _make_stub()
    ctrl = HeadMaskController.attach(m)
    x = torch.randn(1, 5, hidden)

    with torch.no_grad():
        baseline = m.model.layers[0].self_attn.q_proj(x).clone()
        cfg = add_head(empty_mask_cfg(mask_qkv=("q",), scale_factor=1.0), 0, 1)
        with ctrl.active(cfg):
            scaled = m.model.layers[0].self_attn.q_proj(x).clone()

    assert torch.allclose(baseline, scaled, atol=1e-6)
    ctrl.detach()


def test_scale_zero_zeroes_head_slice():
    torch.manual_seed(0)
    m, hidden, head_dim = _make_stub(n_heads=4, head_dim=8)
    ctrl = HeadMaskController.attach(m)
    x = torch.randn(1, 5, hidden)

    with torch.no_grad():
        cfg = add_head(empty_mask_cfg(mask_qkv=("q",), scale_factor=0.0), 0, 2)
        with ctrl.active(cfg):
            out = m.model.layers[0].self_attn.q_proj(x)
        head2 = out.view(1, 5, 4, head_dim)[:, :, 2, :]
        assert torch.all(head2.abs() < 1e-7)
        # Other heads are untouched
        head1 = out.view(1, 5, 4, head_dim)[:, :, 1, :]
        assert head1.abs().sum() > 0
    ctrl.detach()


def test_no_cfg_is_passthrough():
    torch.manual_seed(0)
    m, hidden, _ = _make_stub()
    ctrl = HeadMaskController.attach(m)
    x = torch.randn(1, 4, hidden)
    with torch.no_grad():
        baseline = m.model.layers[0].self_attn.q_proj(x).clone()
        with ctrl.active(None):
            out = m.model.layers[0].self_attn.q_proj(x).clone()
    assert torch.allclose(baseline, out)
    ctrl.detach()
