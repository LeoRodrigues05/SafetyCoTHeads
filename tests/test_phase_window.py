"""Unit tests for ``HeadMaskController.phase_window`` token-gating logic.

These tests bypass the full attention stack and exercise the controller's
gating predicate directly, plus a synthetic single-layer module to verify
that the q-proj hook is only applied when the current token index falls
inside the configured window.
"""
from __future__ import annotations

import torch
from torch import nn

from safety_cot_heads.models.custom_llama import HeadMaskController


class _StubAttn(nn.Module):
    """Minimal attention shim exposing the attributes the controller probes."""

    def __init__(self, hidden: int = 16, num_heads: int = 4, num_kv: int = 2):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv
        self.head_dim = hidden // num_heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, (hidden // num_heads) * num_kv, bias=False)
        self.v_proj = nn.Linear(hidden, (hidden // num_heads) * num_kv, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        # init q-proj to identity-like so we can detect zero-mask easily
        with torch.no_grad():
            self.q_proj.weight.fill_(0.0)
            self.q_proj.weight += torch.eye(hidden)


def _build_controller() -> tuple[HeadMaskController, _StubAttn]:
    attn = _StubAttn()
    ctrl = HeadMaskController(model=nn.Module(), attn_layers=[attn])
    # Manually attach the q-proj hook (mirrors what attach() does).
    attn.q_proj.register_forward_hook(ctrl._make_proj_hook(0, "q"))
    return ctrl, attn


def test_window_unset_is_always_active() -> None:
    ctrl, attn = _build_controller()
    cfg = {"head_mask": {(0, 0): "qkv"}, "mask_type": "scale_mask",
           "scale_factor": 0.0}
    x = torch.ones(1, 1, 16)
    with ctrl.active(cfg):
        # token index defaults to -1 but no window → mask still fires
        out = attn.q_proj(x)
    # head 0 occupies dims [0:4]; scale_mask with factor 0 zeroes them.
    assert torch.allclose(out[..., :4], torch.zeros(4))
    assert torch.allclose(out[..., 4:], torch.ones(12))


def test_window_gates_outside_range() -> None:
    ctrl, attn = _build_controller()
    cfg = {"head_mask": {(0, 0): "qkv"}, "mask_type": "scale_mask",
           "scale_factor": 0.0, "phase_window": (10, 20)}
    x = torch.ones(1, 1, 16)
    with ctrl.active(cfg):
        ctrl.set_token_index(5)         # outside window
        out_outside = attn.q_proj(x)
        ctrl.set_token_index(15)        # inside window
        out_inside = attn.q_proj(x)
        ctrl.set_token_index(20)        # right boundary is exclusive
        out_boundary = attn.q_proj(x)
    # outside → identity (no masking)
    assert torch.allclose(out_outside, torch.ones(1, 1, 16))
    # inside → head 0 dims masked to zero
    assert torch.allclose(out_inside[..., :4], torch.zeros(4))
    # boundary (== end) → bypassed
    assert torch.allclose(out_boundary, torch.ones(1, 1, 16))


def test_active_context_resets_token_index() -> None:
    ctrl, _ = _build_controller()
    with ctrl.active({"head_mask": {}, "phase_window": (0, 1)}):
        ctrl.set_token_index(5)
        assert ctrl._token_idx == 5
    # after exit, restored to default sentinel
    assert ctrl._token_idx == -1


def test_phase_window_predicate() -> None:
    ctrl, _ = _build_controller()
    with ctrl.active({"head_mask": {(0, 0): "q"}, "phase_window": (3, 6)}):
        ctrl.set_token_index(2); assert not ctrl._phase_window_active()
        ctrl.set_token_index(3); assert ctrl._phase_window_active()
        ctrl.set_token_index(5); assert ctrl._phase_window_active()
        ctrl.set_token_index(6); assert not ctrl._phase_window_active()
