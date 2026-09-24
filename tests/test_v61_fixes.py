"""Regression tests for the v6.1 correctness fixes.

* parser: truncated traces under a pre-filled ``<think>`` are malformed, not answers
* coherence gate: the classifier label is applied; repetition-only variant kept
* input-hash resume + pruning
* random-target controls (heads / neurons / directions)
* relative steering dose
* ``ablate_all`` removes the direction from every residual position (tiny real
  Llama and OLMo-3), where the legacy ``ablate`` mode does not
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "common"))
import _bootstrap  # noqa: E402,F401  (src/ + every scripts/<group>/ on sys.path)

from safety_cot_heads.direction_a_v6.parsing import parse_completion, prompt_prefills_trace
from safety_cot_heads.analysis.coherence import canonical_is_clean
from safety_cot_heads.attribution.random_heads import layer_matched, layer_matched_neurons
from safety_cot_heads.interventions.steering import (
    build_activation_addition_cfg, random_direction_like)


# ---------------------------------------------------------------- parser ----
OLMO_PROMPT = "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>"
QWEN_PROMPT = "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n"


def test_prompt_prefills_trace():
    assert prompt_prefills_trace(OLMO_PROMPT)
    assert prompt_prefills_trace("<｜Assistant｜><think>\n")
    assert not prompt_prefills_trace(QWEN_PROMPT)
    assert not prompt_prefills_trace("<think>x</think> done")
    assert not prompt_prefills_trace(None)


def test_prefilled_truncated_trace_is_malformed_not_answer():
    comp = "Okay, the user wants X. First, I should consider the steps. Step one is"
    p = parse_completion(comp, trace_prefilled=True)
    assert p.trace_kind == "malformed_explicit"
    assert p.answer_text == "" and p.answer_is_empty
    assert p.trace_text.startswith("Okay")


def test_prefilled_terminated_trace_unchanged():
    p = parse_completion("reasoning here.</think>Final answer.", trace_prefilled=True)
    assert p.trace_kind == "explicit" and p.answer_text == "Final answer."


def test_unprefilled_no_tags_is_still_prose():
    p = parse_completion("Sure. Here is how. Done.", trace_prefilled=False)
    assert p.trace_kind == "prose_prefix" and p.answer_text.startswith("Sure")


# ---------------------------------------------------------- coherence ------
def test_gate_applies_classifier_label():
    g = canonical_is_clean(is_empty=False, repeat3=0.1, gibberish_label="word salad")
    assert not g["is_clean"] and "gibberish:word salad" in g["fail_reasons"]
    g2 = canonical_is_clean(is_empty=False, repeat3=0.1, gibberish_label="word salad",
                            use_gibberish=False)
    assert g2["is_clean"] and g2["gate_version"].endswith("repetition-only")


def test_gate_mild_gibberish_and_repetition():
    assert canonical_is_clean(is_empty=False, repeat3=0.1, gibberish_label="mild gibberish")["is_clean"]
    assert not canonical_is_clean(is_empty=False, repeat3=0.9, gibberish_label="clean")["is_clean"]
    assert not canonical_is_clean(is_empty=True, repeat3=0.0, gibberish_label="clean")["is_clean"]


# ------------------------------------------------------ resume / prune ------
def test_resume_todo_and_prune(tmp_path):
    import v6_common as C
    out = tmp_path / "judge.jsonl"
    rows = [{"id": "a", "prompt": "p", "completion": "x"},
            {"id": "b", "prompt": "p", "completion": "y"},
            {"id": "c", "prompt": "p", "completion": "z"}]
    for r in rows:
        r["input_sha256"] = C.input_sha(r["prompt"], r["completion"])
    C.write_jsonl(out, [
        {"id": "a", "input_sha256": rows[0]["input_sha256"]},      # current
        {"id": "b", "input_sha256": "stale"},                      # input changed
        {"id": "p::p000::think", "parent_id": "c"},                # legacy prefix row
    ])
    todo, stale = C.resume_todo(rows, out)
    assert {r["id"] for r in todo} == {"b", "c"} and stale == {"b"}
    assert C.prune_rows(out, {"b", "c"}, tag="t") == 2
    assert [r["id"] for r in C.read_jsonl(out)] == ["a"]
    assert len(C.read_jsonl(out.with_name("judge.pruned_t.jsonl"))) == 2


def test_coherence_version_forces_recompute(tmp_path):
    import v6_common as C
    out = tmp_path / "coh.jsonl"
    C.write_jsonl(out, [{"id": "a", "is_clean": True}])            # v6.0 row, no version
    rows = [{"id": "a", "prompt": "", "completion": "t", "input_sha256": "h"}]
    todo, _ = C.resume_todo(rows, out, "coherence_gate_version", "v6.1")
    assert [r["id"] for r in todo] == ["a"]


# ---------------------------------------------------- random controls -------
def test_layer_matched_neurons_matches_histogram_and_excludes_reference():
    ref = [(31, 5), (31, 9), (30, 1)]
    r1 = layer_matched_neurons(ref, intermediate_size=64, seed=0)
    r2 = layer_matched_neurons(ref, intermediate_size=64, seed=0)
    assert r1 == r2
    assert sorted(l for l, _ in r1) == sorted(l for l, _ in ref)
    assert not set(r1) & set(ref)


def test_layer_matched_heads_exclude_reference():
    ref = [(0, 1), (0, 2), (3, 4)]
    r = layer_matched(ref, n_heads_per_layer=4, seed=1, exclude_reference=True)
    assert sorted(l for l, _ in r) == [0, 0, 3] and not set(r) & set(ref)


def test_random_direction_norm_matched_and_seeded():
    v = torch.randn(32) * 5
    a, b = random_direction_like(v, 7), random_direction_like(v, 7)
    assert torch.allclose(a, b) and abs(float(a.norm()) - float(v.norm())) < 1e-4
    assert abs(float(torch.nn.functional.cosine_similarity(a, v, dim=0))) < 0.9


def test_relative_dose_scales_by_direction_norm():
    v = torch.ones(4) * 2.0                                          # ||v|| = 4
    absolute = build_activation_addition_cfg(direction=v, layer=0, alpha=-8.0)
    relative = build_activation_addition_cfg(direction=v, layer=0, alpha=-1.5,
                                             dose_mode="relative")
    assert absolute["alpha"] == -8.0 and absolute["dose_mode"] == "absolute"
    assert relative["alpha"] == pytest.approx(-6.0) and relative["direction_norm"] == pytest.approx(4.0)


# ----------------------------------------------------- full ablation --------
def _tiny(arch: str):
    tf = pytest.importorskip("transformers")
    common = dict(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
    if arch == "llama":
        return tf.LlamaForCausalLM(tf.LlamaConfig(**common)).eval()
    if not hasattr(tf, "Olmo3Config"):
        pytest.skip("transformers without Olmo3")
    return tf.Olmo3ForCausalLM(tf.Olmo3Config(**common)).eval()


def _residual_projections(model, v, mode):
    from safety_cot_heads.models.neuron_and_steer import SteeringController
    ctrl = SteeringController.attach(model)
    final_in = {}
    h = model.model.norm.register_forward_pre_hook(
        lambda m, a: final_in.__setitem__("x", a[0].detach()))
    try:
        with torch.no_grad(), ctrl.active({"mode": mode, "direction": v,
                                           "layers": list(range(3)), "alpha": 1.0}):
            out = model(torch.randint(0, 64, (2, 7)), output_hidden_states=True)
    finally:
        h.remove()
        ctrl.detach()
    vh = v / v.norm()
    per_layer = [float((hs @ vh).abs().max()) for hs in out.hidden_states[1:-1]]
    return per_layer + [float((final_in["x"] @ vh).abs().max())]


@pytest.mark.parametrize("arch", ["llama", "olmo3"])
def test_ablate_all_removes_direction_from_every_residual_position(arch):
    torch.manual_seed(0)
    model = _tiny(arch)
    v = torch.randn(32)
    full = _residual_projections(model, v, "ablate_all")
    legacy = _residual_projections(model, v, "ablate")
    assert max(full) < 1e-4, full
    # the layer-input-only ablation leaves the direction in the last residual
    assert legacy[-1] > 1e-3, legacy
