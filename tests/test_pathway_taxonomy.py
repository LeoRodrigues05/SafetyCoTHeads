"""Unit tests for the pathway-taxonomy aggregator (Direction A v4)."""
from __future__ import annotations

from safety_cot_heads.direction_a.pathway_taxonomy import (
    PATHWAY_VECTOR_FIELDS, pathway_vector, summarise_pathways,
)
from safety_cot_heads.judging.judge_prompts import PATHWAY_LABELS


def _zero_labels() -> dict:
    return {lbl: False for lbl in PATHWAY_LABELS}


def _row(parent: str, idx: int, *, labels: dict, condition: str = "C") -> dict:
    return {
        "id": f"{parent}::p{idx:03d}::think",
        "parent_id": parent,
        "traj_prefix_idx": idx,
        "traj_prefix_kind": "think",
        "condition": condition,
        "model": "test",
        "dataset": "test",
        "seed": 0,
        "judge_flat": {"pathway_labels": labels},
    }


def test_empty_input_returns_no_vectors() -> None:
    assert pathway_vector([]) == []


def test_direct_execution_completion() -> None:
    # 4 prefixes, all execution-positive, no recognition / rationalisation
    rows = []
    for i in range(4):
        labs = _zero_labels()
        labs["task_decomposition"] = True
        labs["operational_detail"] = True
        rows.append(_row("p0", i, labels=labs))
    vecs = pathway_vector(rows)
    assert len(vecs) == 1
    v = vecs[0]
    for f in PATHWAY_VECTOR_FIELDS:
        assert f in v
    assert v["execution_rate"] == 1.0
    assert v["operational_detail_rate"] == 1.0
    assert v["recognition_rate"] == 0.0
    assert v["rationalisation_rate"] == 0.0
    assert v["first_execution_idx_norm"] == 0.0
    assert v["dominant_pathway"] in ("direct_execution", "recognition_loss")


def test_refusal_loss_pathway() -> None:
    # refusal initiated at idx 1, suppressed at idx 3, execution at idx 4
    rows = []
    labs0 = _zero_labels(); labs0["risk_acknowledgment"] = True
    labs1 = _zero_labels(); labs1["refusal_initiation"] = True
    labs2 = _zero_labels()
    labs3 = _zero_labels(); labs3["refusal_suppression"] = True
    labs4 = _zero_labels(); labs4["operational_detail"] = True
    for i, l in enumerate([labs0, labs1, labs2, labs3, labs4]):
        rows.append(_row("p1", i, labels=l))
    v = pathway_vector(rows)[0]
    assert v["refusal_initiated"] == 1
    assert v["refusal_suppressed"] == 1
    assert v["dominant_pathway"] == "refusal_loss"


def test_rationalised_compliance() -> None:
    # ≥30% rationalisation AND ≥30% execution; recognition present so
    # ``recognition_loss`` precedence does not fire.
    rows = []
    for i in range(10):
        labs = _zero_labels()
        if i < 2:
            labs["risk_acknowledgment"] = True
        if i < 4:
            labs["educational_framing"] = True
        if i >= 6:
            labs["procedural_planning"] = True
        rows.append(_row("p2", i, labels=labs))
    v = pathway_vector(rows)[0]
    assert v["rationalisation_rate"] >= 0.3
    assert v["execution_rate"] >= 0.3
    assert v["recognition_rate"] >= 0.1
    assert v["dominant_pathway"] == "rationalised_compliance"


def test_recognition_to_execution_ordering() -> None:
    rows = [
        _row("p3", 0, labels={**_zero_labels(), "risk_acknowledgment": True}),
        _row("p3", 1, labels={**_zero_labels(), "operational_detail": True}),
    ]
    v = pathway_vector(rows)[0]
    assert v["recognition_to_execution"] == 1


def test_unparsed_rows_handled() -> None:
    rows = [{"id": "p4::p000::think", "parent_id": "p4",
             "traj_prefix_idx": 0, "traj_prefix_kind": "think",
             "condition": "C", "judge_flat": None}]
    v = pathway_vector(rows)[0]
    assert v["n_parsed"] == 0
    assert v["dominant_pathway"] == "none"


def test_summarise_groups_by_condition() -> None:
    labs_exec = _zero_labels(); labs_exec["operational_detail"] = True
    rows = [
        _row("pA", 0, labels=labs_exec, condition="cond_a"),
        _row("pB", 0, labels=_zero_labels(), condition="cond_b"),
    ]
    s = summarise_pathways(pathway_vector(rows))
    assert set(s.keys()) == {"cond_a", "cond_b"}
    assert s["cond_a"]["mean"]["operational_detail_rate"] == 1.0
    assert s["cond_b"]["mean"]["operational_detail_rate"] == 0.0
