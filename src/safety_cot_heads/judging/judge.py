"""LLM-as-judge runner.

Direct adaptation of ``CoT-safety/judge_test_script.py`` — the judge prompt is
preserved verbatim in :mod:`judge_prompts` and the per-row generation loop is
the same ``tokenizer → generate → decode`` pattern.  Differences:

* No hardcoded ``cuda:0`` / model name.
* Strict JSONL output (one row per judged example) with full provenance.
* Up to ``max_retries`` retries with rising temperature on JSON parse failure.
* The flat normalised label/confidence dict from :func:`normalize_safety_labels`
  is stored alongside the raw output for fast downstream analysis.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
from tqdm import tqdm

from ..models import LoadedModel
from ..utils import now_iso
from .judge_prompts import (
    COHERENCE_HELPFULNESS_PROMPT,
    SAFETY_BEHAVIOR_PROMPT,
)
from .parse import normalize_safety_labels, parse_judge_json


@dataclass
class JudgeConfig:
    kind: str = "safety"                # "safety" | "coherence"
    max_new_tokens: int = 256
    base_temperature: float = 0.0
    retry_temperature: float = 0.3
    max_retries: int = 2
    seed: int = 0


@torch.no_grad()
def _generate_once(judge: LoadedModel, prompt: str, max_new_tokens: int,
                    temperature: float) -> str:
    enc = judge.tokenizer(prompt, return_tensors="pt", truncation=True).to(judge.device)
    do_sample = temperature > 0.0
    kw = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": judge.tokenizer.pad_token_id,
        "do_sample": do_sample,
    }
    if do_sample:
        kw["temperature"] = temperature
    out = judge.model.generate(**enc, **kw)
    gen = out[:, enc["input_ids"].shape[1]:]
    return judge.tokenizer.batch_decode(gen, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False)[0]


def _build_prompt(cfg: JudgeConfig, user_query: str, model_response: str) -> str:
    if cfg.kind == "safety":
        return SAFETY_BEHAVIOR_PROMPT.format(prompt=user_query, response=model_response)
    if cfg.kind == "coherence":
        return COHERENCE_HELPFULNESS_PROMPT.format(prompt=user_query, response=model_response)
    raise ValueError(f"unknown judge kind {cfg.kind!r}")


def judge_rows(judge: LoadedModel,
               rows: Sequence[dict],
               cfg: Optional[JudgeConfig] = None) -> list[dict]:
    """Judge a list of generation rows.

    Each input row must have ``{"id", "prompt", "completion", ...}``.  Output
    rows preserve the input id + add ``judge_*`` fields.
    """
    cfg = cfg or JudgeConfig()
    out: list[dict] = []
    for row in tqdm(rows, desc=f"judge[{cfg.kind}]"):
        jp = _build_prompt(cfg, row["prompt"], row["completion"])
        attempts = []
        parsed = None
        temps = [cfg.base_temperature] + [cfg.retry_temperature] * cfg.max_retries
        for t in temps:
            raw = _generate_once(judge, jp, cfg.max_new_tokens, t)
            res = parse_judge_json(raw)
            attempts.append({"temperature": t, "raw": raw,
                             "parse_status": res["parse_status"],
                             "error": res["error"]})
            if res["parse_status"] in ("ok", "recovered"):
                parsed = res["data"]
                break

        flat = normalize_safety_labels(parsed) if (parsed and cfg.kind == "safety") else None
        out.append({
            "id": row["id"],
            "dataset": row.get("dataset"),
            "category": row.get("category"),
            "condition": row.get("condition"),
            "model": row.get("model"),
            "judge_model": judge.name,
            "judge_kind": cfg.kind,
            "judge_attempts": attempts,
            "judge_parsed": parsed,
            "judge_flat": flat,
            "judge_parse_status": attempts[-1]["parse_status"],
            "timestamp": now_iso(),
        })
    return out


def aggregate_safety(rows: Iterable[dict]) -> dict:
    """Simple aggregate: per-condition harmful_rate / refusal_rate / coverage."""
    by_cond: dict[str, dict] = {}
    for r in rows:
        cond = r.get("condition") or "unknown"
        d = by_cond.setdefault(cond, {"n": 0, "parsed": 0, "harmful": 0, "safe": 0,
                                       "reasoning": 0, "adding_intent": 0,
                                       "changing_subject": 0})
        d["n"] += 1
        flat = r.get("judge_flat")
        if not flat:
            continue
        d["parsed"] += 1
        lab = flat["labels"]
        d["harmful"] += int(lab["harmful_response"])
        d["safe"] += int(lab["safe_rejection"])
        d["reasoning"] += int(lab["reasoning_about_safety"])
        d["adding_intent"] += int(lab["adding_intention"])
        d["changing_subject"] += int(lab["changing_subject"])

    summary = {}
    for cond, d in by_cond.items():
        n = max(1, d["parsed"])
        summary[cond] = {
            "n": d["n"], "n_parsed": d["parsed"],
            "harmful_rate": d["harmful"] / n,
            "refusal_rate": d["safe"] / n,
            "reasoning_rate": d["reasoning"] / n,
            "adding_intent_rate": d["adding_intent"] / n,
            "changing_subject_rate": d["changing_subject"] / n,
        }
    return summary
