"""Dual-judge driver (P1.2.3 — stub).

This module wraps two independent judge models behind a single interface so
the trajectory pipeline can later compute judge-vs-judge agreement without
plumbing a second model through every script. The **active default is
single-judge** (Qwen2.5-32B only, per D7 in
`docs/direction_a/direction_a_plan.md`); the secondary judge is only invoked
when ``cfg.enable_secondary=True``.

Both judges run the same prompt template and the same parser
(:func:`safety_cot_heads.judging.parse_judge_json`), then we surface the raw
label sets side-by-side and a per-row agreement flag. We do *not* try to
merge the labels — downstream callers decide whether to take the primary,
the intersection, or report disagreement explicitly.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .judge import JudgeConfig, judge_rows
from ..models import load_model


@dataclass
class DualJudgeConfig:
    """Configuration for the dual-judge driver.

    Attributes
    ----------
    primary
        Fully-specified :class:`JudgeConfig` for the primary judge (Qwen-2.5-32B
        in the locked Phase 1 protocol).
    primary_model
        HF model id of the primary judge.
    secondary
        Optional :class:`JudgeConfig` for the secondary judge. When
        ``enable_secondary`` is False this is ignored.
    secondary_model
        Optional HF model id of the secondary judge (e.g. ``mistralai/Mistral-Large-Instruct-2407``
        or ``meta-llama/Llama-3.1-70B-Instruct``).
    enable_secondary
        Master switch. Default ``False`` — Phase 1 ships single-judge.
    sentinel_field
        Top-level field name written onto each row with the secondary judge's
        parsed output. Defaults to ``"judge_flat_secondary"``.
    """

    primary: JudgeConfig = field(default_factory=JudgeConfig)
    primary_model: str = "Qwen/Qwen2.5-32B-Instruct"
    secondary: Optional[JudgeConfig] = None
    secondary_model: Optional[str] = None
    enable_secondary: bool = False
    sentinel_field: str = "judge_flat_secondary"


class DualJudgeDriver:
    """Run one or two judges over a row set and surface side-by-side labels.

    Usage::

        cfg = DualJudgeConfig(primary=JudgeConfig(...))
        driver = DualJudgeDriver(cfg)
        rows = driver.judge(rows)               # primary only (Phase 1 default)

    To enable the secondary on a validation subset (D7 / §13.4)::

        cfg.enable_secondary = True
        cfg.secondary = JudgeConfig(kind="safety", ...)
        cfg.secondary_model = "meta-llama/Llama-3.1-70B-Instruct"
        driver = DualJudgeDriver(cfg)
        rows = driver.judge(rows)
        agree = driver.agreement_table(rows)    # {label: fraction_agree}

    The driver instantiates judges lazily so a Phase 1 single-judge run never
    pays the secondary-model load cost.
    """

    def __init__(self, cfg: DualJudgeConfig):
        self.cfg = cfg
        self._primary = None
        self._secondary = None

    def _load_primary(self):
        if self._primary is None:
            self._primary = load_model(self.cfg.primary_model,
                                       dtype="auto", load_in_4bit=True,
                                       device_map="auto")
        return self._primary

    def _load_secondary(self):
        if not self.cfg.enable_secondary:
            return None
        if self.cfg.secondary is None or self.cfg.secondary_model is None:
            raise ValueError(
                "enable_secondary=True requires both `secondary` and `secondary_model`."
            )
        if self._secondary is None:
            self._secondary = load_model(self.cfg.secondary_model,
                                         dtype="auto", load_in_4bit=True,
                                         device_map="auto")
        return self._secondary

    def judge(self, rows: Sequence[dict]) -> list[dict]:
        """Judge ``rows`` with the primary (and optionally secondary).

        Returns a new list of rows; never mutates inputs.
        """
        primary = self._load_primary()
        out = judge_rows(primary, list(rows), self.cfg.primary)
        if not self.cfg.enable_secondary:
            return out
        secondary = self._load_secondary()
        sec_rows = judge_rows(secondary, list(rows), self.cfg.secondary)
        sec_by_id = {r["id"]: r for r in sec_rows}
        merged: list[dict] = []
        for r in out:
            srow = sec_by_id.get(r["id"], {})
            r2 = dict(r)
            r2[self.cfg.sentinel_field] = {
                k: v for k, v in srow.items()
                if k.startswith("judge_") or k.startswith("parsed_")
            }
            r2[f"{self.cfg.sentinel_field}_model"] = self.cfg.secondary_model
            merged.append(r2)
        return merged

    @staticmethod
    def agreement_table(rows: Sequence[dict],
                        secondary_field: str = "judge_flat_secondary",
                        label_field: str = "judge_flat") -> dict[str, float]:
        """Per-label exact-match agreement between primary and secondary.

        Operates on the flat 5-label safety schema (booleans). Rows missing
        the secondary block contribute neither numerator nor denominator.
        Returns ``{label_name: fraction_agreement}`` plus ``"_n"`` (rows used).
        """
        from .judge_prompts import LABELS

        agree = {lbl: 0 for lbl in LABELS}
        denom = 0
        for r in rows:
            sec = r.get(secondary_field)
            pri = r.get(label_field)
            if not (isinstance(sec, dict) and isinstance(pri, dict)):
                continue
            sec_labels = sec.get("labels") if "labels" in sec else sec.get(label_field, {}).get("labels", {})
            pri_labels = pri.get("labels", {})
            if not isinstance(sec_labels, dict) or not isinstance(pri_labels, dict):
                continue
            denom += 1
            for lbl in LABELS:
                if bool(pri_labels.get(lbl)) == bool(sec_labels.get(lbl)):
                    agree[lbl] += 1
        if denom == 0:
            return {"_n": 0, **{lbl: float("nan") for lbl in LABELS}}
        return {"_n": denom, **{lbl: agree[lbl] / denom for lbl in LABELS}}


__all__ = ["DualJudgeConfig", "DualJudgeDriver"]
