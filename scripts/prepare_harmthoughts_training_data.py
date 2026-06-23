"""Prepare HarmThoughts → pathway-judge training JSONL.

Downloads ishitakakkar-10/HarmThoughts from HuggingFace, reconstructs
cumulative prefixes from sentence-level annotations, maps HarmThoughts
labels to the repo's 12 pathway labels, and emits training examples in
the same format used by train_pathway_judge.py.

Output schema per row:
    {"user_prompt": <formatted SINGLE_LABEL_JUDGE_PROMPT>,
     "target_json": {"label_present": bool, "confidence": 0.9,
                     "rationale": "annotated in HarmThoughts"}}

Rows include both positive examples from mapped HarmThoughts annotations and
negative examples sampled from other trained pathway labels.

Usage:
    # Print unique llm_annotation values to verify abbreviation mapping:
    python scripts/prepare_harmthoughts_training_data.py --print-annotations

    # Dry-run (sample 50 traces, print stats, do not write files):
    python scripts/prepare_harmthoughts_training_data.py --dry-run

    # Full run (writes train + test JSONL):
    python scripts/prepare_harmthoughts_training_data.py \
        --out-train data/pathway_judge_train.jsonl \
        --out-test  data/pathway_judge_test.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from safety_cot_heads.judging.judge_prompts import (  # noqa: E402
    PATHWAY_LABELS,
    build_single_label_prompt,
)
from safety_cot_heads.utils import get_logger  # noqa: E402

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# HarmThoughts llm_annotation abbreviation → repo label(s)
#
# Each HT sentence can carry one annotation.  The mapping may produce
# multiple repo labels for a single HT label (e.g. DKE → both
# procedural_planning and operational_detail).
#
# Unmapped repo labels (policy_awareness, refusal_maintenance,
# legal_disclaimer_laundering) are intentionally omitted — no HT signal
# for them.
#
# Run --print-annotations first on a freshly downloaded dataset to verify
# these abbreviations are correct before generating training data.
# ---------------------------------------------------------------------------
HT_LABEL_TO_REPO: dict[str, list[str]] = {
    # Strong matches
    "RA":  ["risk_acknowledgment"],          # Risk Acknowledgement
    "IA":  ["intent_assessment"],            # Intent Assessment
    "RS":  ["refusal_suppression"],          # Refusal Suppression
    "IR":  ["benign_motive_invention"],      # Intent Rationalization
    "TD":  ["task_decomposition"],           # Task Decomposition
    # Partial matches — refusal_initiation
    "CC":  ["refusal_initiation"],           # Compliance Check
    "AL":  ["refusal_initiation"],           # Safe Alternative (dataset abbreviates as AL)
    # Partial matches — educational_framing (3 HT sources)
    "CR":  ["educational_framing"],          # Constraint Reframing
    "PA":  ["educational_framing"],          # Persona Adoption
    "CE":  ["educational_framing"],          # Challenge Engineering
    # Partial matches — operational_detail / procedural_planning
    # DKE = Domain Knowledge Extension (paper: "Domain Knowledge Synthesis", dataset: DKE)
    # This is by far the largest category (22,310 rows) — must not be skipped
    "DKE": ["procedural_planning", "operational_detail"],
    "HV":  ["operational_detail"],           # Harmful Verification
    # Partial match — risk_acknowledgment (additional source)
    "ED":  ["risk_acknowledgment"],          # Ethical Deliberation
    # Excluded: PS (Performative Safety) — ambiguous
    # Excluded: OB (Obfuscation) — no clean match
    # Excluded: FL (Filler sentences, dataset abbreviation for "FS") — neutral, no label
}

# Repo labels with no HT training signal (rely on base model knowledge)
UNTRAINED_LABELS = {"refusal_maintenance", "legal_disclaimer_laundering", "policy_awareness"}

# Fixed confidence for HT-derived labels (human-annotated, treat as high quality)
HT_CONFIDENCE = 0.9


def _load_dataset() -> list[dict]:
    """Download HarmThoughts from HuggingFace and return all rows."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.error("datasets package not found — run: pip install datasets")
        raise
    log.info("downloading ishitakakkar-10/HarmThoughts ...")
    ds = load_dataset("ishitakakkar-10/HarmThoughts", split="train")
    rows = list(ds)
    log.info("loaded %d sentence rows from HarmThoughts", len(rows))
    return rows


def _group_by_trace(rows: list[dict]) -> dict[str, list[dict]]:
    """Group rows by trace id, sorted by sentence_id."""
    traces: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        traces[r["id"]].append(r)
    for tid in traces:
        traces[tid].sort(key=lambda r: r["sentence_id"])
    return dict(traces)


def _build_examples(
    traces: dict[str, list[dict]],
    *,
    seed: int = 42,
    test_fraction: float = 0.1,
    negatives_per_positive: int = 2,
) -> tuple[list[dict], list[dict], dict[str, int]]:
    """Build (train, test) example lists from HarmThoughts traces.

    For each annotated sentence:
      - Reconstruct cumulative prefix.
      - Map HT annotation → one or more repo labels (positive examples).
      - Sample up to `negatives_per_positive * len(pos_labels)` other trained
        labels and emit negative examples (label_present=False).  This ensures
        the model sees both classes and doesn't drift toward always predicting
        True.

    Returns (train_examples, test_examples, label_counts).
    """
    trained_labels = [
        lbl for lbl in PATHWAY_LABELS if lbl not in UNTRAINED_LABELS
    ]

    trace_ids = sorted(traces.keys())
    rng = random.Random(seed)
    rng.shuffle(trace_ids)
    n_test = max(1, int(len(trace_ids) * test_fraction))
    test_ids = set(trace_ids[:n_test])

    train: list[dict] = []
    test: list[dict] = []
    skipped_annotations: set[str] = set()
    label_counts: dict[str, int] = defaultdict(int)

    for tid, sentences in traces.items():
        bucket = test if tid in test_ids else train
        query = sentences[0].get("query", "")
        cumulative: list[str] = []
        for sent in sentences:
            cumulative.append(sent.get("sentence", "").strip())
            prefix_text = " ".join(cumulative)
            annotation = (sent.get("llm_annotation") or "").strip()
            if not annotation or annotation not in HT_LABEL_TO_REPO:
                # FL = Filler (neutral), PS = Performative Safety, OB = Obfuscation
                if annotation and annotation not in ("FL", "PS", "OB", ""):
                    skipped_annotations.add(annotation)
                continue

            pos_labels = HT_LABEL_TO_REPO[annotation]

            # Positive examples
            for repo_label in pos_labels:
                user_prompt = build_single_label_prompt(
                    "pathway_single",
                    repo_label,
                    user_query=query,
                    model_response=prefix_text,
                )
                bucket.append({
                    "user_prompt": user_prompt,
                    "target_json": {
                        "label_present": True,
                        "confidence": HT_CONFIDENCE,
                        "rationale": "annotated in HarmThoughts",
                    },
                    "_meta": {
                        "trace_id": tid,
                        "sentence_id": sent.get("sentence_id"),
                        "ht_annotation": annotation,
                        "repo_label": repo_label,
                        "positive": True,
                    },
                })
                label_counts[repo_label] += 1

            # Negative examples — sample other trained labels.  When an HT
            # annotation maps to multiple repo labels, scale the negative
            # sample accordingly so --negatives-per-positive is literal.
            neg_pool = [lbl for lbl in trained_labels if lbl not in pos_labels]
            n_neg = max(0, int(negatives_per_positive)) * len(pos_labels)
            neg_sample = rng.sample(neg_pool, min(n_neg, len(neg_pool)))
            for neg_label in neg_sample:
                user_prompt = build_single_label_prompt(
                    "pathway_single",
                    neg_label,
                    user_query=query,
                    model_response=prefix_text,
                )
                bucket.append({
                    "user_prompt": user_prompt,
                    "target_json": {
                        "label_present": False,
                        "confidence": 0.8,
                        "rationale": "not annotated for this label in HarmThoughts",
                    },
                    "_meta": {
                        "trace_id": tid,
                        "sentence_id": sent.get("sentence_id"),
                        "ht_annotation": annotation,
                        "repo_label": neg_label,
                        "positive": False,
                    },
                })

    if skipped_annotations:
        log.warning(
            "skipped %d unknown HT annotation values: %s",
            len(skipped_annotations),
            sorted(skipped_annotations),
        )
    return train, test, label_counts


def _write_jsonl(path: Path, rows: list[dict], strip_meta: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            out = {k: v for k, v in r.items() if not (strip_meta and k == "_meta")}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    log.info("wrote %d rows → %s", len(rows), path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert HarmThoughts → pathway judge training JSONL")
    ap.add_argument("--out-train", type=Path, default=Path("data/pathway_judge_train.jsonl"))
    ap.add_argument("--out-test", type=Path, default=Path("data/pathway_judge_test.jsonl"))
    ap.add_argument("--test-fraction", type=float, default=0.1,
                    help="Fraction of traces held out for test (default 0.1)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print-annotations", action="store_true",
                    help="Print unique llm_annotation values and exit (use to verify mapping)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build examples but do not write files; print statistics")
    ap.add_argument("--negatives-per-positive", type=int, default=2,
                    help="Negative examples per positive example (default 2, set 0 to disable)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N traces (for quick testing)")
    args = ap.parse_args()

    rows = _load_dataset()

    if args.print_annotations:
        annotations = sorted({r.get("llm_annotation", "") for r in rows})
        print("Unique llm_annotation values:")
        for a in annotations:
            count = sum(1 for r in rows if r.get("llm_annotation") == a)
            mapped = HT_LABEL_TO_REPO.get(a, [])
            print(f"  {a!r:8s}  count={count:6d}  mapped_to={mapped or '(skipped)'}")
        return 0

    traces = _group_by_trace(rows)
    log.info("grouped into %d traces", len(traces))

    if args.limit:
        limited_ids = sorted(traces)[:args.limit]
        traces = {k: traces[k] for k in limited_ids}
        log.info("limiting to %d traces", len(traces))

    train, test, label_counts = _build_examples(
        traces,
        seed=args.seed,
        test_fraction=args.test_fraction,
        negatives_per_positive=args.negatives_per_positive,
    )

    log.info("examples: train=%d  test=%d", len(train), len(test))
    n_positive = sum(1 for r in train + test if r["target_json"]["label_present"])
    n_negative = len(train) + len(test) - n_positive
    log.info("examples by class: positive=%d  negative=%d", n_positive, n_negative)
    log.info("positive per-label counts (train+test):")
    for label in PATHWAY_LABELS:
        n = label_counts.get(label, 0)
        tag = "(untrained — no HT signal)" if label in UNTRAINED_LABELS else ""
        log.info("  %-35s %5d  %s", label, n, tag)

    if args.dry_run:
        log.info("DRY-RUN: skipping file writes")
        return 0

    _write_jsonl(args.out_train, train)
    _write_jsonl(args.out_test, test)
    n_trained = len(PATHWAY_LABELS) - len(UNTRAINED_LABELS)
    log.info("done — %d train / %d test examples across %d trained pathway labels",
             len(train), len(test), n_trained)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
