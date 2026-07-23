#!/usr/bin/env python3
"""Aggregate the 12 fine-grained pathway labels from the in-repo judge output and
render per-family radar (spider) plots of the pathway signature.

Unlike ``aggregate_pathway_labels.py`` (which targets the raw HuggingFace layout),
the per-query labels are already checked out in this repo under

    runs/direction_a_v5/<model>/judge/<dataset>/<condition>/seed0/judge_pathway.jsonl

where each row is one CoT-prefix carrying ``judge_flat.pathway_labels`` (a
{label: bool} dict).  We therefore read them straight from disk -- no HF download.

For each (model, dataset, condition) cell we take the fraction of parsed pathway
rows exhibiting each label; per condition we then average the two datasets
(equal-weight "dataset mean", matching the composite_* figures).  Output is a
tidy CSV plus a 2x2 radar grid (one panel per intervention family, baseline
overlaid as a grey reference).

Usage:
    .venv/bin/python -m scripts.plot_pathway_radar            # olmo3_7b_think
    .venv/bin/python -m scripts.plot_pathway_radar --model qwen3_8b
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
JUDGE_ROOT = ROOT / "runs" / "direction_a_v5"
PLOTS_DEFAULT = ROOT / "runs" / "plots"

# The 12 fine-grained pathway labels, ordered so the radar sweeps recognition ->
# refusal -> rationalisation -> execution (the safety-to-harm progression).
LABELS = [
    "risk_acknowledgment", "policy_awareness", "intent_assessment",
    "refusal_initiation", "refusal_maintenance", "refusal_suppression",
    "benign_motive_invention", "educational_framing", "legal_disclaimer_laundering",
    "task_decomposition", "procedural_planning", "operational_detail",
]
SHORT = {
    "risk_acknowledgment": "risk\nack", "policy_awareness": "policy\naware",
    "intent_assessment": "intent\nassess", "refusal_initiation": "refusal\ninit",
    "refusal_maintenance": "refusal\nmaint", "refusal_suppression": "refusal\nsuppr",
    "benign_motive_invention": "benign\nmotive", "educational_framing": "edu\nframing",
    "legal_disclaimer_laundering": "legal\nlaunder", "task_decomposition": "task\ndecomp",
    "procedural_planning": "proc\nplan", "operational_detail": "op\ndetail",
}

# Family -> ordered conditions (weak -> strong).  Baseline is overlaid on every panel.
FAMILY_CONDS = {
    "Steering": ["steering_a0.5", "steering_a1.0", "steering_a1.5"],
    "SHIPS (heads)": ["ships_top3", "ships_top5", "ships_top8"],
    "Neuron": ["neurons_top256", "neurons_top512", "neurons_top1024"],
    "Directional ablation": ["steering_ablate"],
}
COND_LABEL = {
    "steering_a0.5": "a0.5", "steering_a1.0": "a1.0", "steering_a1.5": "a1.5",
    "ships_top3": "top3", "ships_top5": "top5", "ships_top8": "top8",
    "neurons_top256": "top256", "neurons_top512": "top512", "neurons_top1024": "top1024",
    "steering_ablate": "ablate",
}
# sequential dose colours (weak -> strong) reused per family
DOSE_COLORS = ["#9ecae1", "#4292c6", "#08519c"]

# For the single-circle overlay: one representative condition per family
# (the strongest dose), each in its own family colour.
OVERLAY_REPS = [
    ("steering_a1.5", "Steering a1.5", "#1f77b4"),
    ("steering_ablate", "Dir. ablation", "#ff7f0e"),
    ("ships_top8", "SHIPS top8", "#2ca02c"),
    ("neurons_top1024", "Neuron top1024", "#9467bd"),
]

# --- Cross-model reasoning-vs-non-reasoning radar ---------------------------
# Collapse the 12 fine-grained labels into 6 grouped axes ordered along the
# recognise -> refuse -> comply arc (each axis = mean rate of its members).
GROUPS_6 = [
    ("recognition\n(risk/policy)", ["risk_acknowledgment", "policy_awareness"]),
    ("intent\nassessment", ["intent_assessment"]),
    ("refusal\ninitiation", ["refusal_initiation", "refusal_maintenance"]),
    ("refusal\nsuppression", ["refusal_suppression"]),
    ("rationali-\nsation", ["benign_motive_invention", "educational_framing",
                            "legal_disclaimer_laundering"]),
    ("execution", ["task_decomposition", "procedural_planning", "operational_detail"]),
]

# Two model classes. Colours are the Okabe-Ito CVD-safe blue/orange pair; each
# class member gets a distinct line style so all six models stay on one circle.
REASONING_CLASS = {
    "color": "#1f77b4",
    "models": [
        ("olmo3_7b_think", "OLMo-3 Think", "-"),
        ("qwen3_8b", "Qwen3-8B", "--"),
    ],
}
NONREASONING_CLASS = {
    "color": "#d95f02",
    "models": [
        ("olmo3_7b_base", "OLMo-3 Base", "-"),
        ("olmo3_7b_base_own", "OLMo-3 Base (own dir.)", "--"),
        ("llama31_8b_control", "Llama-3.1 (control)", ":"),
    ],
}


def model_group_vector(model: str) -> np.ndarray | None:
    """Per-model 6-group profile: average each label over all conditions
    (dataset-mean already applied by ``aggregate``), then group into 6 axes."""
    try:
        agg, _, _ = aggregate(model)
    except Exception:
        return None
    if not agg:
        return None
    lab_mean = {l: float(np.mean([agg[c][l] for c in agg])) for l in LABELS}
    return np.array([float(np.mean([lab_mean[l] for l in members]))
                     for _, members in GROUPS_6])


def plot_reasoning_radar(plots_dir: Path) -> Path | None:
    """One circle, six grouped axes, all models overlaid and coloured by class.
    Thin styled lines are individual models; the thick filled line per class is
    the class mean — the two class means trace opposite shapes."""
    labels = [name for name, _ in GROUPS_6]
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)

    vectors: dict[str, np.ndarray] = {}
    for cls in (REASONING_CLASS, NONREASONING_CLASS):
        for key, _, _ in cls["models"]:
            v = model_group_vector(key)
            if v is not None:
                vectors[key] = v
    if not vectors:
        print("[error] no per-model vectors for reasoning radar")
        return None
    rmax = min(1.0, 0.05 * np.ceil(max(v.max() for v in vectors.values()) / 0.05) + 0.05)

    fig, ax = plt.subplots(figsize=(9.5, 9.5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(ang)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, rmax)
    ax.set_rlabel_position(0)
    ax.tick_params(axis="y", labelsize=8, colors="0.5")
    ax.grid(alpha=.4)

    a_closed = np.concatenate([ang, ang[:1]])

    def draw(vec, color, label, lw, alpha, ls="-", ms=0, fill=False):
        vals = np.concatenate([vec, vec[:1]])
        ax.plot(a_closed, vals, color=color, lw=lw, ls=ls, label=label,
                alpha=alpha, marker="o", ms=ms)
        if fill:
            ax.fill(a_closed, vals, color=color, alpha=0.10)

    for cls, cname in ((REASONING_CLASS, "Reasoning models"),
                       (NONREASONING_CLASS, "Non-reasoning models")):
        color = cls["color"]
        member_vecs = []
        for key, disp, ls in cls["models"]:
            if key not in vectors:
                continue
            member_vecs.append(vectors[key])
            draw(vectors[key], color, disp, lw=1.4, alpha=0.55, ls=ls)
        if not member_vecs:
            continue
        mean = np.mean(member_vecs, axis=0)
        draw(mean, color, f"{cname} (mean)", lw=3.2, alpha=1.0, ms=5, fill=True)

    # Two-tier legend: class means bold, members light.
    ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.10), fontsize=9,
              frameon=True, framealpha=0.9)
    fig.suptitle("Pathway signature — reasoning vs non-reasoning\n"
                 "(6 grouped rate axes, recognise → refuse → comply arc; "
                 "mean over conditions & datasets)",
                 fontweight="bold", fontsize=13, y=1.03)
    fig.tight_layout()
    path = plots_dir / "pathway_radar_reasoning.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


def cell_rates(fp: str) -> tuple[dict[str, float], int] | None:
    """{label: fraction of rows with label True}, n_rows for one judge_pathway.jsonl."""
    sums: dict[str, float] = defaultdict(float)
    n = 0
    with open(fp) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            labels = row.get("judge_flat", {}).get("pathway_labels")
            if not isinstance(labels, dict):
                continue
            n += 1
            for lab in LABELS:
                if labels.get(lab) is True:
                    sums[lab] += 1.0
    if n == 0:
        return None
    return {lab: sums[lab] / n for lab in LABELS}, n


def aggregate(model: str) -> dict[str, dict[str, float]]:
    """condition -> {label: dataset-mean rate}, averaging equal-weight over datasets."""
    base = JUDGE_ROOT / model / "judge"
    per_ds: dict[tuple[str, str], dict[str, float]] = {}
    counts: dict[tuple[str, str], int] = {}
    for fp in glob.glob(str(base / "*/*/seed0/judge_pathway.jsonl")):
        parts = fp.split(os.sep)
        ds, cond = parts[-4], parts[-3]
        res = cell_rates(fp)
        if res is None:
            continue
        rates, n = res
        per_ds[(ds, cond)] = rates
        counts[(ds, cond)] = n
    conds = sorted({c for _, c in per_ds})
    out: dict[str, dict[str, float]] = {}
    for cond in conds:
        ds_rates = [per_ds[(ds, cond)] for ds in {d for d, c in per_ds if c == cond}]
        out[cond] = {lab: float(np.mean([r[lab] for r in ds_rates])) for lab in LABELS}
    return out, per_ds, counts


def write_csv(model: str, per_ds: dict, counts: dict, out_csv: Path) -> None:
    cols = ["model", "dataset", "condition"] + LABELS + ["n_rows"]
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for (ds, cond) in sorted(per_ds):
            r = per_ds[(ds, cond)]
            w.writerow([model, ds, cond] + [f"{r[l]:.4f}" for l in LABELS]
                       + [counts[(ds, cond)]])
    print(f"  wrote {out_csv.relative_to(ROOT)}")


def _radar(ax, values, color, label, fill=False, ls="-"):
    ang = np.linspace(0, 2 * np.pi, len(LABELS), endpoint=False)
    ang = np.concatenate([ang, ang[:1]])
    vals = np.concatenate([values, values[:1]])
    ax.plot(ang, vals, color=color, lw=2, ls=ls, label=label, marker="o", ms=3)
    if fill:
        ax.fill(ang, vals, color=color, alpha=0.08)


def plot_radars(model: str, agg: dict, plots_dir: Path, rmax: float) -> Path:
    fams = list(FAMILY_CONDS)
    fig, axes = plt.subplots(2, 2, figsize=(13, 13),
                             subplot_kw=dict(polar=True))
    ang = np.linspace(0, 2 * np.pi, len(LABELS), endpoint=False)
    base = agg.get("baseline")
    for ax, fam in zip(axes.flat, fams):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(ang)
        ax.set_xticklabels([SHORT[l] for l in LABELS], fontsize=8)
        ax.set_ylim(0, rmax)
        ax.set_rlabel_position(0)
        ax.tick_params(axis="y", labelsize=7, colors="0.5")
        ax.grid(alpha=.4)
        if base is not None:
            _radar(ax, np.array([base[l] for l in LABELS]), "0.5",
                   "baseline", ls="--")
        for cond, color in zip(FAMILY_CONDS[fam], DOSE_COLORS):
            if cond not in agg:
                continue
            _radar(ax, np.array([agg[cond][l] for l in LABELS]), color,
                   COND_LABEL.get(cond, cond), fill=(len(FAMILY_CONDS[fam]) == 1))
        ax.set_title(fam, fontweight="bold", fontsize=13, pad=22)
        ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10), fontsize=8)
    fig.suptitle(f"Pathway-label signature — {model}\n"
                 "(rate over parsed CoT-prefixes, dataset mean)",
                 fontweight="bold", fontsize=15, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = plots_dir / f"pathway_radar_{model}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


def plot_overlay_radar(model: str, agg: dict, plots_dir: Path, rmax: float) -> Path:
    """Single circle: baseline + one representative (strongest) condition per
    family, coloured by family.  Deliberately not every condition -- overlaying
    all 11 is unreadable; this keeps the cross-family comparison legible."""
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ang = np.linspace(0, 2 * np.pi, len(LABELS), endpoint=False)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(ang)
    ax.set_xticklabels([SHORT[l] for l in LABELS], fontsize=9)
    ax.set_ylim(0, rmax)
    ax.set_rlabel_position(0)
    ax.tick_params(axis="y", labelsize=7, colors="0.5")
    ax.grid(alpha=.4)
    if "baseline" in agg:
        _radar(ax, np.array([agg["baseline"][l] for l in LABELS]), "0.45",
               "baseline", ls="--")
    for cond, label, color in OVERLAY_REPS:
        if cond not in agg:
            continue
        _radar(ax, np.array([agg[cond][l] for l in LABELS]), color, label,
               fill=True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.08), fontsize=9)
    fig.suptitle(f"Pathway-label signature — {model}\n"
                 "baseline vs. strongest per family (rate over CoT-prefixes, "
                 "dataset mean)", fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    path = plots_dir / f"pathway_radar_overlay_{model}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="olmo3_7b_think")
    ap.add_argument("--plots-dir", default=str(PLOTS_DEFAULT))
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--rmax", type=float, default=0.7,
                    help="radial axis max (labels are rates in [0,1])")
    ap.add_argument("--reasoning", action="store_true",
                    help="render the cross-model reasoning-vs-non-reasoning "
                         "radar (all models on one circle) and exit")
    args = ap.parse_args()

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if args.reasoning:
        return 0 if plot_reasoning_radar(plots_dir) is not None else 1

    out_csv = Path(args.out_csv) if args.out_csv else \
        plots_dir / f"pathway_labels_{args.model}.csv"

    agg, per_ds, counts = aggregate(args.model)
    if not agg:
        print(f"[error] no pathway rows under {JUDGE_ROOT / args.model / 'judge'}")
        return 1
    print(f"loaded {len(per_ds)} (dataset,condition) cells for {args.model}")
    write_csv(args.model, per_ds, counts, out_csv)
    plot_radars(args.model, agg, plots_dir, args.rmax)
    plot_overlay_radar(args.model, agg, plots_dir, args.rmax)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
