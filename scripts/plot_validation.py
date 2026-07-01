"""Render human-vs-judge validation plots for an annotation batch.

Self-contained companion to ``scripts/score_annotations.py``: recomputes the
same human-vs-judge / inter-annotator Cohen's-kappa numbers straight from the
batch (blind ``tasks.json`` + held-out ``judge_labels.json`` + every
``annotations_<name>.jsonl``) and renders a small set of PNGs into
``<batch>/plots/``:

  01_judge_vs_human_ceiling      human-judge kappa vs the human-human ceiling
  02_reliable_subset_lift        judge kappa on all items vs both-humans-agree
  03_harmful_by_model            ASR-label kappa broken down by model
  04_safety_reasoning_categories Tier-2 any-SR + per-category kappa
  05_confusion_headline          human x judge confusion for the 2 headline labels

Only extra dependency is matplotlib (`pip install matplotlib`). The kappa math
is identical to score_annotations.py (hand-rolled, import-free).

Usage:
    python scripts/plot_validation.py --batch data/annotations/batch_v5_002
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

SAFETY_LABELS = ["harmful_response", "safe_rejection", "reasoning_about_safety",
                 "adding_intention", "changing_subject"]
COT_LABELS = ["cot_predicts_unsafe"]
LABELS_BY_TYPE = {"safety_5label": SAFETY_LABELS, "cot_only": COT_LABELS}
SR_CATEGORIES = ["risk_acknowledgment", "policy_boundary", "intent_assessment",
                 "refusal_reasoning", "safer_alternative", "other_safety_reasoning"]


# --------------------------------------------------------------------------- #
def metrics(gold: list[bool], pred: list[bool]) -> dict:
    """agreement + Cohen's kappa + confusion counts for aligned bool lists."""
    n = len(gold)
    if n == 0:
        return {"n": 0, "agree": 0.0, "kappa": 0.0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    tp = sum(g and p for g, p in zip(gold, pred))
    fp = sum(not g and p for g, p in zip(gold, pred))
    fn = sum(g and not p for g, p in zip(gold, pred))
    tn = sum(not g and not p for g, p in zip(gold, pred))
    agree = (tp + tn) / n
    p_e = ((tp + fp) / n) * ((tp + fn) / n) + ((tn + fn) / n) * ((tn + fp) / n)
    kappa = (agree - p_e) / (1 - p_e) if (1 - p_e) > 0 else (1.0 if agree == 1.0 else 0.0)
    return {"n": n, "agree": agree, "kappa": kappa,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def kcolor(k: float) -> str:
    if k < 0.0:  return "#d1495b"
    if k < 0.20: return "#e8813a"
    if k < 0.40: return "#e3b341"
    if k < 0.60: return "#b5c22e"
    if k < 0.80: return "#5aa02c"
    return "#2e7d32"


def parse_id(tid: str) -> dict:
    p = tid.split("::")
    return {"type": p[0], "model": p[1], "dataset": p[2], "cond": p[3]}


# --------------------------------------------------------------------------- #
def compute(batch: Path) -> dict:
    tasks = json.loads((batch / "tasks.json").read_text(encoding="utf-8"))
    judge = json.loads((batch / "judge_labels.json").read_text(encoding="utf-8"))
    T = {t["task_id"]: t for t in tasks}
    seg_by_task = {t["task_id"]: (t.get("segments") or [])
                   for t in tasks if t["task_type"] == "safety_reasoning"}

    recs = []
    for fp in sorted(glob.glob(str(batch / "annotations_*.jsonl"))):
        if fp.endswith(".tmp"):
            continue
        for line in Path(fp).read_text(encoding="utf-8").splitlines():
            if line.strip():
                recs.append(json.loads(line))
    if not recs:
        raise SystemExit(f"no annotations_*.jsonl in {batch} — nothing to plot.")

    annotators = sorted({r["annotator"] for r in recs})
    by_task: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in recs:
        by_task[r["task_id"]][r["annotator"]] = r.get("labels") or {}

    # pooled human-vs-judge (every annotator counts)
    pooled = {}
    for lbl in SAFETY_LABELS + COT_LABELS:
        g, p = [], []
        for tid, m in by_task.items():
            if lbl not in LABELS_BY_TYPE.get(T[tid]["task_type"], []):
                continue
            for ann in annotators:
                if ann in m:
                    hv, jv = m[ann].get(lbl), (judge.get(tid) or {}).get(lbl)
                    if isinstance(hv, bool) and isinstance(jv, bool):
                        g.append(hv); p.append(jv)
        if g:
            pooled[lbl] = metrics(g, p)

    # inter-annotator ceiling (first two annotators)
    inter = {}
    if len(annotators) >= 2:
        a, b = annotators[0], annotators[1]
        for lbl in SAFETY_LABELS + COT_LABELS:
            ga, gb = [], []
            for tid, m in by_task.items():
                if a in m and b in m:
                    va, vb = m[a].get(lbl), m[b].get(lbl)
                    if isinstance(va, bool) and isinstance(vb, bool):
                        ga.append(va); gb.append(vb)
            if ga:
                inter[lbl] = metrics(ga, gb)

    # consensus (both agree) judge-validity
    consensus = {}
    if len(annotators) >= 2:
        a, b = annotators[0], annotators[1]
        for lbl in SAFETY_LABELS + COT_LABELS:
            gold, pred = [], []
            for tid, m in by_task.items():
                if a not in m or b not in m:
                    continue
                va, vb = m[a].get(lbl), m[b].get(lbl)
                jv = (judge.get(tid) or {}).get(lbl)
                if isinstance(va, bool) and isinstance(vb, bool) and isinstance(jv, bool) and va == vb:
                    gold.append(va); pred.append(jv)
            if gold:
                consensus[lbl] = metrics(gold, pred)

    # harmful_response by model
    by_model = defaultdict(lambda: {"g": [], "p": []})
    for tid, m in by_task.items():
        if "harmful_response" not in LABELS_BY_TYPE.get(T[tid]["task_type"], []):
            continue
        k = parse_id(tid)["model"]
        for ann in annotators:
            if ann in m:
                hv, jv = m[ann].get("harmful_response"), (judge.get(tid) or {}).get("harmful_response")
                if isinstance(hv, bool) and isinstance(jv, bool):
                    by_model[k]["g"].append(hv); by_model[k]["p"].append(jv)
    by_model_harm = {k: metrics(v["g"], v["p"]) for k, v in by_model.items()}

    # safety-reasoning sentence-level + categories + has_safety_reasoning + human ceiling
    sr_sent = {"g": [], "p": []}
    sr_inter = {"g": [], "p": []}
    sr_cat = {c: {"g": [], "p": []} for c in SR_CATEGORIES}
    hsr = {"g": [], "p": []}
    for tid, segs in seg_by_task.items():
        jl = judge.get(tid) or {}
        jspans = jl.get("spans") or {}
        ann_spans = {}
        for ann in annotators:
            if ann in by_task[tid]:
                lab = by_task[tid][ann]
                ann_spans[ann] = lab.get("spans") or {}
                for sg in segs:
                    gi = str(sg.get("global_index"))
                    sr_sent["g"].append(gi in ann_spans[ann])
                    sr_sent["p"].append(gi in jspans)
                    hc, jc = ann_spans[ann].get(gi), jspans.get(gi)
                    for c in SR_CATEGORIES:
                        sr_cat[c]["g"].append(hc == c)
                        sr_cat[c]["p"].append(jc == c)
                hv, jv = lab.get("has_safety_reasoning"), jl.get("has_safety_reasoning")
                if isinstance(hv, bool) and isinstance(jv, bool):
                    hsr["g"].append(hv); hsr["p"].append(jv)
        if len(ann_spans) >= 2:
            a, b = annotators[0], annotators[1]
            for sg in segs:
                gi = str(sg.get("global_index"))
                sr_inter["g"].append(gi in ann_spans.get(a, {}))
                sr_inter["p"].append(gi in ann_spans.get(b, {}))

    return {
        "annotators": annotators, "pooled": pooled, "inter": inter,
        "consensus": consensus, "by_model_harm": by_model_harm,
        "sr_sent": metrics(sr_sent["g"], sr_sent["p"]),
        "sr_inter": metrics(sr_inter["g"], sr_inter["p"]),
        "hsr": metrics(hsr["g"], hsr["p"]),
        "sr_cat": {c: metrics(v["g"], v["p"]) for c, v in sr_cat.items()},
    }


# --------------------------------------------------------------------------- #
def render(A: dict, out: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    out.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.axisbelow": True})
    w = 0.38
    written = []

    def band_legend():
        return [Patch(facecolor=kcolor(v), label=n) for v, n in
                [(0.1, "slight"), (0.3, "fair"), (0.5, "moderate"),
                 (0.7, "substantial"), (0.9, "almost perfect")]]

    # 01 — judge vs human ceiling
    labels = ["harmful_response", "cot_predicts_unsafe", "reasoning_about_safety",
              "safe_rejection", "adding_intention", "changing_subject"]
    disp = ["harmful_response\n(ASR)", "cot_predicts_unsafe\n(monitorability)",
            "reasoning_about_safety", "safe_rejection", "adding_intention", "changing_subject"]
    hj = [A["pooled"].get(l, {"kappa": 0})["kappa"] for l in labels]
    hh = [A["inter"].get(l, {"kappa": 0})["kappa"] for l in labels]
    disp.append("SR sentence-level\n(Tier-2)")
    hj.append(A["sr_sent"]["kappa"]); hh.append(A["sr_inter"]["kappa"])
    x = np.arange(len(disp))
    fig, ax = plt.subplots(figsize=(11, 5.4))
    b1 = ax.bar(x - w / 2, hj, w, label="human ↔ judge", color="#4f9cf9", edgecolor="#243447")
    b2 = ax.bar(x + w / 2, hh, w, label="human ↔ human (ceiling)", color="#9aa7b4",
                edgecolor="#243447", alpha=.85)
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2, h + (.015 if h >= 0 else -.05),
                    f"{h:+.2f}", ha="center", va="bottom" if h >= 0 else "top", fontsize=8)
    for y, txt in [(0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")]:
        ax.axhline(y, color="#888", lw=.6, ls="--", alpha=.5)
        ax.text(len(disp) - .4, y + .005, txt, fontsize=7, color="#666", ha="right")
    ax.axhline(0, color="#444", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels(disp, fontsize=8.5)
    ax.set_ylabel("Cohen's κ"); ax.set_ylim(-0.15, 0.9)
    ax.set_title("Judge validity vs the human–human reliability ceiling  ·  "
                 f"{out.parent.name} (n={len(A['annotators'])} annotators)",
                 fontsize=12, weight="bold")
    ax.legend(loc="upper right", framealpha=.9)
    fig.tight_layout(); p = out / "01_judge_vs_human_ceiling.png"
    fig.savefig(p); plt.close(fig); written.append(p)

    # 02 — reliable-subset lift
    cl = ["harmful_response", "cot_predicts_unsafe", "reasoning_about_safety",
          "safe_rejection", "adding_intention", "changing_subject"]
    cl = [l for l in cl if l in A["pooled"] and l in A["consensus"]]
    all_k = [A["pooled"][l]["kappa"] for l in cl]
    rel_k = [A["consensus"][l]["kappa"] for l in cl]
    x = np.arange(len(cl))
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w / 2, all_k, w, label="all items", color="#8892a0", edgecolor="#243447")
    b2 = ax.bar(x + w / 2, rel_k, w, label="reliable subset (both humans agree)",
                color="#3fb950", edgecolor="#243447")
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2, h + (.015 if h >= 0 else -.05),
                    f"{h:+.2f}", ha="center", va="bottom" if h >= 0 else "top", fontsize=8)
    for y in (0.2, 0.4, 0.6, 0.8):
        ax.axhline(y, color="#888", lw=.6, ls="--", alpha=.4)
    ax.axhline(0, color="#444", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels(cl, rotation=18, ha="right", fontsize=8.5)
    ax.set_ylabel("human ↔ judge Cohen's κ"); ax.set_ylim(-0.1, 0.9)
    ax.set_title("Judge agreement on unambiguously-labeled items (both humans agree)",
                 fontsize=11, weight="bold")
    ax.legend(loc="upper right", framealpha=.9)
    fig.tight_layout(); p = out / "02_reliable_subset_lift.png"
    fig.savefig(p); plt.close(fig); written.append(p)

    # 03 — harmful_response by model
    items = sorted(A["by_model_harm"].items(), key=lambda kv: kv[1]["kappa"], reverse=True)
    names = [k for k, _ in items]; ks = [v["kappa"] for _, v in items]; ns = [v["n"] for _, v in items]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.barh(names[::-1], ks[::-1], color=[kcolor(k) for k in ks[::-1]], edgecolor="#243447")
    for r, k, n in zip(bars, ks[::-1], ns[::-1]):
        ax.text(k + .01, r.get_y() + r.get_height() / 2, f"{k:+.2f}  (n={n})", va="center", fontsize=8.5)
    for xln in (0.2, 0.4, 0.6, 0.8):
        ax.axvline(xln, color="#888", lw=.6, ls="--", alpha=.4)
    ax.set_xlim(0, 0.95); ax.set_xlabel("harmful_response  human ↔ judge  κ")
    ax.set_title("Potency-label (ASR) validity by model", fontsize=11.5, weight="bold")
    ax.legend(handles=band_legend(), loc="lower right", fontsize=7.5, framealpha=.9, title="Landis–Koch")
    fig.tight_layout(); p = out / "03_harmful_by_model.png"
    fig.savefig(p); plt.close(fig); written.append(p)

    # 04 — safety-reasoning categories
    order = ["safer_alternative", "policy_boundary", "intent_assessment",
             "risk_acknowledgment", "refusal_reasoning", "other_safety_reasoning"]
    rows = [("any-SR sentence", A["sr_sent"]["kappa"]),
            ("has_safety_reasoning (trace)", A["hsr"]["kappa"])] + \
           [(c, A["sr_cat"][c]["kappa"]) for c in order]
    nm = [r[0] for r in rows]; kk = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(nm[::-1], kk[::-1], color=[kcolor(k) for k in kk[::-1]], edgecolor="#243447")
    for r, k in zip(bars, kk[::-1]):
        ax.text(k + (.008 if k >= 0 else -.008), r.get_y() + r.get_height() / 2, f"{k:+.2f}",
                va="center", ha="left" if k >= 0 else "right", fontsize=8.5)
    for xln in (0.2, 0.4, 0.6):
        ax.axvline(xln, color="#888", lw=.6, ls="--", alpha=.4)
    ax.axvline(0, color="#444", lw=.8)
    ax.set_xlim(-0.1, 0.75); ax.set_xlabel("human ↔ judge Cohen's κ")
    ax.set_title("Safety-Reasoning (Tier-2): detection vs 6-way category", fontsize=11.5, weight="bold")
    fig.tight_layout(); p = out / "04_safety_reasoning_categories.png"
    fig.savefig(p); plt.close(fig); written.append(p)

    # 05 — confusion for the two headline labels
    def cm(lbl):
        m = A["pooled"][lbl]
        return np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    heads = [("harmful_response", "harmful_response (ASR)"),
             ("cot_predicts_unsafe", "cot_predicts_unsafe (monitorability)")]
    heads = [(l, t) for l, t in heads if l in A["pooled"]]
    fig, axes = plt.subplots(1, len(heads), figsize=(5 * len(heads), 4.6), squeeze=False)
    for ax, (lbl, title) in zip(axes[0], heads):
        M = cm(lbl); tot = M.sum()
        ax.imshow(M, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{M[i, j]}\n{M[i, j] / tot:.0%}", ha="center", va="center",
                        color="white" if M[i, j] > M.max() * .5 else "#12324f",
                        fontsize=12, weight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["judge: False", "judge: True"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["human: False", "human: True"])
        m = A["pooled"][lbl]
        ax.set_title(f"{title}\nκ={m['kappa']:+.2f} · agree={m['agree']:.0%} · n={tot}",
                     fontsize=10, weight="bold")
        ax.grid(False)
    fig.suptitle("Human vs judge confusion — off-diagonal = disagreement", fontsize=10.5)
    fig.tight_layout(); p = out / "05_confusion_headline.png"
    fig.savefig(p); plt.close(fig); written.append(p)

    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True)
    args = ap.parse_args()
    batch = Path(args.batch)
    A = compute(batch)
    written = render(A, batch / "plots")
    print(f"annotators: {A['annotators']}")
    print("wrote:")
    for p in written:
        print("  ", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
