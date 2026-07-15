"""Direction A v5 — render the 10 analysis diagrams from judge summaries.

Reads every ``runs/direction_a_v5/<model>/judge/<ds>/<cond>/seed0/summary.json``
(std-metrics OR pathway flavour) plus the fine-tuned pathway-judge eval, and
writes PNGs to ``runs/plots/``. Models with missing data are skipped per-plot
rather than erroring, so partial coverage still produces useful figures.

Usage:
    .venv/bin/python -m scripts.make_v5_plots
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("runs/direction_a_v5")
OUT = Path("runs/plots")
OUT.mkdir(parents=True, exist_ok=True)
EVAL = Path("runs/pathway_judge_14b_lora/eval_sample_n180.json")

COND_ORDER = ["baseline", "ships_top3", "ships_top5", "ships_top8",
              "neurons_top256", "neurons_top512", "neurons_top1024",
              "steering_a0.5", "steering_a1.0", "steering_a1.5", "steering_ablate"]
CLBL = {"baseline": "base", "ships_top3": "ships3", "ships_top5": "ships5",
        "ships_top8": "ships8", "neurons_top256": "neu256", "neurons_top512": "neu512",
        "neurons_top1024": "neu1024", "steering_a0.5": "steer.5",
        "steering_a1.0": "steer1", "steering_a1.5": "steer1.5", "steering_ablate": "ablate"}
DATASETS = ["jbb", "bt"]
MODEL_ORDER = ["llama31_8b_control", "olmo3_7b_base_own", "qwen3_8b",
               "olmo3_7b_think", "olmo3_7b_base", "r1_distill_qwen_7b"]
DISPLAY = {"qwen3_8b": "Qwen3 8B", "llama31_8b_control": "Llama 3.1 8B",
           "olmo3_7b_think": "OLMo-3 7B Think", "olmo3_7b_base": "OLMo-3 7B Base",
           "olmo3_7b_base_own": "OLMo-3 7B Base-own",
           "r1_distill_qwen_7b": "R1-Distill Qwen 7B"}
COL = {m: c for m, c in zip(MODEL_ORDER, plt.get_cmap("tab10").colors)}

# (model, ds, cond) -> merged metric dict
DATA: dict[tuple[str, str, str], dict] = {}


def _load() -> None:
    for f in glob.glob(str(ROOT / "*/judge/*/*/seed0/summary.json")):
        if "_smoke_vm" in f:
            continue
        p = Path(f).parts
        model, ds, cond = p[-6], p[-4], p[-3]
        s = json.load(open(f))
        b = (s.get("per_condition_basic") or {}).get(cond, {})
        coh = s.get("coherence") or {}
        mon = (s.get("monitorability") or {}).get("per_condition", {}).get(cond, {})
        pw = (s.get("per_condition_pathway") or {}).get(cond, {}) or {}
        pwm = pw.get("mean", {})
        DATA.setdefault((model, ds, cond), {}).update({
            "harmful": b.get("harmful_rate"), "refusal": b.get("refusal_rate"),
            "reasoning": b.get("reasoning_rate"), "add_intent": b.get("adding_intent_rate"),
            "chg_subject": b.get("changing_subject_rate"),
            "clean": coh.get("clean_rate"), "empty": coh.get("empty_rate"),
            "asr_clean": coh.get("harmful_among_clean_rate"),
            "asr_final": mon.get("asr_final"), "asr_cot": mon.get("asr_cot_pred"),
            "gap": mon.get("gap"),
            "recognition": pwm.get("recognition_rate"), "refusal_supp": pwm.get("refusal_suppressed"),
            "execution": pwm.get("execution_rate"), "op_detail": pwm.get("operational_detail_rate"),
            "dom": pw.get("dominant_pathway_hist") or {},
        })


def _f(v):
    return np.nan if v is None else float(v)


def val(m, ds, cond, key):
    return _f(DATA.get((m, ds, cond), {}).get(key))


def arr(m, ds, key):
    return np.array([val(m, ds, c, key) for c in COND_ORDER])


def has(m, ds, key):
    return not np.all(np.isnan(arr(m, ds, key)))


def _xticks(ax):
    ax.set_xticks(range(len(COND_ORDER)))
    ax.set_xticklabels([CLBL[c] for c in COND_ORDER], rotation=45, ha="right", fontsize=8)


def save(fig, name, title):
    fig.suptitle(title, fontsize=13, y=1.0)
    fig.tight_layout()
    out = OUT / f"{name}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


# --------------------------------------------------------------------------- #
def d01_asr_vs_condition():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        for m in MODEL_ORDER:
            if not has(m, ds, "asr_clean"):
                continue
            ax.plot(range(len(COND_ORDER)), arr(m, ds, "asr_clean"),
                    marker="o", color=COL[m], label=DISPLAY[m])
        ax.set_title(ds.upper()); ax.grid(alpha=.3); _xticks(ax)
        ax.set_ylabel("ASR (harmful_among_clean)")
    axes[0].legend(fontsize=8)
    save(fig, "01_asr_vs_condition", "1. ASR-clean (potency) vs condition")


FAMS = {"SHIPS top-k": (["ships_top3", "ships_top5", "ships_top8"], [3, 5, 8]),
        "neurons top-k": (["neurons_top256", "neurons_top512", "neurons_top1024"], [256, 512, 1024]),
        "steering alpha": (["steering_a0.5", "steering_a1.0", "steering_a1.5"], [0.5, 1.0, 1.5])}


def d02_dose_response():
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for r, ds in enumerate(DATASETS):
        for c, (fam, (conds, xs)) in enumerate(FAMS.items()):
            ax = axes[r][c]
            for m in MODEL_ORDER:
                ys = [val(m, ds, cd, "asr_clean") for cd in conds]
                if np.all(np.isnan(ys)):
                    continue
                ax.plot(xs, ys, marker="o", color=COL[m], label=DISPLAY[m])
                base = val(m, ds, "baseline", "asr_clean")
                if not np.isnan(base):
                    ax.axhline(base, color=COL[m], ls="--", lw=.8, alpha=.5)
            ax.set_title(f"{fam} — {ds.upper()}"); ax.grid(alpha=.3)
            ax.set_xlabel(fam); ax.set_ylabel("ASR-clean")
    axes[0][0].legend(fontsize=7)
    save(fig, "02_dose_response", "2. Dose-response by intervention family (dashed = baseline)")


def d03_monitor_gap():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        models = [m for m in MODEL_ORDER if has(m, ds, "gap")]
        x = np.arange(len(COND_ORDER)); w = 0.8 / max(len(models), 1)
        for i, m in enumerate(models):
            ax.bar(x + i * w, arr(m, ds, "gap"), w, color=COL[m], label=DISPLAY[m])
        ax.axhline(0, color="k", lw=.8); ax.set_title(ds.upper()); ax.grid(alpha=.3, axis="y")
        ax.set_ylabel("gap = ASR_cot_pred - ASR_final"); _xticks(ax)
    axes[0].legend(fontsize=8)
    save(fig, "03_monitorability_gap", "3. Monitorability gap vs condition (>0 = CoT over-warns)")


def d04_asr_scatter():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, ds in zip(axes, DATASETS):
        for m in MODEL_ORDER:
            xs = [val(m, ds, c, "asr_final") for c in COND_ORDER]
            ys = [val(m, ds, c, "asr_cot") for c in COND_ORDER]
            if np.all(np.isnan(xs)):
                continue
            ax.scatter(xs, ys, color=COL[m], label=DISPLAY[m], s=45, alpha=.8, edgecolor="w")
        lim = [0, 1]
        ax.plot(lim, lim, "k--", lw=.8, alpha=.6)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal"); ax.grid(alpha=.3)
        ax.set_title(ds.upper()); ax.set_xlabel("ASR final"); ax.set_ylabel("ASR CoT-pred")
    axes[0].legend(fontsize=8)
    save(fig, "04_asr_final_vs_cot", "4. ASR final vs CoT-predicted (above line = CoT over-warns)")


def d05_frontier():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, ds in zip(axes, DATASETS):
        for m in MODEL_ORDER:
            xs = [val(m, ds, c, "refusal") for c in COND_ORDER]
            ys = [val(m, ds, c, "harmful") for c in COND_ORDER]
            if np.all(np.isnan(xs)):
                continue
            ax.scatter(xs, ys, color=COL[m], label=DISPLAY[m], s=45, alpha=.8, edgecolor="w")
            xb, yb = val(m, ds, "baseline", "refusal"), val(m, ds, "baseline", "harmful")
            if not np.isnan(xb):
                ax.scatter([xb], [yb], color=COL[m], marker="*", s=240, edgecolor="k", zorder=5)
        ax.set_title(ds.upper()); ax.grid(alpha=.3)
        ax.set_xlabel("refusal rate"); ax.set_ylabel("harmful rate")
    axes[0].legend(fontsize=8)
    save(fig, "05_refusal_harmful_frontier", "5. Refusal vs harmful frontier (star = baseline)")


def d06_coherence():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        for m in MODEL_ORDER:
            if not has(m, ds, "clean"):
                continue
            ax.plot(range(len(COND_ORDER)), arr(m, ds, "clean"),
                    marker="o", color=COL[m], label=DISPLAY[m])
        ax.set_ylim(0, 1.02); ax.set_title(ds.upper()); ax.grid(alpha=.3); _xticks(ax)
        ax.set_ylabel("coherence clean_rate")
    axes[0].legend(fontsize=8)
    save(fig, "06_coherence", "6. Output coherence (clean_rate) vs condition")


def d07_content_profile():
    metrics = [("reasoning", "reasoning_about_safety"), ("add_intent", "adding_intention"),
               ("chg_subject", "changing_subject")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    for r, ds in enumerate(DATASETS):
        for c, (key, lbl) in enumerate(metrics):
            ax = axes[r][c]
            for m in MODEL_ORDER:
                if not has(m, ds, key):
                    continue
                ax.plot(range(len(COND_ORDER)), arr(m, ds, key),
                        marker=".", color=COL[m], label=DISPLAY[m])
            ax.set_title(f"{lbl} — {ds.upper()}"); ax.grid(alpha=.3); ax.set_ylim(0, 1.02)
            _xticks(ax)
    axes[0][0].legend(fontsize=7)
    save(fig, "07_content_profile", "7. 5-label content profile vs condition")


PW_KEYS = [("recognition", "recognition"), ("refusal_supp", "refusal supp."),
           ("execution", "execution"), ("op_detail", "op. detail")]


def _pw_models():
    return [m for m in MODEL_ORDER
            if any(has(m, ds, "execution") for ds in DATASETS)]


def d08_pathway_heatmap():
    models = _pw_models()
    if not models:
        print("  skip 08: no pathway data"); return
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 6), squeeze=False)
    for ax, m in zip(axes[0], models):
        conds = [c for c in COND_ORDER
                 if any(not np.isnan(val(m, ds, c, "execution")) for ds in DATASETS)]
        mat = np.full((len(conds), len(PW_KEYS)), np.nan)
        for i, c in enumerate(conds):
            for j, (key, _) in enumerate(PW_KEYS):
                vals = [val(m, ds, c, key) for ds in DATASETS]
                vals = [v for v in vals if not np.isnan(v)]
                if vals:
                    mat[i, j] = np.mean(vals)
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(PW_KEYS))); ax.set_xticklabels([l for _, l in PW_KEYS], rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(conds))); ax.set_yticklabels([CLBL[c] for c in conds], fontsize=8)
        ax.set_title(DISPLAY[m], fontsize=10)
        for i in range(len(conds)):
            for j in range(len(PW_KEYS)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            fontsize=7, color="w" if mat[i, j] < .6 else "k")
        fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    save(fig, "08_pathway_heatmap", "8. Pathway vector (dataset-avg) by condition")


def d09_dominant_pathway():
    models = _pw_models()
    if not models:
        print("  skip 09: no pathway data"); return
    cats = ["recognition_loss", "sanitised_compliance", "rationalised_compliance",
            "direct_execution", "none"]
    cmap = plt.get_cmap("Set2")
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 6), squeeze=False)
    for ax, m in zip(axes[0], models):
        conds = [c for c in COND_ORDER
                 if any(not np.isnan(val(m, ds, c, "execution")) for ds in DATASETS)]
        bottoms = np.zeros(len(conds))
        seen_cats = set()
        for ci, cat in enumerate(cats):
            fracs = []
            for c in conds:
                tot, hit = 0, 0
                for ds in DATASETS:
                    dom = DATA.get((m, ds, c), {}).get("dom", {})
                    tot += sum(dom.values()); hit += dom.get(cat, 0)
                fracs.append(hit / tot if tot else 0.0)
            fracs = np.array(fracs)
            if fracs.sum() > 0:
                seen_cats.add(cat)
            ax.bar(range(len(conds)), fracs, bottom=bottoms, color=cmap(ci), label=cat)
            bottoms += fracs
        ax.set_xticks(range(len(conds))); ax.set_xticklabels([CLBL[c] for c in conds], rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.02); ax.set_title(DISPLAY[m], fontsize=10); ax.set_ylabel("fraction")
    axes[0][-1].legend(fontsize=7, loc="upper right")
    save(fig, "09_dominant_pathway", "9. Dominant-pathway composition by condition")


def d10_judge_eval():
    if not EVAL.exists():
        print("  skip 10: no eval file"); return
    ev = json.load(open(EVAL))
    pl = ev["per_label"]
    labels = sorted(pl)
    ft = [pl[l]["finetuned"]["f1"] for l in labels]
    bl = [pl[l]["baseline"]["f1"] for l in labels]
    ftk = [pl[l]["finetuned"]["cohen_kappa"] for l in labels]
    blk = [pl[l]["baseline"]["cohen_kappa"] for l in labels]
    x = np.arange(len(labels)); w = .38
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (a, b, ttl) in zip(axes, [(ft, bl, "F1"), (ftk, blk, "Cohen's kappa")]):
        ax.bar(x - w / 2, a, w, label="fine-tuned 14B", color="#137a43")
        ax.bar(x + w / 2, b, w, label="Qwen3-30B baseline", color="#9a2020")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{ttl} per label"); ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8)
    o_ft, o_bl = ev["overall_finetuned"], ev["overall_baseline"]
    save(fig, "10_judge_eval",
         f"10. Fine-tuned pathway judge eval (n={o_ft['n']}): "
         f"overall F1 {o_ft['f1']:.3f} vs {o_bl['f1']:.3f}, "
         f"kappa {o_ft['cohen_kappa']:.3f} vs {o_bl['cohen_kappa']:.3f}")


def main() -> int:
    _load()
    print(f"loaded {len(DATA)} cells")
    for fn in (d01_asr_vs_condition, d02_dose_response, d03_monitor_gap,
               d04_asr_scatter, d05_frontier, d06_coherence, d07_content_profile,
               d08_pathway_heatmap, d09_dominant_pathway, d10_judge_eval):
        try:
            fn()
        except Exception as e:  # keep going so one bad plot doesn't sink the rest
            print(f"  ERROR {fn.__name__}: {e}")
    print(f"done -> {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
