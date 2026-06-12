"""K-sweep for neuron-ablation dose-response on Llama-3.1-8B-Instruct.

For each K in --k-values:
  * top-K positive-contrast neurons from the discovery ranking
  * K random neurons drawn from the *same* layers as the top-K positive set
    (layer-matched control)

Plus a single baseline condition. Generates on N JBB prompts, scores refusal
rate + avg word count, and also records the L2 logit-diff of the ablated vs
baseline forward pass on a small sanity batch.

Writes:
  runs/analysis/neuron_ksweep/summary.json
  runs/analysis/neuron_ksweep/sweep.csv
  runs/analysis/neuron_ksweep/completions_<label>.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.verify_neuron_ablation import (  # type: ignore
    logit_diff_sanity,
    looks_like_refusal,
    random_neurons,
    run_condition,
    select_neurons,
)

from safety_cot_heads.data import load_jailbreakbench
from safety_cot_heads.models import load_model
from safety_cot_heads.utils import ensure_dir, get_logger, json_dump, set_seed

log = get_logger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ranking",
        default="runs/direction_a/16-neuron-discovery-llama31/neuron_ranking.json")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--k-values", type=int, nargs="+",
                    default=[32, 64, 128, 256, 512, 1024])
    ap.add_argument("--out-dir", default="runs/analysis/neuron_ksweep")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=200,
                    help="Larger window so we can see whether refusal "
                         "decays into substantive jailbroken content.")
    ap.add_argument("--skip-random", action="store_true",
                    help="Skip the layer-matched random controls.")
    args = ap.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    log.info("loading ranking from %s", args.ranking)
    rk = json.load(open(args.ranking))
    ranking = rk["ranked_neurons"]
    n_pos_total = sum(1 for n in ranking if n["mean_score"] > 0)
    log.info("ranking: %d total neurons, %d with positive contrast",
             len(ranking), n_pos_total)

    log.info("loading model %s", args.model)
    lm = load_model(args.model, dtype="bfloat16")
    intermediate = int(lm.model.config.intermediate_size)
    n_layers = int(lm.model.config.num_hidden_layers)
    log.info("model loaded: layers=%d intermediate=%d total_neurons=%d",
             n_layers, intermediate, n_layers * intermediate)

    # build (label, neurons) for each K
    sweep: list[tuple[str, list[tuple[int, int]]]] = [("baseline", [])]
    set_layer_summary: dict[str, list[int]] = {"baseline": []}
    for K in args.k_values:
        if K > n_pos_total:
            log.warning("K=%d exceeds positive-contrast pool (%d); clipping",
                        K, n_pos_total)
            K_eff = n_pos_total
        else:
            K_eff = K
        pos_set = select_neurons(ranking, top_k=K_eff, positive_only=True)
        layers_in_pos = sorted({l for (l, _) in pos_set})
        sweep.append((f"top{K}_positive", pos_set))
        set_layer_summary[f"top{K}_positive"] = layers_in_pos
        if not args.skip_random:
            rand_set = random_neurons(layers_in_pos, intermediate, K_eff,
                                      rng=rng)
            sweep.append((f"random{K}_same_layers", rand_set))
            set_layer_summary[f"random{K}_same_layers"] = layers_in_pos

    for label, neurons in sweep:
        log.info("  set %-28s n=%4d layers=%s",
                 label, len(neurons),
                 sorted({l for (l, _) in neurons}) if neurons else [])

    # ---- 1. logit-diff sanity check on first 4 prompts for each set ----
    jbb = load_jailbreakbench(n=args.n)
    sanity_prompts = [r["prompt"] for r in jbb[:4]]
    log.info("running logit-diff sanity check on %d prompts...",
             len(sanity_prompts))
    sanity: dict[str, dict] = {}
    for label, neurons in sweep:
        if not neurons:
            continue
        s = logit_diff_sanity(lm, sanity_prompts, neurons)
        sanity[label] = s
        log.info("  %-28s mean L2 logit-diff %.3f  top1-changed %d/%d",
                 label, s["mean_l2_logit_diff"],
                 s["top1_token_changed_count"], s["n_prompts"])
    json_dump(out_dir / "sanity_logit_diff.json", sanity)

    # ---- 2. generation per K ----
    summary = []
    for label, neurons in sweep:
        log.info("=== %-28s (%d neurons) ===",
                 label, len(neurons))
        res = run_condition(lm, jbb, label=label,
                            neurons=neurons if neurons else None,
                            max_new_tokens=args.max_new_tokens)
        jpath = out_dir / f"completions_{label}.jsonl"
        with open(jpath, "w") as f:
            for r in res["rows"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info("  -> refusal=%.1f%% avg_words=%.1f  (-> %s)",
                 100 * res["refusal_rate"], res["avg_words"], jpath.name)
        summary.append({
            "label": label,
            "n": res["n"],
            "n_neurons_ablated": res["n_neurons_ablated"],
            "refusal_rate": res["refusal_rate"],
            "avg_words": res["avg_words"],
        })

        # show a couple of examples for quick eyeball
        for r in res["rows"][:2]:
            head = (r["completion"] or "").replace("\n", " ")[:200]
            print(f"    [{label}] id={r['id']}  | {head}")

    # ---- 3. write CSV + summary ----
    csv_path = out_dir / "sweep.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "n_neurons", "refusal_rate", "avg_words",
                    "mean_l2_logit_diff", "top1_changed", "kind", "K"])
        for s in summary:
            label = s["label"]
            san = sanity.get(label, {})
            if label == "baseline":
                kind, K = "baseline", 0
            elif label.startswith("top"):
                kind = "top_positive"
                K = int(label.replace("top", "").replace("_positive", ""))
            elif label.startswith("random"):
                kind = "random_same_layers"
                K = int(label.replace("random", "").replace("_same_layers", ""))
            else:
                kind, K = "?", 0
            w.writerow([
                label, s["n_neurons_ablated"],
                f"{s['refusal_rate']:.4f}", f"{s['avg_words']:.2f}",
                f"{san.get('mean_l2_logit_diff', float('nan')):.4f}",
                san.get("top1_token_changed_count", 0),
                kind, K,
            ])

    json_dump(out_dir / "summary.json", {
        "model": args.model,
        "ranking_path": args.ranking,
        "n_prompts": args.n,
        "k_values": args.k_values,
        "intermediate_size": intermediate,
        "n_layers": n_layers,
        "n_positive_in_ranking": n_pos_total,
        "set_layers": set_layer_summary,
        "sanity_logit_diff": sanity,
        "conditions": summary,
    })

    # ---- 4. printed table ----
    print()
    print("================== NEURON ABLATION K-SWEEP ==================")
    print(f" model:   {args.model}")
    print(f" prompts: {args.n} from JBB | max_new_tokens={args.max_new_tokens}")
    print(f" K values: {args.k_values}")
    print()
    print(f"  {'condition':28s} {'#neurons':>9s} {'refusal':>9s} "
          f"{'avg_words':>10s} {'L2_diff':>10s} {'top1Δ':>6s}")
    print(f"  {'-'*28} {'-'*9} {'-'*9} {'-'*10} {'-'*10} {'-'*6}")
    for s in summary:
        san = sanity.get(s["label"], {})
        l2 = san.get("mean_l2_logit_diff", float("nan"))
        t1 = san.get("top1_token_changed_count", 0)
        print(
            f"  {s['label']:28s} {s['n_neurons_ablated']:>9d} "
            f"{100*s['refusal_rate']:>8.1f}% {s['avg_words']:>10.1f} "
            f"{l2:>10.3f} {t1:>6d}"
        )
    print()
    print(f"Outputs in: {out_dir}")
    print(f"CSV:        {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
