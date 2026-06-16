"""Compute per-head template-anchoring (rho_tpl) and residualise a head ranking.

Phase 1 / P1.5 diagnostic. Loads a model, computes rho_tpl over a prompt set,
optionally residualises an existing SHIPS / Sahara discovery ranking on
rho_tpl, and writes the augmented ranking + raw anchoring table.

Usage (matches other scripts/run_*.py):

    python -m scripts.run_template_anchoring \\
        --config configs/experiments/direction_a_ships/12-template-anchoring-llama31.yaml \\
        [--ranking runs/direction_a/01-ships-discovery-llama31/ships_dataset_ranking.json] \\
        [--dry-run]

YAML schema:

    seed: 0
    model:
      name: meta-llama/Llama-3.1-8B-Instruct
      dtype: bfloat16
      attn_implementation: eager
    dataset:
      name: maliciousinstruct       # or jailbreakbench / beavertails
      n: 100
    anchoring:
      prompt_template: "## Query:{q}\\n## Answer:"
      extra_template_substrings: []
      max_prompt_len: 1024
      normalise: true
    ranking_score_key: score        # field name inside ranking entries
    output:
      dir: runs/direction_a/12-template-anchoring-llama31
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _cli import cfg_to_dict, load_cfg                # noqa: E402

from safety_cot_heads.attribution import (
    TemplateAnchoringConfig,
    compute_head_template_anchoring,
    residualize_on_template_anchoring,
)
from safety_cot_heads.data import (
    load_beavertails, load_jailbreakbench, load_maliciousinstruct,
)
from safety_cot_heads.models import load_model
from safety_cot_heads.utils import (
    ensure_dir, get_logger, json_dump, jsonl_write, set_seed,
)

log = get_logger(__name__)


def _load_dataset(name: str, split_n: int | None, **kw):
    if name == "maliciousinstruct": return load_maliciousinstruct(n=split_n)
    if name == "jailbreakbench":    return load_jailbreakbench(n=split_n)
    if name == "beavertails":
        return load_beavertails(
            categories=kw.get("categories"),
            n_per_category=kw.get("n_per_category"),
        )
    raise ValueError(f"unknown dataset {name!r}")


def _load_ranking(path: Path) -> tuple[list[dict], dict]:
    """Load a SHIPS/Sahara ranking JSON. Returns (entries, full_doc)."""
    doc = json.loads(path.read_text())
    if isinstance(doc, list):
        return list(doc), {"ranking": doc}
    for key in ("dataset_ranking", "ranked_heads", "ranking"):
        if key in doc and isinstance(doc[key], list):
            return list(doc[key]), doc
    raise ValueError(f"could not find ranking entries in {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ranking", default=None,
                    help="Optional path to ships/sahara ranking JSON to residualise.")
    ap.add_argument("--overrides", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    set_seed(int(cfg.get("seed", 0)))
    out_dir = Path(cfg.output.dir)
    ensure_dir(out_dir)

    rows = _load_dataset(
        cfg.dataset.name,
        cfg.dataset.get("n"),
        categories=cfg.dataset.get("categories"),
        n_per_category=cfg.dataset.get("n_per_category"),
    )
    log.info("loaded %d prompts from %s", len(rows), cfg.dataset.name)

    if args.dry_run:
        log.info("DRY-RUN: would load %s and score %d prompts", cfg.model.name, len(rows))
        json_dump(out_dir / "dryrun.json",
                  {"plan": cfg_to_dict(cfg), "n_prompts": len(rows),
                   "ranking": args.ranking})
        return 0

    lm = load_model(
        cfg.model.name,
        dtype=cfg.model.get("dtype", "auto"),
        attn_implementation=cfg.model.get("attn_implementation", "eager"),
        load_in_4bit=bool(cfg.model.get("load_in_4bit", False)),
        device_map=cfg.model.get("device_map"),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
    )
    anc_cfg = TemplateAnchoringConfig(
        prompt_template=cfg.get("anchoring", {}).get(
            "prompt_template", "## Query:{q}\n## Answer:"),
        extra_template_substrings=tuple(
            cfg.get("anchoring", {}).get("extra_template_substrings", []) or []),
        max_prompt_len=cfg.get("anchoring", {}).get("max_prompt_len", 1024),
        normalise=bool(cfg.get("anchoring", {}).get("normalise", True)),
    )
    anchoring = compute_head_template_anchoring(lm.model, lm.tokenizer, rows, anc_cfg)
    log.info("computed rho_tpl for %d (layer, head) entries", len(anchoring))

    anchoring_rows = [
        {"layer": li, "head": hi, "rho_tpl": v}
        for (li, hi), v in sorted(anchoring.items())
    ]
    jsonl_write(out_dir / "template_anchoring.jsonl", anchoring_rows)

    if args.ranking:
        rank_entries, rank_doc = _load_ranking(Path(args.ranking))
        score_key = cfg.get("ranking_score_key", "score")
        resid = residualize_on_template_anchoring(rank_entries, anchoring, score_key=score_key)
        out_doc = {
            **{k: v for k, v in rank_doc.items()
               if k not in ("dataset_ranking", "ranked_heads", "ranking")},
            "source_ranking": str(args.ranking),
            "score_key": score_key,
            "n_anchoring_prompts": len(rows),
            "anchoring_config": cfg_to_dict(cfg)["anchoring"],
            "ranking_residualized": resid,
        }
        json_dump(out_dir / "ranking_residualized.json", out_doc)
        log.info("residualised %d ranking entries → %s",
                 len(resid), out_dir / "ranking_residualized.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
