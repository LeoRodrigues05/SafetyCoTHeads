# Legacy analysis notebooks

These three notebooks drove the original (pre-migration) post-ablation analysis.
They were moved here unchanged for reference; the equivalent functionality now
lives in the package proper:

| notebook                  | replaced by                                                                                 |
| ------------------------- | ------------------------------------------------------------------------------------------- |
| `full_judge.ipynb`        | `scripts/run_judge.py` + `safety_cot_heads.judging` (now with batched gen + chat template)   |
| `judge_w_ablations.ipynb` | `safety_cot_heads.analysis.paired.paired_judge_compare` + `per_category_breakdown`           |
| `post_ablation.ipynb`     | `safety_cot_heads.analysis.coherence` + `paired.harmful_among_coherent`                       |

End-to-end runner that strings them together:
`scripts/run_evaluation.py` (gibberish filter → judge summary → paired deltas).

Notebook-only tricks that were ported into the codebase:

- Granite/Qwen JSON cleanups (`(Float)`, `true.`/`false.`, smart quotes, double
  whitespace) — now in `judging.parse._granite_clean`.
- Gibberish filter (`madhurjindal/autonlp-Gibberish-Detector-492513457`) — now
  in `analysis.coherence.classify_gibberish`.
- Paired drop-and-realign by parse failure index — replaced by id-based join
  in `analysis.paired`.
- Multi-seed random head sets `[7-7, 10-22, 20-29, 27-2, 29-26]` — still TODO;
  see `docs/richer_evaluation.md` §11.

Do not modify the notebooks; they're read-only history.
