# Paper outline — structural plan

Working title: **"One Number Isn't Enough: A Decomposable Metric for Comparing
White-Box Safety Interventions on Reasoning Models."**
Target: ACL Rolling Review (ARR), long paper — **8 pages body** + unlimited references
+ appendix. This file is the global structural plan; the paragraph-by-paragraph outline
lives as comments inside each `NNname.tex` section file.

## One-sentence thesis
Inference-time white-box safety interventions are each reported with a bespoke
single-axis ASR, so they can't be compared; we give the first head-to-head comparison on
shared axes **and** a decomposable metric (Potency × Quality × Safety-Reasoning →
Selective-Failure Score) and show a lone ASR demonstrably mis-ranks methods.

## Section map and page budget

| File | Section | Budget | Core job |
|---|---|---|---|
| `00abs` | Abstract | — | gap → dual contribution → setup → 5 findings → release |
| `01intro` | Introduction | ~1.25 pg | the gap; why one ASR hides 3 collapses; dual contribution; findings teaser; **Fig 1** |
| `02related` | Related Work | ~0.75 pg | 5 buckets; show existing evals don't cover the gap |
| `03prelim` | Preliminaries | ~2.0 pg | 3 axes + composite (eqs); the grid; judges + validation; **Fig 2, Tab 1, Tab 2** |
| `04findings` | Findings | ~2.75 pg | F1–F5 in 4 subsections + ablation; **Figs 3–5, Tab 3–4** |
| `05discussion` | Discussion | ~0.5 pg | what it buys; mechanism as "why"; defence-side; scope |
| `06conclusion` | Conclusion | ~0.25 pg | restate contribution + honest-null framing |
| `07limitations` | Limitations | ~0.4 pg | ACL-required, unnumbered |
| `08acknowledgments` | Acknowledgments | — | remove for anonymous review |
| `09appendix` | Appendix | unlimited | full per-cell table, judge prompts+κ, extra figures |

## The five findings (the spine of §4)
- **F1** a single coherence-gated ASR mis-ranks methods (Kendall τ vs SFS as low as 0.73).
- **F2** baseline-correction is essential on non-safe base models (OLMo-3-base: raw HAC
  0.63–0.74 for *all ten* interventions, but P≈0 for most).
- **F3** coherence-gating separates "removed safety" from "broke the model"
  (Llama head-ablation Q≈0.44).
- **F4** families separate once decomposed (Steering SFS 0.50 > dir-ablation 0.42 >
  heads 0.37 > neuron 0.26); steering shows a clean dose-response.
- **F5** the CoT monitor does *not* covertly fail for these suppressive interventions —
  reframes pre-registered H3 (an honest null the metric reports rather than hides).

## Figure & table inventory — "where do the diagrams go"
Main body (≤ ~5 floats to fit 8 pages). Sources are in `figures/` (copied from
`runs/plots/`) or the composite report.

| Float | Where | Content | Source | Status |
|---|---|---|---|---|
| **Fig 1** (teaser) | §1, top of p.1–2, wide | Raw ASR vs SFS scatter — same ASR, wildly different SFS | `figures/composite_03_raw_asr_vs_sfs.png` | ✅ have |
| **Fig 2** | §3.1, col | Metric schematic: 3 axes → SFS pipeline | TikZ / drawn | ✗ to make |
| **Tab 1** | §3.3, col | The grid: 5 models × conditions × 2 datasets | hand table | ✗ to write |
| **Tab 2** | §3.4, col | Judge validation κ (pathway 0.96 vs 30B 0.21; SR/5-label human) | validation report | ✗ to write |
| **Fig 3** | §4.1 (F2), col | Baseline-correction: OLMo-base raw HAC (all ~0.7) vs P (~0) | derive from CSV | ✗ to make |
| **Tab 3** | §4.1 (F1)+ablation | Per-model Kendall τ: raw ASR / P / P·Q vs SFS | composite report §2–3 | ✅ have data |
| **Fig 4** | §4.2 (F4), wide | Steering dose-response P/Q/S/SFS across α, 4 models | `figures/composite_01_steering_dose_response.png` | ✅ have |
| **Tab 4** | §4.2 (F4), col | Family-pooled mean P,Q,S,SFS,rawASR | composite report §4 | ✅ have data |
| **Fig 5** | §4.3 (F5), col | Monitorability: gap / S by family, covert rate ≈ 0 | derive (or composite_08) | ✗ to make |

Appendix floats (already rendered): `composite_02_family_mean_sfs`,
`composite_04_model_family_sfs_heatmap`, `composite_05_pqs_pareto_3d`,
`composite_06_pq_pareto_facets`, `composite_07_pareto_frontier_clean`,
`composite_08_metric_diagnostics`; full per-cell table from
`runs/direction_a_v5/composite_cells.csv`.

## Open structural decisions (flag before drafting prose)
1. **Teaser**: Fig 1 = the raw-ASR-vs-SFS scatter (abstract, all methods) vs a concrete
   two-trace example (same prompt, two methods, same ASR, different failure). The scatter
   is ready; the two-trace example is more visceral. Could do both (scatter as Fig 1,
   example as a boxed vignette).
2. **Metric in §3 vs its own section**: currently folded into Preliminaries. If reviewers
   want the metric foregrounded, promote 3.1–3.2 to a standalone "The Metric" section.
3. **Pareto**: keep the (P,Q,S) vector + Pareto framing in the body (§3.2 + one appendix
   figure) or feature a Pareto figure in §4. Currently appendix to save space.
4. **Datasets**: JBB + BeaverTails reported pooled or split? Pooled in body, split in
   appendix (recommended).

## Build
Main file `acl_latex.tex` → `\input{}`s the numbered section files. Compile with
`pdflatex acl_latex && bibtex acl_latex && pdflatex acl_latex ×2`. `custom.bib` holds our
refs (currently placeholder stubs — verify before submission). `anthology.bib` is the ACL
Anthology bib (optional; enable via `\bibliography{anthology,custom}`).
</content>
