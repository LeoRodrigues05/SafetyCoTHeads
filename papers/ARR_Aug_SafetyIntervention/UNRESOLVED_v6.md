# Unresolved items — claims still depending on unfinished experiments (2026-07-23)

The completed v6 bundle (`runs/direction_a_v6/reports/`) covers **only the two
explicit-trace arms** (Qwen3-8B, OLMo-3-Think), still on **v5 answer labels**.
The items below were **NOT changed** because the required completed result does
not exist yet; each is flagged for the author.

## A. Blocked on the currently-running Finding-2 rerun (`scripts/rerun_finding2_p04.sh`)

The pathway + safety-reasoning judges are being re-run with corrected inputs.
`reasoning_metrics.csv` is empty (2 stale rows) and `prose_prefix_sensitivity.csv`
has 0 rows. Everything below is pending that job (ETA ~6–8 h from 20:47 UTC):

1. **Tab. `cot-signature`** (5-indicator family pathway signature) — all values.
2. **Tab. `appendix-pathway`** (full 12-label OLMo-3-Think/JailbreakBench profile).
3. **Radar figures** `pathway_radar_overlay_olmo3_7b_think.png`,
   `pathway_radar_olmo3_7b_think.png`, `pathway_radar_reasoning.png`.
4. **§4.2.2 / §4.3 pathway correlations** (Spearman ρ of exec/op-detail/recognition
   with Potency; refusal-suppression change ≤0.01).
5. **Safety-reasoning verbalization rates**: "86–90% of traces vs 15–35% of prefixes"
   (§4.3) and "spans occupy 37–39% vs 5–16%" (appendix `cot-other`).
6. **Rationalization-flat claim** "0.10→0.12→0.09" across steering doses (appendix).
7. **Prefix-sensitivity monitorability** for the non-explicit arms (0 rows in
   `prose_prefix_sensitivity.csv`).

## B. Blocked because Llama-3.1-8B and both OLMo-3-Base arms are not in the v6 bundle

v6 re-aggregated only the 2 explicit arms. Every pooled/multi-arm number that
includes Llama or the Base arms is still the **v5** value:

8. **Headline pooled family SFS** (Tab. `intervention_family`; abstract & intro
   **0.67 / 0.47 / 0.26 / 0.23**; P/Q/S rows) — pools 3 safety-trained arms.
9. **`fig:family-model` heatmap** and **`tab:appendix-model-family`** Llama and
   both OLMo-3-Base rows.
10. **Kendall table** `tab:appendix-kendall` and the §4.1 τ values, incl. the
    headline **τ=0.73** (OLMo-3-Base transfer) in the abstract/intro/§4.1.
11. **Dose-level table** `tab:appendix-dose` (pools 3 arms; e.g. steering P
    .264/.437/.595 ≠ the 2-arm v6 .050/.228/.500).
12. **Grid-average susceptibility** for Llama (0.57) and the §4.1 OLMo-3-Base
    baseline-harm narrative (HAC 66%, 0.65–0.81 band, P 0–.45); baseline-correction
    figures `fig_baseline_correction.png`, `fig_gap_by_family.png`,
    `composite_03_raw_asr_vs_sfs.png`.
13. **§4.2.1 gap-across-100-cells** claims (96/100 non-positive; 4 positive, max
    +0.036; family means −0.107…−0.075; most-negative −0.448 SHIPS/Llama;
    ρ≈0.00 pooled) — span all 5 arms + prefix sensitivity.

## C. Structural data gap — OLMo-3-Think steering α=1.5 has no closed CoT

At α=1.5, OLMo-3-Think emits **no closed `</think>`** (0/100 JailbreakBench,
0/98 BeaverTails within the 2,048-token budget), so `n_pairs=0` and **S/SFS are
undefined** in v6. This is not fixed by the running job (no trace/answer split
exists); it needs a larger generation budget or a revised trace-parse. It blocks:

14. **`fig:teaser` SFS = 0.88** (OLMo-3-Think steering α=1.5) — intro figure and caption.
15. **§4.2 OLMo-3-Think** "mean SFS **0.64** … rises to **0.88** at α=1.5" (P=0.77,
    Q=0.99 *are* confirmed by v6; only S/SFS are missing).
16. **`tab:appendix-model-family`** OLMo-3-Think **Steer .64** (bold).
17. **Grid-average** OLMo-3-Think **0.34** (appendix `comp-other`) — depends on the
    missing α=1.5 SFS; v6 partial (α=1.5 excluded) gives ~0.28.
18. OLMo-3-Think per-arm **Kendall τ** (1.00 / 0.96) — 10-condition ranking needs
    the α=1.5 SFS.

## D. Global caveat affecting even the updated explicit-model numbers

19. **`answer_source=v5`.** P, Q, and HAC for *all* cells (including the two Qwen
    updates in the changelog) still use v5 answer labels. The v6 answer-input
    correction (`--answer-source v6`, after the B200 answer stage) is pending and
    may shift these again. Treat the current explicit-model SFS as "v6 paired-S on
    v5 answers," not final v6.
