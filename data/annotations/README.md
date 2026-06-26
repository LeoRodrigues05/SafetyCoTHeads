# Human annotation — judge validation

Tooling to verify the v5 LLM judges are *valid* by having a human apply the **same
rubric** and computing **human-vs-judge Cohen's κ**. These instruments map onto the
three evaluation axes ([../../docs/general/EVALUATION_FRAMEWORK.md](../../docs/general/EVALUATION_FRAMEWORK.md));
the current batch `batch_v5_002` validates all three task types:

- **safety_5label** 〔Potency〕 — `harmful_response` / `safe_rejection` /
  `reasoning_about_safety` / `adding_intention` / `changing_subject` (drives every ASR
  number). Pick HARMFUL **or** SAFE (one), then tick any independent behaviors.
- **cot_only** 〔Safety-Reasoning / monitorability〕 — `cot_predicts_unsafe`. The
  annotator sees the **reasoning trace only**; the final answer is hidden.
- **safety_reasoning** 〔Safety-Reasoning · Tier 2, sentence-level〕 — over the indexed
  trace, mark **each sentence** that contains safety reasoning and assign one of 6
  categories (`risk_acknowledgment`, `policy_boundary`, `intent_assessment`,
  `refusal_reasoning`, `safer_alternative`, `other_safety_reasoning`). Scored as
  per-sentence κ (any-SR), per-category κ, and per-trace `has_safety_reasoning`.

The 12-label pathway judge is out of scope here (already κ≈0.96 vs HarmThoughts gold).

## Workflow

```bash
# 1. Build a blind batch (stratified across model×dataset×condition, prefer-unsafe).
#    Writes tasks.json (BLIND), judge_labels.json (held out), manifest.json.
#    --n-sr adds Tier-2 sentence-level safety-reasoning traces (keep small; each is a
#    full 20-50-sentence trace to annotate).
python -m scripts.make_annotation_batch --n-safety 80 --n-cot 40 --n-sr 20 \
    --batch-id batch_v5_002 --seed 0

# 2. Annotate in the browser. Zero-dependency stdlib server; auto-saves every label
#    to annotations_<name>.jsonl as you go (resume-safe). Use the Type filter to
#    switch between safety / cot / safety-reasoning.
python -m scripts.annotate_server --batch data/annotations/batch_v5_002
#    -> open http://127.0.0.1:8765/  (remote box: ssh -L 8765:127.0.0.1:8765 <host>)

# 3. Score: human-vs-judge κ + F1/agreement (+ inter-annotator κ if ≥2 annotators).
#    Reports per-label safety/cot κ AND sentence-level + per-category SR κ.
python -m scripts.score_annotations --batch data/annotations/batch_v5_002

# 4. Version-control the result.
git add data/annotations/batch_v5_002 && \
  git commit -m "human annotations: batch_v5_002" && git push
```

## Files in a batch dir
| file | committed | contents |
|------|-----------|----------|
| `tasks.json` | yes | **blind** items — prompt + response/trace, **no judge labels** |
| `judge_labels.json` | yes | held-out judge labels keyed by `task_id` (scoring only) |
| `manifest.json` | yes | seed, source root, per-stratum counts, judge models |
| `annotations_<name>.jsonl` | yes | one JSON object per task, latest write wins (upsert) |
| `validation_report.{html,json}` | yes | κ / F1 / agreement report |

`data/annotations/` is **not** gitignored (unlike `runs/`), so everything here is
committable. The rubric shown in the UI mirrors
`src/safety_cot_heads/judging/judge_prompts.py` verbatim
(`SAFETY_BEHAVIOR_PROMPT`, `SAFETY_LABEL_DEFINITIONS`, `COT_ONLY_PREDICTION_PROMPT`).

## Two annotators on the *same* batch (consistency)
For an honest validity claim, have a 2nd annotator label the same batch. The scorer
then reports **inter-annotator κ** (human↔human) — the reliability ceiling that the
human↔judge κ cannot exceed. If human↔human κ is itself low, the label is ambiguous
and a low human↔judge κ doesn't necessarily indict the judge.

This works because the batch is **deterministic and committed**: `tasks.json` /
`judge_labels.json` / `manifest.json` are version-controlled, so everyone who clones
and pulls gets the **identical** set of queries — don't rebuild the batch locally.
Each annotator writes their own `annotations_<name>.jsonl` (filenames don't collide),
so two people commit non-conflicting files against the same `tasks.json`. To make a
*new* shared batch, one person runs `make_annotation_batch` (fixed `--seed`) and
commits the three batch files; everyone else just pulls.
