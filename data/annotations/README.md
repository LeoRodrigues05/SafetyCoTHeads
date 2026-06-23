# Human annotation — judge validation

Tooling to verify the v5 LLM judges are *valid* by having a human apply the **same
rubric** and computing **human-vs-judge Cohen's κ**. First batch targets the two
unvalidated outputs of the Qwen3-30B std judge:

- **safety_5label** — `harmful_response` / `safe_rejection` / `reasoning_about_safety` /
  `adding_intention` / `changing_subject` (drives every ASR number).
- **cot_only** — `cot_predicts_unsafe` (the monitorability-gap signal). The annotator
  sees the **reasoning trace only**; the final answer is hidden.

The 12-label pathway judge is out of scope here (already κ≈0.96 vs HarmThoughts gold).

## Workflow

```bash
# 1. Build a blind batch (stratified across model×dataset×condition, prefer-unsafe).
#    Writes tasks.json (BLIND), judge_labels.json (held out), manifest.json.
python -m scripts.make_annotation_batch --n-safety 80 --n-cot 40 --seed 0

# 2. Annotate in the browser. Zero-dependency stdlib server; auto-saves every label
#    to annotations_<name>.jsonl as you go (resume-safe).
python -m scripts.annotate_server --batch data/annotations/batch_v5_001
#    -> open http://127.0.0.1:8765/  (remote box: ssh -L 8765:127.0.0.1:8765 <host>)

# 3. Score: human-vs-judge κ + F1/agreement (+ inter-annotator κ if ≥2 annotators).
python -m scripts.score_annotations --batch data/annotations/batch_v5_001

# 4. Version-control the result.
git add data/annotations/batch_v5_001 && \
  git commit -m "human annotations: batch_v5_001" && git push
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

## Recommended: two annotators
For an honest validity claim, have a 2nd annotator label the same batch. The scorer
then reports **inter-annotator κ** (human↔human) — the reliability ceiling that the
human↔judge κ cannot exceed. If human↔human κ is itself low, the label is ambiguous
and a low human↔judge κ doesn't necessarily indict the judge.
