# Annotation setup — guide for a second annotator

This guide gets a new person from a fresh `git clone` to submitting human
annotations that validate our LLM judges. **You do not need a GPU, model weights,
or the heavy Python dependencies** — annotation runs on the standard library and a
browser. (Only the maintainer scoring the results needs `scikit-learn`.)

Background on what you're validating and the full metric mapping is in
[data/annotations/README.md](../../data/annotations/README.md). This doc is just the
setup + run steps.

---

## 1. Prerequisites
- `git`
- **Python 3.9+** (standard library only — no `pip install` required to annotate)
- A web browser

## 2. Get the repo and the annotation batch
```bash
git clone <repo-url> SafetyCoTHeads
cd SafetyCoTHeads
git checkout human-annotation-tool        # branch that carries the tool + batch
                                           # (skip if it has been merged to main)
```
The blind task batch is already committed at
`data/annotations/batch_v5_001/` (`tasks.json` = 120 items, no judge labels). You
do **not** rebuild it — everyone annotates the same committed set.

## 3. Launch the annotation app
```bash
python -m scripts.annotate_server --batch data/annotations/batch_v5_001
# -> open the printed URL, e.g. http://127.0.0.1:8765/
```
- Working on a remote machine? Forward the port from your laptop:
  `ssh -L 8765:127.0.0.1:8765 <user>@<host>` then open `http://127.0.0.1:8765/`.
- The server is pure stdlib `http.server` — nothing to install.

## 4. Annotate
1. **Sign in** with your name/initials when prompted (this names your output file).
2. For each item, read the prompt + response (or, for cot items, the reasoning
   **trace only** — the final answer is hidden) and apply the rubric shown in the
   side panel (it mirrors the exact judge rubric).
   - **safety** items: pick **HARMFUL** or **SAFE rejection** (exactly one), then
     tick any of the three independent behaviors.
   - **cot** items: pick **will be UNSAFE** or **will be SAFE**.
3. Click **Save & next** (or press Enter). Every save is written to disk
   immediately — you can stop and resume anytime; your progress reloads.
4. Optional: add a note or tick "hard / ambiguous case" for tricky items.

Shortcuts: `h`/`s` harmful/safe · `r a c` behaviors · `u`/`f` unsafe/safe ·
`Enter` save · `[` `]` navigate. There's also an **Export** button that downloads
a backup copy of your file.

## 5. Submit your annotations (version control)
Your labels are saved to `data/annotations/batch_v5_001/annotations_<yourname>.jsonl`.
Commit just that file and push (it won't collide with other annotators — each file
is named by annotator):
```bash
git add data/annotations/batch_v5_001/annotations_<yourname>.jsonl
git commit -m "human annotations: batch_v5_001 (<yourname>)"
git push        # or open a PR
```

## 6. (Maintainer) Score human vs judge
Once one or more annotators have submitted:
```bash
pip install scikit-learn        # only dependency the scorer needs
python -m scripts.score_annotations --batch data/annotations/batch_v5_001
# -> writes validation_report.html + .json in the batch dir
```
This reports per-label **human-vs-judge Cohen's κ** (plus F1/agreement), and — if
two or more annotators labelled the shared batch — **inter-annotator κ** (the
human↔human reliability ceiling). Two annotators are strongly recommended so a low
κ can be attributed to judge error vs. label ambiguity.

---

### Notes
- All annotators receive the **same 120 items** (one committed `tasks.json`). This
  is deliberate: it yields both human-vs-judge validity and human-vs-human
  agreement on the overlap.
- Judge labels are held out in `judge_labels.json` and are **never sent to the
  browser**, so annotation is blind.
- To create a *new* batch (different sample/size) you do need the full env
  (`pip install -e .`, because sampling reconstructs CoT traces via the package):
  `python -m scripts.make_annotation_batch --n-safety 80 --n-cot 40 --seed 0`.
