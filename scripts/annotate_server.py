"""Local, zero-dependency human-annotation server (Python stdlib http.server).

Serves the annotation page and auto-persists each submitted label to
``data/annotations/<batch_id>/annotations_<annotator>.jsonl`` (a git-committable
path). Each task is upserted (latest write per task_id wins), so re-saving fixes a
mistake instead of duplicating, and restarting the server resumes from disk.

Held-out judge labels are NEVER sent to the browser — the page only ever sees the
blind ``tasks.json``, keeping the annotation blind.

Usage:
    python -m scripts.annotate_server --batch data/annotations/batch_v5_001
    # then open the printed http://127.0.0.1:8765/ in a browser

For a remote box, forward the port:  ssh -L 8765:127.0.0.1:8765 <host>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

TEMPLATE = Path(__file__).parent / "templates" / "annotate.html"
_SAFE = re.compile(r"[^A-Za-z0-9_-]")
_LOCK = threading.Lock()


class State:
    def __init__(self, batch_dir: str):
        self.batch_dir = Path(batch_dir)
        tasks_path = self.batch_dir / "tasks.json"
        if not tasks_path.exists():
            raise SystemExit(f"no tasks.json under {self.batch_dir} — run "
                             f"`python -m scripts.make_annotation_batch` first")
        self.tasks = json.loads(tasks_path.read_text(encoding="utf-8"))


def _ann_path(state: State, annotator: str):
    name = _SAFE.sub("", annotator or "")[:64] or "anon"
    return name, state.batch_dir / f"annotations_{name}.jsonl"


def _read_ann(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    out[r["task_id"]] = r
                except Exception:
                    pass
    return out


def _write_ann(path: Path, by_id: dict[str, dict]) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in by_id.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the console quiet
            pass

        def _send(self, code: int, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                self._send(200, TEMPLATE.read_bytes(), "text/html; charset=utf-8")
            elif u.path == "/api/tasks":
                self._send(200, json.dumps({"batch_id": state.batch_dir.name,
                                            "tasks": state.tasks}))
            elif u.path == "/api/progress":
                annotator = (parse_qs(u.query).get("annotator") or [""])[0]
                name, path = _ann_path(state, annotator)
                self._send(200, json.dumps({"annotator": name, "done": _read_ann(path)}))
            else:
                self._send(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            if urlparse(self.path).path != "/api/annotate":
                self._send(404, json.dumps({"error": "not found"}))
                return
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                self._send(400, json.dumps({"error": "bad json"}))
                return
            task_id = body.get("task_id")
            if not task_id:
                self._send(400, json.dumps({"error": "missing task_id"}))
                return
            name, path = _ann_path(state, body.get("annotator") or "")
            rec = {
                "task_id": task_id,
                "task_type": body.get("task_type"),
                "annotator": name,
                "labels": body.get("labels") or {},
                "notes": (body.get("notes") or "").strip(),
                "hard_case": bool(body.get("hard_case")),
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            with _LOCK:
                by_id = _read_ann(path)
                by_id[task_id] = rec
                _write_ann(path, by_id)
                n_done = len(by_id)
            self._send(200, json.dumps({"ok": True, "n_done": n_done,
                                        "file": str(path), "annotator": name}))

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True, help="batch dir, e.g. data/annotations/batch_v5_001")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    state = State(args.batch)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    url = f"http://{args.host}:{args.port}/"
    print(f"\n  Annotation server for batch '{state.batch_dir.name}' "
          f"({len(state.tasks)} tasks)")
    print(f"  Open: {url}")
    print(f"  Saving to: {state.batch_dir}/annotations_<name>.jsonl  (auto-saved per item)")
    print("  Ctrl-C to stop.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped. To version-control the annotations:")
        print(f"  git add {state.batch_dir}/annotations_*.jsonl && "
              f"git commit -m 'human annotations: {state.batch_dir.name}' && git push")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
