#!/usr/bin/env python
"""Upload the Direction A v5 per-query results to a Hugging Face dataset repo.

The ~5.9 GB of per-query completions + judge outputs (jsonl) is too large for GitHub;
this uploads them (plus the aggregate metrics, so the dataset is self-contained) to a
HF dataset repo using the resumable, multi-threaded ``upload_large_folder`` API.

Prereqs (already true on this machine):
    pip install -U huggingface_hub          # >=0.24 for upload_large_folder
    huggingface-cli login                    # or hf auth login  (token cached)

Usage:
    # Safe default: create PRIVATE, upload, then flip visibility once verified.
    python scripts/upload_results_to_hf.py --repo-id LeoRodrigues05/safety-cot-interventions-v5

    # Make it public (or gated) only after reviewing the harmful-content warning:
    python scripts/upload_results_to_hf.py --repo-id <id> --public
    hf repo settings <id> --repo-type dataset --gated auto     # gated-access alternative

Notes:
    * Excludes internal/underscore working dirs (_stale_*, _smoke_vm, _orch_state, _sr_logs)
      and any model-weight files.
    * The dataset card (scripts/hf_dataset_card.md) is uploaded as README.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_large_folder

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOLDER = REPO_ROOT / "runs" / "direction_a_v5"
CARD = REPO_ROOT / "scripts" / "hf_dataset_card.md"

# Excluded from the release: internal working dirs and any weights.
IGNORE = [
    "_stale_steering_pre_dose_fix/*", "_stale_steering_pre_dose_fix/**",
    "_smoke_vm/*", "_smoke_vm/**",
    "_orch_state/*", "_orch_state/**",
    "_sr_logs/*", "_sr_logs/**",
    "*.pt", "*.safetensors", "*.pth", "*.bin",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default="LeoRodrigues05/safety-cot-interventions-v5")
    ap.add_argument("--folder", default=str(DEFAULT_FOLDER))
    ap.add_argument("--public", action="store_true",
                    help="Create/keep the repo PUBLIC. Default is PRIVATE (recommended: "
                         "review harmful content, then flip to public or gated).")
    ap.add_argument("--card", default=str(CARD))
    args = ap.parse_args()

    api = HfApi()
    who = api.whoami()["name"]
    print(f"Logged in as: {who}")
    print(f"Repo:   {args.repo_id}  (private={not args.public})")
    print(f"Folder: {args.folder}")

    create_repo(args.repo_id, repo_type="dataset", private=not args.public, exist_ok=True)

    # Card first so the repo has a README even if the big upload is interrupted.
    if Path(args.card).exists():
        api.upload_file(
            path_or_fileobj=args.card,
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Add dataset card",
        )
        print("Uploaded dataset card -> README.md")

    upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=args.folder,
        ignore_patterns=IGNORE,
        print_report=True,
    )
    print(f"\nDone. https://huggingface.co/datasets/{args.repo_id}")
    if not args.public:
        print("Repo is PRIVATE. To publish after review:\n"
              f"  hf repo settings {args.repo_id} --repo-type dataset --private false\n"
              f"  # or gated:  hf repo settings {args.repo_id} --repo-type dataset --gated auto")


if __name__ == "__main__":
    main()
