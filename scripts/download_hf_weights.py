#!/usr/bin/env python3
"""Download the timm weights of the pending queue runs into a folder, to copy to a server without internet.

Usage
  python scripts/download_hf_weights.py queue.jsonl TARGET_DIR            lists the models and sizes only
  python scripts/download_hf_weights.py queue.jsonl TARGET_DIR --download  downloads them

TARGET_DIR gets the Hugging Face hub layout (TARGET_DIR/hub/models--timm--<name>/...), so copying
TARGET_DIR/hub/ into the server's cache folder (HF_HOME/hub, usually ~/.cache/huggingface/hub) is enough.
Only runs that have no probs file in the folder given by --done-dir (default the current folder) and
whose model is not in TARGET_DIR yet are listed. Names that start with hf-hub: are skipped.
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("queue")
ap.add_argument("target")
ap.add_argument("--download", action="store_true")
ap.add_argument("--skip", default="", help="comma separated model names that are already on the server")
ap.add_argument("--done-dir", default=".", help="folder with probs-<tag>.npz files, to skip finished runs")
args = ap.parse_args()
skip = {n for n in args.skip.split(",") if n}

names = []
for raw in Path(args.queue).read_text().splitlines():
    raw = raw.strip()
    if not raw or raw.startswith("#"):
        continue
    job = json.loads(raw)
    name = job.get("MODEL_NAME")
    if not name or name.startswith("hf-hub:") or name in names or name in skip:
        continue
    if (Path(args.done_dir) / f"probs-{job['RUN_TAG']}.npz").exists() or (Path(args.done_dir) / "archive" / f"probs-{job['RUN_TAG']}.npz").exists():
        continue
    names.append(name)

api = HfApi()
target = Path(args.target) / "hub"
plan, total = [], 0
for name in names:
    repo = f"timm/{name}"
    if (target / f"models--timm--{name}" / "snapshots").exists():
        print(f"already there  {repo}")
        continue
    info = api.model_info(repo, files_metadata=True)
    sizes = {s.rfilename: s.size or 0 for s in info.siblings}
    weights = "model.safetensors" if "model.safetensors" in sizes else "pytorch_model.bin"
    patterns = ["*.json", weights]
    size = sum(sz for fn, sz in sizes.items() if fn.endswith(".json") or fn == weights)
    plan.append((repo, patterns))
    total += size
    print(f"{size / 1e9:6.2f} GB  {repo}  ({weights})")
print(f"total {total / 1e9:.1f} GB for {len(plan)} models into {target}")

if args.download:
    for repo, patterns in plan:
        print("downloading", repo, flush=True)
        snapshot_download(repo, cache_dir=str(target), allow_patterns=patterns)
    print("done, copy", target, "to the server hub cache folder")
else:
    print("dry run, add --download to fetch them")
