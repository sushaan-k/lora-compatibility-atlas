#!/usr/bin/env python3
"""Rebuild a panel's task files from the public Natural Instructions release."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TREE = "https://api.github.com/repos/allenai/natural-instructions/git/trees/master?recursive=1"
RAW = "https://raw.githubusercontent.com/allenai/natural-instructions/master/tasks/"
SEED = 5


def index():
    with urllib.request.urlopen(TREE) as r:
        tree = json.load(r)["tree"]
    return [x["path"] for x in tree if x["path"].startswith("tasks/task")]


def build(num, paths, n_instances):
    match = [p for p in paths if re.match(rf"tasks/task{num}_", p)]
    if not match:
        raise SystemExit(f"task{num} not found upstream")
    with urllib.request.urlopen(RAW + match[0].split("/")[-1]) as r:
        src = json.load(r)
    pool = [{"input": i["input"], "output": i["output"]} for i in src["Instances"]]
    order = list(range(len(pool)))
    random.Random(SEED).shuffle(order)
    return {"definition": src["Definition"][0],
            "instances": [pool[i] for i in order[:n_instances]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="panel config naming the tasks")
    ap.add_argument("--out", default=str(REPO / "experiments_gpu/tasks"))
    ap.add_argument("--instances", type=int, default=800)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    paths = index()
    for a in cfg["adapters"]:
        name = Path(a["task_json"]).name
        dst = out / name
        if dst.exists():
            print("have", name); continue
        num = re.sub(r"\D", "", name)
        dst.write_text(json.dumps(build(num, paths, args.instances)))
        print("wrote", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
