#!/usr/bin/env python3
"""Downstream selection on the task-diverse TinyLlama-12 library (one adapter per
task, including a GSM8K math adapter and a text-to-SQL adapter). Shows that
selecting merges by the atlas score delivers working multi-task merges far more
often than random selection, and preserves the individual task capabilities.
All quantities are recomputed from the released results_public CSV (CPU only)."""
import ast
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(HERE, "results_public", "tinyllama_12x", "results.csv")
CFG = os.path.join(HERE, "peft_atlas_lite", "configs", "multifamily", "tinyllama_12x.json")


def parse(x):
    try:
        return ast.literal_eval(x)
    except Exception:
        try:
            return json.loads(x)
        except Exception:
            return {}


def main():
    df = pd.read_csv(CSV)
    df = df[df["predicted_score"].notna()].copy()
    df["y"] = df["feasible"].astype(int)
    df["m"] = df["margins"].apply(parse)
    task2name = {a["task"]: a["name"] for a in json.load(open(CFG))["adapters"]}

    print(f"library: 12 task-diverse adapters, {len(df)} merge events, "
          f"base feasible rate {df['y'].mean():.3f}")

    print("\n=== screen-then-merge delivery (all constituent tasks retained) ===")
    for qk, g in df.groupby("query_kind"):
        g = g.sort_values("predicted_score")
        k = max(1, int(round(0.20 * len(g))))
        atlas, base = g.head(k)["y"].mean(), g["y"].mean()
        print(f"  {qk:7s}: atlas top-20% = {atlas:.2f}  random = {base:.2f}  "
              f"lift = {atlas / max(base, 1e-9):.1f}x   (n={len(g)})")

    print("\n=== per-task capability retention (atlas-picked vs random) ===")
    for task in ("math_reasoning", "sql_generation"):
        name = task2name[task]
        sub = df[df["adapters"].str.contains(name)].copy()
        sub["ret"] = sub["m"].apply(lambda d: 1 if d.get(name, -1) >= 0 else 0)
        sub = sub.sort_values("predicted_score")
        k = max(1, int(round(0.20 * len(sub))))
        atlas, base = sub.head(k)["ret"].mean(), sub["ret"].mean()
        print(f"  {task:16s} (adapter '{name}'): retained in atlas top-20% = {atlas:.2f} "
              f"vs random = {base:.2f}   (n={len(sub)} merges contain it)")


if __name__ == "__main__":
    main()
