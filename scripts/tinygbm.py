#!/usr/bin/env python3
"""Cold-start atlas-vs-GBM replication on the TinyLlama stability sweep."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "results_public/tinyllama_stability/runs"
SEED = 0


def auroc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0) + 0.5 * (diff == 0)).mean())


def main():
    frames = []
    for d in sorted(RUNS.iterdir()):
        f = d / "results.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df = df[df.error.isna()] if "error" in df else df
        m = re.match(r"tau(?P<tau>[0-9p]+)_seed(?P<seed>\d+)", d.name)
        df["cell"] = d.name
        df["tau"] = float(m.group("tau").replace("p", "."))
        df["seed"] = m.group("seed")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["arity"] = df.adapters.apply(lambda a: len(a.split("+")))
    print(f"{len(df)} queries over {df.cell.nunique()} cells")

    y = df.feasible.to_numpy(int)
    # cold start: arity + operator + cell identity only, no atlas score
    X = pd.get_dummies(df[["arity", "method", "tau"]], columns=["method"]).astype(float)
    groups = df.cell.to_numpy()
    gbm_scores = np.full(len(df), np.nan)
    gkf = GroupKFold(n_splits=df.cell.nunique())
    for tr, te in gkf.split(X, y, groups=groups):
        clf = GradientBoostingClassifier(n_estimators=500, max_depth=3,
                                         learning_rate=0.05, subsample=0.8,
                                         random_state=SEED)
        clf.fit(X.iloc[tr], y[tr])
        gbm_scores[te] = clf.predict_proba(X.iloc[te])[:, 1]

    out = {"n_queries": int(len(df)), "n_cells": int(df.cell.nunique()),
           "features": list(X.columns), "folds": "leave-one-cell-out", "per_operator": {}}
    for op in sorted(df.method.unique()):
        m = (df.method == op).to_numpy()
        a = auroc(y[m], -df.predicted_score.to_numpy(float)[m])
        g = auroc(y[m], gbm_scores[m])
        out["per_operator"][op] = {"n": int(m.sum()), "atlas": a, "gbm": g,
                                   "atlas_minus_gbm": (a - g) if (a and g) else None}
        print(f"  {op:16s} atlas {a:.3f}  gbm {g:.3f}  diff {a-g:+.3f}")

    od = REPO / "results_public/tinyllama_stability"
    (od / "coldstart.json").write_text(json.dumps(out, indent=1))
    print("wrote", od / "coldstart.json")


if __name__ == "__main__":
    main()
