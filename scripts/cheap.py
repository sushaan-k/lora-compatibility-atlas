#!/usr/bin/env python3
"""Generator for the cheap-screen rows of tab:qwen_balanced_baselines."""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent


def auroc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0) + 0.5 * (diff == 0)).mean())


def main():
    bal = REPO / "results_public/qwen25_m50_balanced"
    df = pd.read_csv(bal / "results.csv")
    df["members"] = df.adapters.apply(lambda a: sorted(a.split("+")))
    summ = json.loads((bal / "summary.json").read_text())

    gains = {}  # (slice_id, method) -> {adapter: gain}
    for item in summ["component_summaries"]:
        meth = list(item["metrics_by_method"].keys())[0]
        b = item["base_losses"]["calibration"]
        s = item["single_losses"]["calibration"]
        gains[(item["slice_id"], meth)] = {a: b[a] - s[a] for a in b if a in s}

    fp = pd.read_csv(REPO / "results_public/qwen25_m50_facepairs/results.csv")
    pair_feas = {}
    for _, r in fp[fp.query_kind == "pair"].iterrows():
        pair_feas[(r.method, "+".join(sorted(r.adapters.split("+"))))] = int(r.feasible)

    out = {"convention": __doc__.strip().split("\n\n")[1], "per_operator": {}}
    for meth in sorted(df.method.unique()):
        sub = df[df.method == meth]
        g = {sl: gains[(sl, meth)] for sl in sub.slice_id.unique()}
        y = sub.feasible.to_numpy(int)
        min_gain = np.array([min(g[r.slice_id].get(a, np.nan) for a in r.members)
                             for _, r in sub.iterrows()])
        mean_gain = np.array([np.mean([g[r.slice_id].get(a, np.nan) for a in r.members])
                              for _, r in sub.iterrows()])
        arity = sub.members.apply(len).to_numpy(float)

        learned = np.full(len(sub), np.nan)
        X = np.column_stack([arity, min_gain, mean_gain])
        slices = sub.slice_id.to_numpy()
        for held in np.unique(slices):
            tr, te = slices != held, slices == held
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
            learned[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]

        trip = sub[sub.query_kind == "triple"]
        clique = [sum(pair_feas.get((meth, "+".join(p)), 0)
                      for p in combinations(r.members, 2)) for _, r in trip.iterrows()]

        out["per_operator"][meth] = {
            "n": int(len(sub)), "n_triples": int(len(trip)),
            "single_gain_auroc": auroc(y, min_gain),
            "learned_auroc": auroc(y, learned),
            "clique_face_closure_auroc_triples": auroc(trip.feasible.to_numpy(int), clique),
        }
        print(meth, {k: round(v, 3) for k, v in out["per_operator"][meth].items()
                     if isinstance(v, float)})

    (bal / "cheap_baselines.json").write_text(json.dumps(out, indent=1))
    print("wrote", bal / "cheap_baselines.json")


if __name__ == "__main__":
    main()
