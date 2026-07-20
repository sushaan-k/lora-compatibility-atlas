#!/usr/bin/env python3
"""Dependence-aware uncertainty for the m=30 and m=50 amortized index panels.

scale_panel_analyze.py reports subset-level pooled and triples-only AUROC at
each retention threshold tau, and the pooled Pearson fidelity on triple member
events, but no uncertainty. As in capability_dependence_ci.py, the subsets are
not independent draws: every pair or triple is a "+"-joined combination of a
small adapter pool, so queries sharing an adapter share correlated screening
errors. This script rebuilds the per-member events exactly as
scale_panel_analyze.py does (same coefficients, same restriction to the query
face, same exclusion of tasks whose answer loss is NaN at every design point),
asserts that the rebuilt aggregates reproduce the published scale_analysis.json
values, and then reports the two dependence-aware summaries used for the m=16
panel:

  1. Leave-one-adapter-out (LOAO): drop every subset containing adapter a,
     recompute the AUROC, report the min-to-max range over defined drops.

  2. Adapter-block bootstrap: resample the scored adapter pool with
     replacement (2000 reps, seed 0); keep a subset iff all of its members
     were drawn; weight it by the product of its members' resample
     multiplicities; recompute the weighted AUROC (and, for fidelity, the
     weighted Pearson correlation over triple member events). The 2.5/97.5
     percentile interval is the block-bootstrap CI.

The label and score definitions are byte-for-byte those of
scale_panel_analyze.py: a member is retained at tau iff
L_answer <= base_La - tau*gain; a subset is feasible iff every member is; the
screening score is minus the worst member's predicted deficit; AUROC is
Mann-Whitney with 0.5 credit for ties.

Run:  python3 scripts/scale_panel_ci.py
Out:  results_public/scale_m30/scale_ci.json, results_public/scale_m50/scale_ci.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "peft_atlas_lite"))
from run_peft_atlas_lite import fit_quadratic_model, predict_quadratic  # noqa: E402

TAU = 0.70
N_BOOT = 2000
SEED = 0

PANELS = [
    ("results_public/scale_m30", "Mistral-7B m=30 linear"),
    ("results_public/scale_m50", "Mistral-7B m=50 linear"),
]


def auroc_np(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0) + 0.5 * (diff == 0)).mean())


def weighted_auroc_np(labels, scores, weights):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    p, n = labels & (weights > 0), (~labels) & (weights > 0)
    if not p.any() or not n.any():
        return None
    sp, wp = scores[p], weights[p]
    sn, wn = scores[n], weights[n]
    diff = sp[:, None] - sn[None, :]
    wins = (weights_outer := wp[:, None] * wn[None, :]) * ((diff > 0) + 0.5 * (diff == 0))
    return float(wins.sum() / weights_outer.sum())


def weighted_pearson(x, y, w):
    x, y, w = (np.asarray(a, dtype=float) for a in (x, y, w))
    keep = w > 0
    x, y, w = x[keep], y[keep], w[keep]
    if len(x) < 3:
        return None
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    vx = np.average((x - mx) ** 2, weights=w)
    vy = np.average((y - my) ** 2, weights=w)
    if vx == 0 or vy == 0:
        return None
    return float(cov / np.sqrt(vx * vy))


def build_events(panel_dir):
    """Rebuild per-member events exactly as scale_panel_analyze.py main()."""
    z = np.load(panel_dir / "answer_jets_raw.npz", allow_pickle=True)
    grid = [list(w) for w in z["grid"]]
    adapters = [str(x) for x in z["names"]]
    vals = z["values"]
    excluded = [adapters[i] for i in range(len(adapters)) if np.isnan(vals[i]).any()]
    tasks = [t for t in adapters if t not in excluded]
    coeffs = {t: fit_quadratic_model(grid, list(vals[adapters.index(t)]))[0] for t in tasks}

    rows = json.loads((panel_dir / "validation_rows.json").read_text())["rows"]
    base = {r["task"]: r for r in rows if r["merge"] == "base"}
    alone = {r["task"]: r for r in rows if r["merge"] == "alone"}
    gains = {t: base[t]["L_answer"] - alone[t]["L_answer"] for t in tasks}

    def weights_for(merge):
        mem = merge.split("+")
        w = [0.0] * len(adapters)
        for t in mem:
            w[adapters.index(t)] = 1.0 / len(mem)
        return w

    per_member = []
    for r in rows:
        if "+" not in r["merge"] or r["task"] not in r["merge"].split("+"):
            continue
        if r["task"] in excluded:
            continue
        per_member.append({
            "task": r["task"], "merge": r["merge"],
            "order": r["merge"].count("+") + 1,
            "pred_L_answer": predict_quadratic(coeffs[r["task"]], weights_for(r["merge"])),
            "L_answer": r["L_answer"], "base_La": base[r["task"]]["L_answer"],
            "gain": gains[r["task"]]})

    by_merge = defaultdict(list)
    for e in per_member:
        by_merge[e["merge"]].append(e)
    subsets = [{"merge": m, "order": v[0]["order"], "adapters": m.split("+"), "members": v}
               for m, v in by_merge.items()
               if len(v) == len(m.split("+")) and not any(t in excluded for t in m.split("+"))]
    return tasks, excluded, per_member, subsets


def subset_label_score(subsets, tau):
    labels = [all(m["L_answer"] <= m["base_La"] - tau * m["gain"] for m in s["members"])
              for s in subsets]
    scores = [-max(m["pred_L_answer"] - (m["base_La"] - tau * m["gain"]) for m in s["members"])
              for s in subsets]
    return labels, scores


def block_bootstrap(subsets, tasks, tau, triple_events, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    labels, scores = subset_label_score(subsets, tau)
    idx = {a: i for i, a in enumerate(tasks)}
    sub_ad = [[idx[a] for a in s["adapters"]] for s in subsets]
    trip_flag = np.array([s["order"] == 3 for s in subsets])
    # Fidelity events keep merges with an excluded co-member (as in
    # scale_panel_analyze.py); excluded adapters never enter the resample
    # pool, so they contribute no multiplicity factor.
    ev_ad = [[idx[a] for a in e["merge"].split("+") if a in idx] for e in triple_events]
    ev_p = [e["pred_L_answer"] for e in triple_events]
    ev_m = [e["L_answer"] for e in triple_events]
    n = len(tasks)
    au_pool, au_trip, fids = [], [], []
    for _ in range(n_boot):
        mult = np.bincount(rng.integers(0, n, size=n), minlength=n)
        w = np.array([np.prod(mult[m]) for m in sub_ad], dtype=float)
        if (w > 0).sum() < 5:
            continue
        a = weighted_auroc_np(labels, scores, w)
        if a is not None:
            au_pool.append(a)
        at = weighted_auroc_np(np.asarray(labels)[trip_flag],
                               np.asarray(scores)[trip_flag], w[trip_flag])
        if at is not None:
            au_trip.append(at)
        we = np.array([np.prod(mult[m]) for m in ev_ad], dtype=float)
        f = weighted_pearson(ev_p, ev_m, we)
        if f is not None:
            fids.append(f)

    def ci(v):
        return {"n_valid": len(v), "median": float(np.median(v)),
                "ci_lo_2.5": float(np.percentile(v, 2.5)),
                "ci_hi_97.5": float(np.percentile(v, 97.5))} if v else None
    return {"auroc_pooled": ci(au_pool), "auroc_triples": ci(au_trip),
            "fidelity_triples_pooled": ci(fids)}


def loao(subsets, tasks, tau):
    vals = []
    for a in tasks:
        keep = [s for s in subsets if a not in s["adapters"]]
        labels, scores = subset_label_score(keep, tau)
        v = auroc_np(labels, scores)
        if v is not None:
            vals.append(v)
    return {"min": min(vals), "max": max(vals), "n_defined": len(vals),
            "n_undefined": len(tasks) - len(vals)}


def main():
    for sub, label in PANELS:
        pdir = REPO / sub
        tasks, excluded, per_member, subsets = build_events(pdir)
        pub = json.loads((pdir / "scale_analysis.json").read_text())

        labels, scores = subset_label_score(subsets, TAU)
        au = auroc_np(labels, scores)
        trips = [i for i, s in enumerate(subsets) if s["order"] == 3]
        au_t = auroc_np([labels[i] for i in trips], [scores[i] for i in trips])
        triple_events = [e for e in per_member if e["order"] == 3]
        fid = weighted_pearson([e["pred_L_answer"] for e in triple_events],
                               [e["L_answer"] for e in triple_events],
                               np.ones(len(triple_events)))

        ref = pub["by_answer_tau"]["0.7"]
        assert abs(au - ref["auroc_pooled"]) < 1e-9, (label, au, ref["auroc_pooled"])
        assert abs(au_t - ref["auroc_triples"]) < 1e-9, (label, au_t, ref["auroc_triples"])
        assert abs(fid - pub["fidelity_triples"]["pooled"]) < 1e-9
        assert len(subsets) == pub["n_subsets"] and excluded == pub["excluded_tasks"]
        print(f"{label}: reproduced published AUROC {au:.4f}/{au_t:.4f}, fidelity {fid:.4f}")

        out = {"panel": label, "tau": TAU, "n_subsets": len(subsets),
               "n_scored_tasks": len(tasks), "excluded_tasks": excluded,
               "published_reproduced": {"auroc_pooled": au, "auroc_triples": au_t,
                                        "fidelity_triples_pooled": fid},
               "loao_auroc_pooled": loao(subsets, tasks, TAU),
               "adapter_block_bootstrap": block_bootstrap(subsets, tasks, TAU, triple_events)}
        (pdir / "scale_ci.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out["adapter_block_bootstrap"], indent=1))
        print("loao:", out["loao_auroc_pooled"])


if __name__ == "__main__":
    main()
