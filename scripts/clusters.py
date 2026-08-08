#!/usr/bin/env python3
"""Clustered atlas: the constructive answer to the m=50 negative certification."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments_gpu"))
import decide as expB  # noqa: E402


def top_eigenspace(H, r):
    w, V = np.linalg.eigh(H)
    idx = np.argsort(w)[::-1][:r]
    return V[:, idx]


def subspace_affinity(U, W):
    """Mean squared cosine of principal angles between two orthonormal bases:
    (1/r) * ||U^T W||_F^2, in [0,1], 1 iff the subspaces coincide."""
    M = U.T @ W
    return float(np.sum(M * M) / U.shape[1])


def balanced_similarity_partition(aff, size, rng):
    """Greedy balanced partition into charts of `size` adapters: repeatedly seed a new chart with the least-assigned adapter and grow it by highest mean affi"""
    m = aff.shape[0]
    remaining = set(range(m))
    charts = []
    order = list(rng.permutation(m))
    while remaining:
        seed = next(i for i in order if i in remaining)
        chart = [seed]
        remaining.discard(seed)
        while len(chart) < size and remaining:
            cand = max(remaining, key=lambda j: np.mean([aff[j, c] for c in chart]))
            chart.append(cand)
            remaining.discard(cand)
        charts.append(chart)
    return charts


def random_partition(m, size, rng):
    perm = rng.permutation(m).tolist()
    return [perm[i:i + size] for i in range(0, m, size)]


def cluster_sigma_adim(Hs, members, epsilons=(0.20, 0.10)):
    """Smallest shared rank whose strict max-task residual falls below each eps,
    using one projector from the cluster mean Hessian on the FULL d-dim space.
    Tests whether the adapters' full-dimensional curvatures share a low-rank
    subspace (they need not, even when each small chart is itself low-rank)."""
    sub = [Hs[t] for t in members]
    d = sub[0].shape[0]
    out = {}
    residuals = []
    for r in range(1, min(len(sub), d) + 1):
        res = expB.shared_projector_residual(sub, r)
        residuals.append((r, float(res)))
    for eps in epsilons:
        hit = next((r for r, res in residuals if res < eps), None)
        out[f"sigma_adim_{eps:.2f}"] = hit
    out["residuals"] = residuals
    return out


def subblock_sigma_adim(Hs, members, epsilons=(0.20, 0.10)):
    """Shared rank of the jets RESTRICTED to the cluster's own sub-simplex: the Hessian sub-block on the cluster coordinates."""
    d = Hs[0].shape[0]
    cix = [i for i in members if i < d]
    if len(cix) < 2:
        return {f"sigma_adim_{e:.2f}": None for e in epsilons}
    blocks = [np.asarray(Hs[t])[np.ix_(cix, cix)] for t in members]
    out = {}
    for r in range(1, len(cix) + 1):
        res = expB.shared_projector_residual(blocks, r)
        if all(f"sigma_adim_{e:.2f}" in out for e in epsilons):
            break
        for e in epsilons:
            key = f"sigma_adim_{e:.2f}"
            if key not in out and res < e:
                out[key] = r
    for e in epsilons:
        out.setdefault(f"sigma_adim_{e:.2f}", None)
    return out


def core_table_cost(order, size):
    """Number of subset scores in an (order)-core table over a chart of `size`
    adapters: all subsets up to that order."""
    return int(sum(_c(size, k) for k in range(1, order + 1)))


def _c(n, k):
    from math import comb
    return comb(n, k) if 0 <= k <= n else 0


def load_query_subsets():
    """Real pair/triple queries from the released balanced m=50 panel, as index
    sets into the adapter name list (for the coverage tradeoff)."""
    import csv
    path = REPO / "results_public" / "qwen25_m50_balanced" / "results.csv"
    if not path.exists():
        return None
    subs = []
    for r in csv.DictReader(open(path)):
        if r["query_kind"].strip().lower() in ("pair", "triple"):
            subs.append(tuple(sorted(r["adapters"].split("+"))))
    return subs


def main():
    z = np.load(REPO / "results_public" / "shared_chart_decision_m50" / "qwen50_jets.npz",
                allow_pickle=True)
    alphas = np.asarray(z["alpha"], float)
    bs = [np.asarray(b, float) for b in z["b"]]
    Hs = [np.asarray(H, float) for H in z["H"]]
    names = [str(x) for x in z["names"]]
    m, d = len(names), Hs[0].shape[0]

    # library-wide baseline
    lib = cluster_sigma_adim(Hs, list(range(m)))
    print(f"library-wide (m={m}, d={d}): sigma-adim_0.20 = {lib['sigma_adim_0.20']}, "
          f"_0.10 = {lib['sigma_adim_0.10']}", flush=True)

    # adapter subspace descriptors and affinity
    r0 = 3
    U = [top_eigenspace(H, r0) for H in Hs]
    aff = np.array([[subspace_affinity(U[i], U[j]) for j in range(m)] for i in range(m)])

    rng = np.random.default_rng(20260711)
    result = {"library": {"m": m, "d": d, "sigma_adim_0.20": lib["sigma_adim_0.20"],
                          "sigma_adim_0.10": lib["sigma_adim_0.10"]},
              "clusterings": {}}

    query_subs = load_query_subsets()
    name_to_idx = {nm: i for i, nm in enumerate(names)}

    def eval_partition(charts):
        """Per-chart sigma-adim (0.20), effective per-chart rank (fallback to
        chart dim when uncertified), coverage of real queries, and total
        (r+1)-core scores across charts."""
        cl_of = {}
        for ci, c in enumerate(charts):
            for t in c:
                cl_of[t] = ci
        r20 = [cluster_sigma_adim(Hs, c)["sigma_adim_0.20"] for c in charts]
        sub20 = [subblock_sigma_adim(Hs, c)["sigma_adim_0.20"] for c in charts]
        eff = [(r if r else len(c) - 1) for r, c in zip(sub20, charts)]
        core = sum(core_table_cost(e + 1, len(c)) for e, c in zip(eff, charts))
        cov = None
        if query_subs:
            inside = counted = 0
            for sub in query_subs:
                idx = [name_to_idx[a] for a in sub if a in name_to_idx]
                if len(idx) != len(sub):
                    continue
                counted += 1
                if len({cl_of[t] for t in idx}) == 1:
                    inside += 1
            cov = inside / counted if counted else None
        certified = [r for r in sub20 if r]
        return {"r20": r20, "subblock20": sub20, "eff": eff,
                "n_certified": len(certified), "n_charts": len(charts),
                "median_eff": float(np.median(eff)), "max_eff": int(max(eff)),
                "median_fulldim_r20":
                    float(np.median([r for r in r20 if r])) if any(r20) else None,
                "fulldim_n_certified": len([r for r in r20 if r]),
                "total_core_scores": core, "within_query_frac": cov}

    lib_core = core_table_cost((lib["sigma_adim_0.20"] or d) + 1, m)
    result["library"]["core_scores_at_certified_rank"] = lib_core

    for size in (5, 8, 10, 13, 25):
        clus = eval_partition(balanced_similarity_partition(aff, size, np.random.default_rng(1)))
        rand = [eval_partition(random_partition(m, size, np.random.default_rng(s)))
                for s in range(20)]
        rand_med_eff = float(np.median([r["median_eff"] for r in rand]))
        rand_max_eff = float(np.median([r["max_eff"] for r in rand]))
        rand_ncert = float(np.median([r["n_certified"] for r in rand]))
        rand_core = float(np.median([r["total_core_scores"] for r in rand]))
        result["clusterings"][f"size{size}"] = {
            "chart_size": size, "n_charts": clus["n_charts"],
            "clustered_subblock_sigma_adim_0.20": clus["subblock20"],
            "clustered_median_subblock_eff_rank": clus["median_eff"],
            "clustered_max_subblock_eff_rank": clus["max_eff"],
            "clustered_n_certified_subblock": clus["n_certified"],
            "clustered_fulldim_sigma_adim_0.20": clus["r20"],
            "clustered_median_fulldim_r20": clus["median_fulldim_r20"],
            "clustered_fulldim_n_certified": clus["fulldim_n_certified"],
            "clustered_total_core_scores": clus["total_core_scores"],
            "random_median_subblock_eff_rank": rand_med_eff,
            "random_median_max_subblock_eff_rank": rand_max_eff,
            "random_median_n_certified_subblock": rand_ncert,
            "random_median_total_core_scores": rand_core,
            "library_core_scores": lib_core,
            "within_cluster_query_fraction": clus["within_query_frac"],
        }
        print(f"size={size} ({clus['n_charts']} charts): SUBBLOCK eff-rank "
              f"median {clus['median_eff']} max {clus['max_eff']} "
              f"({clus['n_certified']}/{clus['n_charts']} certified) | "
              f"FULLDIM shared-rank certified {clus['fulldim_n_certified']}/{clus['n_charts']} "
              f"median {clus['median_fulldim_r20']} | "
              f"random subblock median {rand_med_eff} | "
              f"within-cluster queries {clus['within_query_frac']}", flush=True)

    out = REPO / "results_public" / "clustered_atlas"
    out.mkdir(parents=True, exist_ok=True)
    (out / "clusters.json").write_text(json.dumps(result, indent=2))
    print("\nwrote", out / "clusters.json")


if __name__ == "__main__":
    main()
