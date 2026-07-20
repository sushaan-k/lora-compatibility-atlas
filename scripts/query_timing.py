#!/usr/bin/env python3
"""CPU wall-clock timing for atlas minimax queries on the shared 7-simplex chart.

The paper claims atlas queries are answered "without a GPU"; this script measures
that claim on the released m=8 shared-chart jets (qwen8_jets.npz and
tinyllama_jets.npz). It imports the EXACT production solver from
experiments_gpu/expB_shared_chart_decision.py -- minimax_score, the SLSQP
epigraph formulation of rho_S = min over the FULL shared 8-adapter simplex of
max_{t in S} q_t, with the production defaults (restarts=6, seed=0, maxiter=300,
ftol=1e-10, min_weight=0.0). For query sizes |S| <= d+1 = 8, expB's decide path
computes the full score as a single minimax_score call (core_score hits its
max_core >= |S| branch), so timing minimax_score directly IS the production
full-score query path.

Two things are timed per substrate:
  1. Direct minimax queries for |S| in {2, 3, 5, 8}: every subset of that size
     (28 / 56 / 56 / 1; if a size ever exceeded 100 subsets we would take the
     first 100 in lexicographic order and flag it -- at m=8 no size does).
     One warmup call per size is excluded. Median / p90 / max ms per query.
  2. The core-table lookup path for r=2: precompute rho_T for every subset T
     with |T| <= r+1 = 3 (8 + 28 + 56 = 92 solver calls, timed once), then
     answer threshold queries "is max over triples T in S of rho_T <= tau" by
     pure table lookup. Because rho is monotone in S (the max over a superset
     is pointwise larger, so the min is larger), the max over triples equals
     core_score's max over all |T| <= 3 whenever |S| >= 3. Lookup latency is
     the median over 1000 lookups (cycling lexicographically through the
     5-subsets, and separately the single 8-set), tau = 0.65 from the
     production tau grid.

No extrapolation is performed: the m=50 (d=49) jets are not on disk, so nothing
is claimed about that regime (see "caveats" in the output JSON).
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy

REPO = Path(__file__).resolve().parents[1]
JETS = {
    "qwen8": REPO / "results_public/shared_chart_decision/qwen8_jets.npz",
    "tinyllama": REPO / "results_public/shared_chart_decision/tinyllama_jets.npz",
}
OUT = REPO / "results_public/query_timing/query_timing.json"
SUBSET_SIZES = (2, 3, 5, 8)
MAX_SUBSETS_PER_SIZE = 100
R = 2                      # production rank (r2 decision JSONs)
TAU = 0.65                 # from the production tau grid 0.55,0.65,0.70,0.75,0.85
N_LOOKUPS = 1000
MIN_WEIGHT = 0.0           # expB decide default


def load_expB():
    """Import the production module so minimax_score is reused verbatim."""
    path = REPO / "experiments_gpu" / "expB_shared_chart_decision.py"
    spec = importlib.util.spec_from_file_location("expB_shared_chart_decision", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_jets(path):
    z = np.load(path, allow_pickle=True)
    return (np.asarray(z["alpha"], float), list(np.asarray(z["b"], float)),
            list(np.asarray(z["H"], float)), [str(x) for x in z["names"]])


def percentiles_ms(times_s):
    a = np.asarray(times_s) * 1e3
    return {"median_ms": float(np.median(a)), "p90_ms": float(np.percentile(a, 90)),
            "max_ms": float(a.max()), "n_queries": int(a.size)}


def time_minimax_queries(expB, alphas, bs, Hs):
    n = len(alphas)
    per_size = {}
    for size in SUBSET_SIZES:
        subsets = list(itertools.combinations(range(n), size))  # lexicographic
        truncated = len(subsets) > MAX_SUBSETS_PER_SIZE
        if truncated:
            subsets = subsets[:MAX_SUBSETS_PER_SIZE]
        # one warmup call, excluded from timing
        expB.minimax_score(list(subsets[0]), alphas, bs, Hs, min_weight=MIN_WEIGHT)
        times = []
        for S in subsets:
            t0 = time.perf_counter()
            expB.minimax_score(list(S), alphas, bs, Hs, min_weight=MIN_WEIGHT)
            times.append(time.perf_counter() - t0)
        stats = percentiles_ms(times)
        stats["n_subsets_available"] = int(len(list(itertools.combinations(range(n), size))))
        stats["truncated_to_first_100_lexicographic"] = bool(truncated)
        per_size[str(size)] = stats
    return per_size


def build_core_table(expB, alphas, bs, Hs, r):
    """Precompute rho_T for all |T| <= r+1 (what the (r+1)-core screening uses)."""
    n = len(alphas)
    table = {}
    t0 = time.perf_counter()
    for size in range(1, r + 2):
        for T in itertools.combinations(range(n), size):
            table[T] = expB.minimax_score(list(T), alphas, bs, Hs, min_weight=MIN_WEIGHT)
    build_s = time.perf_counter() - t0
    return table, build_s


def time_table_lookups(table, n, query_size, r, tau):
    """Threshold query by lookup: max over (r+1)-subsets of S of table[T] <= tau.
    Cycles lexicographically through all subsets of query_size; one warmup."""
    queries = list(itertools.combinations(range(n), query_size))

    def lookup(S):
        best = -np.inf
        for T in itertools.combinations(S, r + 1):
            v = table[T]
            if v > best:
                best = v
        return best <= tau

    lookup(queries[0])  # warmup
    times = []
    for i in range(N_LOOKUPS):
        S = queries[i % len(queries)]
        t0 = time.perf_counter()
        lookup(S)
        times.append(time.perf_counter() - t0)
    return percentiles_ms(times)


def main():
    expB = load_expB()
    result = {
        "description": "CPU wall-clock timing of the production minimax atlas query "
                       "(expB_shared_chart_decision.minimax_score: SLSQP epigraph over the "
                       "full shared 8-adapter simplex, restarts=6, maxiter=300, ftol=1e-10, "
                       "min_weight=0.0) and the r=2 core-table lookup path, on the released "
                       "m=8 shared-chart jets.",
        "platform": {
            "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip(),
            "os": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "settings": {
            "subset_sizes": list(SUBSET_SIZES),
            "max_subsets_per_size": MAX_SUBSETS_PER_SIZE,
            "subset_order": "lexicographic (itertools.combinations); truncation to the "
                            "first 100 would apply only if a size exceeded 100 subsets -- "
                            "at m=8 the counts are 28/56/56/1, so none were truncated",
            "solver": "expB_shared_chart_decision.minimax_score (imported, not reimplemented)",
            "restarts": 6, "seed": 0, "min_weight": MIN_WEIGHT,
            "r": R, "tau": TAU, "n_lookups": N_LOOKUPS,
            "warmup": "one untimed call per subset size and before lookup timing",
            "timer": "time.perf_counter",
        },
        "substrates": {},
        "caveats": [
            "Measured only on the released m=8 (d=7) shared-chart jets. The m=50 (d=49) "
            "jets are not on disk, so NO timing is reported or extrapolated for d=49; "
            "SLSQP cost grows with dimension and constraint count, so d=49 latency must "
            "be measured, not inferred from these numbers.",
            "Single machine (Apple M4 Pro), single process, default BLAS threading; at "
            "d=7 the linear algebra is negligible and the cost is SLSQP iteration overhead.",
            "Each timed query includes all 6 SLSQP restarts, matching the production "
            "defaults in expB's decide path.",
        ],
    }
    for name, path in JETS.items():
        alphas, bs, Hs, names = load_jets(path)
        n = len(alphas)
        print(f"[{name}] m={n} adapters, d={Hs[0].shape[0]} chart dims "
              f"({path.name})", flush=True)
        per_size = time_minimax_queries(expB, alphas, bs, Hs)
        for size in SUBSET_SIZES:
            s = per_size[str(size)]
            print(f"  |S|={size}: median {s['median_ms']:.2f} ms  p90 {s['p90_ms']:.2f} ms  "
                  f"max {s['max_ms']:.2f} ms  ({s['n_queries']} queries)", flush=True)
        table, build_s = build_core_table(expB, alphas, bs, Hs, R)
        lookup5 = time_table_lookups(table, n, 5, R, TAU)
        lookup8 = time_table_lookups(table, n, 8, R, TAU)
        print(f"  core table (r={R}, {len(table)} entries, |T|<={R+1}): built in "
              f"{build_s:.2f} s", flush=True)
        print(f"  table lookup |S|=5: median {lookup5['median_ms']*1e3:.1f} us over "
              f"{N_LOOKUPS} lookups; |S|=8: median {lookup8['median_ms']*1e3:.1f} us",
              flush=True)
        result["substrates"][name] = {
            "jets_file": str(path.relative_to(REPO)),
            "n_adapters": n, "chart_dim": int(Hs[0].shape[0]),
            "adapter_names": names,
            "minimax_query_ms_per_subset_size": per_size,
            "core_table": {
                "r": R, "core_order": R + 1, "n_entries": len(table),
                "entry_sizes": "all subsets with |T| <= r+1 (sizes 1,2,3)",
                "build_seconds": float(build_s),
                "n_solver_calls": len(table),
                "threshold_lookup_size5": lookup5,
                "threshold_lookup_size8": lookup8,
                "lookup_semantics": f"max over (r+1)-subsets of S of precomputed rho_T, "
                                    f"compared to tau={TAU}; equals core_score's max over "
                                    f"all |T|<=r+1 by monotonicity of rho in S",
            },
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
