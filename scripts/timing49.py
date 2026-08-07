#!/usr/bin/env python3
"""CPU wall-clock timing for atlas minimax queries at library scale (d=49)."""
from __future__ import annotations

import importlib.util
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "peft_atlas_lite"))
from run_peft_atlas_lite import (fit_quadratic_model,  # noqa: E402
                                 hessian_from_quadratic_coeff)

QUERY_SIZES = (2, 3, 5)
N_PER_SIZE = 20
SEED = 0


def load_expB():
    path = REPO / "experiments_gpu" / "decide.py"
    spec = importlib.util.spec_from_file_location("expB", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    np.seterr(all="ignore")
    expB = load_expB()

    z = np.load(REPO / "results_public/scale_m50/answer_jets_raw.npz", allow_pickle=True)
    grid = [list(w) for w in z["grid"]]
    names = [str(x) for x in z["names"]]
    vals = z["values"]
    d = len(names) - 1

    # The solver indexes tasks and adapters by one index set (n = len(alphas)),
    # so every adapter slot must be present; the six unscoreable tasks get
    # placeholder entries and are never included in a query's member set.
    alphas = [0.0] * len(names)
    bs = [np.zeros(d) for _ in names]
    Hs = [np.zeros((d, d)) for _ in names]
    scored = []
    for i, nm in enumerate(names):
        if np.isnan(vals[i]).any():
            continue
        coeff, _ = fit_quadratic_model(grid, list(vals[i]))
        alphas[i] = float(coeff[0])
        bs[i] = np.asarray(coeff[1:1 + d], dtype=float)
        Hs[i] = np.asarray(hessian_from_quadratic_coeff(coeff, d), dtype=float)
        scored.append(i)
    print(f"fitted {len(scored)} task models on the d={d} chart", flush=True)

    rng = np.random.default_rng(SEED)
    out = {
        "description": ("CPU wall-clock timing of the production minimax atlas query "
                        "(decide.minimax_score, restarts=6, seed=0, "
                        "maxiter=300, ftol=1e-10, min_weight=0.0) on the released Mistral "
                        "m=50 answer-token index, chart dimension d=49."),
        "platform": {"cpu": platform.processor() or platform.machine(),
                     "os": platform.platform(), "python": platform.python_version(),
                     "numpy": np.__version__},
        "settings": {"substrate": "results_public/scale_m50/answer_jets_raw.npz",
                     "n_adapters": len(names), "n_scored_tasks": len(scored),
                     "chart_dim": d, "restarts": 6, "seed": SEED, "min_weight": 0.0,
                     "query_sizes": list(QUERY_SIZES), "n_queries_per_size": N_PER_SIZE,
                     "query_selection": f"uniform without replacement, seed {SEED}",
                     "warmup": "one untimed call per size", "timer": "time.perf_counter"},
        "minimax_query_ms_per_subset_size": {},
    }

    for k in QUERY_SIZES:
        subsets = [tuple(sorted(rng.choice(scored, size=k, replace=False).tolist()))
                   for _ in range(N_PER_SIZE)]
        expB.minimax_score(list(subsets[0]), alphas, bs, Hs, min_weight=0.0)  # warmup
        times = []
        for S in subsets:
            t0 = time.perf_counter()
            expB.minimax_score(list(S), alphas, bs, Hs, min_weight=0.0)
            times.append((time.perf_counter() - t0) * 1000.0)
        times = np.asarray(times)
        out["minimax_query_ms_per_subset_size"][str(k)] = {
            "median_ms": float(np.median(times)), "p90_ms": float(np.percentile(times, 90)),
            "max_ms": float(times.max()), "n_queries": len(times)}
        print(f"  |S|={k}: median {np.median(times):.0f} ms, p90 {np.percentile(times,90):.0f} ms",
              flush=True)

    out["caveats"] = [
        "Measured on the Mistral m=50 answer-token index, the only released fifty-adapter "
        "chart whose fitted models are on disk; the Qwen m=50 panel's jets are not released, "
        "so its query latency is not reported here.",
        "min_weight=0.0. At the default floor 0.2 a five-way query exhausts the coefficient "
        "mass on the selected adapters and reduces to a single equal-floor point, so the "
        "zero-floor timing is the upper bound.",
        "Single machine, single process, default BLAS threading; each timed query includes "
        "all 6 SLSQP restarts, matching the production decide path.",
    ]
    od = REPO / "results_public" / "query_timing"
    od.mkdir(parents=True, exist_ok=True)
    (od / "timing49.json").write_text(json.dumps(out, indent=1))
    print("wrote", od / "timing49.json")


if __name__ == "__main__":
    main()
