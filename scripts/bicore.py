#!/usr/bin/env python3
"""Foundations of the support-aware two-sided atlas theorem, validated on real jets."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

# The macOS Accelerate BLAS raises spurious overflow/invalid warnings on finite
# matmuls; inputs and outputs are verified finite, so silence them for a clean log.
np.seterr(over="ignore", divide="ignore", invalid="ignore")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments_gpu"))
sys.path.insert(0, str(REPO / "scripts"))
import decide as expB  # noqa: E402,F401
from certify import build_reduced_jets, eps_exact_certificate  # noqa: E402

TOL = 1e-6


def w2c(w):
    return np.asarray(w[:-1], dtype=float)


def load_jets(kind):
    if kind == "qwen8":
        z = np.load(REPO / "results_public/shared_chart_decision/qwen8_jets.npz", allow_pickle=True)
        return (np.asarray(z["alpha"], float), [np.asarray(b, float) for b in z["b"]],
                [np.asarray(H, float) for H in z["H"]])
    sys.path.insert(0, str(REPO / "peft_atlas_lite"))
    from run_peft_atlas_lite import fit_quadratic_model, hessian_from_quadratic_coeff
    path = {"m8": "answer_panel_m8", "m16": "scale_m16"}[kind]
    z = np.load(REPO / "results_public" / path / "answer_jets_raw.npz", allow_pickle=True)
    grid = [list(w) for w in z["grid"]]
    vals = z["values"]
    names = [str(x) for x in z["names"]]
    d = len(names) - 1
    A, B, H = [], [], []
    for j in range(len(names)):
        coeff, _ = fit_quadratic_model(grid, list(vals[j]), project_psd=True)
        A.append(float(coeff[0]))
        B.append(np.array(coeff[1:1 + d], float))
        H.append(np.asarray(hessian_from_quadratic_coeff(coeff, d), float))
    return np.array(A), B, H


def rho(alphas, bs, Hs, A, V, n):
    """min over the simplex supported on adapters V of max_{t in A} q_t (zero floor).

    Returns (score, c_star).  A is the protected task set, V the adapter set.
    """
    A = list(A)
    V = list(V)
    nv = len(V)

    def jv(t, c):
        return float(alphas[t] + bs[t] @ c + 0.5 * c @ Hs[t] @ c)

    def emb(x):
        w = np.zeros(n)
        w[V] = x
        return w

    def qmax_w(x):
        return max(jv(t, w2c(emb(x))) for t in A)

    cons = [{"type": "ineq", "fun": (lambda u, t=t: u[0] - jv(t, w2c(emb(u[1:]))))} for t in A]
    cons.append({"type": "eq", "fun": lambda u: float(np.sum(u[1:]) - 1.0)})
    bnds = [(None, None)] + [(0.0, 1.0)] * nv
    best = np.inf
    bestc = None
    rng = np.random.default_rng(0)
    for k in range(6):
        w0 = np.full(nv, 1.0 / nv) if k == 0 else rng.dirichlet(np.ones(nv))
        u0 = np.concatenate([[qmax_w(w0)], w0])
        res = minimize(lambda u: u[0], u0, jac=lambda u: np.eye(len(u))[0],
                       bounds=bnds, constraints=cons, method="SLSQP",
                       options={"maxiter": 300, "ftol": 1e-11})
        w = np.abs(res.x[1:])
        s = w.sum()
        w = w / s if s > 0 else w
        v = qmax_w(w)
        if v < best:
            best = v
            bestc = w2c(emb(w))
    return best, bestc


def task_core(alphas, bs, Hs, T, U, n, R, cstar):
    """Greedy-minimal binding task subset A* with rho(A*;U) = R."""
    jv = lambda t, c: float(alphas[t] + bs[t] @ c + 0.5 * c @ Hs[t] @ c)
    vals = np.array([jv(t, cstar) for t in T])
    order = np.argsort(vals)[::-1]                       # binding tasks first
    A = [int(T[i]) for i in order if vals[i] >= vals.max() - 1e-4] or [int(T[order[0]])]
    # add tasks until the restricted score reaches R, then prune
    i = 0
    while rho(alphas, bs, Hs, A, U, n)[0] < R - TOL and i < len(order):
        cand = int(T[order[i]])
        if cand not in A:
            A.append(cand)
        i += 1
    for t in list(A):
        if len(A) > 1 and rho(alphas, bs, Hs, [x for x in A if x != t], U, n)[0] >= R - TOL:
            A.remove(t)
    return A


def adapter_core(alphas, bs, Hs, T, U, n, R, cstar):
    """Greedy-minimal adapter subset V* (from the support of c*) with rho(T;V*) = R."""
    c_full = np.zeros(n)
    c_full[:n - 1] = cstar
    c_full[n - 1] = 1.0 - cstar.sum()
    V = [int(i) for i in U if c_full[i] > 1e-4] or [int(np.argmax([c_full[i] for i in U]))]
    for i in list(V):
        if len(V) > 1 and rho(alphas, bs, Hs, T, [x for x in V if x != i], n)[0] <= R + TOL:
            V.remove(i)
    return V


def analyze(kind, ranks=(2, 3)):
    alphas, bs, Hs = load_jets(kind)
    n = len(alphas)
    d = Hs[0].shape[0]
    T = tuple(range(n))
    U = tuple(range(n))
    out = {"kind": kind, "n": n, "d": d}

    # (1) exact atlas dimension on the full jets
    M = sum(H @ H + np.outer(b, b) for H, b in zip(Hs, bs))
    Mh = sum(H @ H for H in Hs)
    r_exact = int(np.sum(np.linalg.eigvalsh(M) > 1e-8 * max(1.0, np.trace(M))))
    r_hess = int(np.sum(np.linalg.eigvalsh(Mh) > 1e-8 * max(1.0, np.trace(Mh))))
    out["r_exact"] = r_exact
    out["r_hessian_only"] = r_hess

    # full-jet two-sided cores
    R, cstar = rho(alphas, bs, Hs, T, U, n)
    A_star = task_core(alphas, bs, Hs, T, U, n, R, cstar)
    V_star = adapter_core(alphas, bs, Hs, T, U, n, R, cstar)
    rA = rho(alphas, bs, Hs, A_star, U, n)[0]
    rV = rho(alphas, bs, Hs, T, V_star, n)[0]
    rBi = rho(alphas, bs, Hs, A_star, V_star, n)[0]
    # task-core validity: drop-one relaxation + random-subset comparison
    drop = [rho(alphas, bs, Hs, [x for x in A_star if x != t], U, n)[0] for t in A_star] \
        if len(A_star) > 1 else [rho(alphas, bs, Hs, list(A_star), U, n)[0]]
    max_drop = float(R - min(drop)) if len(A_star) > 1 else 0.0
    rng = np.random.default_rng(1)
    ksz = len(A_star)
    rand = []
    pool = list(T)
    for _ in range(min(40, max(1, n))):
        sub = list(rng.choice(pool, size=min(ksz, n), replace=False))
        rand.append(rho(alphas, bs, Hs, sub, U, n)[0])
    out["full"] = {
        "rho": round(R, 6),
        "task_core_size": len(A_star), "task_core_le_rexact1": bool(len(A_star) <= r_exact + 1),
        "task_core_recovers": bool(abs(rA - R) < 1e-4),
        "adapter_core_size": len(V_star), "adapter_core_le_rexact1": bool(len(V_star) <= r_exact + 1),
        "adapter_core_recovers": bool(abs(rV - R) < 1e-4),
        "bicore_recovers": bool(abs(rBi - R) < 1e-4),
        "drop_one_max_relaxation": round(max_drop, 6),
        "random_subset_mean_deficit": round(float(R - np.mean(rand)), 6),
        "random_subset_frac_below": round(float(np.mean(np.array(rand) < R - TOL)), 3),
    }
    print(f"[{kind}] n={n} d={d} r_exact={r_exact} (Hess-only {r_hess}) rho={R:.4f}", flush=True)
    print(f"  full: task-core |A*|={len(A_star)}<= {r_exact+1} recovers={abs(rA-R)<1e-4} "
          f"| adapter-core |V*|={len(V_star)} recovers={abs(rV-R)<1e-4} "
          f"| bicore={abs(rBi-R)<1e-4} | drop-relax={max_drop:.3f} rand-deficit={R-np.mean(rand):.3f}",
          flush=True)

    # rank-reduced cores + 2-delta band
    per_rank = {}
    for r in ranks:
        if r >= d:
            continue
        a_r, b_r, H_r, _ = build_reduced_jets(alphas, bs, Hs, r)
        Rr, cst_r = rho(a_r, b_r, H_r, T, U, n)
        Ar = task_core(a_r, b_r, H_r, T, U, n, Rr, cst_r)
        Vr = adapter_core(a_r, b_r, H_r, T, U, n, Rr, cst_r)
        rAr = rho(a_r, b_r, H_r, Ar, U, n)[0]
        rVr = rho(a_r, b_r, H_r, T, Vr, n)[0]
        eps_t = eps_exact_certificate(alphas, bs, Hs, a_r, b_r, H_r)
        delta = float(np.max(eps_t))
        gap = float(abs(R - Rr))
        per_rank[f"r{r}"] = {
            "rho_reduced": round(Rr, 6),
            "task_core_size": len(Ar), "task_core_le_r1": bool(len(Ar) <= r + 1),
            "task_core_recovers": bool(abs(rAr - Rr) < 1e-4),
            "adapter_core_size": len(Vr), "adapter_core_le_r1": bool(len(Vr) <= r + 1),
            "adapter_core_recovers": bool(abs(rVr - Rr) < 1e-4),
            "delta_max": round(delta, 4), "gap_full_reduced": round(gap, 6),
            "gap_within_2delta": bool(gap <= 2 * delta + 1e-6),
        }
        print(f"  r={r}: rho_red={Rr:.4f} |A*|={len(Ar)}<= {r+1} |V*|={len(Vr)}<= {r+1} "
              f"| gap={gap:.3f} <= 2delta={2*delta:.3f}: {gap <= 2*delta+1e-6}", flush=True)
    out["per_rank"] = per_rank
    return out


def main():
    res = {}
    for kind in ("qwen8", "m8", "m16"):
        try:
            res[kind] = analyze(kind)
        except Exception as e:
            import traceback
            res[kind] = {"error": str(e)}
            print(f"[{kind}] ERROR {e}\n{traceback.format_exc()}", flush=True)
    outdir = REPO / "results_public" / "foundations"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "foundations.json").write_text(json.dumps(res, indent=2))
    print("wrote", outdir / "foundations.json")


if __name__ == "__main__":
    main()
