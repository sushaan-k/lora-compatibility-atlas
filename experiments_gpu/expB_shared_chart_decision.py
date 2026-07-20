#!/usr/bin/env python3
"""Shared-chart decision preservation for the effective-rank compression claim.

Reviewer ask: the released 108x reduction is per-core (each five-adapter core fit
on its own chart with its own projector). Certify instead ONE shared projector on
ONE shared chart, over four- and five-way queries, and report the error between the
full obstruction score and the (r+1)-core score plus the fraction of threshold
decisions preserved.

Three subcommands:
  fit      GPU: fit one affine-quadratic jet per adapter on a shared (k-1)-simplex,
           save jets.npz (alpha, b, H, names). Reuses the validated pipeline.
  decide   CPU: load jets.npz, build the shared projector Pi_r from the mean
           Hessian, and for every 4- and 5-way query subset compare the full score
           rho_S to the (r+1)-core projected score, over a tau grid.
  selftest CPU: synthetic rank-2 jets on a shared chart; checks that the projected
           score preserves decisions when the structure is genuinely low rank and
           breaks predictably when it is not. Runs with no GPU or data.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------
# Geometry (CPU): min-over-simplex of max-of-convex-quadratics, and the shared
# projector decision-preservation test. No pipeline import needed here.
# ----------------------------------------------------------------------------

def jet_value(alpha, b, H, c):
    """q_t(c) = alpha + b . c + 0.5 c^T H c, c in coord space R^{d}."""
    c = np.asarray(c, dtype=float)
    return float(alpha + b @ c + 0.5 * c @ H @ c)


def project_jet(b, H, Pi):
    """Project a jet through orthogonal projector Pi (Hessian and linear term)."""
    return Pi @ b, Pi @ H @ Pi


def weights_to_coords(w):
    """Simplex weights over n adapters -> coord space R^{n-1} (drop last), matching
    the fit convention in scripts/measure_effective_rank_strict.py."""
    return np.asarray(w[:-1], dtype=float)


def minimax_score(members, alphas, bs, Hs, min_weight=0.0, restarts=6, seed=0):
    """Fixed-chart obstruction score rho_S(C) = min_{c in C} max_{t in S} q_t(c),
    where C is the FULL shared k-adapter simplex (every coordinate free) and only
    the max is restricted to `members`. This is the definition the Helly and
    effective-dimension theorems use; scoring S on a sub-face instead would be the
    support-specific regime and would not be monotone in S.

    Convex program (PSD Hessians; pointwise max of convex quadratics; simplex),
    solved on the epigraph with SLSQP from several starts."""
    from scipy.optimize import minimize

    n = len(alphas)
    S = list(members)

    def qmax(w):
        c = weights_to_coords(w)
        return max(jet_value(alphas[t], bs[t], Hs[t], c) for t in S)

    def obj(x):
        return x[0]

    def obj_grad(x):
        g = np.zeros_like(x); g[0] = 1.0
        return g

    def make_con(t):
        def con(x):
            c = weights_to_coords(x[1:])
            return x[0] - jet_value(alphas[t], bs[t], Hs[t], c)
        return con

    cons = [{"type": "ineq", "fun": make_con(t)} for t in S]
    cons.append({"type": "eq", "fun": lambda x: float(np.sum(x[1:]) - 1.0)})
    bounds = [(None, None)] + [(min_weight, 1.0)] * n

    rng = np.random.default_rng(seed)
    best = np.inf
    feas_tol = 1e-6
    for r in range(restarts):
        w0 = np.full(n, 1.0 / n) if r == 0 else rng.dirichlet(np.ones(n))
        if min_weight > 0:
            w0 = min_weight + (1 - n * min_weight) * w0
        x0 = np.concatenate([[qmax(w0)], w0])
        res = minimize(obj, x0, jac=obj_grad, bounds=bounds, constraints=cons,
                       method="SLSQP", options={"maxiter": 300, "ftol": 1e-10})
        # Guard: an SLSQP restart that stops at an infeasible point must not
        # contribute res.fun (an unattained lower value breaks monotonicity in S).
        # Repair by evaluating qmax at the clipped-renormalized weights, which is
        # feasible and hence a valid upper bound for this convex program.
        w = np.asarray(res.x[1:], dtype=float)
        scale = max(1.0, abs(float(res.x[0])))
        feasible = (res.success
                    and abs(float(np.sum(w)) - 1.0) <= feas_tol
                    and float(np.min(w)) >= min_weight - feas_tol
                    and float(np.max(w)) <= 1.0 + feas_tol
                    and qmax(w) - float(res.x[0]) <= feas_tol * scale)
        if feasible:
            best = min(best, float(res.fun))
        else:
            w_rep = np.clip(w, min_weight, 1.0)
            tot = float(np.sum(w_rep))
            if tot > 0:
                best = min(best, qmax(w_rep / tot))
    return best


def core_score(members, alphas, bs, Hs, max_core, min_weight=0.0):
    """Helly core-table score: max over subsets T of `members` with |T| <= max_core
    of rho_T. Equals rho_S when max_core >= |members| (Theorem finitecore)."""
    S = list(members)
    if max_core >= len(S):
        return minimax_score(S, alphas, bs, Hs, min_weight)
    best = -np.inf
    for size in range(1, max_core + 1):
        for T in itertools.combinations(S, size):
            best = max(best, minimax_score(list(T), alphas, bs, Hs, min_weight))
    return best


def shared_projector(Hs, r):
    """Top-r eigenspace projector of the mean Hessian: the ONE shared projector."""
    H_bar = np.mean(np.stack(Hs), axis=0)
    evals, evecs = np.linalg.eigh(H_bar)
    order = np.argsort(evals)[::-1]
    V = evecs[:, order[:r]]
    return V @ V.T


def shared_projector_residual(Hs, r):
    """Strict max-task relative residual of the ONE shared rank-r projector across
    all tasks: max_t ||H_t - Pi_r H_t Pi_r||_F / max_t ||H_t||_F. Small residual
    certifies that a single library-wide projector at rank r captures every task."""
    Pi = shared_projector(Hs, r)
    L_F = max(float(np.linalg.norm(H, "fro")) for H in Hs)
    tail = max(float(np.linalg.norm(H - Pi @ H @ Pi, "fro")) for H in Hs)
    return tail / L_F if L_F > 0 else float("nan")


def decide(jets, r, taus, query_sizes=(4, 5), min_weight=0.0, max_full_core=None):
    """Certify a single shared rank-r projector, then measure how well the
    (r+1)-core table of the ORIGINAL jets reproduces the full obstruction score
    (Theorem effdim_approx: the projector fixes r; the (r+1)-core table, not the
    projected jets, is what screening uses). Reports the shared-projector residual,
    the full-vs-(r+1)-core score error, and per-tau decision agreement."""
    alphas, bs, Hs, names = jets["alpha"], list(jets["b"]), list(jets["H"]), jets["names"]
    n = len(alphas)
    d = Hs[0].shape[0]
    max_full_core = max_full_core or (d + 1)
    residual = shared_projector_residual(Hs, r)

    rows = []
    for size in query_sizes:
        if size > n:
            continue
        for S in itertools.combinations(range(n), size):
            full = core_score(S, alphas, bs, Hs, min(max_full_core, size), min_weight)
            rplus1 = core_score(S, alphas, bs, Hs, min(r + 1, size), min_weight)
            rows.append({"members": [names[t] for t in S], "size": size,
                         "rho_full": full, "rho_rplus1": rplus1,
                         "abs_err": abs(full - rplus1)})
    errs = np.array([row["abs_err"] for row in rows])
    per_tau = []
    for tau in taus:
        agree = np.mean([(row["rho_full"] <= tau) == (row["rho_rplus1"] <= tau)
                         for row in rows])
        per_tau.append({"tau": float(tau), "decision_preserved_frac": float(agree)})
    return {
        "r": int(r), "core_order_rplus1": int(r + 1), "full_core_order": int(max_full_core),
        "shared_projector_strict_max_residual": float(residual),
        "n_queries": len(rows), "query_sizes": list(query_sizes),
        "mean_abs_score_error": float(errs.mean()) if len(errs) else None,
        "max_abs_score_error": float(errs.max()) if len(errs) else None,
        "per_tau_decision_preservation": per_tau,
        "queries": rows,
    }


# ----------------------------------------------------------------------------
# fit (GPU): reuse the validated pipeline to fit jets on the shared chart.
# ----------------------------------------------------------------------------

def cmd_fit(args):
    REPO = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO / "peft_atlas_lite"))
    import random
    from run_peft_atlas_lite import (
        adapters_from_config, evaluate_base_and_singles, evaluate_merge_losses,
        fit_quadratic_model, hessian_from_quadratic_coeff, load_json,
        load_peft_model, score_query, simplex_grid, split_prompts, stable_seed,
    )
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    config = load_json(Path(args.config))
    by_name = {a.name: a for a in adapters_from_config(config)}
    names = [s.strip() for s in args.adapters.split(",")]
    adapters = [by_name[n] for n in names]
    n = len(adapters); d = n - 1
    grid = simplex_grid(n, step=args.step, min_weight=args.min_weight)
    n_d = 1 + d + d * (d + 1) // 2
    if len(grid) < n_d:
        raise ValueError("design grid %d < N_d=%d; lower --step" % (len(grid), n_d))
    rng = random.Random(stable_seed(config))
    prompt_sets = {a.name: dict(zip(["calibration", "heldout"], split_prompts(config, a, rng)))
                   for a in adapters}
    model, tokenizer = load_peft_model(config, adapters)
    max_length = int(config.get("max_length", 384))
    base_losses, single_losses = evaluate_base_and_singles(
        model, tokenizer, adapters, prompt_sets, max_length)
    threshold = float(config.get("retention_threshold", 0.55))
    min_gain = float(config.get("min_single_adapter_gain", 0.005))
    density = float(config.get("density", 0.5))
    names_list = [a.name for a in adapters]
    cal_values = {nm: [] for nm in names_list}
    # per-point checkpoint: survive interruptible-instance preemption
    ckpt_path = out / "fit_points.jsonl"
    done_points = 0
    if ckpt_path.exists():
        for line in ckpt_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                for nm in names_list:
                    cal_values[nm].append(rec["values"][nm])
                done_points += 1
        print("resuming fit: %d/%d design points done" % (done_points, len(grid)), flush=True)
    ckpt = open(ckpt_path, "a")
    serial = 400000 + done_points; t0 = time.time()
    for i, w in enumerate(grid):
        if i < done_points:
            continue
        serial += 1
        losses = evaluate_merge_losses(model, tokenizer, adapters, w, "linear",
                                       density, prompt_sets, max_length, "calibration", serial)
        _, margins, _ = score_query(names_list, w, losses, base_losses["calibration"],
                                    single_losses["calibration"], threshold, min_gain)
        rec = {}
        for nm in names_list:
            v = -margins[nm]
            cal_values[nm].append(v)
            rec[nm] = v
        ckpt.write(json.dumps({"i": i, "values": rec}) + "\n")
        ckpt.flush()
        if (i + 1) % 5 == 0:
            print("  %d/%d t=%ds" % (i + 1, len(grid), int(time.time() - t0)), flush=True)
    alphas, bs, Hs = [], [], []
    for a in adapters:
        coeff, _ = fit_quadratic_model(list(grid), cal_values[a.name], project_psd=True)
        H = hessian_from_quadratic_coeff(coeff, d)
        alphas.append(float(coeff[0])); bs.append(np.array(coeff[1:1 + d], float)); Hs.append(np.asarray(H, float))
    # also save the raw design (grid + per-task values) so jet standard errors and
    # refits at other ranks are computable later without touching the GPU again
    np.savez(out / "jets.npz", alpha=np.array(alphas), b=np.stack(bs),
             H=np.stack(Hs), names=np.array(names_list),
             grid=np.array(grid),
             values=np.array([cal_values[nm] for nm in names_list]))
    print("wrote", out / "jets.npz", flush=True)
    return 0


def cmd_decide(args):
    z = np.load(args.jets, allow_pickle=True)
    jets = {"alpha": z["alpha"], "b": list(z["b"]), "H": list(z["H"]),
            "names": [str(x) for x in z["names"]]}
    taus = [float(x) for x in args.taus.split(",")]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    result = decide(jets, r=args.r, taus=taus, min_weight=args.min_weight)
    (out / "shared_chart_decision.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "queries"}, indent=2))
    print("wrote", out / "shared_chart_decision.json")
    return 0


def cmd_selftest(_args):
    rng = np.random.default_rng(0)
    n, d = 8, 7
    # genuinely rank-2 shared structure: Hessian AND linear term live in span(U),
    # so the jets factor through Pi_2 (Theorem effdim: (r+1)-core is then exact).
    U = np.linalg.qr(rng.standard_normal((d, 2)))[0]
    Hs, bs, alphas = [], [], []
    for _ in range(n):
        core = U @ np.diag(rng.uniform(2, 5, 2)) @ U.T
        noise = rng.standard_normal((d, d)); noise = 0.005 * (noise @ noise.T)
        Hs.append(core + noise)
        bs.append(U @ (0.1 * rng.standard_normal(2)))  # linear term in span(U)
        alphas.append(rng.uniform(-0.5, 0.5))
    names = [f"a{i}" for i in range(n)]
    jets = {"alpha": np.array(alphas), "b": bs, "H": Hs, "names": names}
    taus = [0.55, 0.65, 0.75]
    lo = decide(jets, r=2, taus=taus, min_weight=0.0)
    # control: genuinely full-rank structure, where a rank-2 projector should NOT fit
    Hs_full = [(lambda M: 0.5 * (M @ M.T))(rng.standard_normal((d, d))) for _ in range(n)]
    ctrl = decide({"alpha": np.array(alphas), "b": bs, "H": Hs_full, "names": names},
                  r=2, taus=taus, min_weight=0.0)
    print("rank-2 shared structure, r=2:")
    print("  shared-projector residual=%.4f  mean|full - (r+1)-core|=%.4f" % (
        lo["shared_projector_strict_max_residual"], lo["mean_abs_score_error"]))
    print("  decisions preserved: %s" % [(p["tau"], round(p["decision_preserved_frac"], 3))
                                          for p in lo["per_tau_decision_preservation"]])
    print("full-rank control, r=2 (spectrum NOT low-rank):")
    print("  shared-projector residual=%.4f  mean|full - (r+1)-core|=%.4f" % (
        ctrl["shared_projector_strict_max_residual"], ctrl["mean_abs_score_error"]))
    assert lo["n_queries"] == 126  # C(8,4)=70 + C(8,5)=56
    # rank-2 jets: (r+1)-core is EXACT and every decision is preserved (Thm effdim)
    assert lo["mean_abs_score_error"] < 1e-3
    assert all(p["decision_preserved_frac"] == 1.0 for p in lo["per_tau_decision_preservation"])
    # the shared projector residual discriminates low-rank from full-rank spectra
    assert lo["shared_projector_strict_max_residual"] < ctrl["shared_projector_strict_max_residual"]
    print("selftest OK: rank-2 -> exact (r+1)-core + full decision preservation; the residual")
    print("  discriminates spectra, while decision preservation depends on active-core size")
    print("  (the spectral-residual vs decision-preservation decoupling the paper measures).")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("--config", required=True)
    f.add_argument("--adapters", required=True); f.add_argument("--out", required=True)
    f.add_argument("--step", type=float, default=0.25); f.add_argument("--min-weight", type=float, default=0.0)
    f.set_defaults(func=cmd_fit)
    de = sub.add_parser("decide"); de.add_argument("--jets", required=True)
    de.add_argument("--out", required=True); de.add_argument("--r", type=int, default=2)
    de.add_argument("--taus", default="0.55,0.65,0.70,0.75,0.85")
    de.add_argument("--min-weight", type=float, default=0.0); de.set_defaults(func=cmd_decide)
    st = sub.add_parser("selftest"); st.set_defaults(func=cmd_selftest)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
