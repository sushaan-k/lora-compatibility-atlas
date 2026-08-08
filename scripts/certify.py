#!/usr/bin/env python3
"""Interval certification for the shared-chart rank-r reduced jets."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments_gpu"))
import decide as expB  # noqa: E402

SUBSET_SIZES = (2, 3, 4, 5)
N_TAU = 201
TAU_PAD = 0.05


def all_query_subsets(n):
    subs = []
    for size in SUBSET_SIZES:
        subs.extend(tuple(S) for S in itertools.combinations(range(n), size))
    subs.append(tuple(range(n)))  # the full set (m = 8)
    return subs


def guarded_minimax_score(members, alphas, bs, Hs, min_weight=0.0, restarts=6,
                          seed=0, feas_tol=1e-6):
    """expB.minimax_score's exact SLSQP epigraph program (same constraints, bounds, starts, rng seed, options), plus a feasibility guard per restart."""
    from scipy.optimize import minimize

    n = len(alphas)
    S = list(members)

    def qmax(w):
        c = expB.weights_to_coords(w)
        return max(expB.jet_value(alphas[t], bs[t], Hs[t], c) for t in S)

    def obj(x):
        return x[0]

    def obj_grad(x):
        g = np.zeros_like(x)
        g[0] = 1.0
        return g

    def make_con(t):
        def con(x):
            c = expB.weights_to_coords(x[1:])
            return x[0] - expB.jet_value(alphas[t], bs[t], Hs[t], c)
        return con

    cons = [{"type": "ineq", "fun": make_con(t)} for t in S]
    cons.append({"type": "eq", "fun": lambda x: float(np.sum(x[1:]) - 1.0)})
    bounds = [(None, None)] + [(min_weight, 1.0)] * n

    rng = np.random.default_rng(seed)
    raw = np.inf
    guarded = np.inf
    n_infeasible = 0
    for r in range(restarts):
        w0 = np.full(n, 1.0 / n) if r == 0 else rng.dirichlet(np.ones(n))
        if min_weight > 0:
            w0 = min_weight + (1 - n * min_weight) * w0
        x0 = np.concatenate([[qmax(w0)], w0])
        res = minimize(obj, x0, jac=obj_grad, bounds=bounds, constraints=cons,
                       method="SLSQP", options={"maxiter": 300, "ftol": 1e-10})
        raw = min(raw, float(res.fun))
        w = np.asarray(res.x[1:], dtype=float)
        scale = max(1.0, abs(float(res.x[0])))
        feasible = (res.success
                    and abs(float(np.sum(w)) - 1.0) <= feas_tol
                    and float(np.min(w)) >= min_weight - feas_tol
                    and float(np.max(w)) <= 1.0 + feas_tol
                    and qmax(w) - float(res.x[0]) <= feas_tol * scale)
        if feasible:
            guarded = min(guarded, float(res.fun))
        else:
            n_infeasible += 1
            w_rep = np.clip(w, min_weight, 1.0)
            tot = float(np.sum(w_rep))
            if tot > 0:
                guarded = min(guarded, qmax(w_rep / tot))
    return float(guarded), float(raw), n_infeasible


class MemoScorer:
    """Memoized wrapper around the guarded solver (expB program + feasibility
    guard). rho() returns the guarded value; raw expB values are kept so every
    guarded-vs-raw divergence can be reported as a solver artifact."""

    def __init__(self, alphas, bs, Hs, min_weight=0.0):
        self.alphas, self.bs, self.Hs = alphas, list(bs), list(Hs)
        self.min_weight = min_weight
        self.cache = {}

    def _solve(self, S):
        key = frozenset(S)
        if key not in self.cache:
            self.cache[key] = guarded_minimax_score(
                sorted(S), self.alphas, self.bs, self.Hs, min_weight=self.min_weight)
        return self.cache[key]

    def rho(self, S):
        return self._solve(S)[0]

    def rho_raw(self, S):
        return self._solve(S)[1]

    def solver_artifacts(self, tol=1e-8):
        return [{"subset": sorted(k), "guarded": v[0], "raw_expB": v[1],
                 "n_infeasible_restarts": v[2]}
                for k, v in self.cache.items() if abs(v[0] - v[1]) > tol]

    def core(self, S, max_core):
        """expB.core_score semantics: max over sub-cores T, |T| <= max_core."""
        S = sorted(S)
        if max_core >= len(S):
            return self.rho(S)
        best = -np.inf
        for size in range(1, max_core + 1):
            for T in itertools.combinations(S, size):
                best = max(best, self.rho(T))
        return best


def build_reduced_jets(alphas, bs, Hs, r):
    """Rank-r reduced jets exactly as expB constructs them: shared projector
    from the mean Hessian, project_jet on (b, H); alpha unchanged."""
    Pi = expB.shared_projector(Hs, r)
    bs_red, Hs_red = [], []
    for b, H in zip(bs, Hs):
        b_r, H_r = expB.project_jet(b, H, Pi)
        bs_red.append(b_r)
        Hs_red.append(H_r)
    return np.array(alphas, dtype=float), bs_red, Hs_red, Pi


def exact_sup_abs_quadratic_on_corner_simplex(dA, db, dH, tol=1e-9):
    """Exact sup over C = {c >= 0, sum_i c_i <= 1} of |dA + db.c + 0.5 c' dH c|."""
    d = len(db)
    verts = [np.zeros(d)] + [np.eye(d)[i] for i in range(d)]

    def g(x):
        return float(dA + db @ x + 0.5 * x @ dH @ x)

    vals = [g(v) for v in verts]
    for kmask in range(1, 2 ** (d + 1)):
        F = [verts[i] for i in range(d + 1) if (kmask >> i) & 1]
        k = len(F) - 1
        if k < 1:
            continue
        v0 = F[0]
        M = np.stack([f - v0 for f in F[1:]], axis=1)      # d x k
        A = M.T @ dH @ M
        rhs = -(M.T @ db + M.T @ dH @ v0)
        try:
            cond_ok = np.linalg.cond(A) < 1e12
        except np.linalg.LinAlgError:
            cond_ok = False
        if cond_ok:
            mu = np.linalg.solve(A, rhs)
            inside = (mu.min() >= -tol) and (mu.sum() <= 1.0 + tol)
            if inside:
                vals.append(g(v0 + M @ mu))
        else:
            mu, res, _, _ = np.linalg.lstsq(A, rhs, rcond=None)
            if np.linalg.norm(A @ mu - rhs) <= tol * (1.0 + np.linalg.norm(rhs)):
                # Singular restricted Hessian: the stationary set is an affine
                # subspace on which the quadratic is constant, but its value is a
                # valid extremum candidate only if the set meets the face.  Check
                # by LP feasibility: mu >= 0, sum mu <= 1, A mu = rhs.
                from scipy.optimize import linprog
                lp = linprog(np.zeros(k), A_ub=np.ones((1, k)), b_ub=[1.0 + tol],
                             A_eq=A, b_eq=rhs, bounds=[(-tol, None)] * k,
                             method="highs")
                if lp.status == 0:
                    vals.append(g(v0 + M @ mu))
    max_g, min_g = max(vals), min(vals)
    return max(abs(max_g), abs(min_g)), max_g, min_g


def eps_exact_certificate(alphas, bs, Hs, alphas_red, bs_red, Hs_red):
    """Exact per-task sup_C |q_t - q^red_t| by face enumeration."""
    out = []
    for a, b, H, ar, br, Hr in zip(alphas, bs, Hs, alphas_red, bs_red, Hs_red):
        sup_abs, _, _ = exact_sup_abs_quadratic_on_corner_simplex(
            float(a - ar), np.asarray(b) - np.asarray(br),
            np.asarray(H) - np.asarray(Hr))
        out.append(sup_abs)
    return np.array(out)


def eps_certificate(alphas, bs, Hs, alphas_red, bs_red, Hs_red):
    """eps_t = |dA_t| + max_i |db_t,i| + 0.5 max_ij |dH_t,ij| per task."""
    eps_t = []
    for a, b, H, ar, br, Hr in zip(alphas, bs, Hs, alphas_red, bs_red, Hs_red):
        dA = abs(float(a) - float(ar))  # 0 by construction, computed anyway
        db = np.max(np.abs(np.asarray(b) - np.asarray(br)))
        dH = np.max(np.abs(np.asarray(H) - np.asarray(Hr)))
        eps_t.append(dA + float(db) + 0.5 * float(dH))
    return np.array(eps_t)


def analyze(substrate, jets_path, ranks, prior_dir=None, verbose=True):
    z = np.load(jets_path, allow_pickle=True)
    alphas = np.asarray(z["alpha"], dtype=float)
    bs = [np.asarray(b, dtype=float) for b in z["b"]]
    Hs = [np.asarray(H, dtype=float) for H in z["H"]]
    names = [str(x) for x in z["names"]]
    n = len(alphas)
    subsets = all_query_subsets(n)

    full_scorer = MemoScorer(alphas, bs, Hs)  # shared across ranks
    t0 = time.time()
    rho_full = np.array([full_scorer.rho(S) for S in subsets])
    if verbose:
        print("[%s] rho_full for %d subsets in %.1fs (%d distinct solves)"
              % (substrate, len(subsets), time.time() - t0, len(full_scorer.cache)),
              flush=True)

    # cross-check against the prior expB decide outputs (4- and 5-way rho_full)
    prior_check = None
    if prior_dir is not None:
        pj = Path(prior_dir) / ("%s_shared_chart_decision_r1.json" % substrate)
        if pj.exists():
            prior = json.loads(pj.read_text())
            name_to_idx = {nm: i for i, nm in enumerate(names)}
            diffs = []
            for row in prior["queries"]:
                S = tuple(sorted(name_to_idx[m] for m in row["members"]))
                diffs.append((abs(full_scorer.rho(S) - row["rho_full"]), S, row))
            worst = max(diffs, key=lambda x: x[0])
            prior_check = {"n_prior_queries": len(diffs),
                           "max_abs_diff_vs_prior_rho_full": float(worst[0]),
                           "worst_query": {
                               "members_idx": list(worst[1]),
                               "this_run_rho_full": float(full_scorer.rho(worst[1])),
                               "prior_rho_full": float(worst[2]["rho_full"]),
                               "prior_rho_rplus1": float(worst[2]["rho_rplus1"]),
                               "prior_violates_subset_monotonicity": bool(
                                   worst[2]["rho_full"] < worst[2]["rho_rplus1"] - 1e-9)}}
            if verbose:
                print("[%s] prior expB rho_full cross-check: max |diff| = %.3e over %d queries"
                      % (substrate, prior_check["max_abs_diff_vs_prior_rho_full"],
                         len(diffs)), flush=True)

    spread_full = float(rho_full.max() - rho_full.min())
    out = {"substrate": substrate, "jets_path": str(jets_path), "n_adapters": n,
           "names": names, "n_subsets": len(subsets),
           "subset_sizes": list(SUBSET_SIZES) + [n],
           "rho_full_min": float(rho_full.min()), "rho_full_max": float(rho_full.max()),
           "score_spread_full": spread_full,
           "full_jet_solver_artifacts": full_scorer.solver_artifacts(),
           "prior_expB_cross_check": prior_check, "per_rank": {}}

    for r in ranks:
        t0 = time.time()
        alphas_red, bs_red, Hs_red, _Pi = build_reduced_jets(alphas, bs, Hs, r)
        eps_t = eps_certificate(alphas, bs, Hs, alphas_red, bs_red, Hs_red)
        eps = float(eps_t.max())

        # Exact per-task sup by face enumeration, with a sampling tightness /
        # validity check: sampled sup <= eps_exact_t <= closed-form eps_t.
        eps_exact_t = eps_exact_certificate(alphas, bs, Hs,
                                            alphas_red, bs_red, Hs_red)
        rng = np.random.default_rng(12345)
        w_samp = rng.dirichlet(np.ones(n), size=20000)
        c_samp = w_samp[:, :-1]
        samp_ok, closed_ok = True, True
        samp_gap = []
        for t in range(n):
            dA = float(alphas[t] - alphas_red[t])
            db = np.asarray(bs[t]) - np.asarray(bs_red[t])
            dH = np.asarray(Hs[t]) - np.asarray(Hs_red[t])
            vals = np.abs(dA + c_samp @ db
                          + 0.5 * np.einsum("ij,jk,ik->i", c_samp, dH, c_samp))
            samp_max = float(vals.max())
            samp_gap.append(eps_exact_t[t] - samp_max)
            samp_ok &= samp_max <= eps_exact_t[t] + 1e-7
            closed_ok &= eps_exact_t[t] <= eps_t[t] + 1e-9
        eps_exact = float(eps_exact_t.max())

        # Per-subset certificates eps_S = max_{t in S} eps_exact_t.
        eps_S = np.array([max(eps_exact_t[t] for t in S) for S in subsets])
        red_scorer = MemoScorer(alphas_red, bs_red, Hs_red)
        rho_red = np.array([red_scorer.rho(S) for S in subsets])
        rho_red_raw = np.array([red_scorer.rho_raw(S) for S in subsets])
        rho_core = np.array([red_scorer.core(S, r + 1) for S in subsets])

        core_disc = np.abs(rho_red - rho_core)
        gap_full_red = np.abs(rho_full - rho_red)
        gap_full_core = np.abs(rho_full - rho_core)
        i_worst = int(np.argmax(gap_full_red))

        # Per-subset bound validity: |rho_full - rho_reduced| <= eps_S for every S.
        per_subset_bound_ok = bool(np.all(gap_full_red <= eps_S + 1e-9))

        # Coverage at the deployed threshold (retention deficit tau = 0): the
        # fraction of subsets whose reduced-core score clears 0 by more than
        # its own certificate eps_S, so the full-jet decision is certified.
        cov_tau0 = float(np.mean(np.abs(rho_core - 0.0) > eps_S))

        # tau sweep (per-subset certificates)
        scores_all = np.concatenate([rho_full, rho_core])
        tau_lo = float(scores_all.min() - TAU_PAD)
        tau_hi = float(scores_all.max() + TAU_PAD)
        taus = np.linspace(tau_lo, tau_hi, N_TAU)
        per_tau = []
        certified_violations = 0
        worst_certified_margin = 0.0  # max (|rho_core - tau| - eps_S) among violations
        worst_band_agree = 1.0
        n_taus_with_band = 0
        min_overall_agree = 1.0
        for tau in taus:
            dec_full = rho_full <= tau
            dec_core = rho_core <= tau
            match = dec_full == dec_core
            agree = float(np.mean(match))
            min_overall_agree = min(min_overall_agree, agree)
            certified = np.abs(rho_core - tau) > eps_S
            viol = int(np.sum(certified & ~match))
            certified_violations += viol
            if viol:
                worst_certified_margin = max(
                    worst_certified_margin,
                    float(np.max((np.abs(rho_core - tau) - eps_S)[certified & ~match])))
            band = ~certified
            n_band = int(np.sum(band))
            band_agree = float(np.mean(match[band])) if n_band else None
            if n_band:
                n_taus_with_band += 1
                worst_band_agree = min(worst_band_agree, band_agree)
            per_tau.append({"tau": float(tau), "agree_frac": agree,
                            "n_uncertified": n_band, "band_agree_frac": band_agree,
                            "n_certified_violations": viol})

        rank_out = {
            "r": int(r), "core_order": int(r + 1),
            "eps_per_task": [float(x) for x in eps_t],
            "eps_max": eps, "eps_median": float(np.median(eps_t)),
            "eps_exact_per_task": [float(x) for x in eps_exact_t],
            "eps_exact_max": eps_exact,
            "eps_exact_median": float(np.median(eps_exact_t)),
            "eps_exact_le_closed_form": bool(closed_ok),
            "sampled_sup_le_eps_exact_20k": bool(samp_ok),
            "min_sampling_slack": float(min(samp_gap)),
            "per_subset_eps_bound_holds": per_subset_bound_ok,
            "eps_exact_over_spread_full": eps_exact / spread_full
                if spread_full > 0 else None,
            "coverage_at_tau0_per_subset_eps": cov_tau0,
            "dA_is_zero_by_construction": True,
            "shared_projector_strict_max_residual":
                float(expB.shared_projector_residual(Hs, r)),
            "max_abs_rho_full_minus_rho_reduced": float(gap_full_red.max()),
            "max_abs_rho_full_minus_rho_reduced_raw_expB": float(
                np.max(np.abs(rho_full - rho_red_raw))),
            "reduced_jet_solver_artifacts": red_scorer.solver_artifacts(),
            "bound_holds_max_gap_le_eps": bool(gap_full_red.max() <= eps),
            "worst_subset_full_vs_reduced": {
                "members": [names[t] for t in subsets[i_worst]],
                "rho_full": float(rho_full[i_worst]),
                "rho_reduced": float(rho_red[i_worst]),
                "gap": float(gap_full_red[i_worst])},
            "core_identity_max_abs_discrepancy": float(core_disc.max()),
            "max_abs_rho_full_minus_rho_core": float(gap_full_core.max()),
            "eps_over_spread_full": eps / spread_full if spread_full > 0 else None,
            "uncertified_band_width_tau_units": 2.0 * eps,
            "band_width_over_tau_range": 2.0 * eps / (tau_hi - tau_lo),
            "tau_sweep": {"tau_lo": tau_lo, "tau_hi": tau_hi, "n_taus": N_TAU,
                          "min_overall_agree_frac": min_overall_agree,
                          "total_certified_violations": int(certified_violations),
                          "worst_certified_violation_margin": worst_certified_margin,
                          "certified_agreement_is_100pct": certified_violations == 0,
                          "n_taus_with_nonempty_band": n_taus_with_band,
                          "worst_band_agree_frac":
                              worst_band_agree if n_taus_with_band else None,
                          "per_tau": per_tau},
            "subsets": [{"members_idx": list(S), "size": len(S),
                         "rho_full": float(rf), "rho_reduced": float(rr),
                         "rho_reduced_raw_expB": float(rw),
                         "rho_core_reduced": float(rc)}
                        for S, rf, rr, rw, rc in zip(subsets, rho_full, rho_red,
                                                     rho_red_raw, rho_core)],
        }
        out["per_rank"]["r%d" % r] = rank_out
        if verbose:
            print("[%s] r=%d done in %.1fs: eps=%.4g (median eps_t=%.4g)  "
                  "max|full-red|=%.4g (<=eps: %s)  core-disc=%.3g  "
                  "eps/spread=%.4g  cert-violations=%d  worst-band-agree=%s"
                  % (substrate, r, time.time() - t0, eps, rank_out["eps_median"],
                     gap_full_red.max(), rank_out["bound_holds_max_gap_le_eps"],
                     core_disc.max(), rank_out["eps_over_spread_full"],
                     certified_violations,
                     ("%.4f" % worst_band_agree) if n_taus_with_band else "n/a (band always empty)"),
                  flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    default_jets_dir = REPO / "results_public" / "shared_chart_decision"
    default_out = REPO / "results_public" / "interval_certification" / "certify.json"
    ap.add_argument("--jets-dir", default=str(default_jets_dir))
    ap.add_argument("--out", default=str(default_out))
    ap.add_argument("--ranks", default="1,2,3")
    args = ap.parse_args()

    ranks = [int(x) for x in args.ranks.split(",")]
    jets_dir = Path(args.jets_dir)
    results = {"description": "Per-library interval certificate for shared-chart "
                              "rank-r reduced jets (see script docstring)",
               "certificate": "eps_t = |dA_t| + max_i|db_t,i| + 0.5 max_ij|dH_t,ij|; "
                              "eps = max_t eps_t bounds |rho_S(full)-rho_S(reduced)| "
                              "for every subset S on the shared chart",
               "solver": "decide.minimax_score program "
                         "(SLSQP epigraph, full 8-simplex, restarts=6, seed=0, "
                         "min_weight=0) with a per-restart feasibility guard; "
                         "raw expB values recorded alongside, divergences "
                         "listed as solver artifacts",
               "substrates": {}}
    for substrate in ("qwen8", "tinyllama"):
        jets_path = jets_dir / ("%s_jets.npz" % substrate)
        results["substrates"][substrate] = analyze(
            substrate, jets_path, ranks, prior_dir=jets_dir)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print("\nwrote", out_path)

    # compact summary table
    print("\n===== SUMMARY =====")
    hdr = ("substrate", "r", "eps", "med_eps_t", "max|f-red|", "<=eps",
           "core_disc", "eps/spread", "cert_viol", "worst_band_agree", "band_2eps")
    print(("%-10s %2s %10s %10s %11s %5s %10s %10s %9s %16s %10s") % hdr)
    for substrate, sub in results["substrates"].items():
        for rk, rr in sub["per_rank"].items():
            ts = rr["tau_sweep"]
            wba = ts["worst_band_agree_frac"]
            print("%-10s %2d %10.4g %10.4g %11.4g %5s %10.3g %10.4g %9d %16s %10.4g"
                  % (substrate, rr["r"], rr["eps_max"], rr["eps_median"],
                     rr["max_abs_rho_full_minus_rho_reduced"],
                     "yes" if rr["bound_holds_max_gap_le_eps"] else "NO",
                     rr["core_identity_max_abs_discrepancy"],
                     rr["eps_over_spread_full"],
                     ts["total_certified_violations"],
                     ("%.4f" % wba) if wba is not None else "n/a",
                     rr["uncertified_band_width_tau_units"]))
    for substrate, sub in results["substrates"].items():
        arts = list(sub["full_jet_solver_artifacts"])
        for rk, rr in sub["per_rank"].items():
            arts += rr["reduced_jet_solver_artifacts"]
        if arts:
            print("\n[%s] %d solver artifact(s) where raw expB minimax_score "
                  "accepted an infeasible SLSQP restart:" % (substrate, len(arts)))
            for a in arts:
                print("  subset=%s raw=%.6g guarded=%.6g (%d infeasible restart(s))"
                      % (a["subset"], a["raw_expB"], a["guarded"],
                         a["n_infeasible_restarts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
