"""Measured deployed-loss envelopes for the target certificate (Theorem 2.3).

For each dense cartography chart (three adapters, 91-point simplex grid,
measured answer loss per task at every point), fit the task models on the six
poised design points with the released fit code, measure the one-sided
envelopes eps_t^+ = sup (f - qhat)_+ and eps_t^- = sup (qhat - f)_+ over the
85 held-out grid points, then answer every task-subset query with the padded
certificates U_true / L_true of the target certificate and report which
verdicts survive the measured slack.

Thresholds are the panel's own retention rule at tau = 0.70 on the answer
loss: accept task t iff L_c <= L0_t - tau * (L0_t - L1_t), i.e. deficit <= 0
after shifting each model by its per-task threshold.

Reads  results_public/cartography/carto_raw.json
       results_public/answer_panel_m8/answer_panel_analysis.json  (gains)
Writes results_public/deployed_envelopes/deployed_envelopes.json
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[1]
_cand = [REPO / "peft_atlas_lite", REPO.parent / "supplement" / "peft_atlas_lite"]
sys.path.insert(0, str(next(p for p in _cand if p.exists())))
from run_peft_atlas_lite import fit_quadratic_model, predict_quadratic  # noqa: E402

TAUS = [0.70, 0.85]
DESIGN = [
    (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
    (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5),
]


def exact_min(fun, grid, extra=()):
    """Exact minimum of a convex function over the 2-simplex (SLSQP, multi-start)."""
    starts = sorted(grid + list(extra), key=fun)[:6]
    best, bx = np.inf, None
    cons = [{"type": "eq", "fun": lambda c: sum(c) - 1.0}]
    for s in starts:
        r = minimize(fun, s, method="SLSQP", bounds=[(0, 1)] * 3, constraints=cons)
        if r.success and r.fun < best:
            best, bx = float(r.fun), [float(v) for v in r.x]
    return best, bx


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--carto", default="results_public/cartography/carto_raw.json")
    ap.add_argument("--gains", default="results_public/answer_panel_m8/answer_panel_analysis.json")
    ap.add_argument("--out", default="results_public/deployed_envelopes")
    args = ap.parse_args()
    carto = json.load(open(REPO / args.carto))
    gains = json.load(open(REPO / args.gains))["answer_token_gains"]
    grid = [tuple(p) for p in carto["grid"]]
    didx = [grid.index(p) for p in DESIGN]
    heldout = [i for i in range(len(grid)) if i not in didx]
    n_charts = len({k.split(":")[0] for k in carto["points"]})

    out = {"taus": TAUS, "n_charts": n_charts, "surfaces": [], "queries": []}
    for ch in range(n_charts):
        pts = [carto["points"][f"{ch}:{i}"] for i in range(len(grid))]
        tasks = list(pts[0].keys())
        models, eps, theta, f_vals = {}, {}, {}, {}
        for j, t in enumerate(tasks):
            f = np.array([pts[i][t]["answer"] for i in range(len(grid))])
            coeff, meta = fit_quadratic_model([grid[i] for i in didx], [f[i] for i in didx])
            q = np.array([predict_quadratic(coeff, grid[i]) for i in range(len(grid))])
            # Uniform envelope over ALL grid points, design included: the PSD
            # projection means the fit need not interpolate the design, so a
            # held-out-only envelope is not uniform on projected surfaces.
            r = f - q
            models[t], f_vals[t] = coeff, f
            eps[t] = (max(0.0, float(r.max())), max(0.0, float(-r.min())))
            vlosses = [f[grid.index(DESIGN[v])] for v in range(3)]
            # A task's own vertex need not be its minimum (a peer adapter can
            # transfer better, e.g. task1661 on the new-at-fifty charts);
            # record the observation rather than treating it as an error.
            own_vertex_minimal = vlosses[j] == min(vlosses)
            L1 = vlosses[j]                       # the task's own single-adapter vertex
            L0 = L1 + gains[t]                    # base loss = alone + measured gain
            theta[t] = {tau: L0 - tau * (L0 - L1) for tau in TAUS}
            out["surfaces"].append({
                "chart": ch, "task": t, "eps_plus": eps[t][0], "eps_minus": eps[t][1],
                "surface_span": float(f.max() - f.min()), "projected": meta.get("psd_projected", False),
                "L1": float(L1), "L0": float(L0), "own_vertex_minimal": bool(own_vertex_minimal),
                "threshold": {str(k): float(v) for k, v in theta[t].items()},
            })

        for tau in TAUS:
          th = {t: theta[t][tau] for t in tasks}
          for k in range(1, len(tasks) + 1):
            for T in combinations(tasks, k):
                def smax(c, T=T):  # shifted fitted max (deficit form)
                    return max(predict_quadratic(models[t], c) - th[t] for t in T)
                rho, cstar = exact_min(smax, list(grid))
                # dual weights from the active set at cstar (uniform is valid; refine by KKT LS)
                act = [t for t in T if predict_quadratic(models[t], cstar) - th[t] >= rho - 1e-6]
                lam = {t: (1.0 / len(act) if t in act else 0.0) for t in T}
                def wsum(c, lam=lam, T=T):
                    return sum(lam[t] * (predict_quadratic(models[t], c) - th[t]) for t in T)
                wmin, _ = exact_min(wsum, list(grid))
                L_true = wmin - sum(lam[t] * eps[t][1] for t in T)
                U_true = min(max(predict_quadratic(models[t], c) - th[t] + eps[t][0] for t in T)
                             for c in grid)
                rho_f = float(min(max(f_vals[t][i] - th[t] for t in T) for i in range(len(grid))))
                verdict = "accept" if U_true <= 0 else ("reject" if L_true > 0 else "inconclusive")
                out["queries"].append({
                    "chart": ch, "tau": tau, "T": list(T), "rho_fitted": rho, "rho_deployed_grid": rho_f,
                    "L_true": float(L_true), "U_true": float(U_true),
                    "fitted_accept": bool(rho <= 0),
                    "deployed_accept": bool(rho_f <= 0), "padded_verdict": verdict,
                })

    out["summary"] = {}
    for tau in TAUS:
        qs = [q for q in out["queries"] if q["tau"] == tau]
        certified = [q for q in qs if q["padded_verdict"] != "inconclusive"]
        correct = [q for q in certified
                   if (q["padded_verdict"] == "accept") == q["deployed_accept"]]
        bracket_ok = [q for q in qs if q["L_true"] <= q["rho_deployed_grid"] <= q["U_true"]]
        fit_match = [q for q in qs if q["fitted_accept"] == q["deployed_accept"]]
        out["summary"][str(tau)] = {
            "n_queries": len(qs), "n_accepts_fitted": sum(q["fitted_accept"] for q in qs),
            "n_certified": len(certified), "n_certified_correct": len(correct),
            "n_bracket_contains_deployed": len(bracket_ok),
            "n_fitted_matches_deployed": len(fit_match),
            "verdicts": {v: sum(q["padded_verdict"] == v for q in qs)
                         for v in ("accept", "reject", "inconclusive")},
        }
    out["summary"]["eps_plus_all"] = sorted(round(s["eps_plus"], 3) for s in out["surfaces"])
    out["summary"]["eps_minus_all"] = sorted(round(s["eps_minus"], 3) for s in out["surfaces"])
    od = REPO / args.out
    od.mkdir(parents=True, exist_ok=True)
    (od / "deployed_envelopes.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["summary"], indent=2))
    for s in out["surfaces"]:
        print(f"chart{s['chart']} {s['task']}: eps+={s['eps_plus']:.3f} eps-={s['eps_minus']:.3f} "
              f"span={s['surface_span']:.2f} proj={s['projected']}")


if __name__ == "__main__":
    main()
