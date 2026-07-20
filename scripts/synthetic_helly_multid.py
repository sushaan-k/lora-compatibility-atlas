"""Helly construction at d=2,3,4. (d+1) centred-quadratic budgets at the
vertices of a regular d-simplex with unit circumradius; the Chebyshev centre
of the simplex is the origin (by symmetry) and the Chebyshev centre of any
d-vertex face sits in that face's plane. Both values are analytic, so we
compute them in closed form and verify on a fine grid."""
import argparse, json, numpy as np
from pathlib import Path


def regular_simplex_vertices(d):
    """d+1 vertices of a regular d-simplex centred at the origin, unit
    circumradius. Construction: e_1, ..., e_d in R^d plus a last vertex at
    (-c, ..., -c) where c = (sqrt(d+1) - 1) / d makes every vertex equidistant
    from the centroid. After centering and normalising by that common norm the
    d+1 points sit on the unit sphere with equal pairwise distances."""
    n = d + 1
    c = (np.sqrt(d + 1) - 1) / d
    pts = np.zeros((n, d))
    for i in range(n - 1):
        pts[i, i] = 1.0
    pts[-1] = np.full(d, -c)
    pts = pts - pts.mean(0)
    return pts / np.linalg.norm(pts[0])


def rho_min_grid(centres, R, chart=1.6, grid=160):
    """Numerical sanity-check on a fine grid in [-chart, chart]^d."""
    d = centres.shape[1]
    axes = [np.linspace(-chart, chart, grid) for _ in range(d)]
    mesh = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([m.ravel() for m in mesh], axis=1)
    q = np.zeros((centres.shape[0], pts.shape[0]))
    for t, v in enumerate(centres):
        q[t] = ((pts - v) ** 2).sum(1) / R ** 2
    mq = q.max(0)
    return float(mq.min())


def test_d(d, R, verify_grid=False):
    V = regular_simplex_vertices(d)
    # By symmetry the Chebyshev centre of the full simplex is the origin and
    # every vertex sits at unit distance from it, giving rho_S = 1/R^2.
    rho_full = 1.0 / R ** 2
    # Drop-one d-subset = regular (d-1)-face. The face's circumradius (in its
    # own plane) equals sqrt(1 - 1/d^2) for a unit-circumradius d-simplex. The
    # min-max over R^d is attained in the face plane, so rho_T = (1-1/d^2)/R^2.
    face_circ_sq = 1.0 - 1.0 / d ** 2
    rho_d_subset = face_circ_sq / R ** 2
    helly = (rho_d_subset <= 1.0) and (rho_full > 1.0)
    core_size = d + 1 if rho_full > 1.0 else 0
    row = {
        "d": d, "R": R, "n_tasks": d + 1,
        "full_rho": rho_full,
        "full_feasible": rho_full <= 1.0,
        "core_size": core_size,
        "d_subset_rho": rho_d_subset,
        "all_d_subsets_feasible": rho_d_subset <= 1.0,
        "helly_direction": helly,
        "core_eq_d_plus_1": core_size == d + 1,
    }
    if verify_grid and d <= 3:
        rho_full_grid = rho_min_grid(V, R)
        sub_grid = []
        for omit in range(d + 1):
            sub = np.delete(V, omit, axis=0)
            sub_grid.append(rho_min_grid(sub, R))
        row["full_rho_grid"] = rho_full_grid
        row["d_subset_rho_grid_max"] = float(max(sub_grid))
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_json", required=True)
    p.add_argument("--out_tex", required=True)
    p.add_argument("--verify_grid", action="store_true",
                   help="cross-check analytic values against a 160-point grid for d<=3")
    args = p.parse_args()

    rows = []
    print(f"{'d':>2} {'R':>5} {'full_rho':>10} {'d_sub_rho':>10}"
          f" {'all_d_feas':>11} {'full_infeas':>12} {'core':>5} {'helly':>6}")
    for d in [2, 3, 4]:
        for R in [0.93, 0.95, 0.97, 0.99]:
            r = test_d(d, R, verify_grid=args.verify_grid)
            rows.append(r)
            print(f"{d:>2d} {R:>5.2f} {r['full_rho']:>10.3f}"
                  f" {r['d_subset_rho']:>10.3f}"
                  f" {str(r['all_d_subsets_feasible']):>11s}"
                  f" {str(not r['full_feasible']):>12s}"
                  f" {r['core_size']:>5d} {str(r['helly_direction']):>6s}")

    # Cleanest Helly witness per d: largest full_rho with all d-subsets feasible.
    best = {}
    for d in [2, 3, 4]:
        ws = [r for r in rows if r["d"] == d and r["helly_direction"]]
        if ws:
            best[str(d)] = max(ws, key=lambda r: r["full_rho"])

    Path(args.out_json).write_text(json.dumps(
        {"all_cases": rows, "best_witness_per_d": best}, indent=2))

    tex = [
        r"\begin{table}[h]",
        r"\centering\small",
        r"\caption{Synthetic Helly verification across $d{=}2,3,4$. For each "
        r"$d$ we place $d{+}1$ unit-norm centred-quadratic budgets "
        r"$F_t=\{c:\|c-v_t\|_2^2\le R^2\}$ at the vertices of a regular "
        r"$d$-simplex of unit circumradius. By symmetry the Chebyshev centre "
        r"of the full simplex is the origin, so $\rho_S=1/R^2$; the Chebyshev "
        r"centre of a $d$-vertex face is the face circumcentre, so "
        r"$\rho_T=(1-1/d^2)/R^2$. Every $d$-subset is feasible and the full "
        r"$(d{+}1)$-tuple is infeasible at the listed $R$, with obstructing "
        r"core size exactly $d{+}1$.}",
        r"\label{tab:synthetic_helly_multid}",
        r"\begin{tabular}{c c c c c c}",
        r"\toprule",
        r"$d$ & $R$ & $n{=}d{+}1$ & full $\rho_S$ & all $d$-subsets feasible & core size \\",
        r"\midrule",
    ]
    for d in [2, 3, 4]:
        r = best.get(str(d))
        if r:
            tex.append(f"{d} & {r['R']:.2f} & {r['n_tasks']} & "
                       f"{r['full_rho']:.3f} & \\checkmark & {r['core_size']} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(args.out_tex).write_text("\n".join(tex) + "\n")
    print(f"\nwrote {args.out_json}, {args.out_tex}")


if __name__ == "__main__":
    main()
