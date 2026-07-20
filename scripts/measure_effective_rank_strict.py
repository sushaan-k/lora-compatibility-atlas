"""Strict max-task spectral diagnostic on a real LoRA sub-library.

For each rank r in {1, ..., d}, computes:
  (a) the strict max-task spectral residual
        max_t ||H_t - Pi_r H_t Pi_r||_F  /  max_t ||H_t||_F,
  (b) the first-order affine residual max_t ||(I - Pi_r) b_t||_2 * D,
      with D the chart diameter, and
  (c) saves the full per-task Hessians as .npz so the diagnostic can be
      recomputed without re-fitting.

The strict-max residual is what the sigma-adim_eps definition uses; the
mean-Hessian eigenvalue trace alone is not sufficient.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import random
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "peft_atlas_lite"))

from run_peft_atlas_lite import (
    adapters_from_config, evaluate_base_and_singles, evaluate_merge_losses,
    fit_quadratic_model, hessian_from_quadratic_coeff, load_json,
    load_peft_model, score_query, simplex_grid, split_prompts, stable_seed,
)


def select_adapters(config, names):
    by_name = {a.name: a for a in adapters_from_config(config)}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise KeyError(f"adapters not in manifest: {missing}")
    return [by_name[n] for n in names]


def b_from_quadratic_coeff(coeff, d):
    """Extract linear coefficients b from the affine-quadratic fit.

    Layout: [const, b_1..b_d, H entries...]"""
    return np.array(coeff[1:1+d], dtype=float)


def chart_diameter(grid):
    """Maximum L2 distance between any two design points in coord space."""
    pts = np.array(grid)
    d_plus_1 = pts.shape[1]
    coords = pts[:, :d_plus_1 - 1]
    dmax = 0.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dmax = max(dmax, float(np.linalg.norm(coords[i] - coords[j])))
    return dmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapters", required=True)
    ap.add_argument("--step", type=float, default=0.25)
    ap.add_argument("--min-weight", type=float, default=0.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = load_json(Path(args.config))
    names = [n.strip() for n in args.adapters.split(",")]
    adapters = select_adapters(config, names)
    n = len(adapters)
    d = n - 1
    print("d=%d m=%d adapters=%s" % (d, n, [a.name for a in adapters]))

    grid = simplex_grid(n, step=args.step, min_weight=args.min_weight)
    print("design grid: %d points" % len(grid))
    n_d = 1 + d + d * (d + 1) // 2
    if len(grid) < n_d:
        raise ValueError("grid %d < N_d=%d; lower --step" % (len(grid), n_d))

    rng = random.Random(stable_seed(config))
    prompt_sets = {}
    for a in adapters:
        cal, held = split_prompts(config, a, rng)
        prompt_sets[a.name] = {"calibration": cal, "heldout": held}

    print("loading PEFT model...")
    model, tokenizer = load_peft_model(config, adapters)

    print("evaluating base + single-adapter losses...")
    base_losses, single_losses = evaluate_base_and_singles(
        model, tokenizer, adapters, prompt_sets, int(config.get("max_length", 384)),
    )

    threshold = float(config.get("retention_threshold", 0.55))
    min_gain = float(config.get("min_single_adapter_gain", 0.005))
    density = float(config.get("density", 0.5))
    max_length = int(config.get("max_length", 384))
    method = "linear"

    print("evaluating %d grid x %d tasks..." % (len(grid), n))
    cal_values = {a.name: [] for a in adapters}
    serial = 300000
    t0 = time.time()
    for i, weights in enumerate(grid):
        serial += 1
        losses = evaluate_merge_losses(
            model, tokenizer, adapters, weights, method, density,
            prompt_sets, max_length, "calibration", serial,
        )
        names_list = [a.name for a in adapters]
        _, margins, _ = score_query(
            names_list, weights, losses,
            base_losses["calibration"], single_losses["calibration"],
            threshold, min_gain,
        )
        for nm in names_list:
            cal_values[nm].append(-margins[nm])
        if (i + 1) % 5 == 0:
            print("  %d/%d t=%ds" % (i + 1, len(grid), int(time.time() - t0)))

    print("fitting per-adapter PSD-projected quadratics...")
    H_list = []
    b_list = []
    per_adapter = []
    coord_dim = n - 1
    for a in adapters:
        coeff, info = fit_quadratic_model(list(grid), cal_values[a.name], project_psd=True)
        H = hessian_from_quadratic_coeff(coeff, coord_dim)
        b = b_from_quadratic_coeff(coeff, coord_dim)
        eigvals = np.linalg.eigvalsh(H)
        H_list.append(H)
        b_list.append(b)
        nrm_H = float(np.linalg.norm(H, "fro"))
        nrm_b = float(np.linalg.norm(b))
        per_adapter.append({
            "name": a.name,
            "hessian_frobenius": nrm_H,
            "hessian_eigvals": eigvals.tolist(),
            "linear_b_norm": nrm_b,
            "min_eigenvalue_before_psd": float(info.get("min_eigenvalue_before", 0.0)),
            "psd_projected": bool(info.get("psd_projected", False)),
            "psd_adjustment_frob": float(info.get("psd_adjustment_frobenius", 0.0)),
        })
        print("  %-30s ||H||_F=%.3f ||b||=%.3f" % (a.name[:30], nrm_H, nrm_b))

    H_stack = np.stack(H_list)
    H_bar = H_stack.mean(axis=0)
    eigvals, eigvecs = np.linalg.eigh(H_bar)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    L_F = max(float(np.linalg.norm(H, "fro")) for H in H_list)
    D = chart_diameter(grid)

    print("L_F=%.3f D=%.3f (chart diameter)" % (L_F, D))
    strict = []
    for r in range(0, d + 1):
        if r == 0:
            tail_H = max(float(np.linalg.norm(H, "fro")) for H in H_list)
            tail_b = max(float(np.linalg.norm(b)) for b in b_list)
        else:
            Pi = eigvecs[:, :r] @ eigvecs[:, :r].T
            tail_H = max(float(np.linalg.norm(H - Pi @ H @ Pi, "fro")) for H in H_list)
            tail_b = max(float(np.linalg.norm(b - Pi @ b)) for b in b_list)
        relative = tail_H / L_F if L_F > 0 else float("nan")
        first_order = tail_b * D
        strict.append({
            "r": int(r),
            "max_task_hessian_residual_frobenius": tail_H,
            "max_task_hessian_residual_relative": relative,
            "max_task_first_order_residual_times_D": first_order,
        })
        print("  r=%d: ||H-PHP||_F max/L_F=%.4f ||(I-P)b||*D max=%.4f"
              % (r, relative, first_order))

    per_eps = []
    for eps in [0.01, 0.05, 0.10, 0.20]:
        r_found = d
        for entry in strict:
            if entry["r"] >= 1 and entry["max_task_hessian_residual_relative"] <= eps:
                r_found = entry["r"]
                break
        per_eps.append({"eps": eps, "r_eps_strict_max": r_found})
        print("  sigma-adim_eps at eps=%.2f: r=%d" % (eps, r_found))

    summary = {
        "adapters": [a.name for a in adapters],
        "n_adapters_m": n,
        "chart_dim_d": d,
        "n_design_points": len(grid),
        "N_d_required": n_d,
        "per_adapter": per_adapter,
        "spectrum": {
            "eigvals_H_bar": eigvals.tolist(),
            "L_F_max_task_frobenius": L_F,
            "chart_diameter_D": D,
            "trace_H_bar": float(np.trace(H_bar)),
        },
        "strict_max_residuals_per_r": strict,
        "sigma_adim_eps_strict": per_eps,
    }
    (out / "effective_rank_strict.json").write_text(json.dumps(summary, indent=2))
    np.savez(out / "hessians.npz",
             H_stack=H_stack, H_bar=H_bar, b_stack=np.stack(b_list),
             names=np.array([a.name for a in adapters]))
    print("wrote %s" % (out / "effective_rank_strict.json"))
    print("wrote %s" % (out / "hessians.npz"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
