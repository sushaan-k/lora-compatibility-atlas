#!/usr/bin/env python3
"""Strict spectral effective dimension for the Mistral m=30 and m=50 charts.

Same definition as measure_effective_rank_strict.py: fit the PSD-projected
quadratic per scored task from the released jets, take Pi_r as the top-r
eigenspace of the mean Hessian, and report the strict max-task residual
max_t ||H_t - Pi_r H_t Pi_r||_F / max_t ||H_t||_F, recording the first rank at
which it falls below 0.20 and 0.10. Tasks excluded by the panels (answer loss
NaN at every design point) are skipped, matching the index.

Run:  python3 scripts/scale_adim.py
Out:  results_public/scale_m30/sigma_adim.json, results_public/scale_m50/sigma_adim.json
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "peft_atlas_lite"))
from run_peft_atlas_lite import fit_quadratic_model, hessian_from_quadratic_coeff  # noqa: E402


def main():
    np.seterr(all="ignore")
    for sub in ("results_public/scale_m30", "results_public/scale_m50"):
        pdir = REPO / sub
        z = np.load(pdir / "answer_jets_raw.npz", allow_pickle=True)
        grid = [list(w) for w in z["grid"]]
        names = [str(x) for x in z["names"]]
        vals = z["values"]
        d = len(names) - 1
        H_list = []
        for i in range(len(names)):
            if np.isnan(vals[i]).any():
                continue
            coeff, _ = fit_quadratic_model(grid, list(vals[i]))
            H_list.append(np.asarray(hessian_from_quadratic_coeff(coeff, d), float))
        H_bar = np.stack(H_list).mean(axis=0)
        eigvals, eigvecs = np.linalg.eigh(H_bar)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]
        L_F = max(float(np.linalg.norm(H, "fro")) for H in H_list)
        residuals, r20, r10 = [], None, None
        for r in range(0, d + 1):
            if r == 0:
                rel = 1.0
            else:
                Pi = eigvecs[:, :r] @ eigvecs[:, :r].T
                rel = max(float(np.linalg.norm(H - Pi @ H @ Pi, "fro")) for H in H_list) / L_F
            residuals.append(rel)
            if r20 is None and rel <= 0.20:
                r20 = r
            if r10 is None and rel <= 0.10:
                r10 = r
        out = {"n_tasks": len(H_list), "d": d,
               "first_r_below_0.20": r20, "first_r_below_0.10": r10,
               "residual_at_r3": residuals[3], "residual_at_r10": residuals[min(10, d)],
               "residuals": [round(x, 4) for x in residuals]}
        (pdir / "sigma_adim.json").write_text(json.dumps(out, indent=1))
        print(sub, "-> r0.20 =", r20, ", r0.10 =", r10,
              ", res@r3 =", round(residuals[3], 3), ", res@r10 =", round(residuals[min(10, d)], 3))


if __name__ == "__main__":
    main()
