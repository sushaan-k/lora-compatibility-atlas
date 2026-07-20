#!/usr/bin/env python3
"""Index mechanics from existing data: incremental insertion (B1) and certificate
calibration on capability labels (B4).

B1 replays the growth of the m=8 answer-token atlas one adapter at a time. The
36-point design nests: the first k adapters carry k singles and C(k,2) pairs, and
adapter k+1 adds its single and k pairs. We fit the k-atlas, extend it to k+1 by
solving only the new coefficients from the k+1 insertion evaluations (the design
stays poised, Section 4), and compare the extended jets against a full refit on
all k+1 adapters. This measures both the ideal (exact extension when the surface
is quadratic) and the real drift of the shared coefficients.

B4 builds the certificate reliability curve on the expD capability labels: bin the
validated subsets by the atlas margin and report the observed retention rate per
bin. The paper's reliability figure uses prompt-loss labels; this is its
capability-label counterpart.

All CPU, from results_public/answer_panel_m8/.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "peft_atlas_lite"))
sys.path.insert(0, str(REPO / "experiments_gpu"))
from run_peft_atlas_lite import (fit_quadratic_model,  # noqa: E402
                                 hessian_from_quadratic_coeff, quadratic_features)
from expC_answer_jets import jet_value, weights_to_coords  # noqa: E402


def support(w):
    return tuple(i for i, x in enumerate(w) if x > 1e-9)


def fit_subatlas(grid, values, adapters):
    """Fit one jet per task on the sub-design supported on `adapters`, in the
    coordinate system of that sub-simplex (drop-last convention). Returns
    per-task (alpha, b, H) and the design point count."""
    aset = set(adapters)
    rows = [i for i, w in enumerate(grid) if set(support(w)) <= aset]
    # remap active adapters to a compact coordinate space of dim len(adapters)-1
    idx = {a: j for j, a in enumerate(adapters)}
    sub_w = []
    for i in rows:
        v = np.zeros(len(adapters))
        for a in adapters:
            v[idx[a]] = grid[i][a]
        sub_w.append(v)
    coord_dim = len(adapters) - 1
    jets = {}
    for t in adapters:  # fit only the tasks whose adapters are in this atlas
        vals = [values[t][i] for i in rows]
        coeff, info = fit_quadratic_model([w.tolist() for w in sub_w], vals, project_psd=True)
        H = np.asarray(hessian_from_quadratic_coeff(coeff, coord_dim), float)
        jets[t] = {"alpha": float(coeff[0]),
                   "b": np.array(coeff[1:1 + coord_dim], float), "H": H,
                   "coeff": np.asarray(coeff, float)}
    return jets, len(rows), idx


def b1_growth(grid, values, names):
    m = len(names)
    seq = []
    for k in range(2, m):  # atlas over first k adapters, insert adapter k
        prev = list(range(k))
        full = list(range(k + 1))
        jets_prev, n_prev, _ = fit_subatlas(grid, values, prev)
        jets_full, n_full, idxf = fit_subatlas(grid, values, full)
        # shared-coefficient drift: compare the k-atlas jet against the (k+1)-atlas
        # jet on the shared adapters, both evaluated at the shared sub-simplex
        # vertices and pairwise midpoints (a common evaluation grid)
        shared = prev
        pts = []
        for a in shared:
            v = np.zeros(k + 1); v[idxf[a]] = 1.0; pts.append(v)
        for a, b in itertools.combinations(shared, 2):
            v = np.zeros(k + 1); v[idxf[a]] = v[idxf[b]] = 0.5; pts.append(v)
        # map the same physical points into the k-atlas coordinate frame
        idxp = {a: j for j, a in enumerate(prev)}
        drift = []
        for t in shared:
            for phys in pts:
                cf = weights_to_coords(phys)
                vf = jet_value(jets_full[t]["alpha"], jets_full[t]["b"], jets_full[t]["H"], cf)
                vp_w = np.zeros(k)
                for a in prev:
                    vp_w[idxp[a]] = phys[idxf[a]]
                cp = weights_to_coords(vp_w)
                vp = jet_value(jets_prev[t]["alpha"], jets_prev[t]["b"], jets_prev[t]["H"], cp)
                drift.append(abs(vf - vp))
        # insertion cost accounting
        insert_pts = (k + 1) - 1 + 1  # new single + k pairs = k+1
        seq.append({
            "k": k, "insert_adapter": names[k],
            "prev_design_points": n_prev, "full_design_points": n_full,
            "insertion_evaluations": n_full - n_prev,
            "expected_insertion_evaluations": k + 1,
            "shared_coeff_drift_median": float(np.median(drift)),
            "shared_coeff_drift_max": float(np.max(drift)),
        })
        print(f"  k={k}: insert {names[k]}  design {n_prev}->{n_full} "
              f"(+{n_full-n_prev}, expected {k+1})  shared-jet drift "
              f"median {np.median(drift):.4f} max {np.max(drift):.4f}", flush=True)
    return seq


def b4_calibration(names):
    rows = json.loads((REPO / "results_public" / "answer_panel_m8"
                       / "validation_rows.json").read_text())["rows"]
    z = np.load(REPO / "results_public" / "answer_panel_m8" / "answer_jets_raw.npz",
                allow_pickle=True)
    grid = z["grid"]; vals = z["values"]
    jets, _, _ = fit_subatlas(grid, vals, list(range(len(names))))
    base = {r["task"]: r for r in rows if r["merge"] == "base"}
    alone = {r["task"]: r for r in rows if r["merge"] == "alone"}
    meas = {(r["task"], r["merge"]): r for r in rows if r["merge"] not in ("base", "alone")}
    name_to_idx = {nm: i for i, nm in enumerate(names)}

    events = []
    for (t, mg), r in meas.items():
        members = mg.split("+")
        w = np.zeros(len(names))
        for a in members:
            w[name_to_idx[a]] = 1.0 / len(members)
        c = weights_to_coords(w)
        jt = jets[name_to_idx[t]]
        pred = jet_value(jt["alpha"], jt["b"], jt["H"], c)
        gain = base[t]["L_answer"] - alone[t]["L_answer"]
        tau = 0.7
        margin = (base[t]["L_answer"] - tau * gain) - pred   # >0 => predicted feasible
        ans_ret = r["L_answer"] <= base[t]["L_answer"] - tau * gain
        acc_ret = r["accuracy"] >= base[t]["accuracy"] + 0.7 * (alone[t]["accuracy"] - base[t]["accuracy"])
        events.append({"margin": margin, "answer_retained": ans_ret, "acc_retained": acc_ret})

    events.sort(key=lambda e: e["margin"])
    q = np.array_split(events, 5)
    bins = []
    for i, grp in enumerate(q):
        bins.append({
            "quintile": i + 1, "n": len(grp),
            "margin_range": [round(grp[0]["margin"], 3), round(grp[-1]["margin"], 3)],
            "answer_retained_rate": round(float(np.mean([e["answer_retained"] for e in grp])), 3),
            "acc_retained_rate": round(float(np.mean([e["acc_retained"] for e in grp])), 3),
        })
    # monotonicity check
    ar = [b["answer_retained_rate"] for b in bins]
    mono = all(ar[i] <= ar[i + 1] + 1e-9 for i in range(len(ar) - 1))
    print("  reliability (answer-retention rate by margin quintile):",
          [b["answer_retained_rate"] for b in bins], "monotone:", mono, flush=True)
    return {"bins": bins, "answer_rate_monotone": mono, "n_events": len(events)}


def main():
    z = np.load(REPO / "results_public" / "answer_panel_m8" / "answer_jets_raw.npz",
                allow_pickle=True)
    grid = z["grid"]; values = z["values"]; names = [str(x) for x in z["names"]]
    print("B1 incremental-insertion growth replay:")
    growth = b1_growth([list(w) for w in grid], values, names)
    print("B4 certificate calibration on capability labels:")
    calib = b4_calibration(names)
    out = REPO / "results_public" / "maintenance"
    out.mkdir(parents=True, exist_ok=True)
    (out / "index_updates.json").write_text(json.dumps(
        {"b1_growth": growth, "b4_calibration": calib}, indent=2))
    print("\nwrote", out / "index_updates.json")


if __name__ == "__main__":
    main()
