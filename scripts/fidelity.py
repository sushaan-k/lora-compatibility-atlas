#!/usr/bin/env python3
"""Generator for results_public/capability_ci/fidelity.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
N_BOOT = 2000
SEED = 0


def weighted_pearson(x, y, w):
    x, y, w = (np.asarray(a, dtype=float) for a in (x, y, w))
    keep = w > 0
    x, y, w = x[keep], y[keep], w[keep]
    if len(x) < 3:
        return None
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    vx = np.average((x - mx) ** 2, weights=w)
    vy = np.average((y - my) ** 2, weights=w)
    if vx == 0 or vy == 0:
        return None
    return float(cov / np.sqrt(vx * vy))


def main():
    panel = json.loads((REPO / "results_public/scale_m16/answer_panel_analysis.json").read_text())
    adapters = panel["names"]
    events = [e for e in panel["per_member"] if e["order"] == 3]
    p = [e["pred_L_answer"] for e in events]
    m = [e["L_answer"] for e in events]
    r = weighted_pearson(p, m, np.ones(len(events)))
    published = panel["r_pred_vs_measured_L_answer_triples_only"]
    assert abs(r - published) < 1e-9, (r, published)

    loo = []
    for a in adapters:
        keep = [i for i, e in enumerate(events) if a not in e["merge"].split("+")]
        v = weighted_pearson([p[i] for i in keep], [m[i] for i in keep], np.ones(len(keep)))
        if v is not None:
            loo.append(v)

    idx = {a: i for i, a in enumerate(adapters)}
    ev_ad = [[idx[t] for t in e["merge"].split("+")] for e in events]
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(N_BOOT):
        mult = np.bincount(rng.integers(0, len(adapters), size=len(adapters)),
                           minlength=len(adapters))
        w = np.array([np.prod(mult[mem]) for mem in ev_ad], dtype=float)
        v = weighted_pearson(p, m, w)
        if v is not None:
            boots.append(v)

    out = {
        "triple_r_published": published,
        "triple_r_recomputed": r,
        "n_triple_members": len(events),
        "loo_range": [min(loo), max(loo)],
        "n_loo": len(loo),
        "adapter_block_bootstrap": {
            "n_valid": len(boots),
            "median": float(np.median(boots)),
            "ci_lo_2.5": float(np.percentile(boots, 2.5)),
            "ci_hi_97.5": float(np.percentile(boots, 97.5)),
        },
    }
    od = REPO / "results_public/capability_ci"
    od.mkdir(exist_ok=True)
    (od / "fidelity.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
