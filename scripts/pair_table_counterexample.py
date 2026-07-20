"""Numeric verification of the pair-table counterexample (Proposition: pair scores
do not determine triple scores).

Two families of convex quadratic task models on the same chart:
  A: q_t(c) = |c - v_t|^2, centers at the vertices of a unit equilateral triangle.
  B: q_1, q_2 as in A with |v_1 - v_2| = 1; q_3(c) = 4 |c - v_3'|^2 with
     |v_3' - v_1| = |v_3' - v_2| = 3/4.

For two weighted paraboloids w_i d_i^2, w_j d_j^2 at center distance L the pair
minimax is w_i w_j L^2 / (sqrt(w_i) + sqrt(w_j))^2, attained on the segment.
Both families therefore share the pair table rho_12 = rho_13 = rho_23 = 1/4
(and all singleton scores 0).  The triple scores differ.

Writes results_public/pair_table_counterexample/pair_table_counterexample.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[1]

S3 = np.sqrt(3.0)
FAMILIES = {
    "A": [((0.0, 0.0), 1.0), ((1.0, 0.0), 1.0), ((0.5, S3 / 2), 1.0)],
    "B": [((0.0, 0.0), 1.0), ((1.0, 0.0), 1.0), ((0.5, np.sqrt(5.0) / 4), 4.0)],
}


def q(center, w, c):
    return w * ((c[0] - center[0]) ** 2 + (c[1] - center[1]) ** 2)


def minimax(models, idx):
    def f(c):
        return max(q(*models[i], c) for i in idx)
    best = np.inf
    for s in [(0.3, 0.3), (0.5, 0.3), (0.5, 0.5), (0.7, 0.2), (0.2, 0.6), (0.5, 0.1)]:
        r = minimize(f, s, method="Nelder-Mead",
                     options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 20000})
        best = min(best, float(r.fun))
    return best


def main():
    out = {}
    for name, models in FAMILIES.items():
        pairs = {f"{i+1}{j+1}": minimax(models, [i, j])
                 for i in range(3) for j in range(i + 1, 3)}
        triple = minimax(models, [0, 1, 2])
        out[name] = {"pairs": pairs, "triple": triple}
        print(name, "pairs:", {k: round(v, 10) for k, v in pairs.items()},
              "triple:", round(triple, 10))
    a, b = out["A"], out["B"]
    pair_gap = max(abs(a["pairs"][k] - b["pairs"][k]) for k in a["pairs"])
    exact_gap = max(abs(v - 0.25) for fam in (a, b) for v in fam["pairs"].values())
    triple_gap = abs(a["triple"] - b["triple"])
    print(f"max pair-table deviation between families: {pair_gap:.2e}")
    print(f"max pair deviation from closed form 1/4:   {exact_gap:.2e}")
    print(f"triple-score gap |rho_A - rho_B|:          {triple_gap:.6f}")
    assert exact_gap < 1e-9 and pair_gap < 1e-9, "pair tables must match"
    assert triple_gap > 1e-3, "triples must differ"
    out["summary"] = {"pair_table_value": 0.25, "triple_A": a["triple"],
                      "triple_B": b["triple"], "triple_gap": triple_gap}
    od = REPO / "results_public/pair_table_counterexample"
    od.mkdir(exist_ok=True)
    (od / "pair_table_counterexample.json").write_text(json.dumps(out, indent=2))
    print("verified and saved")


if __name__ == "__main__":
    main()
