#!/usr/bin/env python3
"""Recompute the paper's headline numbers from the released data and check them."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R = ROOT / "results_public"


def num(path, *keys):
    d = json.loads((R / path).read_text())
    for k in keys:
        d = d[k]
    return d


# (label, computed value, published value, tolerance)
def checks():
    out = []
    a = num("scale_m50/scale_analysis.json", "by_answer_tau", "0.7", "auroc_pooled")
    out.append(("m=50 screening AUROC", a, 0.95, 0.005))
    out.append(("m=50 triples AUROC",
                num("scale_m50/scale_analysis.json", "by_answer_tau", "0.7", "auroc_triples"),
                0.94, 0.005))
    out.append(("m=30 screening AUROC",
                num("scale_m30/scale_analysis.json", "by_answer_tau", "0.7", "auroc_pooled"),
                0.94, 0.005))
    out.append(("m=50 subsets", num("scale_m50/scale_analysis.json", "n_subsets"), 1096, 0))
    out.append(("m=50 scored tasks", num("scale_m50/scale_analysis.json", "n_tasks"), 44, 0))
    out.append(("m=16 triple fidelity",
                num("capability_ci/m16_fidelity_ci.json", "triple_r_recomputed"), 0.96, 0.005))
    out.append(("m=50 sigma-adim", num("scale_m50/sigma_adim.json", "first_r_below_0.20"), 5, 0))
    out.append(("m=30 sigma-adim", num("scale_m30/sigma_adim.json", "first_r_below_0.20"), 6, 0))
    s = json.loads((R / "intervention_15q/summary.json").read_text())
    out.append(("15-query improvements", s["n_worst_improved"], 13, 0))
    out.append(("15-query median gain", s["median_worst_delta_nats"], 2.25, 0.01))
    e = json.loads((R / "deployed_envelopes/deployed_envelopes.json").read_text())["summary"]
    out.append(("m=8 envelope certified",
                e["0.7"]["n_certified"] + e["0.85"]["n_certified"], 41, 0))
    e2 = json.loads((R / "deployed_envelopes_m50/deployed_envelopes.json").read_text())["summary"]
    out.append(("m=50 envelope certified",
                e2["0.7"]["n_certified"] + e2["0.85"]["n_certified"], 32, 0))
    t = num("query_timing/timing49.json",
            "minimax_query_ms_per_subset_size", "5", "median_ms")
    out.append(("d=49 five-way median ms", t, 464, 250))
    g = num("tinyllama_stability/coldstart.json", "per_operator", "linear", "atlas")
    out.append(("TinyLlama atlas AUROC", g, 0.93, 0.01))
    return out


SCRIPTS = ["panelci.py", "fidelity.py", "adim.py", "cheap.py", "tinygbm.py", "envelopes.py"]


def main():
    if "--rerun" in sys.argv:
        for s in SCRIPTS:
            print(f"running {s}")
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / s)],
                               capture_output=True, text=True)
            if r.returncode:
                print(f"  FAILED: {r.stderr.strip().splitlines()[-1:]}")
                return 1

    bad = 0
    for label, got, want, tol in checks():
        ok = abs(float(got) - float(want)) <= tol
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'}  {label:26s} {got}  (paper: {want})")
    print("\nall headline numbers match" if not bad else f"\n{bad} mismatch(es)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
