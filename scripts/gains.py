#!/usr/bin/env python3
"""Single-adapter gain distribution per panel, from the released run summaries."""
import json
import os
import statistics as st
from pathlib import Path

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_public")
FLOOR = 0.005


def gains_from_summary(path):
    s = json.load(open(path))
    comps = s.get("component_summaries") or [s]
    gains = {}
    for c in comps:
        bl = (c.get("base_losses") or {}).get("heldout") or (c.get("base_losses") or {}).get("calibration") or {}
        sl = (c.get("single_losses") or {}).get("heldout") or (c.get("single_losses") or {}).get("calibration") or {}
        for a in bl:
            if a in sl:
                gains[a] = bl[a] - sl[a]
    return gains


def main():
    for panel in ["qwen25_m50_balanced", "mistral_10x", "phi2_10x", "pythia410m_10x", "pythia1b_10x"]:
        p = Path(BASE) / panel / "summary.json"
        if not p.exists():
            print(f"{panel}: no summary.json"); continue
        g = sorted(gains_from_summary(p).values())
        if not g:
            print(f"{panel}: no base/single losses in summary"); continue
        n = len(g)
        print(f"{panel:22s} n={n:3d}  min={g[0]:+.3f}  median={st.median(g):+.3f}  max={g[-1]:+.3f}  "
              f"nonpositive={sum(1 for x in g if x <= 0):3d}  below_floor({FLOOR})={sum(1 for x in g if x < FLOOR):3d}")


if __name__ == "__main__":
    main()
