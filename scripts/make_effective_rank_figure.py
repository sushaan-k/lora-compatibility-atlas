#!/usr/bin/env python3
"""Mean-Hessian eigenvalue spectra (figs_png/effective_rank_spectrum.pdf).

Reads the strict effective-rank diagnostics written by
measure_effective_rank_strict.py (results_public/effective_rank_strict/) and
plots the eigenvalues of the mean Hessian H-bar for the TinyLlama and Qwen
five-adapter sub-libraries.  Log y axis (spectra span two orders of
magnitude), one shared y label, and headroom above the top bar.

Rendered at final physical size (6.2in wide; the .tex includes it at
0.95\\linewidth on a 6.5in TMLR text block) with the shared 8pt boxed style.
"""
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from fig_style import apply_style, BLUE, ORANGE

import matplotlib.pyplot as plt

PANELS = [
    ("TinyLlama", REPO / "results_public/effective_rank_strict/tinyllama.json", BLUE),
    ("Qwen2.5", REPO / "results_public/effective_rank_strict/qwen.json", ORANGE),
]

apply_style()
fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.2), layout="constrained")

for ax, (name, path, color) in zip(axes, PANELS):
    d = json.loads(path.read_text())
    eigvals = d["spectrum"]["eigvals_H_bar"]
    trace = d["spectrum"]["trace_H_bar"]
    m = d["n_adapters_m"]
    dim = d["chart_dim_d"]
    frac1 = 100.0 * eigvals[0] / trace
    idx = range(1, len(eigvals) + 1)
    bottom = 10.0 ** math.floor(math.log10(min(eigvals) / 2.0))
    ax.bar(idx, eigvals, color=color, edgecolor="white",
           linewidth=0.6, zorder=3)
    ax.set_yscale("log")
    # headroom so the tallest bar is not flush with the top spine
    ax.set_ylim(bottom, max(eigvals) * 1.4)
    ax.set_xticks(list(idx))
    ax.set_xlabel("eigenvalue index")
    ax.set_title(f"{name} $m{{=}}{m}$ ($d{{=}}{dim}$)\n"
                 f"{frac1:.1f}% energy in $\\lambda_1$", fontsize=9.5)

fig.supylabel("eigenvalue of $\\bar{H}$", fontsize=8)

out = REPO / "figs_png" / "effective_rank_spectrum.pdf"
fig.savefig(out)
plt.close(fig)
print("wrote", out)
