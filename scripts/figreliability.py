#!/usr/bin/env python3
"""Reliability figure (figs_png/reliability.pdf): empirical feasibility rate against atlas-margin bucket median,"""
from __future__ import annotations

import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from style import apply_style, BLUE

import matplotlib.pyplot as plt

# Same six panels as the supplement figure; the local copies are byte-identical
# to the supplement campaign CSVs.
DATASETS = {
    "Qwen2.5-Instruct (m=50, balanced slices)":
        REPO / "results_public/qwen25_m50_balanced/results.csv",
    "TinyLlama-v0.3 (m=12)":
        REPO / "results_public/tinyllama_12x/results.csv",
    "Mistral-7B (m=10, MIT adapters)":
        REPO / "results_public/mistral_10x/results.csv",
    "Phi-2 (m=10, MIT adapters)":
        REPO / "results_public/phi2_10x/results.csv",
    "Pythia-410M (m=10)":
        REPO / "results_public/pythia410m_10x/results.csv",
    "Pythia-1B (m=10)":
        REPO / "results_public/pythia1b_10x/results.csv",
}


def load(path):
    df = pd.read_csv(path)
    df["predicted_score"] = pd.to_numeric(df["predicted_score"], errors="coerce")
    df["heldout_score"] = pd.to_numeric(df["heldout_score"], errors="coerce")
    df["feasible"] = pd.to_numeric(df["feasible"], errors="coerce").astype(int)
    return df.dropna(subset=["predicted_score", "heldout_score", "feasible"])


def main():
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(5.85, 3.6), layout="constrained",
                             squeeze=False)
    axes = axes.ravel()
    bucket_ranges = {}
    for ax, (name, path) in zip(axes, DATASETS.items()):
        df = load(path)
        sub = df[df["method"] == "linear"].copy()
        if sub.empty:
            sub = df.copy()
        margin = -sub["predicted_score"].to_numpy()
        feasible = sub["feasible"].to_numpy().astype(int)
        # Bucket by margin quantile: sixths (quartiles only below n=60)
        n_buckets = 6 if len(sub) >= 60 else 4
        qs = np.quantile(margin, np.linspace(0, 1, n_buckets + 1))
        qs[0] -= 1e-9
        qs[-1] += 1e-9
        buckets = np.clip(np.digitize(margin, qs) - 1, 0, n_buckets - 1)
        x_centers, y_emp, errs, counts = [], [], [], []
        for b in range(n_buckets):
            mask = buckets == b
            if mask.sum() < 2:
                continue
            p = float(feasible[mask].mean())
            n = int(mask.sum())
            z = 1.96  # Wilson 95%
            den = 1 + z**2 / n
            centre = (p + z**2 / (2 * n)) / den
            halfw = z * sqrt(p * (1 - p) / n + z**2 / (4 * n * n)) / den
            lo = max(0.0, centre - halfw)
            hi = min(1.0, centre + halfw)
            x_centers.append(float(np.median(margin[mask])))
            y_emp.append(p)
            errs.append([max(0.0, p - lo), max(0.0, hi - p)])
            counts.append(n)
        bucket_ranges[name] = (min(counts), max(counts), n_buckets)
        ax.errorbar(np.array(x_centers), np.array(y_emp),
                    yerr=np.array(errs).T, fmt="o-", color=BLUE,
                    ecolor="0.45", elinewidth=0.7, capsize=1.8,
                    markersize=3.5, lw=1.1, zorder=3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(name, fontsize=7, pad=3)
    for ax in axes[len(DATASETS):]:
        ax.set_visible(False)
    fig.supylabel("Empirical feasibility rate", fontsize=8)
    fig.supxlabel("Atlas margin $-\\widehat\\rho_S(C)$ (nats, bucket median)",
                  fontsize=8)
    out = REPO / "figs_png" / "reliability.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)
    for name, (lo, hi, nb) in bucket_ranges.items():
        print(f"  {name}: {nb} buckets, sizes {lo}-{hi}")


if __name__ == "__main__":
    main()
