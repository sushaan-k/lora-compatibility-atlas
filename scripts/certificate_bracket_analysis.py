#!/usr/bin/env python3
"""Post-hoc analyses over the released per-query certificate CSVs.

All quantities here are recomputed from results_public/*/results.csv with no model
forward passes (CPU only).  Three things are produced:

  (1) Transfer gap: AUROC/AUPRC pooled and per operator, split by whether the
      query's fitted Hessian was PSD-projected (psd_projected).  This quantifies
      how much the projected surrogate costs on the projected subset.
  (2) Budget operating point: recall and precision when only the lowest-score
      fraction of candidates is sent to held-out validation.
  (3) Certificate-bracket confidence: the primal-dual bracket width
      (certificate_gap = primal_upper - dual_lower) as an observable confidence
      signal -- selective AUROC vs coverage, AUROC by bracket quartile, and the
      concentration of screen errors in wide-bracket queries.  Plus a
      leave-one-slice-out Platt calibration (Brier / ECE) reconfirmation.

Score convention: predicted_score is the minimax deficit rho_S (lower = more
feasible), so the ranking score for a feasible-positive label is -predicted_score.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results_public")

QWEN = os.path.join(RESULTS, "qwen25_m50_balanced", "results.csv")
MULTIFAMILY = {
    "TinyLlama-v0.3": "tinyllama_12x",
    "Mistral-7B-Instruct-v0.2": "mistral_10x",
    "Phi-2": "phi2_10x",
    "Pythia-410M": "pythia410m_10x",
    "Pythia-1B": "pythia1b_10x",
}


def _auroc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return roc_auc_score(y, s)


def _auprc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return average_precision_score(y, s)


def load(path):
    df = pd.read_csv(path)
    df = df[df["predicted_score"].notna()].copy()
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"].astype(str).str.len() == 0)]
    df["y"] = df["feasible"].astype(int)
    df["score"] = -df["predicted_score"].astype(float)  # higher = more feasible
    df["proj"] = df["psd_projected"].astype(float).astype(int)
    return df


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def transfer_gap(df):
    section("(1) TRANSFER GAP  --  AUROC / AUPRC by PSD-projection status")
    rows = []
    groups = [("POOLED", df)] + [(m, df[df["method"] == m]) for m in sorted(df["method"].unique())]
    for name, g in groups:
        proj = g[g["proj"] == 1]
        nonp = g[g["proj"] == 0]
        rows.append({
            "group": name,
            "n": len(g),
            "feas": int(g["y"].sum()),
            "proj_rate": round(g["proj"].mean(), 3),
            "AUROC_all": round(_auroc(g["y"], g["score"]), 3),
            "AUROC_nonproj": round(_auroc(nonp["y"], nonp["score"]), 3) if len(nonp) else float("nan"),
            "AUROC_proj": round(_auroc(proj["y"], proj["score"]), 3) if len(proj) else float("nan"),
            "AUPRC_all": round(_auprc(g["y"], g["score"]), 3),
        })
    print(pd.DataFrame(rows).to_string(index=False))


def budget_operating_point(df, budgets=(0.10, 0.20, 0.30)):
    section("(2) BUDGET OPERATING POINT  --  validate lowest-score fraction")
    g = df.sort_values("predicted_score", ascending=True).reset_index(drop=True)
    total_feasible = int(g["y"].sum())
    n = len(g)
    rows = []
    for b in budgets:
        k = int(round(b * n))
        sel = g.iloc[:k]
        tp = int(sel["y"].sum())
        rows.append({
            "budget_frac": b,
            "k_validated": k,
            "precision": round(tp / k, 3) if k else float("nan"),
            "recall": round(tp / total_feasible, 3) if total_feasible else float("nan"),
            "frac_candidates_skipped": round(1 - b, 3),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n[base rate feasible = {total_feasible}/{n} = {total_feasible/n:.3f}]")
    # cost-weighted skip: triples are more expensive than pairs; report the
    # validated set's pair/triple composition at 20% to contextualise GPU savings.
    k20 = int(round(0.20 * n))
    sel20 = g.iloc[:k20]
    comp = sel20["query_kind"].value_counts(normalize=True).round(3).to_dict()
    full = g["query_kind"].value_counts(normalize=True).round(3).to_dict()
    print(f"[query-kind mix among validated 20%: {comp}  vs full pool: {full}]")


def bracket_confidence(df):
    section("(3) CERTIFICATE BRACKET WIDTH  --  confidence / selective signal")
    g = df.copy()
    gap = g["certificate_gap"].astype(float)
    print(f"certificate_gap: n={gap.notna().sum()}  min={gap.min():.4f}  "
          f"median={gap.median():.4f}  mean={gap.mean():.4f}  max={gap.max():.4f}  "
          f"frac<=1e-6={float((gap <= 1e-6).mean()):.3f}")
    # correlation with the hard (projected) queries
    print(f"corr(gap, psd_projected) = {np.corrcoef(gap, g['proj'])[0,1]:.3f}")
    print(f"mean gap | projected   = {gap[g['proj']==1].mean():.4f}")
    print(f"mean gap | unprojected = {gap[g['proj']==0].mean():.4f}")

    # AUROC by bracket quartile (tightest Q1 ... widest Q4)
    g = g.assign(gap=gap)
    g["q"] = pd.qcut(g["gap"].rank(method="first"), 4, labels=["Q1_tight", "Q2", "Q3", "Q4_wide"])
    print("\nAUROC within bracket quartile (tightest -> widest):")
    for q, sub in g.groupby("q", observed=True):
        print(f"  {q}: n={len(sub):4d}  feas={int(sub['y'].sum()):4d}  "
              f"AUROC={_auroc(sub['y'], sub['score']):.3f}")

    # Selective AUROC vs coverage: keep the tightest-gap fraction, screen there.
    print("\nSelective AUROC by coverage (keep tightest-bracket fraction):")
    order = g.sort_values("gap", ascending=True).reset_index(drop=True)
    for cov in (0.25, 0.50, 0.75, 1.00):
        k = int(round(cov * len(order)))
        sub = order.iloc[:k]
        print(f"  coverage={cov:.2f}  n={k:4d}  AUROC={_auroc(sub['y'], sub['score']):.3f}")

    # Screen errors concentrate in wide brackets.  Decision = feasible if score
    # above the balanced-accuracy-optimal threshold; report error rate by quartile.
    s = g["score"].values
    y = g["y"].values
    thr_grid = np.quantile(s, np.linspace(0.01, 0.99, 99))
    best_thr, best_bacc = thr_grid[0], -1
    for t in thr_grid:
        pred = (s >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
        tpr = tp / (tp + fn) if (tp + fn) else 0
        tnr = tn / (tn + fp) if (tn + fp) else 0
        bacc = 0.5 * (tpr + tnr)
        if bacc > best_bacc:
            best_bacc, best_thr = bacc, t
    g["pred"] = (g["score"] >= best_thr).astype(int)
    g["err"] = (g["pred"] != g["y"]).astype(int)
    print(f"\nScreen decision at balanced-accuracy-optimal threshold "
          f"(bacc={best_bacc:.3f}); error rate by bracket quartile:")
    for q, sub in g.groupby("q", observed=True):
        print(f"  {q}: error_rate={sub['err'].mean():.3f}  (n={len(sub)})")
    # if you validate the widest-bracket X% you catch this share of all errors:
    order_err = g.sort_values("gap", ascending=False).reset_index(drop=True)
    total_err = int(g["err"].sum())
    for frac in (0.20, 0.30):
        k = int(round(frac * len(order_err)))
        caught = int(order_err.iloc[:k]["err"].sum())
        print(f"  validate widest-bracket {int(frac*100)}% -> catches "
              f"{caught}/{total_err} = {caught/total_err:.3f} of all screen errors")


def platt_calibration(df):
    section("(4) LEAVE-ONE-SLICE-OUT PLATT CALIBRATION (reconfirm Brier / ECE)")
    if "slice_id" not in df.columns or df["slice_id"].isna().all():
        print("no slice_id; skipping")
        return
    g = df[df["slice_id"].notna()].copy()
    slices = sorted(g["slice_id"].unique())
    probs = np.full(len(g), np.nan)
    idx = {ix: i for i, ix in enumerate(g.index)}
    for sl in slices:
        tr = g[g["slice_id"] != sl]
        te = g[g["slice_id"] == sl]
        if len(np.unique(tr["y"])) < 2:
            continue
        lr = LogisticRegression(max_iter=1000)
        lr.fit(tr["score"].values.reshape(-1, 1), tr["y"].values)
        p = lr.predict_proba(te["score"].values.reshape(-1, 1))[:, 1]
        for j, ix in enumerate(te.index):
            probs[idx[ix]] = p[j]
    y = g["y"].values
    mask = ~np.isnan(probs)
    y, probs = y[mask], probs[mask]
    brier = float(np.mean((probs - y) ** 2))
    base_rate = y.mean()
    base_brier = float(base_rate * (1 - base_rate))  # constant base-rate predictor
    # 10-bin ECE
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        if m.sum():
            ece += (m.mean()) * abs(probs[m].mean() - y[m].mean())
    print(f"slices={len(slices)}  n={len(y)}  base_rate={base_rate:.3f}")
    print(f"Platt Brier      = {brier:.3f}")
    print(f"base-rate Brier  = {base_brier:.3f}")
    print(f"10-bin ECE       = {ece:.3f}")


def multifamily_auprc():
    section("(5) MULTIFAMILY AUPRC RANGES (per operator within each panel)")
    rows = []
    for label, sub in MULTIFAMILY.items():
        path = os.path.join(RESULTS, sub, "results.csv")
        if not os.path.exists(path):
            rows.append({"panel": label, "status": "missing"})
            continue
        df = load(path)
        aurocs, auprcs = [], []
        for m, g in df.groupby("method"):
            if len(np.unique(g["y"])) < 2:
                continue
            aurocs.append(_auroc(g["y"], g["score"]))
            auprcs.append(_auprc(g["y"], g["score"]))
        rows.append({
            "panel": label, "n": len(df), "feas": int(df["y"].sum()),
            "base_rate": round(df["y"].mean(), 3),
            "AUROC_range": f"{min(aurocs):.2f}-{max(aurocs):.2f}" if aurocs else "na",
            "AUPRC_range": f"{min(auprcs):.2f}-{max(auprcs):.2f}" if auprcs else "na",
        })
    print(pd.DataFrame(rows).to_string(index=False))


def summary_savings():
    section("(6) summary.json fields mentioning savings / cost / budget")
    path = os.path.join(RESULTS, "qwen25_m50_balanced", "summary.json")
    if not os.path.exists(path):
        print("no summary.json")
        return
    d = json.load(open(path))
    def walk(o, pre=""):
        if isinstance(o, dict):
            for k, v in o.items():
                kk = f"{pre}.{k}" if pre else k
                if any(t in k.lower() for t in ("sav", "cost", "budget", "gpu", "recall", "precision", "fraction", "skip")):
                    if not isinstance(v, (dict, list)):
                        print(f"  {kk} = {v}")
                walk(v, kk)
        elif isinstance(o, list):
            for i, v in enumerate(o[:3]):
                walk(v, f"{pre}[{i}]")
    walk(d)
    print("  [top-level keys:", list(d.keys()), "]")


if __name__ == "__main__":
    df = load(QWEN)
    print(f"Loaded headline panel: {QWEN}  ({len(df)} rows, {int(df['y'].sum())} feasible)")
    transfer_gap(df)
    budget_operating_point(df)
    bracket_confidence(df)
    platt_calibration(df)
    multifamily_auprc()
    summary_savings()
