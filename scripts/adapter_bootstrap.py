#!/usr/bin/env python3
"""Adapter-level pigeonhole bootstrap (Owen, 2007) for the Qwen m=50 panel.

Events (pairs/triples) share adapters, so query-level resampling understates
dependence. The pigeonhole bootstrap resamples the ADAPTER pool: draw adapter
counts w_a ~ Multinomial(50, uniform) and weight each event by the product of
its adapters' counts; events containing an unsampled adapter drop out. Report
percentile intervals for per-operator and pooled AUROC, and the pooled
20%-budget operating point.
"""
import csv, os, sys
import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_public")
rng = np.random.default_rng(20260709)
B = 2000

rows = list(csv.DictReader(open(f"{BASE}/qwen25_m50_balanced/results.csv")))
rows = [r for r in rows if not r.get("error")]
adapters = sorted({a for r in rows for a in r["adapters"].split("+")})
aidx = {a: i for i, a in enumerate(adapters)}
m = len(adapters)
print(f"events={len(rows)}  adapters={m}  feasible={sum(int(r['feasible']) for r in rows)}")

methods = sorted({r["method"] for r in rows})
# score: negate predicted_score so higher = more likely feasible
ev = {
    "method": np.array([r["method"] for r in rows]),
    "y": np.array([int(r["feasible"]) for r in rows]),
    "s": np.array([-float(r["predicted_score"]) for r in rows]),
    "mem": [ [aidx[a] for a in r["adapters"].split("+")] for r in rows ],
}

def wauroc(y, s, w):
    pos, neg = (y == 1) & (w > 0), (y == 0) & (w > 0)
    if not pos.any() or not neg.any():
        return np.nan
    # rank-based weighted AUROC
    order = np.argsort(s, kind="mergesort")
    ys, ws = y[order], w[order]
    cw_neg = np.cumsum(np.where(ys == 0, ws, 0.0))
    # ties: group by score value
    sv = s[order]
    auc_num = 0.0
    i = 0
    total_neg = cw_neg[-1]
    neg_before = 0.0
    while i < len(sv):
        j = i
        while j < len(sv) and sv[j] == sv[i]:
            j += 1
        grp_pos = np.where(ys[i:j] == 1, ws[i:j], 0.0).sum()
        grp_neg = np.where(ys[i:j] == 0, ws[i:j], 0.0).sum()
        auc_num += grp_pos * (neg_before + 0.5 * grp_neg)
        neg_before += grp_neg
        i = j
    total_pos = np.where(y == 1, w, 0.0).sum()
    return auc_num / (total_pos * total_neg)

def event_weights(counts):
    return np.array([np.prod([counts[a] for a in mm]) for mm in ev["mem"]], dtype=float)

# --- point estimates (all weights 1): must match published numbers
w1 = np.ones(len(rows))
print("\npoint estimates (must match paper):")
for meth in methods:
    msk = ev["method"] == meth
    print(f"  {meth:12s} AUROC {wauroc(ev['y'][msk], ev['s'][msk], w1[msk]):.3f}")
print(f"  {'pooled':12s} AUROC {wauroc(ev['y'], ev['s'], w1):.3f}")

# --- 20%-budget pooled operating point (unweighted, must match 45.5/77.6)
order = np.argsort(-ev["s"])
k = round(0.20 * len(rows))
tp = ev["y"][order][:k].sum()
print(f"  20% budget: recall {tp/ev['y'].sum():.3f} precision {tp/k:.3f}")

# --- pigeonhole bootstrap
boot = {meth: [] for meth in methods}
boot["pooled"] = []
boot_op = []
for b in range(B):
    counts = rng.multinomial(m, np.ones(m) / m)
    w = event_weights(counts)
    for meth in methods:
        msk = ev["method"] == meth
        boot[meth].append(wauroc(ev["y"][msk], ev["s"][msk], w[msk]))
    boot["pooled"].append(wauroc(ev["y"], ev["s"], w))
    # weighted 20%-budget operating point
    o = np.argsort(-ev["s"])
    wo, yo = w[o], ev["y"][o]
    cw = np.cumsum(wo)
    kk = np.searchsorted(cw, 0.20 * cw[-1])
    tpw = (wo[:kk] * yo[:kk]).sum()
    posw = (w * ev["y"]).sum()
    boot_op.append((tpw / posw if posw > 0 else np.nan, tpw / cw[kk - 1] if kk > 0 else np.nan))

print(f"\npigeonhole bootstrap, B={B}, 95% percentile intervals:")
for keym in methods + ["pooled"]:
    arr = np.array(boot[keym])
    arr = arr[~np.isnan(arr)]
    lo, hi = np.percentile(arr, [2.5, 97.5])
    print(f"  {keym:12s} [{lo:.3f}, {hi:.3f}]  (excludes 0.5: {lo > 0.5})")
op = np.array(boot_op)
op = op[~np.isnan(op).any(axis=1)]
rlo, rhi = np.percentile(op[:, 0], [2.5, 97.5])
plo, phi = np.percentile(op[:, 1], [2.5, 97.5])
print(f"  20%-budget recall    [{rlo:.3f}, {rhi:.3f}]")
print(f"  20%-budget precision [{plo:.3f}, {phi:.3f}]")
