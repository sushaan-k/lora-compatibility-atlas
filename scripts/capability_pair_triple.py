#!/usr/bin/env python3
"""Pair-only vs. triple-only screening AUROC/AUPRC for the answer-token panels.

A reviewer asked for the answer-token capability panels (tab:capability) to be
split by subset arity: pairs are part of the poised fit design, while triples
are the genuine off-design extrapolation. The published table reports a single
pooled AUROC/AUPRC over the 120 pairs + 120 triples (m=16) or 28 pairs + 56
triples (m=8). This script recomputes the pooled numbers and, in addition,
reports the PAIR-ONLY and TRIPLE-ONLY AUROC and AUPRC separately, together with
the feasible/infeasible counts in each arity bucket at tau=0.70.

The subset reconstruction, feasibility label, screening score, and AUROC are the
byte-for-byte definitions used in scripts/capability_dependence_ci.py (which in
turn copies experiments_gpu/expD_answer_panel.py):

  label(subset)  = every member retains >= tau of its answer-token gain
                   member retained iff  L_answer <= base_La - tau*gain
  score(subset)  = -max_member (pred_L_answer - (base_La - tau*gain))
  AUROC          = Mann-Whitney with 0.5 credit for ties, positive class=feasible

AUPRC is the codebase's canonical step-wise average precision (auc_pr in
peft_atlas_lite/run_peft_atlas_lite.py), the same metric quoted in the AUPRC
column of tab:capability. Both AUROC and AUPRC are undefined for a single-class
bucket (AUROC: either class empty; AUPRC: no positives); those are reported as
null and flagged.

All quantities are reconstructed from the per_member records already stored in
each panel's answer_panel_analysis.json -- no GPU or model weights needed.

Run:  python scripts/capability_pair_triple.py
Out:  results_public/capability_ci/pair_triple_auroc.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TAU = 0.70

# The two linear answer-token panels the reviewer asked about, plus the two
# m=8 operator panels for context (kept when present).
PANELS = [
    ("mistral_m16_linear", "results_public/scale_m16", "Mistral-7B m=16 linear"),
    ("mistral_m8_linear", "results_public/answer_panel_m8", "Mistral-7B m=8 linear"),
]


# ----------------------------------------------------------------- primitives
# (auroc / retained_answer / pred_deficit / build_subsets copied verbatim from
#  scripts/capability_dependence_ci.py; auc_pr copied verbatim from
#  peft_atlas_lite/run_peft_atlas_lite.py.)

def auroc(labels, scores):
    """Mann-Whitney U / (|pos||neg|) with 0.5 credit for ties. Positive class is
    the feasible (retained) subset. Returns None if either class is empty."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def auc_pr(labels, scores):
    """Step-wise average precision (codebase canonical AUPRC). Returns None if
    there are no positives (undefined). Sort is descending by score; ties keep
    input order via Python's stable sort, matching run_peft_atlas_lite."""
    if not labels or not any(labels):
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    total_pos = sum(1 for label in labels if label)
    prev_recall = 0.0
    area = 0.0
    tp = 0
    fp = 0
    for _, label in ordered:
        if label:
            tp += 1
        else:
            fp += 1
        recall = tp / total_pos
        precision = tp / max(tp + fp, 1)
        area += (recall - prev_recall) * precision
        prev_recall = recall
    return area


def retained_answer(rec, tau):
    """Member keeps at least tau of its answer-token gain (expD definition)."""
    return rec["L_answer"] <= rec["base_La"] - tau * rec["gain"]


def pred_deficit(rec, tau):
    """Predicted retention deficit (expD definition); >0 => predicted reject."""
    return rec["pred_L_answer"] - (rec["base_La"] - tau * rec["gain"])


def build_subsets(per_member):
    """Group per_member rows into subsets keyed by `merge`, keeping only complete
    subsets (member count equals the recorded order)."""
    by_merge = defaultdict(list)
    for r in per_member:
        by_merge[r["merge"]].append(r)
    subsets = []
    for merge, members in by_merge.items():
        order = members[0]["order"]
        if len(members) == order:
            subsets.append({"merge": merge,
                            "order": order,
                            "adapters": merge.split("+"),
                            "members": members})
    return subsets


def label_score(subsets, tau):
    labels = [all(retained_answer(m, tau) for m in s["members"]) for s in subsets]
    scores = [-max(pred_deficit(m, tau) for m in s["members"]) for s in subsets]
    return labels, scores


def bucket_metrics(subsets, tau):
    """AUROC, AUPRC and feasibility counts for one arity bucket (or pooled)."""
    labels, scores = label_score(subsets, tau)
    n = len(labels)
    n_feasible = int(sum(labels))
    n_infeasible = n - n_feasible
    au = auroc(labels, scores)
    ap = auc_pr(labels, scores)
    single_class = (n_feasible == 0 or n_infeasible == 0)
    return {
        "n_subsets": n,
        "n_feasible": n_feasible,
        "n_infeasible": n_infeasible,
        "feasible_rate": (n_feasible / n) if n else None,
        "auroc": au,
        "auprc": ap,
        "single_class": single_class,
        "auroc_undefined": au is None,
        "auprc_undefined": ap is None,
    }


# ----------------------------------------------------------------------- main

def analyze_panel(subdir, label, tau=TAU):
    d = json.loads((REPO / subdir / "answer_panel_analysis.json").read_text())
    subsets = build_subsets(d["per_member"])
    pairs = [s for s in subsets if s["order"] == 2]
    triples = [s for s in subsets if s["order"] == 3]

    combined = bucket_metrics(subsets, tau)
    pair = bucket_metrics(pairs, tau)
    triple = bucket_metrics(triples, tau)

    # cross-check against the AUROC values already stored in the panel json
    published = d.get("by_answer_tau", {}).get(str(tau), {})
    checks = {
        "auroc_pooled_published": published.get("auroc_pooled"),
        "auroc_pairs_published": published.get("auroc_pairs"),
        "auroc_triples_published": published.get("auroc_triples"),
        "reproduces_pooled": _close(combined["auroc"], published.get("auroc_pooled")),
        "reproduces_pairs": _close(pair["auroc"], published.get("auroc_pairs")),
        "reproduces_triples": _close(triple["auroc"], published.get("auroc_triples")),
    }

    return {
        "label": label,
        "results_dir": subdir,
        "tau": tau,
        "n_adapters": len(d["names"]),
        "buckets": {"combined": combined, "pairs": pair, "triples": triple},
        "published_crosscheck": checks,
    }


def _close(a, b):
    """None-aware near-equality. True iff both None, or both defined and equal
    to 1e-12; False if exactly one is None."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) < 1e-9


def main():
    out = {
        "tau": TAU,
        "description": "Pair-only vs triple-only screening AUROC/AUPRC for the "
                       "answer-token capability panels. Pairs are on the poised "
                       "fit design; triples are off-design extrapolation. AUROC "
                       "and AUPRC are the capability_dependence_ci / expD and "
                       "peft_atlas_lite definitions; both are null (undefined) "
                       "for single-class buckets.",
        "auprc_definition": "step-wise average precision (auc_pr, "
                            "peft_atlas_lite/run_peft_atlas_lite.py)",
        "panels": {},
    }
    rows = []
    for key, subdir, label in PANELS:
        if not (REPO / subdir / "answer_panel_analysis.json").exists():
            continue
        res = analyze_panel(subdir, label)
        out["panels"][key] = res
        for arity in ("pairs", "triples", "combined"):
            b = res["buckets"][arity]
            rows.append((label, arity, b))

    out_dir = REPO / "results_public" / "capability_ci"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pair_triple_auroc.json").write_text(json.dumps(out, indent=1))

    # console table
    hdr = f"{'panel':22s} {'arity':8s} {'n':>4s} {'n_infeas':>8s} {'AUROC':>8s} {'AUPRC':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for label, arity, b in rows:
        au = "undef" if b["auroc"] is None else f"{b['auroc']:.4f}"
        ap = "undef" if b["auprc"] is None else f"{b['auprc']:.4f}"
        flag = "  <- single-class" if b["single_class"] else ""
        print(f"{label:22s} {arity:8s} {b['n_subsets']:4d} {b['n_infeasible']:8d} "
              f"{au:>8s} {ap:>8s}{flag}")
    print("\nwrote", out_dir / "pair_triple_auroc.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
