#!/usr/bin/env python3
"""Dependence-aware uncertainty for the answer-token capability panels."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
TAU = 0.70
N_BOOT = 2000
SEED = 0

# panel key -> (results_public subdir, human label). The first two are the
# linear capability panels quoted in tab:capability; the operator panels are
# included when present.
PANELS = [
    ("mistral_m8_linear", "results_public/answer_panel_m8", "Mistral-7B m=8 linear"),
    ("mistral_m16_linear", "results_public/scale_m16", "Mistral-7B m=16 linear"),
    ("mistral_m8_ties", "results_public/operators_ties", "Mistral-7B m=8 TIES"),
    ("mistral_m8_dare_ties", "results_public/operators_dare_ties", "Mistral-7B m=8 DARE-TIES"),
    ("qwen25_1p5b_linear", "results_public/second_base_qwen", "Qwen2.5-1.5B m=8 linear"),
]


# ----------------------------------------------------------------- primitives

def auroc(labels, scores):
    """Exact copy of experiments_gpu/surfaces.py:auroc.

    Mann-Whitney U / (|pos| |neg|) with 0.5 credit for ties. Positive class is
    the feasible (retained) subset. Returns None if either class is empty."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def weighted_auroc(labels, scores, weights):
    """Weighted Mann-Whitney AUROC."""
    pos = [(s, w) for s, l, w in zip(scores, labels, weights) if l and w > 0]
    neg = [(s, w) for s, l, w in zip(scores, labels, weights) if (not l) and w > 0]
    wp = sum(w for _, w in pos)
    wn = sum(w for _, w in neg)
    if wp == 0 or wn == 0:
        return None
    wins = 0.0
    for sp, wpi in pos:
        for sn, wni in neg:
            wins += wpi * wni * ((sp > sn) + 0.5 * (sp == sn))
    return wins / (wp * wn)


def retained_answer(rec, tau):
    """Member keeps at least tau of its answer-token gain (expD definition)."""
    return rec["L_answer"] <= rec["base_La"] - tau * rec["gain"]


def pred_deficit(rec, tau):
    """Predicted retention deficit (expD definition); >0 => predicted reject."""
    return rec["pred_L_answer"] - (rec["base_La"] - tau * rec["gain"])


def build_subsets(per_member):
    """Group per_member rows into subsets keyed by `merge`. Members of a subset
    are exactly the rows sharing that merge string; the adapter membership is
    the "+"-split of the merge. Keep only complete subsets (member count equals
    the recorded order), matching build_events' per_subset."""
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


def subset_label_score(subsets, tau):
    """Per-subset feasibility label and screening score at tau."""
    labels = [all(retained_answer(m, tau) for m in s["members"]) for s in subsets]
    scores = [-max(pred_deficit(m, tau) for m in s["members"]) for s in subsets]
    return labels, scores


# ------------------------------------------------------------------- analyses

def leave_one_adapter_out(subsets, adapters, tau):
    """Drop every subset containing adapter a, recompute pooled AUROC. Report
    the per-adapter AUROC and the min-to-max range over defined drops."""
    per_adapter = {}
    for a in adapters:
        keep = [s for s in subsets if a not in s["adapters"]]
        labels = [all(retained_answer(m, tau) for m in s["members"]) for s in keep]
        scores = [-max(pred_deficit(m, tau) for m in s["members"]) for s in keep]
        au = auroc(labels, scores)
        per_adapter[a] = {
            "auroc": au,
            "n_subsets": len(keep),
            "n_retained": int(sum(labels)),
            "n_negative": int(len(labels) - sum(labels)),
        }
    defined = [v["auroc"] for v in per_adapter.values() if v["auroc"] is not None]
    undefined = [a for a, v in per_adapter.items() if v["auroc"] is None]
    rng = {
        "min": min(defined) if defined else None,
        "max": max(defined) if defined else None,
        "n_defined": len(defined),
        "n_undefined": len(undefined),
        "undefined_adapters": undefined,
    }
    return per_adapter, rng


def adapter_block_bootstrap(subsets, adapters, tau, n_boot=N_BOOT, seed=SEED):
    """Resample adapters with replacement; keep subsets whose members were all
    drawn; weight each survivor by the product of its members' multiplicities;
    recompute the weighted AUROC. Skip degenerate reps (<5 surviving subsets or
    one label class empty). Return the 2.5/97.5 percentile interval."""
    rng = np.random.default_rng(seed)
    labels, scores = subset_label_score(subsets, tau)
    adapters = list(adapters)
    idx = {a: i for i, a in enumerate(adapters)}
    # precompute member adapter indices per subset
    sub_adapters = [[idx[a] for a in s["adapters"]] for s in subsets]
    n = len(adapters)
    aurocs = []
    n_skip_few = n_skip_class = 0
    for _ in range(n_boot):
        draw = rng.integers(0, n, size=n)
        mult = np.bincount(draw, minlength=n)  # multiplicity per adapter
        weights = []
        for mem in sub_adapters:
            w = 1
            for j in mem:
                w *= int(mult[j])
            weights.append(w)  # 0 if any member not drawn
        n_kept = sum(1 for w in weights if w > 0)
        if n_kept < 5:
            n_skip_few += 1
            continue
        au = weighted_auroc(labels, scores, weights)
        if au is None:
            n_skip_class += 1
            continue
        aurocs.append(au)
    out = {
        "n_boot": n_boot,
        "n_valid": len(aurocs),
        "n_skipped_few_subsets": n_skip_few,
        "n_skipped_one_class": n_skip_class,
    }
    if aurocs:
        arr = np.array(aurocs)
        out.update({
            "ci_lo_2.5": float(np.percentile(arr, 2.5)),
            "ci_hi_97.5": float(np.percentile(arr, 97.5)),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
        })
    else:
        out.update({"ci_lo_2.5": None, "ci_hi_97.5": None,
                    "median": None, "mean": None})
    return out


# ----------------------------------------------------------------------- main

def analyze_panel(key, subdir, label, tau=TAU):
    path = REPO / subdir / "answer_panel_analysis.json"
    d = json.loads(path.read_text())
    names = [str(x) for x in d["names"]]
    subsets = build_subsets(d["per_member"])

    labels, scores = subset_label_score(subsets, tau)
    au = auroc(labels, scores)
    n = len(labels)
    n_ret = int(sum(labels))
    n_neg = n - n_ret

    published = d.get("by_answer_tau", {}).get(str(tau), {})
    pub_au = published.get("auroc_pooled")
    reproduced = (pub_au is not None and au is not None
                  and abs(au - pub_au) < 1e-12)

    per_adapter, loao = leave_one_adapter_out(subsets, names, tau)
    boot = adapter_block_bootstrap(subsets, names, tau)

    negatives = [s["merge"] for s, l in zip(subsets, labels) if not l]
    neg_adapter_counts = Counter()
    for m in negatives:
        for a in m.split("+"):
            neg_adapter_counts[a] += 1

    return {
        "label": label,
        "results_dir": subdir,
        "tau": tau,
        "n_adapters": len(names),
        "adapters": names,
        "n_subsets": n,
        "auroc_pooled_reproduced": au,
        "auroc_pooled_published": pub_au,
        "reproduces_published": reproduced,
        "n_retained": n_ret,
        "n_negative": n_neg,
        "feasible_rate": n_ret / n if n else None,
        "unstable_flag": n_neg <= 5,
        "negative_subsets": negatives,
        "negative_adapter_counts": dict(neg_adapter_counts),
        "leave_one_adapter_out": {
            "per_adapter": per_adapter,
            "range": loao,
        },
        "adapter_block_bootstrap": boot,
    }


def main():
    out = {"tau": TAU, "n_boot": N_BOOT, "seed": SEED,
           "description": "Dependence-aware uncertainty for answer-token "
                          "capability panels (adapter-block resampling).",
           "panels": {}}
    lines = []
    for key, subdir, label in PANELS:
        if not (REPO / subdir / "answer_panel_analysis.json").exists():
            continue
        res = analyze_panel(key, subdir, label)
        out["panels"][key] = res
        lo = res["adapter_block_bootstrap"]["ci_lo_2.5"]
        hi = res["adapter_block_bootstrap"]["ci_hi_97.5"]
        rr = res["leave_one_adapter_out"]["range"]
        flag = "  [UNSTABLE: <=5 negatives]" if res["unstable_flag"] else ""
        lines.append(
            f"{res['label']:26s} AUROC={res['auroc_pooled_reproduced']:.4f} "
            f"(pub {res['auroc_pooled_published']:.4f}, "
            f"repro={res['reproduces_published']})  "
            f"neg={res['n_negative']}/{res['n_subsets']}  "
            f"LOAO=[{_fmt(rr['min'])},{_fmt(rr['max'])}]"
            f"{' undef=' + str(rr['n_undefined']) if rr['n_undefined'] else ''}  "
            f"block-boot 95% CI=[{_fmt(lo)},{_fmt(hi)}]{flag}")

    out_dir = REPO / "results_public" / "capability_ci"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "capability_ci.json").write_text(json.dumps(out, indent=1))
    print("\n".join(lines))
    print("\nwrote", out_dir / "capability_ci.json")
    return 0


def _fmt(x):
    return "n/a" if x is None else f"{x:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
