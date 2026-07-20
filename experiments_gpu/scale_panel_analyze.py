"""Analysis for scale panels where the scored tasks are a subset of the merge
adapters (m=30/m=50 amortized builds).

Differs from expD_answer_panel.py analyze, which assumes tasks == adapters:
here the chart dimension and merge-weight vectors come from the adapter order
of the fit grid, tasks whose answer loss is NaN (prompt exceeds the context
limit, truncating the answer tokens) are excluded with documented cause, and
subsets containing an excluded member are dropped from subset-level screening.
Label and score definitions match expD: a member is retained at tau iff
L_answer <= base_La - tau * gain; a subset is retained iff every member is;
the subset score is the worst member's predicted deficit.

Usage: python3 scale_panel_analyze.py --raw X.npz --rows rows.json --out DIR
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "peft_atlas_lite"))
sys.path.insert(0, "/root/atlas/peft_atlas_lite")
from run_peft_atlas_lite import fit_quadratic_model, predict_quadratic  # noqa: E402

TAU_GRID = [0.55, 0.65, 0.7, 0.75, 0.85]


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) >= 3 else float("nan")


def auroc(labels, scores):
    pos = [s for l, s in zip(labels, scores) if l]
    neg = [s for l, s in zip(labels, scores) if not l]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(wins / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.raw, allow_pickle=True)
    grid = [list(w) for w in z["grid"]]
    adapters = [str(x) for x in z["names"]]          # full adapter order
    vals = z["values"]
    excluded = [adapters[i] for i in range(len(adapters)) if np.isnan(vals[i]).any()]
    tasks = [t for t in adapters if t not in excluded]
    coeffs = {t: fit_quadratic_model(grid, list(vals[adapters.index(t)]))[0] for t in tasks}

    rows = json.loads(Path(args.rows).read_text())["rows"]
    base = {r["task"]: r for r in rows if r["merge"] == "base"}
    alone = {r["task"]: r for r in rows if r["merge"] == "alone"}
    gains = {t: base[t]["L_answer"] - alone[t]["L_answer"] for t in tasks}

    def weights_for(merge):
        mem = merge.split("+")
        w = [0.0] * len(adapters)
        for t in mem:
            w[adapters.index(t)] = 1.0 / len(mem)
        return w

    per_member, by_merge = [], defaultdict(list)
    for r in rows:
        if "+" not in r["merge"] or r["task"] not in r["merge"].split("+"):
            continue
        if r["task"] in excluded:
            continue
        rec = {"task": r["task"], "merge": r["merge"],
               "order": r["merge"].count("+") + 1,
               "pred_L_answer": predict_quadratic(coeffs[r["task"]], weights_for(r["merge"])),
               "L_answer": r["L_answer"], "accuracy": r["accuracy"],
               "base_La": base[r["task"]]["L_answer"], "gain": gains[r["task"]],
               "base_acc": base[r["task"]]["accuracy"], "alone_acc": alone[r["task"]]["accuracy"]}
        per_member.append(rec)
        by_merge[r["merge"]].append(rec)

    per_subset = [{"merge": m, "order": v[0]["order"], "members": v}
                  for m, v in by_merge.items()
                  if len(v) == len([t for t in m.split("+")])
                  and not any(t in excluded for t in m.split("+"))]
    n_dropped = len(by_merge) - len(per_subset)

    def fid(evs):
        p = [e["pred_L_answer"] for e in evs]
        m = [e["L_answer"] for e in evs]
        by = defaultdict(list)
        for e in evs:
            by[e["task"]].append((e["pred_L_answer"], e["L_answer"]))
        wr = [pearson([x for x, _ in v], [y for _, y in v])
              for v in by.values() if len(v) >= 8]
        return {"pooled": pearson(p, m),
                "within_task_median": float(np.median(wr)) if wr else None,
                "n_events": len(evs), "n_tasks_with_events": len(wr)}

    result = {"n_adapters": len(adapters), "n_tasks": len(tasks),
              "excluded_tasks": excluded,
              "excluded_cause": "answer loss NaN at every point: prompt exceeds the context limit, truncating the answer tokens",
              "answer_token_gains": gains,
              "n_member_events": len(per_member), "n_subsets": len(per_subset),
              "n_subsets_dropped_excluded_member": n_dropped,
              "fidelity_all": fid(per_member),
              "fidelity_pairs": fid([e for e in per_member if e["order"] == 2]),
              "fidelity_triples": fid([e for e in per_member if e["order"] == 3]),
              "by_answer_tau": {}, "by_acc_tau": {}}

    for tau in TAU_GRID:
        labels = [all(m["L_answer"] <= m["base_La"] - tau * m["gain"] for m in s["members"])
                  for s in per_subset]
        scores = [-max(m["pred_L_answer"] - (m["base_La"] - tau * m["gain"]) for m in s["members"])
                  for s in per_subset]
        trips = [i for i, s in enumerate(per_subset) if s["order"] == 3]
        result["by_answer_tau"][str(tau)] = {
            "n_retained": int(sum(labels)), "n": len(labels),
            "auroc_pooled": auroc(labels, scores),
            "auroc_triples": auroc([labels[i] for i in trips], [scores[i] for i in trips])}
    for tau in [0.5, 0.7, 0.9]:
        labels = [e["accuracy"] >= e["base_acc"] + tau * (e["alone_acc"] - e["base_acc"])
                  for e in per_member]
        scores = [-e["pred_L_answer"] for e in per_member]
        result["by_acc_tau"][str(tau)] = {
            "n_retained": int(sum(labels)), "n": len(labels),
            "auroc_events_predicted": auroc(labels, scores)}

    od = Path(args.out)
    od.mkdir(exist_ok=True)
    (od / "scale_analysis.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in
                      ("n_adapters", "n_tasks", "excluded_tasks", "fidelity_triples")}, indent=1))
    for tau, v in result["by_answer_tau"].items():
        print("tau", tau, v)


if __name__ == "__main__":
    main()
