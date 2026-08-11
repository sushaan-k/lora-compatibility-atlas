#!/usr/bin/env python3
"""Operating-point analysis: does measured answer-token loss recover accuracy-retaining merges at useful precisio"""

import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PANEL_PATH = ROOT / "results_public" / "answer_token_panel" / "answer_token_panel.json"
OUT_DIR = ROOT / "results_public" / "answer_operating_point"
OUT_PATH = OUT_DIR / "operating.json"

TAUS = [0.5, 0.7, 0.9]
BUDGETS = [round(0.1 * i, 1) for i in range(1, 10)]


def pearson(x, y):
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = math.sqrt(sum((a - mx) ** 2 for a in x))
    syy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sxy / (sxx * syy)


def auroc(labels, scores):
    """AUROC of scores against boolean labels, Mann-Whitney with midranks.

    Returns None if only one class is present.
    """
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        midrank = (i + j) / 2.0 + 1.0  # 1-indexed midrank
        for k in range(i, j + 1):
            ranks[order[k]] = midrank
        i = j + 1
    rank_sum_pos = sum(r for r, l in zip(ranks, labels) if l)
    n_pos, n_neg = len(pos), len(neg)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def screening_curve(records, labels, loss_key):
    """Rank records by loss ascending; precision/recall of retained records
    among the top-k fraction, for each budget fraction k.

    Selection size = ceil(k * N). Ties broken by stable sort order.
    """
    n = len(records)
    n_retained = sum(labels)
    order = sorted(range(n), key=lambda i: records[i][loss_key])
    curve = []
    for k in BUDGETS:
        n_sel = max(1, math.ceil(k * n))
        sel = order[:n_sel]
        tp = sum(1 for i in sel if labels[i])
        curve.append({
            "budget_fraction": k,
            "n_selected": n_sel,
            "n_retained_selected": tp,
            "precision": tp / n_sel,
            "recall": tp / n_retained if n_retained else None,
        })
    return curve


def main():
    panel = json.loads(PANEL_PATH.read_text())
    rows = panel["rows"]
    base = {r["task"]: r for r in rows if r["merge"] == "base"}
    alone = {r["task"]: r for r in rows if r["merge"] == "alone"}
    merges = [r for r in rows if r["merge"] not in ("base", "alone")]

    # ---- structural checks -------------------------------------------------
    assert len(merges) == 45, f"expected 45 merge rows, got {len(merges)}"
    in_merge = [r for r in merges if r["task"] in r["merge"].split("+")]
    out_of_merge = [r for r in merges if r["task"] not in r["merge"].split("+")]
    unordered = Counter(frozenset(r["merge"].split("+")) for r in merges)
    structure = {
        "n_merge_rows": len(merges),
        "n_in_merge_rows": len(in_merge),
        "n_out_of_merge_rows": len(out_of_merge),
        "task_is_first_member_always": all(
            r["merge"].split("+")[0] == r["task"] for r in merges),
        "n_distinct_unordered_merges": len(unordered),
        "rows_per_unordered_merge": dict(Counter(unordered.values())),
        "n_pair_rows": sum(1 for r in merges if len(r["merge"].split("+")) == 2),
        "n_triple_rows": sum(1 for r in merges if len(r["merge"].split("+")) == 3),
        "note": ("Every (merge,task) row evaluates a member task of the merge, "
                 "so the in-merge restriction keeps all 45 records and the "
                 "all-(merge,task) pooled variant is identical to it."),
    }

    # sanity: replicate the published pooled correlations
    acc = [r["accuracy"] for r in in_merge]
    r_answer = pearson([r["L_answer"] for r in in_merge], acc)
    r_input = pearson([r["L_input"] for r in in_merge], acc)
    sanity = {
        "pooled_r_Lanswer_acc_recomputed": r_answer,
        "pooled_r_Linput_acc_recomputed": r_input,
        "pooled_r_Lanswer_acc_published": panel["pooled"]["r_Lanswer_acc"],
        "pooled_r_Linput_acc_published": panel["pooled"]["r_Linput_acc"],
    }
    assert abs(r_answer - panel["pooled"]["r_Lanswer_acc"]) < 1e-9
    assert abs(r_input - panel["pooled"]["r_Linput_acc"]) < 1e-9

    variants = {"in_merge": in_merge, "all_merge_task": merges}
    results = {}
    for tau in TAUS:
        tau_key = f"tau_acc_{tau}"
        tau_res = {}
        for vname, recs in variants.items():
            labels = []
            for r in recs:
                b_acc = base[r["task"]]["accuracy"]
                a_acc = alone[r["task"]]["accuracy"]
                labels.append(r["accuracy"] >= b_acc + tau * (a_acc - b_acc))
            n_pos = sum(labels)
            n_neg = len(labels) - n_pos

            per_task_auroc = {}
            for t in sorted(base):
                idx = [i for i, r in enumerate(recs) if r["task"] == t]
                t_labels = [labels[i] for i in idx]
                t_ans = auroc(t_labels, [-recs[i]["L_answer"] for i in idx])
                t_inp = auroc(t_labels, [-recs[i]["L_input"] for i in idx])
                per_task_auroc[t] = {
                    "n": len(idx),
                    "n_retained": sum(t_labels),
                    "both_classes": 0 < sum(t_labels) < len(idx),
                    "auroc_neg_answer_loss": t_ans,
                    "auroc_neg_input_loss": t_inp,
                }

            answer_curve = screening_curve(recs, labels, "L_answer")
            input_curve = screening_curve(recs, labels, "L_input")

            # operating point: smallest budget where answer-loss ranking
            # recovers at least half of the retained records
            op = None
            for a_pt, i_pt in zip(answer_curve, input_curve):
                if a_pt["recall"] is not None and a_pt["recall"] >= 0.5:
                    op = {
                        "budget_fraction": a_pt["budget_fraction"],
                        "n_selected": a_pt["n_selected"],
                        "answer_loss_precision": a_pt["precision"],
                        "answer_loss_recall": a_pt["recall"],
                        "input_loss_precision_same_budget": i_pt["precision"],
                        "input_loss_recall_same_budget": i_pt["recall"],
                    }
                    break

            tau_res[vname] = {
                "n_records": len(recs),
                "n_retained": n_pos,
                "n_not_retained": n_neg,
                "both_classes_present": n_pos > 0 and n_neg > 0,
                "auroc_pooled_neg_answer_loss": auroc(labels, [-r["L_answer"] for r in recs]),
                "auroc_pooled_neg_input_loss": auroc(labels, [-r["L_input"] for r in recs]),
                "auroc_per_task": per_task_auroc,
                "screening_curve_answer_loss": answer_curve,
                "screening_curve_input_loss": input_curve,
                "operating_point_half_recall": op,
            }
        results[tau_key] = tau_res

    out = {
        "source_panel": str(PANEL_PATH),
        "label_convention": ("retained iff accuracy >= base_acc + tau_acc * "
                             "(alone_acc - base_acc), per surfaces.py "
                             "cmd_analyze"),
        "score_convention": ("screens rank (merge,task) records by measured "
                             "loss ascending (lower retention loss = keep); "
                             "AUROC uses -loss as the score"),
        "selection_rule": "top ceil(k*N) records at budget fraction k",
        "per_task_baselines": {
            t: {"base_acc": base[t]["accuracy"], "alone_acc": alone[t]["accuracy"]}
            for t in sorted(base)
        },
        "structure": structure,
        "sanity_pooled_pearson": sanity,
        "results": results,
        "caveats": [
            "n=45 (merge,task) records over only 20 distinct merged models; "
            "records sharing a merged model are not independent.",
            "Single base model (Mistral-7B) and 5 adapters; no evidence this "
            "generalizes across bases or adapter pools.",
            "Tasks are classification-style with option-token answers; "
            "answer-token loss is unusually well aligned with the accuracy "
            "metric on such tasks, so this is a favorable setting for the "
            "answer-loss screen.",
            "Equal-weight merges only; operating points may shift under "
            "tuned merge weights.",
            "The panel contains no out-of-merge (merge,task) evaluations, so "
            "the in-merge and all-(merge,task) variants coincide exactly.",
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))

    # ---- printed summary ---------------------------------------------------
    print(f"panel: {PANEL_PATH}")
    print(f"records: {len(merges)} (all in-merge; {structure['n_distinct_unordered_merges']} distinct merged models)")
    print(f"sanity pooled Pearson: r(L_answer,acc)={r_answer:+.3f}  r(L_input,acc)={r_input:+.3f}")
    for tau in TAUS:
        res = results[f"tau_acc_{tau}"]["in_merge"]
        print(f"\n== tau_acc={tau} (in-merge, n={res['n_records']}) ==")
        print(f"  retained={res['n_retained']}  not-retained={res['n_not_retained']}")
        print(f"  AUROC pooled: -L_answer={res['auroc_pooled_neg_answer_loss']:.3f}  "
              f"-L_input={res['auroc_pooled_neg_input_loss']:.3f}")
        for t, pt in res["auroc_per_task"].items():
            if pt["both_classes"]:
                print(f"    {t}: n={pt['n']} retained={pt['n_retained']} "
                      f"AUROC -L_answer={pt['auroc_neg_answer_loss']:.3f} "
                      f"-L_input={pt['auroc_neg_input_loss']:.3f}")
            else:
                print(f"    {t}: n={pt['n']} retained={pt['n_retained']} (one class; AUROC undefined)")
        print("  screening (budget k: answer P/R vs input P/R):")
        for a_pt, i_pt in zip(res["screening_curve_answer_loss"],
                              res["screening_curve_input_loss"]):
            print(f"    k={a_pt['budget_fraction']:.1f} (n={a_pt['n_selected']:2d}): "
                  f"answer P={a_pt['precision']:.3f} R={a_pt['recall']:.3f} | "
                  f"input P={i_pt['precision']:.3f} R={i_pt['recall']:.3f}")
        op = res["operating_point_half_recall"]
        if op:
            print(f"  operating point (first budget with answer-loss recall >= 0.5): "
                  f"k={op['budget_fraction']:.1f} -> answer P={op['answer_loss_precision']:.3f} "
                  f"R={op['answer_loss_recall']:.3f}; input at same budget "
                  f"P={op['input_loss_precision_same_budget']:.3f} "
                  f"R={op['input_loss_recall_same_budget']:.3f}")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
