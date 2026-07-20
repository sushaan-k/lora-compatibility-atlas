#!/usr/bin/env python3
"""Reproduce Table 14 (validation-budget yield on public triple queries) from the
released per-event CSVs, then compute a hybrid pair-graph+atlas screen.

Protocol per the table caption: sort all public triples by the screen, take the
top budget%, report pooled recall/precision. Pair-graph ties are broken in
expectation (uniform within tie classes). Hybrid = pair-clique count desc,
then atlas score asc (deterministic tie-break).
"""
import csv, math, os
from collections import defaultdict

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_public")
PANELS = ["qwen25_m50_balanced", "tinyllama_12x", "mistral_10x", "phi2_10x",
          "pythia410m_10x", "pythia1b_10x"]

def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))

rows = []
for p in PANELS:
    for r in load(f"{BASE}/{p}/results.csv"):
        r["panel"] = p
        rows.append(r)
# face-pair completion run for the Qwen slices
facepairs = load(f"{BASE}/qwen25_m50_facepairs/results.csv")
for r in facepairs:
    r["panel"] = "qwen25_m50_balanced"

def keyify(r):
    ads = tuple(sorted(r["adapters"].split("+")))
    return (r["panel"], r["method"], ads)

# pair feasibility lookup (panel, method, pair) -> feasible; facepairs fills gaps
pairfeas = {}
pairpred = {}
for r in rows + facepairs:
    if r["query_kind"] == "pair" and not r.get("error"):
        k = keyify(r)
        pairfeas[k] = max(pairfeas.get(k, 0), int(r["feasible"]))
        # atlas-PREDICTED pair feasibility (predicted_score <= 0): uses no held-out
        # label, so a hybrid built on it stays inside the validation budget.
        pairpred[k] = max(pairpred.get(k, 0), int(float(r["predicted_score"]) <= 0.0))

triples = [r for r in rows if r["query_kind"] == "triple" and not r.get("error")]
print(f"triples: {len(triples)}, feasible: {sum(int(r['feasible']) for r in triples)}")

def subpair_count(r, table=pairfeas):
    a, b, c = sorted(r["adapters"].split("+"))
    cnt, found = 0, 0
    for pr in ((a, b), (a, c), (b, c)):
        k = (r["panel"], r["method"], pr)
        if k in table:
            found += 1
            cnt += table[k]
    return cnt, found

missing = sum(1 for r in triples if subpair_count(r)[1] < 3)
print(f"triples with <3 evaluated sub-pairs: {missing}")

N = len(triples)
POS = sum(int(r["feasible"]) for r in triples)
BUDGETS = [0.10, 0.20, 0.30, 0.50]

def eval_sorted(order):
    """order: list of (feasible int) already sorted best-first."""
    out = []
    for b in BUDGETS:
        k = round(N * b)
        tp = sum(order[:k])
        out.append((tp / POS, tp / k))
    return out

def eval_tied(classes):
    """classes: list of (n_class, pos_class) best-first; expectation within ties."""
    out = []
    for b in BUDGETS:
        k = round(N * b)
        taken, tp = 0, 0.0
        for n_c, p_c in classes:
            if taken >= k:
                break
            take = min(n_c, k - taken)
            tp += p_c * take / n_c
            taken += take
        out.append((tp / POS, tp / k))
    return out

def show(name, res):
    print(f"{name:28s} " + "  ".join(f"{r:.2f}/{p:.2f}" for r, p in res))

# --- Atlas: predicted_score ascending (lower rho = more compatible)
atl = sorted(triples, key=lambda r: float(r["predicted_score"]))
show("Atlas (pooled sort)", eval_sorted([int(r["feasible"]) for r in atl]))

# --- Pair graph: clique count desc, expectation within ties
byc = defaultdict(lambda: [0, 0])
for r in triples:
    c, _ = subpair_count(r)
    byc[c][0] += 1
    byc[c][1] += int(r["feasible"])
classes = [(byc[c][0], byc[c][1]) for c in sorted(byc, reverse=True)]
print("clique classes (count: n, pos):", {c: tuple(byc[c]) for c in sorted(byc, reverse=True)})
show("Pair graph (exp ties)", eval_tied(classes))

# --- Random
show("Random", [(b, POS / N) for b in BUDGETS])

# --- Hybrid: clique count desc, atlas score asc within class
hyb = sorted(triples, key=lambda r: (-subpair_count(r, pairfeas)[0], float(r["predicted_score"])))
show("Hybrid (held-out labels)", eval_sorted([int(r["feasible"]) for r in hyb]))

# Budget-fair hybrid: pair feasibility from ATLAS PREDICTIONS (no held-out label).
hyb_pred = sorted(triples, key=lambda r: (-subpair_count(r, pairpred)[0], float(r["predicted_score"])))
show("Hybrid (predicted pairs, fair)", eval_sorted([int(r["feasible"]) for r in hyb_pred]))

# --- Also try per-method budget allocation to see which matches the table
def eval_grouped(keyfn, scorefn):
    sel = []
    groups = defaultdict(list)
    for r in triples:
        groups[keyfn(r)].append(r)
    res = []
    for b in BUDGETS:
        tp = tot = 0
        for g in groups.values():
            k = round(len(g) * b)
            gs = sorted(g, key=scorefn)[:k]
            tp += sum(int(r["feasible"]) for r in gs)
            tot += k
        res.append((tp / POS, tp / tot))
    return res

show("Atlas (per-method budget)", eval_grouped(lambda r: r["method"], lambda r: float(r["predicted_score"])))
show("Atlas (per-panel+method)", eval_grouped(lambda r: (r["panel"], r["method"]), lambda r: float(r["predicted_score"])))
print("\nTable 14 says:  Atlas 0.29/0.71 0.54/0.67 0.66/0.54 0.82/0.40 | Pair 0.34/0.85 0.59/0.73 0.75/0.62 0.92/0.45")
