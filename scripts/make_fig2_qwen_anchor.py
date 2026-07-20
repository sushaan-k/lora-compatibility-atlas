#!/usr/bin/env python3
"""Anchor figure for the balanced Qwen m=50 campaign (fig2_qwen_anchor.pdf).

Adapted from the supplement's qwen_m50_empirical_postprocess.py so that it
writes ONLY the figure (the .tex tables in figs_png/ are left untouched).
Panels:
  (a) atlas AUROC per operator, dot-and-whisker (central 95% slice-bootstrap range);
  (b) atlas AUPRC per operator against the feasible base rate;
  (c) feasible recall against validation budget per operator;
  (d) atlas against cheap baselines (AUROC), markers per estimator.

Error bars in (a) and (d): 95% percentile interval from a cluster bootstrap
over the four predeclared campaign slices (slices resampled with replacement,
1000 draws, seed 20260503) - the same procedure as Table qwen_clustered_ci.

Rendered at final physical size (6.2in wide; the .tex includes it at
0.95\\linewidth on a 6.5in TMLR text block) with the shared 8pt boxed style.
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from fig_style import (apply_style, OPERATOR_COLORS, OPERATOR_LABELS,
                       OPERATOR_LINESTYLES, OPERATOR_MARKERS)

import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score

METHODS = ["linear", "ties", "dare_ties", "magnitude_prune"]
BUDGETS = [0.10, 0.20, 0.30, 0.50, 0.80]
RUN_DIR = REPO / "results_public" / "qwen25_m50_balanced"
FACE_PAIRS_DIR = REPO / "results_public" / "qwen25_m50_facepairs"
N_BOOT = 1000
BOOT_SEED = 20260503


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: str) -> bool:
    return value in {"1", "true", "True", "yes"}


def parse_float(value: str | None) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def adapters(row: dict[str, str]) -> list[str]:
    return [part for part in row["adapters"].split("+") if part]


def normalized_slice(row: dict[str, str]) -> str:
    value = row.get("slice_id") or row.get("source_run", "")
    return value.replace("_facepairs", "")


def score(row: dict[str, str]) -> float:
    return -parse_float(row["predicted_score"])


def auroc(labels, scores) -> float | None:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    keep = np.isfinite(s)
    y, s = y[keep], s[keep]
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def auprc(labels, scores) -> float | None:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    keep = np.isfinite(s)
    y, s = y[keep], s[keep]
    if y.sum() == 0 or len(y) == 0:
        return None
    return float(average_precision_score(y, s))


def metric_block(rows, score_name: str = "atlas"):
    rows = list(rows)
    labels = [parse_bool(row["feasible"]) for row in rows]
    if score_name == "atlas":
        scores = [score(row) for row in rows]
    else:
        scores = [parse_float(row.get(score_name)) for row in rows]
    return {
        "n": len(rows),
        "n_pos": sum(labels),
        "auroc": auroc(labels, scores),
        "auprc": auprc(labels, scores),
        "feasible_rate": sum(labels) / max(len(labels), 1),
    }


def load_summary(run_dir: Path):
    path = run_dir / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def attach_single_gain_features(rows, summary) -> None:
    components = {item.get("source_run"): item for item in summary.get("component_summaries", [])}
    for row in rows:
        comp = components.get(row.get("source_run", ""))
        gains = []
        if comp:
            base = comp.get("base_losses", {}).get("calibration", {})
            single = comp.get("single_losses", {}).get("calibration", {})
            for name in adapters(row):
                if name in base and name in single:
                    gains.append(float(base[name]) - float(single[name]))
        row["single_gain_min_score"] = str(min(gains)) if gains else ""
        row["single_gain_mean_score"] = str(sum(gains) / len(gains)) if gains else ""


def add_pair_clique_scores(rows, extra_pair_rows=None) -> None:
    pair_feasible = {}
    for row in [*rows, *(extra_pair_rows or [])]:
        if row["query_kind"] != "pair":
            continue
        key = (normalized_slice(row), row["method"], tuple(sorted(adapters(row))))
        pair_feasible[key] = parse_bool(row["feasible"])
    for row in rows:
        if row["query_kind"] != "triple":
            row["pair_clique_binary_score"] = ""
            continue
        names = sorted(adapters(row))
        face_keys = [
            (normalized_slice(row), row["method"], tuple(face))
            for face in [(names[0], names[1]), (names[0], names[2]), (names[1], names[2])]
        ]
        present = [key in pair_feasible for key in face_keys]
        pair_ok = [pair_feasible[key] for key in face_keys if key in pair_feasible]
        row["pair_clique_binary_score"] = "1" if all(present) and all(pair_ok) else "0"


def slice_bootstrap(rows, n: int, seed: int):
    rng = random.Random(seed)
    slices = sorted({row.get("slice_id", row.get("source_run", "")) for row in rows})
    by_slice = {sid: [row for row in rows
                      if row.get("slice_id", row.get("source_run", "")) == sid]
                for sid in slices}
    out = {}
    for method in METHODS:
        values_auc = []
        for _ in range(n):
            picked = [rng.choice(slices) for _ in slices]
            sample = [row for sid in picked for row in by_slice[sid] if row["method"] == method]
            block = metric_block(sample)
            if block["auroc"] is not None:
                values_auc.append(float(block["auroc"]))
        lo, hi = np.percentile(values_auc, [2.5, 97.5])
        out[method] = {"auroc_ci": [float(lo), float(hi)]}
    return out


def operating_points(rows):
    out = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        labels = [parse_bool(row["feasible"]) for row in method_rows]
        n_pos = sum(labels)
        ordered = sorted(method_rows, key=score, reverse=True)
        table = []
        for budget in BUDGETS:
            take = max(1, math.ceil(budget * len(ordered)))
            selected = ordered[:take]
            tp = sum(1 for row in selected if parse_bool(row["feasible"]))
            table.append({"budget": budget, "recall": tp / max(n_pos, 1)})
        out[method] = table
    return out


def baseline_metrics(rows):
    out = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        triples = [row for row in method_rows if row["query_kind"] == "triple"]
        out[method] = {
            "atlas_all": metric_block(method_rows),
            "single_gain_min_all": metric_block(method_rows, "single_gain_min_score"),
            "pair_clique_triples": metric_block(
                [row for row in triples if row.get("pair_clique_binary_score") != ""],
                "pair_clique_binary_score"),
        }
    return out


def cheap_features(row):
    arity = 2.0 if row["query_kind"] == "pair" else 3.0
    single_min = parse_float(row.get("single_gain_min_score"))
    single_mean = parse_float(row.get("single_gain_mean_score"))
    if not math.isfinite(single_min):
        single_min = 0.0
    if not math.isfinite(single_mean):
        single_mean = 0.0
    return [arity, single_min, single_mean, single_mean - single_min]


def learned_baseline(rows):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        slices = sorted({row.get("slice_id", row.get("source_run", "")) for row in method_rows})
        y_all, s_all = [], []
        for held_slice in slices:
            train = [row for row in method_rows
                     if row.get("slice_id", row.get("source_run", "")) != held_slice]
            test = [row for row in method_rows
                    if row.get("slice_id", row.get("source_run", "")) == held_slice]
            if not train or not test:
                continue
            y_train = np.asarray([1 if parse_bool(row["feasible"]) else 0 for row in train])
            if len(np.unique(y_train)) < 2:
                continue
            x_train = np.nan_to_num(np.asarray([cheap_features(r) for r in train], dtype=float),
                                    nan=0.0, posinf=100.0, neginf=-100.0)
            x_test = np.nan_to_num(np.asarray([cheap_features(r) for r in test], dtype=float),
                                   nan=0.0, posinf=100.0, neginf=-100.0)
            x_train = np.clip(x_train, -100.0, 100.0)
            x_test = np.clip(x_test, -100.0, 100.0)
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000,
                                   random_state=20260503, solver="liblinear"),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                model.fit(x_train, y_train)
                scores = model.predict_proba(x_test)[:, 1]
            y_all.extend(parse_bool(row["feasible"]) for row in test)
            s_all.extend(float(v) for v in scores)
        out[method] = {"auroc": auroc(y_all, s_all)}
    return out


def main() -> None:
    rows = read_rows(RUN_DIR / "results.csv")
    face_pair_rows = read_rows(FACE_PAIRS_DIR / "results.csv") if FACE_PAIRS_DIR.exists() else []
    summary = load_summary(RUN_DIR)
    attach_single_gain_features(rows, summary)
    add_pair_clique_scores(rows, face_pair_rows)

    baselines = baseline_metrics(rows)
    learned = learned_baseline(rows)
    operating = operating_points(rows)
    slice_ci = slice_bootstrap(rows, N_BOOT, BOOT_SEED)

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 4.4), layout="constrained")
    x = np.arange(len(METHODS))
    labels = [OPERATOR_LABELS[m] for m in METHODS]
    colors = [OPERATOR_COLORS[m] for m in METHODS]

    # ---- (a) atlas AUROC: dot-and-whisker, honest on the 0.45-1.0 range ----
    ax = axes[0, 0]
    for i, m in enumerate(METHODS):
        v = baselines[m]["atlas_all"]["auroc"]
        lo, hi = slice_ci[m]["auroc_ci"]
        ax.errorbar([i], [v], yerr=[[max(0.0, v - lo)], [max(0.0, hi - v)]],
                    fmt=OPERATOR_MARKERS[m], color=OPERATOR_COLORS[m],
                    ecolor="0.3", elinewidth=0.8, capsize=2.5, markersize=5,
                    zorder=3)
    ax.axhline(0.5, color="0.4", linestyle=":", linewidth=0.7)
    ax.text(3.42, 0.503, "chance", fontsize=6, color="0.35", ha="right", va="bottom")
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(0.45, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUROC")
    ax.set_title("(a) Atlas AUROC (central 95% slice-bootstrap range)")

    # ---- (b) AUPRC lift over the feasible base rate ----
    ax = axes[0, 1]
    ap_values = [baselines[m]["atlas_all"]["auprc"] for m in METHODS]
    base_rates = [baselines[m]["atlas_all"]["feasible_rate"] for m in METHODS]
    width = 0.36
    ax.bar(x - width / 2, ap_values, width, color=colors, edgecolor="white",
           linewidth=0.6, zorder=3)
    ax.bar(x + width / 2, base_rates, width, color="0.75", edgecolor="white",
           linewidth=0.6, zorder=3, label="feasible base rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUPRC")
    ax.set_title("(b) AUPRC (atlas vs base rate)")
    ax.legend(frameon=True, loc="upper right")

    # ---- (c) operating frontier: recall against validation budget ----
    ax = axes[1, 0]
    budgets = [cell["budget"] for cell in operating["linear"]]
    for m in METHODS:
        recalls = [cell["recall"] for cell in operating[m]]
        ax.plot(budgets, recalls, marker=OPERATOR_MARKERS[m],
                linestyle=OPERATOR_LINESTYLES[m], linewidth=1.1, markersize=3.5,
                color=OPERATOR_COLORS[m], label=OPERATOR_LABELS[m], zorder=3)
    ax.plot([0.08, 0.82], [0.08, 0.82], color="0.55", linestyle=(0, (1, 1.5)),
            linewidth=0.9, label="random", zorder=2)
    ax.set_xlim(0.08, 0.82)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Validation budget (fraction of subsets)")
    ax.set_ylabel("Feasible recall")
    ax.set_title("(c) Recall vs validation budget")
    # region below the random diagonal at the lower right is empty
    ax.legend(frameon=True, loc="lower right", handlelength=1.8,
              handletextpad=0.4, labelspacing=0.3, borderaxespad=0.4)

    # ---- (d) cheap baselines: color = operator, marker = estimator ----
    ax = axes[1, 1]
    estimators = [
        ("atlas", "o", True),
        ("single-gain floor", "s", False),
        ("learned baseline", "^", False),
        ("clique count (triples)", "D", False),
    ]
    offsets = np.linspace(-0.27, 0.27, 4)
    for i, m in enumerate(METHODS):
        vals = [
            baselines[m]["atlas_all"]["auroc"],
            baselines[m]["single_gain_min_all"]["auroc"],
            learned[m]["auroc"],
            baselines[m]["pair_clique_triples"]["auroc"],
        ]
        col = OPERATOR_COLORS[m]
        for (name, marker, filled), dx, v in zip(estimators, offsets, vals):
            if v is None:
                continue
            kw = dict(marker=marker, markersize=4.5, linestyle="none", zorder=3)
            if filled:
                lo, hi = slice_ci[m]["auroc_ci"]
                ax.errorbar([i + dx], [v],
                            yerr=[[max(0.0, v - lo)], [max(0.0, hi - v)]],
                            color=col, ecolor="0.3", elinewidth=0.8,
                            capsize=2, **kw)
            else:
                ax.plot([i + dx], [v], color=col, markerfacecolor="white",
                        markeredgecolor=col, markeredgewidth=0.9, **kw)
    ax.axhline(0.5, color="0.4", linestyle=":", linewidth=0.7)
    ax.text(3.47, 0.503, "chance", fontsize=6, color="0.35", ha="right", va="bottom")
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(0.45, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUROC")
    ax.set_title("(d) Low-cost baselines (marker = estimator)")
    handles = [
        plt.Line2D([], [], marker=mk, linestyle="none", markersize=4.5,
                   color="0.2" if filled else "none",
                   markerfacecolor="0.2" if filled else "white",
                   markeredgecolor="0.2", markeredgewidth=0.9, label=name)
        for name, mk, filled in estimators
    ]
    ax.legend(handles=handles, frameon=True, loc="upper right", ncol=2,
              handletextpad=0.2, columnspacing=0.7, labelspacing=0.3,
              borderaxespad=0.3)

    out = REPO / "figs_png" / "fig2_qwen_anchor.pdf"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)
    print(json.dumps({
        "result_count": len(rows),
        "atlas_auroc": {m: baselines[m]["atlas_all"]["auroc"] for m in METHODS},
        "atlas_auroc_slice_ci": {m: slice_ci[m]["auroc_ci"] for m in METHODS},
        "atlas_auprc": {m: baselines[m]["atlas_all"]["auprc"] for m in METHODS},
        "feasible_rate": {m: baselines[m]["atlas_all"]["feasible_rate"] for m in METHODS},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
