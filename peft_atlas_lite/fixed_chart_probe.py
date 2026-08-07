#!/usr/bin/env python3
"""Probe fixed-chart pair/triple compatibility on a selected LoRA triple."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from run_peft_atlas_lite import (
    AdapterSpec,
    adapters_from_config,
    evaluate_base_and_singles,
    evaluate_merge_losses,
    fit_quadratic_scores,
    import_versions,
    load_json,
    load_peft_model,
    predict_quadratic,
    score_query,
    simplex_grid,
    split_prompts,
    stable_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Manifest or config JSON.")
    parser.add_argument("--summary", default=None, help="Optional completed summary JSON with reusable loss tables.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--methods", default="linear", help="Comma-separated PEFT merge methods.")
    parser.add_argument(
        "--triple",
        action="append",
        required=True,
        help="Triple as name1+name2+name3. Can be passed multiple times.",
    )
    parser.add_argument("--step", type=float, default=0.2, help="Shared simplex grid step.")
    parser.add_argument("--min-weight", type=float, default=0.0, help="Shared simplex minimum weight for every adapter.")
    parser.add_argument("--max-length", type=int, default=None, help="Override max sequence length.")
    return parser.parse_args()


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_triples(raw_triples: Iterable[str]) -> List[Tuple[str, str, str]]:
    triples: List[Tuple[str, str, str]] = []
    for raw in raw_triples:
        for chunk in raw.split(";"):
            names = [part.strip() for part in chunk.replace(",", "+").split("+") if part.strip()]
            if not names:
                continue
            if len(names) != 3:
                raise ValueError(f"expected exactly three adapter names in {chunk!r}")
            triples.append((names[0], names[1], names[2]))
    return triples


def selected_adapters(config: Dict[str, Any], names: Sequence[str]) -> List[AdapterSpec]:
    by_name = {adapter.name: adapter for adapter in adapters_from_config(config)}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise KeyError(f"adapter(s) missing from manifest: {missing}")
    return [by_name[name] for name in names]


def prompt_sets_for(
    config: Dict[str, Any],
    adapters: Sequence[AdapterSpec],
    summary: Dict[str, Any] | None,
) -> Dict[str, Dict[str, List[str]]]:
    rng = random.Random(stable_seed(config))
    prompt_sets: Dict[str, Dict[str, List[str]]] = {}
    for adapter in adapters:
        calibration, heldout = split_prompts(config, adapter, rng)
        prompt_sets[adapter.name] = {"calibration": calibration, "heldout": heldout}
    return prompt_sets


def losses_for(
    model: Any,
    tokenizer: Any,
    adapters: Sequence[AdapterSpec],
    prompt_sets: Dict[str, Dict[str, List[str]]],
    config: Dict[str, Any],
    summary: Dict[str, Any] | None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    if summary and "base_losses" in summary and "single_losses" in summary:
        base_losses = {
            split: {adapter.name: float(summary["base_losses"][split][adapter.name]) for adapter in adapters}
            for split in ["calibration", "heldout"]
        }
        single_losses = {
            split: {adapter.name: float(summary["single_losses"][split][adapter.name]) for adapter in adapters}
            for split in ["calibration", "heldout"]
        }
        return base_losses, single_losses
    return evaluate_base_and_singles(
        model,
        tokenizer,
        adapters,
        prompt_sets,
        int(config.get("max_length", 384)),
    )


def subset_result(
    names: Sequence[str],
    grid: Sequence[Sequence[float]],
    task_coeffs: Dict[str, Sequence[float]],
    heldout_task_values: Dict[str, List[float]],
) -> Dict[str, Any]:
    predicted_scores = [
        max(predict_quadratic(task_coeffs[name], weights) for name in names)
        for weights in grid
    ]
    heldout_scores = [
        max(heldout_task_values[name][idx] for name in names)
        for idx, _ in enumerate(grid)
    ]
    pred_idx = min(range(len(grid)), key=lambda idx: predicted_scores[idx])
    held_idx = min(range(len(grid)), key=lambda idx: heldout_scores[idx])
    return {
        "tasks": list(names),
        "predicted_score": float(predicted_scores[pred_idx]),
        "predicted_weights": list(grid[pred_idx]),
        "heldout_score": float(heldout_scores[held_idx]),
        "heldout_weights": list(grid[held_idx]),
        "feasible": bool(heldout_scores[held_idx] <= 0.0),
        "active_obstruction": [
            name for name in names if abs(heldout_task_values[name][held_idx] - heldout_scores[held_idx]) <= 1e-6
        ],
    }


def evaluate_triple(
    model: Any,
    tokenizer: Any,
    triple: Sequence[AdapterSpec],
    method: str,
    grid: Sequence[Sequence[float]],
    config: Dict[str, Any],
    prompt_sets: Dict[str, Dict[str, List[str]]],
    base_losses: Dict[str, Dict[str, float]],
    single_losses: Dict[str, Dict[str, float]],
    serial_start: int,
    min_weight: float,
) -> Tuple[Dict[str, Any], int]:
    threshold = float(config.get("retention_threshold", 0.7))
    min_gain = float(config.get("min_single_adapter_gain", 0.005))
    density = float(config.get("density", 0.5))
    max_length = int(config.get("max_length", 384))
    names = [adapter.name for adapter in triple]
    calibration_task_values: Dict[str, List[float]] = {name: [] for name in names}
    heldout_task_values: Dict[str, List[float]] = {name: [] for name in names}
    serial = serial_start

    for weights in grid:
        serial += 1
        calibration_losses = evaluate_merge_losses(
            model, tokenizer, triple, weights, method, density, prompt_sets, max_length, "calibration", serial
        )
        _, calibration_margins, _ = score_query(
            names, weights, calibration_losses, base_losses["calibration"], single_losses["calibration"], threshold, min_gain
        )
        for name in names:
            calibration_task_values[name].append(-calibration_margins[name])

        serial += 1
        heldout_losses = evaluate_merge_losses(
            model, tokenizer, triple, weights, method, density, prompt_sets, max_length, "heldout", serial
        )
        _, heldout_margins, _ = score_query(
            names, weights, heldout_losses, base_losses["heldout"], single_losses["heldout"], threshold, min_gain
        )
        for name in names:
            heldout_task_values[name].append(-heldout_margins[name])

    task_coeffs = {
        name: fit_quadratic_scores(grid, calibration_task_values[name])
        for name in names
    }
    pairs = [(names[0], names[1]), (names[0], names[2]), (names[1], names[2])]
    pair_rows = [subset_result(pair, grid, task_coeffs, heldout_task_values) for pair in pairs]
    triple_row = subset_result(names, grid, task_coeffs, heldout_task_values)
    return {
        "method": method,
        "adapters": names,
        "chart": f"shared 3-adapter simplex Delta^2; all pair and triple task sets use the same grid with minimum weight {min_weight:.3g}",
        "grid_step": float(config.get("_fixed_chart_step", 0.0)),
        "min_weight": min_weight,
        "grid_size": len(grid),
        "pairs": pair_rows,
        "triple": triple_row,
        "fixed_chart_helly_event": bool(all(row["feasible"] for row in pair_rows) and not triple_row["feasible"]),
    }, serial


def write_tex(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    best = [row for row in rows if row["fixed_chart_helly_event"]]
    chosen = best[0] if best else min(rows, key=lambda row: max(pair["heldout_score"] for pair in row["pairs"]) - row["triple"]["heldout_score"])
    pair_scores = ", ".join(f"{pair['heldout_score']:+.3f}" for pair in chosen["pairs"])
    pair_weights = "; ".join(
        "$(" + ", ".join(f"{w:.1f}" for w in pair["heldout_weights"]) + ")$" for pair in chosen["pairs"]
    )
    triple_weights = "$(" + ", ".join(f"{w:.1f}" for w in chosen["triple"]["heldout_weights"]) + ")$"
    event_text = "yes" if chosen["fixed_chart_helly_event"] else "no"
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\small",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{l l l l l}",
        "\\toprule",
        "method & adapters & pair held-out deficits & triple held-out deficit & Helly event? \\\\",
        "\\midrule",
        (
            f"{chosen['method']} & "
            f"\\texttt{{{chosen['adapters'][0][:10]}}}, \\texttt{{{chosen['adapters'][1][:10]}}}, \\texttt{{{chosen['adapters'][2][:10]}}} & "
            f"{pair_scores} & {chosen['triple']['heldout_score']:+.3f} & {event_text} \\\\"
        ),
        "\\bottomrule",
        "\\end{tabular}}",
        "\\caption{Fixed-chart real-LoRA probe on Qwen $m{=}50$.  All three pair task sets and the triple task set are evaluated in the same shared 3-adapter simplex $\\Delta^2$ with minimum coefficient "
        + f"{chosen.get('min_weight', 0.0):.1f}"
        + ", rather than in support-specific pair and triple charts.  Pair held-out deficits are listed for the three two-task subsets; negative means feasible at the retained-gain threshold.  The displayed row uses pair-optimal weights "
        + pair_weights
        + " and triple-optimal weights "
        + triple_weights
        + ".}",
        "\\label{tab:fixed_chart_probe}",
        "\\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_json(Path(args.config))
    if args.max_length is not None:
        config["max_length"] = args.max_length
    summary = load_json(Path(args.summary)) if args.summary else None
    triples = parse_triples(args.triple)
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    all_names = sorted({name for triple in triples for name in triple})
    adapters = selected_adapters(config, all_names)
    prompt_sets = prompt_sets_for(config, adapters, summary)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.fixed_chart.json").write_text(json.dumps({
        "config": str(Path(args.config)),
        "summary": str(Path(args.summary)) if args.summary else None,
        "triples": triples,
        "methods": methods,
        "step": args.step,
        "min_weight": args.min_weight,
    }, indent=2) + "\n", encoding="utf-8")

    model, tokenizer = load_peft_model(config, adapters)
    base_losses, single_losses = losses_for(model, tokenizer, adapters, prompt_sets, config, summary)
    config["_fixed_chart_step"] = float(args.step)
    grid = simplex_grid(3, float(args.step), min_weight=float(args.min_weight))
    rows: List[Dict[str, Any]] = []
    serial = 100000
    started = time.time()
    by_name = {adapter.name: adapter for adapter in adapters}

    for method in methods:
        for triple_names in triples:
            triple_adapters = [by_name[name] for name in triple_names]
            row, serial = evaluate_triple(
                model,
                tokenizer,
                triple_adapters,
                method,
                grid,
                config,
                prompt_sets,
                base_losses,
                single_losses,
                serial,
                float(args.min_weight),
            )
            rows.append(row)
            dump_json(out_dir / "partial.fixed_chart_probe.json", {"rows": rows})
            if row["fixed_chart_helly_event"]:
                break

    payload = {
        "status": "complete",
        "base_model": config.get("base_model"),
        "retention_threshold": config.get("retention_threshold"),
        "duration_sec": time.time() - started,
        "versions": import_versions(),
        "rows": rows,
        "event_count": sum(1 for row in rows if row["fixed_chart_helly_event"]),
    }
    dump_json(out_dir / "fixed_chart_probe.json", payload)
    write_tex(out_dir / "fixed_chart_probe.tex", rows)


if __name__ == "__main__":
    main()
