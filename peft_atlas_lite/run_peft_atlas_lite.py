#!/usr/bin/env python3
"""Run a public PEFT adapter compatibility-atlas smoke experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    repo: str
    task: str
    probe_key: str


@dataclass
class QueryResult:
    method: str
    query_kind: str
    adapters: List[str]
    predicted_score: float
    heldout_score: float
    feasible: bool
    predicted_weights: List[float]
    heldout_weights: List[float]
    coefficient_regret: float
    margins: Dict[str, float]
    obstructing_tasks: List[str]
    certificate_type: str = "evaluated_coefficient_primal"
    grid_points: int = 0
    primal_upper: Optional[float] = None
    dual_lower: Optional[float] = None
    certificate_gap: Optional[float] = None
    dual_weights: Optional[List[float]] = None
    certificate_message: str = ""
    psd_projected: bool = False
    quadratic_min_eigenvalue: Optional[float] = None
    quadratic_psd_adjustment: Optional[float] = None
    unprojected_jet_max: Optional[float] = None
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a JSON manifest.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Validate only; do not load models.")
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional cap on pair queries.")
    parser.add_argument("--max-triples", type=int, default=None, help="Optional cap on triple queries.")
    parser.add_argument("--methods", default=None, help="Comma-separated merge methods overriding config.")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def adapters_from_config(config: Dict[str, Any]) -> List[AdapterSpec]:
    adapters = []
    seen = set()
    for raw in config.get("adapters", []):
        spec = AdapterSpec(
            name=str(raw["name"]),
            repo=str(raw["repo"]),
            task=str(raw["task"]),
            probe_key=str(raw["probe_key"]),
        )
        if spec.name in seen:
            raise ValueError(f"duplicate adapter name: {spec.name}")
        probe_catalog = config.get("probes", {}) or config.get("probe_counts", {})
        if spec.probe_key not in probe_catalog:
            raise ValueError(f"probe_key {spec.probe_key!r} missing from probes")
        seen.add(spec.name)
        adapters.append(spec)
    if len(adapters) < 2:
        raise ValueError("at least two adapters are required")
    return adapters


def planned_queries(
    adapters: Sequence[AdapterSpec],
    max_pairs: Optional[int],
    max_triples: Optional[int],
    rng: Optional[random.Random] = None,
    shuffle: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Tuple[AdapterSpec, ...]], List[Tuple[AdapterSpec, ...]]]:
    config = config or {}
    design = str(config.get("query_design", config.get("subset_design", "canonical"))).lower()
    pair_subset = config.get("pair_subset")
    triple_subset = config.get("triple_subset")

    if pair_subset is not None:
        pairs = indexed_queries(adapters, pair_subset, 2)
    elif design in {"circulant", "circulant_pairs", "circulant_balanced"}:
        offsets = [int(value) for value in config.get("pair_offsets", [1, 2, 3])]
        pairs = indexed_queries(adapters, circulant_pair_indices(len(adapters), offsets), 2)
    else:
        pairs = list(itertools.combinations(adapters, 2))

    if triple_subset is not None:
        triples = indexed_queries(adapters, triple_subset, 3)
    elif design in {"balanced", "balanced_random", "circulant_balanced"} and max_triples:
        if rng is None:
            raise ValueError("rng is required for balanced triple sampling")
        triples = balanced_queries(adapters, 3, max_triples, rng)
        max_triples = None
    else:
        triples = list(itertools.combinations(adapters, 3))

    if design in {"balanced", "balanced_random"} and max_pairs and pair_subset is None:
        if rng is None:
            raise ValueError("rng is required for balanced pair sampling")
        pairs = balanced_queries(adapters, 2, max_pairs, rng)
        max_pairs = None

    if shuffle:
        if rng is None:
            raise ValueError("rng is required when shuffle=True")
        rng.shuffle(pairs)
        rng.shuffle(triples)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    if max_triples is not None:
        triples = triples[:max_triples]
    return pairs, triples


def indexed_queries(
    adapters: Sequence[AdapterSpec],
    index_rows: Sequence[Sequence[int]],
    arity: int,
) -> List[Tuple[AdapterSpec, ...]]:
    rows: List[Tuple[AdapterSpec, ...]] = []
    for raw in index_rows:
        indices = [int(value) for value in raw]
        if len(indices) != arity:
            raise ValueError(f"expected {arity} indices, got {indices}")
        if len(set(indices)) != arity:
            raise ValueError(f"duplicate adapter index in query: {indices}")
        if min(indices) < 0 or max(indices) >= len(adapters):
            raise ValueError(f"adapter index out of range in query: {indices}")
        rows.append(tuple(adapters[idx] for idx in indices))
    return rows


def circulant_pair_indices(size: int, offsets: Sequence[int]) -> List[List[int]]:
    seen = set()
    rows: List[List[int]] = []
    for i in range(size):
        for offset in offsets:
            if offset <= 0:
                raise ValueError(f"circulant offsets must be positive, got {offset}")
            j = (i + offset) % size
            if i == j:
                continue
            key = tuple(sorted((i, j)))
            if key in seen:
                continue
            seen.add(key)
            rows.append([key[0], key[1]])
    return rows


def balanced_queries(
    adapters: Sequence[AdapterSpec],
    arity: int,
    limit: int,
    rng: random.Random,
) -> List[Tuple[AdapterSpec, ...]]:
    if limit <= 0:
        return []
    remaining = list(itertools.combinations(adapters, arity))
    if limit > len(remaining):
        raise ValueError(f"requested {limit} queries but only {len(remaining)} combinations exist")
    rng.shuffle(remaining)
    counts = {adapter.name: 0 for adapter in adapters}
    selected: List[Tuple[AdapterSpec, ...]] = []
    while len(selected) < limit:
        best_idx = min(
            range(len(remaining)),
            key=lambda idx: (
                max(counts[adapter.name] + 1 for adapter in remaining[idx]),
                sum(counts[adapter.name] for adapter in remaining[idx]),
                idx,
            ),
        )
        query = remaining.pop(best_idx)
        selected.append(query)
        for adapter in query:
            counts[adapter.name] += 1
    return selected


def selection_summary(
    config: Dict[str, Any],
    adapters: Sequence[AdapterSpec],
    pairs: Sequence[Tuple[AdapterSpec, ...]],
    triples: Sequence[Tuple[AdapterSpec, ...]],
) -> Dict[str, Any]:
    def counts_for(rows: Sequence[Tuple[AdapterSpec, ...]]) -> Dict[str, int]:
        counts = {adapter.name: 0 for adapter in adapters}
        for row in rows:
            for adapter in row:
                counts[adapter.name] += 1
        return counts

    def spread(counts: Dict[str, int]) -> Dict[str, Any]:
        values = sorted(counts.values())
        if not values:
            return {"min": 0, "median": 0, "max": 0, "counts": counts}
        return {
            "min": values[0],
            "median": values[len(values) // 2],
            "max": values[-1],
            "counts": counts,
        }

    pair_counts = counts_for(pairs)
    triple_counts = counts_for(triples)
    total_counts = {
        adapter.name: pair_counts[adapter.name] + triple_counts[adapter.name]
        for adapter in adapters
    }
    return {
        "query_design": str(config.get("query_design", config.get("subset_design", "canonical"))),
        "uses_explicit_pair_subset": "pair_subset" in config,
        "uses_explicit_triple_subset": "triple_subset" in config,
        "seed": config.get("seed"),
        "pair_count": len(pairs),
        "triple_count": len(triples),
        "pair_participation": spread(pair_counts),
        "triple_participation": spread(triple_counts),
        "total_participation": spread(total_counts),
        "pairs": [[adapter.name for adapter in row] for row in pairs],
        "triples": [[adapter.name for adapter in row] for row in triples],
    }


def stable_seed(config: Dict[str, Any]) -> int:
    seed = int(config.get("seed", 0))
    manifest_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
    return seed ^ int(manifest_hash[:8], 16)


def split_prompts(config: Dict[str, Any], adapter: AdapterSpec, rng: random.Random) -> Tuple[List[str], List[str]]:
    if "probes" not in config:
        raise ValueError(
            "this config contains probe_counts but not probe text; it can be dry-run for query selection, "
            "but a real evaluation needs a probes mapping"
        )
    limit = int(config.get("prompt_limit_per_split", 4))
    raw_prompts = config["probes"][adapter.probe_key]
    if isinstance(raw_prompts, dict):
        calibration = list(raw_prompts.get("calibration", []))[:limit]
        heldout = list(raw_prompts.get("heldout", []))[:limit]
        if not calibration or not heldout:
            raise ValueError(
                f"probe {adapter.probe_key!r} uses explicit splits and needs non-empty "
                "calibration and heldout lists"
            )
        return calibration, heldout

    prompts = list(raw_prompts)
    if len(prompts) < 2:
        raise ValueError(f"probe {adapter.probe_key!r} needs at least two prompts")
    rng.shuffle(prompts)
    split = max(1, len(prompts) // 2)
    calibration = prompts[:split][:limit]
    heldout = prompts[split:][:limit]
    if not heldout:
        heldout = prompts[:limit]
    return calibration, heldout


def split_count_from_probe_counts(config: Dict[str, Any], adapter: AdapterSpec) -> Dict[str, int]:
    total = int(config.get("probe_counts", {}).get(adapter.probe_key, 0))
    limit = int(config.get("prompt_limit_per_split", 4))
    split = max(1, total // 2)
    calibration_count = min(split, limit)
    heldout_count = min(max(total - split, 0), limit)
    if heldout_count == 0 and total:
        heldout_count = min(total, limit)
    return {
        "calibration_count": calibration_count,
        "heldout_count": heldout_count,
    }


def simplex_grid(size: int, step: float, min_weight: float = 0.0) -> List[List[float]]:
    if size < 2:
        raise ValueError("simplex grid size must be at least 2")
    denom = int(round(1.0 / step))
    if denom <= 0 or abs((1.0 / denom) - step) > 1e-9:
        raise ValueError("step must evenly divide 1.0")
    min_units = int(math.ceil(min_weight * denom - 1e-9))
    if min_units * size > denom:
        raise ValueError("min_weight leaves no simplex grid points")

    rows: List[List[float]] = []

    def rec(accum: List[int], remaining: int, slots: int) -> None:
        if slots == 1:
            if remaining >= min_units:
                rows.append([(value / denom) for value in accum + [remaining]])
            return
        min_remaining = min_units * (slots - 1)
        for value in range(min_units, remaining - min_remaining + 1):
            rec(accum + [value], remaining - value, slots - 1)

    rec([], denom, size)
    return rows


def quadratic_features(weights: Sequence[float]) -> List[float]:
    coords = list(weights[:-1])
    features = [1.0]
    features.extend(coords)
    for i in range(len(coords)):
        for j in range(i, len(coords)):
            features.append(coords[i] * coords[j])
    return features


def hessian_from_quadratic_coeff(coeff: Sequence[float], coord_dim: int):
    import numpy as np

    hessian = np.zeros((coord_dim, coord_dim), dtype=float)
    cursor = 1 + coord_dim
    for i in range(coord_dim):
        for j in range(i, coord_dim):
            value = float(coeff[cursor])
            if i == j:
                hessian[i, i] = 2.0 * value
            else:
                hessian[i, j] = value
                hessian[j, i] = value
            cursor += 1
    return hessian


def replace_quadratic_hessian(coeff: Sequence[float], hessian: Any, coord_dim: int) -> List[float]:
    updated = [float(value) for value in coeff]
    cursor = 1 + coord_dim
    for i in range(coord_dim):
        for j in range(i, coord_dim):
            updated[cursor] = 0.5 * float(hessian[i, i]) if i == j else float(hessian[i, j])
            cursor += 1
    return updated


def fit_quadratic_model(
    weights_grid: Sequence[Sequence[float]],
    values: Sequence[float],
    project_psd: bool = True,
) -> Tuple[List[float], Dict[str, Any]]:
    import numpy as np

    phi = np.asarray([quadratic_features(w) for w in weights_grid], dtype=float)
    y = np.asarray(values, dtype=float)
    coeff, _, _, _ = np.linalg.lstsq(phi, y, rcond=None)
    coeff_list = [float(x) for x in coeff]
    coord_dim = len(weights_grid[0]) - 1
    hessian = hessian_from_quadratic_coeff(coeff_list, coord_dim)
    eigenvalues = np.linalg.eigvalsh(hessian) if coord_dim else np.asarray([0.0])
    min_before = float(eigenvalues.min())
    psd_adjustment = 0.0
    projected = False
    if project_psd and coord_dim and min_before < 0.0:
        evals, evecs = np.linalg.eigh(hessian)
        clipped = np.maximum(evals, 0.0)
        psd_hessian = (evecs * clipped) @ evecs.T
        psd_adjustment = float(np.linalg.norm(psd_hessian - hessian, ord="fro"))
        coeff_list = replace_quadratic_hessian(coeff_list, psd_hessian, coord_dim)
        projected = True
        eigenvalues_after = np.linalg.eigvalsh(psd_hessian)
        min_after = float(eigenvalues_after.min())
    else:
        min_after = min_before
    diagnostics = {
        "coord_dim": coord_dim,
        "project_psd": bool(project_psd),
        "psd_projected": projected,
        "min_eigenvalue_before": min_before,
        "min_eigenvalue_after": min_after,
        "psd_adjustment_frobenius": psd_adjustment,
    }
    return coeff_list, diagnostics


def fit_quadratic_scores(
    weights_grid: Sequence[Sequence[float]],
    values: Sequence[float],
    project_psd: bool = True,
) -> List[float]:
    coeff, _ = fit_quadratic_model(weights_grid, values, project_psd=project_psd)
    return coeff


def predict_quadratic(coeff: Sequence[float], weights: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(coeff, quadratic_features(weights)))


def coords_to_weights(coords: Sequence[float]) -> List[float]:
    values = [float(value) for value in coords]
    values.append(1.0 - sum(values))
    return values


def quadratic_value_grad(coeff: Sequence[float], coords: Sequence[float]):
    import numpy as np

    x = np.asarray(coords, dtype=float)
    coord_dim = len(x)
    alpha = float(coeff[0])
    b = np.asarray(coeff[1 : 1 + coord_dim], dtype=float)
    hessian = hessian_from_quadratic_coeff(coeff, coord_dim)
    value = alpha + float(b @ x) + 0.5 * float(x @ hessian @ x)
    grad = b + hessian @ x
    return value, grad


def simplex_constraints(coord_dim: int, min_weight: float):
    constraints = []
    if coord_dim:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x: 1.0 - min_weight - float(sum(x)),
                "jac": lambda x: -__import__("numpy").ones(coord_dim, dtype=float),
            }
        )
    return constraints


def continuous_primal_dual_certificate(
    task_coeffs: Dict[str, List[float]],
    min_weight: float,
    grid: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    """Solve min_c max_t q_t(c) and the matched Sion dual for fitted jets."""
    import numpy as np
    from scipy.optimize import minimize

    names = list(task_coeffs)
    if not names:
        raise ValueError("continuous certificate needs at least one task")
    feature_tail = len(task_coeffs[names[0]]) - 1
    coord_dim = int(round((-3.0 + math.sqrt(9.0 + 8.0 * feature_tail)) / 2.0)) if feature_tail else 0
    expected_len = 1 + coord_dim + coord_dim * (coord_dim + 1) // 2
    if expected_len != len(task_coeffs[names[0]]):
        raise ValueError(f"could not infer quadratic coordinate dimension from {len(task_coeffs[names[0]])} coefficients")
    support_size = coord_dim + 1
    if min_weight * support_size > 1.0 + 1e-12:
        raise ValueError("minimum simplex weight makes the continuous domain empty")

    def q(name: str, x: Sequence[float]) -> float:
        value, _ = quadratic_value_grad(task_coeffs[name], x)
        return value

    def qgrad(name: str, x: Sequence[float]):
        _, grad = quadratic_value_grad(task_coeffs[name], x)
        return grad

    grid_starts = [np.asarray(weights[:-1], dtype=float) for weights in grid]
    uniform = np.asarray([1.0 / support_size] * coord_dim, dtype=float)
    starts = [uniform]
    if grid_starts:
        best_grid = min(grid_starts, key=lambda x: max(q(name, x) for name in names))
        starts.append(best_grid)

    bounds = [(min_weight, 1.0 - min_weight) for _ in range(coord_dim)] + [(None, None)]
    primal_constraints = []
    if coord_dim:
        primal_constraints.append(
            {
                "type": "ineq",
                "fun": lambda z: 1.0 - min_weight - float(sum(z[:-1])),
                "jac": lambda z: np.concatenate([-np.ones(coord_dim, dtype=float), np.asarray([0.0])]),
            }
        )
    for name in names:
        primal_constraints.append(
            {
                "type": "ineq",
                "fun": lambda z, task=name: float(z[-1]) - q(task, z[:-1]),
                "jac": lambda z, task=name: np.concatenate([-qgrad(task, z[:-1]), np.asarray([1.0])]),
            }
        )

    def primal_obj(z):
        return float(z[-1])

    def primal_jac(z):
        grad = np.zeros(coord_dim + 1, dtype=float)
        grad[-1] = 1.0
        return grad

    primal_candidates = []
    for start in starts:
        start_t = max(q(name, start) for name in names)
        z0 = np.concatenate([start, np.asarray([start_t])])
        result = minimize(
            primal_obj,
            z0,
            jac=primal_jac,
            bounds=bounds,
            constraints=primal_constraints,
            method="SLSQP",
            options={"ftol": 1e-10, "maxiter": 300, "disp": False},
        )
        primal_candidates.append(result)
    primal = min(primal_candidates, key=lambda result: result.fun if result.x is not None else float("inf"))
    if primal.x is None:
        raise ValueError("continuous primal solver returned no iterate")
    primal_x = np.asarray(primal.x[:-1], dtype=float)
    primal_value = max(q(name, primal_x) for name in names)

    def inner_min(lambda_vec: Sequence[float], start_x: np.ndarray | None = None):
        lam = np.asarray(lambda_vec, dtype=float)

        def inner_obj(x):
            return float(sum(lam[idx] * q(name, x) for idx, name in enumerate(names)))

        def inner_jac(x):
            grad = np.zeros(coord_dim, dtype=float)
            for idx, name in enumerate(names):
                grad += lam[idx] * qgrad(name, x)
            return grad

        x0 = np.asarray(start_x if start_x is not None else primal_x, dtype=float)
        result = minimize(
            inner_obj,
            x0,
            jac=inner_jac,
            bounds=[(min_weight, 1.0 - min_weight) for _ in range(coord_dim)],
            constraints=simplex_constraints(coord_dim, min_weight),
            method="SLSQP",
            options={"ftol": 1e-10, "maxiter": 200, "disp": False},
        )
        return float(result.fun), np.asarray(result.x, dtype=float), bool(result.success), str(result.message)

    if len(names) == 1:
        dual_weights = np.asarray([1.0])
        dual_value, dual_x, dual_success, dual_msg = inner_min(dual_weights, primal_x)
    else:
        active = [idx for idx, name in enumerate(names) if abs(q(name, primal_x) - primal_value) <= 1e-5]
        lambda_starts = [np.asarray([1.0 / len(names)] * len(names), dtype=float)]
        if active:
            lam = np.zeros(len(names), dtype=float)
            for idx in active:
                lam[idx] = 1.0 / len(active)
            lambda_starts.append(lam)

        def outer_obj(lambda_vec):
            value, _, _, _ = inner_min(lambda_vec, primal_x)
            return -value

        outer_constraints = [{"type": "eq", "fun": lambda lam: float(sum(lam)) - 1.0}]
        outer_bounds = [(0.0, 1.0) for _ in names]
        outer_candidates = []
        for lam0 in lambda_starts:
            outer_candidates.append(
                minimize(
                    outer_obj,
                    lam0,
                    bounds=outer_bounds,
                    constraints=outer_constraints,
                    method="SLSQP",
                    options={"ftol": 1e-10, "maxiter": 200, "disp": False},
                )
            )
        outer = min(outer_candidates, key=lambda result: result.fun)
        dual_weights = np.maximum(np.asarray(outer.x, dtype=float), 0.0)
        dual_sum = float(dual_weights.sum())
        if dual_sum <= 0:
            dual_weights = np.asarray([1.0 / len(names)] * len(names), dtype=float)
        else:
            dual_weights = dual_weights / dual_sum
        dual_value, dual_x, dual_inner_success, dual_inner_msg = inner_min(dual_weights, primal_x)
        dual_success = bool(outer.success and dual_inner_success)
        dual_msg = f"outer={outer.message}; inner={dual_inner_msg}"

    gap = max(0.0, float(primal_value - dual_value))
    return {
        "primal_value": float(primal_value),
        "primal_weights": coords_to_weights(primal_x),
        "dual_value": float(dual_value),
        "dual_weights": [float(value) for value in dual_weights],
        "dual_inner_weights": coords_to_weights(dual_x),
        "certificate_gap": gap,
        "active_tasks": [names[idx] for idx, value in enumerate(dual_weights) if value > 1e-5],
        "primal_success": bool(primal.success),
        "dual_success": bool(dual_success),
        "message": f"primal={primal.message}; dual={dual_msg}",
    }


def normalized_deficit(loss: float, base_loss: float, single_loss: float, threshold: float, min_gain: float) -> float:
    gain = max(base_loss - single_loss, min_gain)
    target = base_loss - threshold * gain
    return (loss - target) / gain


def score_query(
    adapter_names: Sequence[str],
    weights: Sequence[float],
    losses: Dict[str, float],
    base_losses: Dict[str, float],
    single_losses: Dict[str, float],
    threshold: float,
    min_gain: float,
) -> Tuple[float, Dict[str, float], List[str]]:
    deficits = {}
    for name in adapter_names:
        deficits[name] = normalized_deficit(
            losses[name],
            base_losses[name],
            single_losses[name],
            threshold,
            min_gain,
        )
    max_deficit = max(deficits.values())
    margins = {name: -value for name, value in deficits.items()}
    obstructing = sorted([name for name, value in deficits.items() if abs(value - max_deficit) <= 1e-6])
    return max_deficit, margins, obstructing


def auc_roc(labels: Sequence[bool], scores: Sequence[float]) -> Optional[float]:
    positives = [(score, label) for score, label in zip(scores, labels) if label]
    negatives = [(score, label) for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = len(positives) * len(negatives)
    for ps, _ in positives:
        for ns, _ in negatives:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / total


def auc_pr(labels: Sequence[bool], scores: Sequence[float]) -> Optional[float]:
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


def top_quartile_enrichment(labels: Sequence[bool], scores: Sequence[float]) -> Optional[float]:
    if not labels:
        return None
    base_rate = sum(1 for label in labels if label) / len(labels)
    if base_rate == 0:
        return None
    take = max(1, math.ceil(len(labels) * 0.25))
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    top_rate = sum(1 for _, label in ordered[:take] if label) / take
    return top_rate / base_rate


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return "--"
    return f"{value:.3f}"


def write_summary_tex(path: Path, summary: Dict[str, Any]) -> None:
    metrics = summary.get("best_metrics") or {}
    sentence = summary.get("result_sentence", "The public PEFT atlas-lite run has not completed.")
    lines = [
        f"\\newcommand{{\\PeftAtlasStatus}}{{{summary.get('status', 'unknown')}}}",
        f"\\newcommand{{\\PeftAtlasBaseModel}}{{{summary.get('base_model', 'unknown')}}}",
        f"\\newcommand{{\\PeftAtlasAdapterCount}}{{{summary.get('adapter_count', 0)}}}",
        f"\\newcommand{{\\PeftAtlasPairCount}}{{{summary.get('pair_count', 0)}}}",
        f"\\newcommand{{\\PeftAtlasTripleCount}}{{{summary.get('triple_count', 0)}}}",
        f"\\newcommand{{\\PeftAtlasFeasibleCount}}{{{summary.get('feasible_count', '--')}}}",
        f"\\newcommand{{\\PeftAtlasFullSupportPairFeasibleCount}}{{{summary.get('full_support_pair_feasible_count', '--')}}}",
        f"\\newcommand{{\\PeftAtlasFullSupportTripleFeasibleCount}}{{{summary.get('full_support_triple_feasible_count', '--')}}}",
        f"\\newcommand{{\\PeftAtlasBestMethod}}{{{summary.get('best_method', 'not run')}}}",
        f"\\newcommand{{\\PeftAtlasAUROC}}{{{format_metric(metrics.get('auroc'))}}}",
        f"\\newcommand{{\\PeftAtlasAUPRC}}{{{format_metric(metrics.get('auprc'))}}}",
        f"\\newcommand{{\\PeftAtlasTopQuartile}}{{{format_metric(metrics.get('top_quartile_enrichment'))}}}",
        f"\\newcommand{{\\PeftAtlasTripleObstructions}}{{{summary.get('triple_obstructions', '--')}}}",
        f"\\newcommand{{\\PeftAtlasResultSentence}}{{{sentence}}}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dry_run(config: Dict[str, Any], out_dir: Path, max_pairs: Optional[int], max_triples: Optional[int], methods: Sequence[str]) -> None:
    adapters = adapters_from_config(config)
    rng = random.Random(stable_seed(config))
    shuffle = bool(config.get("shuffle_queries", False))
    pairs, triples = planned_queries(adapters, max_pairs, max_triples, rng, shuffle, config)
    selected = selection_summary(config, adapters, pairs, triples)
    split_counts = {}
    for adapter in adapters:
        if "probes" in config:
            calibration, heldout = split_prompts(config, adapter, rng)
            split_counts[adapter.name] = {
                "calibration_count": len(calibration),
                "heldout_count": len(heldout),
            }
        else:
            split_counts[adapter.name] = split_count_from_probe_counts(config, adapter)
    summary = {
        "status": "pending",
        "experiment_id": config.get("experiment_id", "unknown"),
        "base_model": config.get("base_model", "unknown"),
        "adapter_count": len(adapters),
        "pair_count": len(pairs),
        "triple_count": len(triples),
        "methods": list(methods),
        "query_selection": selected,
        "certificate_implementation": "continuous primal-dual minimax solver over fitted affine-quadratic jets; PEFT labels come from the evaluated coefficient design",
        "evaluation_split_counts": split_counts,
        "best_method": "not run",
        "best_metrics": {},
        "triple_obstructions": "--",
        "result_sentence": "No completed PEFT atlas-lite summary is available.",
    }
    dump_json(out_dir / "dry_run_plan.json", summary)
    dump_json(out_dir / "subset_selection.json", selected)
    write_summary_tex(out_dir / "summary.tex", summary)


def import_versions() -> Dict[str, str]:
    import importlib.metadata as metadata

    versions = {}
    for package in ["torch", "transformers", "peft", "accelerate", "bitsandbytes", "numpy"]:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def configure_torch_runtime() -> None:
    """Keep GPU workers from becoming CPU-thread schedulers."""
    try:
        import torch
    except Exception:
        return

    thread_count = int(os.environ.get("ATLAS_TORCH_NUM_THREADS", "1"))
    if thread_count > 0:
        torch.set_num_threads(thread_count)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(max(1, min(thread_count, 2)))
            except RuntimeError:
                pass
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(os.environ.get("ATLAS_MATMUL_PRECISION", "high"))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def torch_dtype_from_name(name: str):
    import torch

    lowered = name.lower()
    if lowered == "bfloat16":
        return torch.bfloat16
    if lowered == "float16":
        return torch.float16
    if lowered == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def load_peft_model(config: Dict[str, Any], adapters: Sequence[AdapterSpec]):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype = torch_dtype_from_name(str(config.get("dtype", "bfloat16")))
    quantization = str(config.get("quantization", "none")).lower()
    quantization_config = None
    if quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization not in {"none", "false", "0"}:
        raise ValueError(f"unsupported quantization: {quantization}")

    tokenizer_name = config.get("tokenizer_model") or config["base_model"]
    trust_remote_code = bool(config.get("trust_remote_code", False))
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": config.get("device_map", "auto"),
        "trust_remote_code": trust_remote_code,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    base = AutoModelForCausalLM.from_pretrained(config["base_model"], **model_kwargs)
    resize_vocab_to = config.get("resize_vocab_to")
    if resize_vocab_to is not None:
        target_vocab = int(resize_vocab_to)
        if target_vocab > len(tokenizer):
            tokenizer.add_tokens([f"<atlas_extra_{idx}>" for idx in range(target_vocab - len(tokenizer))])
        base.resize_token_embeddings(target_vocab)
    first = adapters[0]
    model = PeftModel.from_pretrained(base, first.repo, adapter_name=first.name, is_trainable=False)
    for adapter in adapters[1:]:
        model.load_adapter(adapter.repo, adapter_name=adapter.name, is_trainable=False)
    model.eval()
    return model, tokenizer


def model_device(model: Any):
    import torch

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mean_nll(model: Any, tokenizer: Any, prompts: Sequence[str], max_length: int) -> float:
    losses = per_prompt_nll(model, tokenizer, prompts, max_length)
    return sum(losses) / max(len(losses), 1)


def per_prompt_nll(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    max_length: int,
    adapter_names: Optional[Sequence[str]] = None,
) -> List[float]:
    import torch
    import torch.nn.functional as F

    if not prompts:
        return []
    device = model_device(model)
    encoded = tokenizer(
        list(prompts),
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    labels = encoded["input_ids"].clone()
    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100
    with torch.no_grad():
        kwargs = {}
        if adapter_names is not None:
            kwargs["adapter_names"] = list(adapter_names)
        logits = model(**encoded, **kwargs).logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    token_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    mask = (shift_labels != -100).float()
    per_sample = token_losses.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return [float(value) for value in per_sample.detach().cpu().tolist()]


def evaluate_base_and_singles(
    model: Any,
    tokenizer: Any,
    adapters: Sequence[AdapterSpec],
    prompt_sets: Dict[str, Dict[str, List[str]]],
    max_length: int,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    base_losses = {"calibration": {}, "heldout": {}}
    single_losses = {"calibration": {}, "heldout": {}}

    with model.disable_adapter():
        for split in ["calibration", "heldout"]:
            prompts: List[str] = []
            adapter_indices: List[str] = []
            for adapter in adapters:
                task_prompts = list(prompt_sets[adapter.name][split])
                prompts.extend(task_prompts)
                adapter_indices.extend([adapter.name] * len(task_prompts))
            grouped = mean_by_group(
                per_prompt_nll(
                    model,
                    tokenizer,
                    prompts,
                    max_length,
                ),
                adapter_indices,
            )
            for adapter in adapters:
                base_losses[split][adapter.name] = grouped.get(adapter.name, float("nan"))

    for split in ["calibration", "heldout"]:
        for adapter in adapters:
            model.set_adapter(adapter.name)
            single_losses[split][adapter.name] = mean_nll(
                model,
                tokenizer,
                prompt_sets[adapter.name][split],
                max_length,
            )
    return base_losses, single_losses


def add_weighted_adapter(model: Any, adapter_names: Sequence[str], weights: Sequence[float], method: str, density: float, merge_name: str) -> None:
    kwargs: Dict[str, Any] = {
        "adapters": list(adapter_names),
        "weights": list(weights),
        "adapter_name": merge_name,
        "combination_type": method,
    }
    if method not in {"linear", "svd", "cat"}:
        kwargs["density"] = density
    model.add_weighted_adapter(**kwargs)
    model.set_adapter(merge_name)


def delete_adapter_safely(model: Any, adapter_name: str, fallback_adapter: Optional[str] = None) -> None:
    if hasattr(model, "delete_adapter"):
        model.delete_adapter(adapter_name)
    if fallback_adapter:
        try:
            model.set_adapter(fallback_adapter)
        except Exception:
            pass


def evaluate_merge_losses(
    model: Any,
    tokenizer: Any,
    query: Sequence[AdapterSpec],
    weights: Sequence[float],
    method: str,
    density: float,
    prompt_sets: Dict[str, Dict[str, List[str]]],
    max_length: int,
    split: str,
    serial: int,
) -> Dict[str, float]:
    merge_name = f"atlas_{method}_{serial}"
    names = [adapter.name for adapter in query]
    add_weighted_adapter(model, names, weights, method, density, merge_name)
    try:
        return {
            adapter.name: mean_nll(model, tokenizer, prompt_sets[adapter.name][split], max_length)
            for adapter in query
        }
    finally:
        delete_adapter_safely(model, merge_name, names[0])


def evaluate_merge_losses_for_splits(
    model: Any,
    tokenizer: Any,
    query: Sequence[AdapterSpec],
    weights: Sequence[float],
    method: str,
    density: float,
    prompt_sets: Dict[str, Dict[str, List[str]]],
    max_length: int,
    serial: int,
    splits: Sequence[str] = ("calibration", "heldout"),
) -> Dict[str, Dict[str, float]]:
    merge_name = f"atlas_{method}_{serial}"
    names = [adapter.name for adapter in query]
    add_weighted_adapter(model, names, weights, method, density, merge_name)
    try:
        return {
            split: {
                adapter.name: mean_nll(
                    model,
                    tokenizer,
                    prompt_sets[adapter.name][split],
                    max_length,
                )
                for adapter in query
            }
            for split in splits
        }
    finally:
        delete_adapter_safely(model, merge_name, names[0])


def mean_by_grid(losses: Sequence[float], grid_indices: Sequence[int], grid_size: int) -> List[float]:
    sums = [0.0 for _ in range(grid_size)]
    counts = [0 for _ in range(grid_size)]
    for loss, grid_idx in zip(losses, grid_indices):
        sums[grid_idx] += float(loss)
        counts[grid_idx] += 1
    return [sums[idx] / max(counts[idx], 1) for idx in range(grid_size)]


def mean_by_group(losses: Sequence[float], group_names: Sequence[str]) -> Dict[str, float]:
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for loss, group_name in zip(losses, group_names):
        sums[group_name] = sums.get(group_name, 0.0) + float(loss)
        counts[group_name] = counts.get(group_name, 0) + 1
    return {group_name: sums[group_name] / max(counts[group_name], 1) for group_name in sums}


def run_query(
    model: Any,
    tokenizer: Any,
    query: Sequence[AdapterSpec],
    method: str,
    grid: Sequence[Sequence[float]],
    config: Dict[str, Any],
    prompt_sets: Dict[str, Dict[str, List[str]]],
    base_losses: Dict[str, Dict[str, float]],
    single_losses: Dict[str, Dict[str, float]],
    serial_start: int,
) -> Tuple[QueryResult, int]:
    threshold = float(config.get("retention_threshold", 0.7))
    min_gain = float(config.get("min_single_adapter_gain", 0.005))
    density = float(config.get("density", 0.5))
    max_length = int(config.get("max_length", 384))
    project_psd = bool(config.get("project_quadratic_hessian_psd", config.get("project_psd", True)))
    adapter_names = [adapter.name for adapter in query]
    query_kind = "pair" if len(query) == 2 else "triple"

    calibration_scores: List[float] = []
    calibration_task_values: Dict[str, List[float]] = {name: [] for name in adapter_names}
    heldout_scores: List[float] = []
    heldout_margins: List[Dict[str, float]] = []
    heldout_obstructions: List[List[str]] = []
    serial = serial_start

    merge_names: List[str] = []
    try:
        for weights in grid:
            serial += 1
            merge_name = f"atlas_{method}_{serial}"
            add_weighted_adapter(model, adapter_names, weights, method, density, merge_name)
            merge_names.append(merge_name)

        heldout_task_values: Dict[str, List[float]] = {name: [] for name in adapter_names}
        for split in ["calibration", "heldout"]:
            split_task_values: Dict[str, List[float]] = {name: [] for name in adapter_names}
            for merge_name in merge_names:
                model.set_adapter(merge_name)
                for adapter in query:
                    loss = mean_nll(
                        model,
                        tokenizer,
                        prompt_sets[adapter.name][split],
                        max_length,
                    )
                    split_task_values[adapter.name].append(
                        normalized_deficit(
                            loss,
                            base_losses[split][adapter.name],
                            single_losses[split][adapter.name],
                            threshold,
                            min_gain,
                        )
                    )
            if split == "calibration":
                calibration_task_values = split_task_values
            else:
                heldout_task_values = split_task_values

        for grid_idx, weights in enumerate(grid):
            calibration_deficits = {name: calibration_task_values[name][grid_idx] for name in adapter_names}
            calibration_score = max(calibration_deficits.values())
            calibration_scores.append(calibration_score)

            heldout_deficits = {name: heldout_task_values[name][grid_idx] for name in adapter_names}
            heldout_score = max(heldout_deficits.values())
            margins = {name: -value for name, value in heldout_deficits.items()}
            obstructions = sorted(
                [name for name, value in heldout_deficits.items() if abs(value - heldout_score) <= 1e-6]
            )
            heldout_scores.append(heldout_score)
            heldout_margins.append(margins)
            heldout_obstructions.append(obstructions)
    finally:
        for merge_name in merge_names:
            delete_adapter_safely(model, merge_name, adapter_names[0])

    task_coeffs: Dict[str, List[float]] = {}
    fit_diagnostics: List[Dict[str, Any]] = []
    for adapter in query:
        coeff, diagnostics = fit_quadratic_model(
            grid,
            calibration_task_values[adapter.name],
            project_psd=project_psd,
        )
        task_coeffs[adapter.name] = coeff
        fit_diagnostics.append(diagnostics)

    design_predicted_scores = []
    for weights in grid:
        design_predicted_scores.append(max(predict_quadratic(task_coeffs[name], weights) for name in adapter_names))
    design_pred_idx = min(range(len(grid)), key=lambda idx: design_predicted_scores[idx])
    held_idx = min(range(len(grid)), key=lambda idx: heldout_scores[idx])
    predicted_score = design_predicted_scores[design_pred_idx]
    predicted_weights = list(grid[design_pred_idx])
    primal_upper = predicted_score
    dual_lower = None
    certificate_gap = None
    dual_weights = None
    certificate_type = "evaluated_coefficient_primal_psd" if project_psd else "evaluated_coefficient_primal_unconstrained"
    certificate_message = ""
    if bool(config.get("solve_continuous_certificate", True)):
        query_min_weight = float(
            config.get(
                "pair_min_weight" if query_kind == "pair" else "triple_min_weight",
                config.get("min_simplex_weight", 0.0),
            )
        )
        try:
            certificate = continuous_primal_dual_certificate(task_coeffs, query_min_weight, grid)
            predicted_score = float(certificate["primal_value"])
            predicted_weights = [float(value) for value in certificate["primal_weights"]]
            primal_upper = float(certificate["primal_value"])
            dual_lower = float(certificate["dual_value"])
            certificate_gap = float(certificate["certificate_gap"])
            dual_weights = [float(value) for value in certificate["dual_weights"]]
            certificate_type = "continuous_primal_dual_psd" if project_psd else "continuous_primal_dual_unconstrained"
            certificate_message = str(certificate["message"])
            design_pred_idx = min(
                range(len(grid)),
                key=lambda idx: sum((float(a) - float(b)) ** 2 for a, b in zip(grid[idx], predicted_weights)),
            )
        except Exception as exc:  # noqa: BLE001 - certificate fallback should not kill GPU runs.
            certificate_message = f"continuous_certificate_failed: {exc!r}"
    heldout_score = heldout_scores[held_idx]
    regret = heldout_scores[design_pred_idx] - heldout_score

    # Ablation: un-projected jet value max_t q~_t(c^) at the projected certificate
    # solution. Fits the true (possibly indefinite) jets and evaluates them at the
    # same primal weights the projected certificate chose.
    unprojected_jet_max = None
    try:
        unproj_coeffs = {
            a.name: fit_quadratic_model(grid, calibration_task_values[a.name], project_psd=False)[0]
            for a in query
        }
        unprojected_jet_max = float(
            max(predict_quadratic(unproj_coeffs[nm], predicted_weights) for nm in adapter_names)
        )
    except Exception:  # noqa: BLE001 - ablation logging must not kill a query
        unprojected_jet_max = None

    result = QueryResult(
        method=method,
        query_kind=query_kind,
        adapters=adapter_names,
        predicted_score=predicted_score,
        heldout_score=heldout_score,
        feasible=heldout_score <= 0.0,
        predicted_weights=predicted_weights,
        heldout_weights=list(grid[held_idx]),
        coefficient_regret=regret,
        margins=heldout_margins[held_idx],
        obstructing_tasks=heldout_obstructions[held_idx],
        certificate_type=certificate_type,
        grid_points=len(grid),
        primal_upper=primal_upper,
        dual_lower=dual_lower,
        certificate_gap=certificate_gap,
        dual_weights=dual_weights,
        certificate_message=certificate_message,
        psd_projected=any(item["psd_projected"] for item in fit_diagnostics),
        quadratic_min_eigenvalue=min(item["min_eigenvalue_before"] for item in fit_diagnostics),
        quadratic_psd_adjustment=max(item["psd_adjustment_frobenius"] for item in fit_diagnostics),
        unprojected_jet_max=unprojected_jet_max,
    )
    return result, serial


def summarize_results(config: Dict[str, Any], adapters: Sequence[AdapterSpec], pairs: Sequence[Tuple[AdapterSpec, ...]], triples: Sequence[Tuple[AdapterSpec, ...]], results: Sequence[QueryResult], errors: Sequence[QueryResult]) -> Dict[str, Any]:
    by_method: Dict[str, List[QueryResult]] = {}
    for result in results:
        by_method.setdefault(result.method, []).append(result)

    metrics_by_method: Dict[str, Dict[str, Optional[float]]] = {}
    for method, rows in by_method.items():
        labels = [row.feasible for row in rows]
        scores = [-row.predicted_score for row in rows]
        metrics_by_method[method] = {
            "auroc": auc_roc(labels, scores),
            "auprc": auc_pr(labels, scores),
            "top_quartile_enrichment": top_quartile_enrichment(labels, scores),
            "mean_coefficient_regret": sum(row.coefficient_regret for row in rows) / max(len(rows), 1),
            "feasible_rate": sum(1 for label in labels if label) / max(len(labels), 1),
        }

    best_method = "not run"
    best_metrics: Dict[str, Optional[float]] = {}
    if metrics_by_method:
        best_method = max(
            metrics_by_method,
            key=lambda method: (
                metrics_by_method[method].get("auroc") if metrics_by_method[method].get("auroc") is not None else -1.0,
                metrics_by_method[method].get("auprc") if metrics_by_method[method].get("auprc") is not None else -1.0,
            ),
        )
        best_metrics = metrics_by_method[best_method]

    pair_lookup = {
        (row.method, tuple(sorted(row.adapters))): row.feasible
        for row in results
        if row.query_kind == "pair"
    }
    triple_obstructions = 0
    for row in results:
        if row.query_kind != "triple" or row.feasible:
            continue
        pair_faces = list(itertools.combinations(sorted(row.adapters), 2))
        if pair_faces and all(pair_lookup.get((row.method, face), False) for face in pair_faces):
            triple_obstructions += 1

    feasible_count = sum(1 for row in results if row.feasible)
    full_support_feasible_count = sum(
        1 for row in results if row.feasible and all(weight > 1e-9 for weight in row.heldout_weights)
    )
    full_support_pair_feasible_count = sum(
        1
        for row in results
        if row.query_kind == "pair" and row.feasible and all(weight > 1e-9 for weight in row.heldout_weights)
    )
    full_support_triple_feasible_count = sum(
        1
        for row in results
        if row.query_kind == "triple" and row.feasible and all(weight > 1e-9 for weight in row.heldout_weights)
    )
    feasible_rate = best_metrics.get("feasible_rate")
    eig_values = [row.quadratic_min_eigenvalue for row in results if row.quadratic_min_eigenvalue is not None]
    psd_adjustments = [
        row.quadratic_psd_adjustment
        for row in results
        if row.quadratic_psd_adjustment is not None
    ]
    convexity_diagnostics = {
        "projected_query_count": sum(1 for row in results if row.psd_projected),
        "min_fitted_hessian_eigenvalue": min(eig_values) if eig_values else None,
        "median_fitted_hessian_eigenvalue": sorted(eig_values)[len(eig_values) // 2] if eig_values else None,
        "max_psd_adjustment_frobenius": max(psd_adjustments) if psd_adjustments else None,
    }
    result_sentence = (
        f"The public PEFT atlas-lite run evaluated {len(results)} completed "
        f"queries over {len(adapters)} adapters with {len(errors)} errors; it isolated "
        f"{feasible_count} feasible subsets at a {format_metric(feasible_rate)} feasible "
        f"base rate, including {full_support_pair_feasible_count} full-support feasible "
        f"pairs and {full_support_triple_feasible_count} full-support feasible triples. "
        f"The best merge method was {best_method} with AUROC "
        f"{format_metric(best_metrics.get('auroc'))}, AUPRC {format_metric(best_metrics.get('auprc'))}, "
        f"and top-quartile enrichment {format_metric(best_metrics.get('top_quartile_enrichment'))}."
    )

    return {
        "status": "completed",
        "experiment_id": config.get("experiment_id", "unknown"),
        "base_model": config.get("base_model", "unknown"),
        "adapter_count": len(adapters),
        "pair_count": len(pairs),
        "triple_count": len(triples),
        "result_count": len(results),
        "error_count": len(errors),
        "best_method": best_method,
        "best_metrics": best_metrics,
        "metrics_by_method": metrics_by_method,
        "feasible_count": feasible_count,
        "full_support_feasible_count": full_support_feasible_count,
        "full_support_pair_feasible_count": full_support_pair_feasible_count,
        "full_support_triple_feasible_count": full_support_triple_feasible_count,
        "triple_obstructions": triple_obstructions,
        "result_sentence": result_sentence,
        "certificate_implementation": "continuous primal-dual minimax solver over fitted affine-quadratic jets; CSV primal_upper, dual_lower, certificate_gap, and dual_weights are populated when the solver succeeds",
        "convexity_diagnostics": convexity_diagnostics,
        "platform": platform.platform(),
        "versions": import_versions(),
    }


def write_results_csv(path: Path, rows: Sequence[QueryResult]) -> None:
    fields = [
        "method",
        "query_kind",
        "adapters",
        "predicted_score",
        "heldout_score",
        "feasible",
        "predicted_weights",
        "heldout_weights",
        "coefficient_regret",
        "margins",
        "obstructing_tasks",
        "certificate_type",
        "grid_points",
        "primal_upper",
        "dual_lower",
        "certificate_gap",
        "dual_weights",
        "certificate_message",
        "psd_projected",
        "quadratic_min_eigenvalue",
        "quadratic_psd_adjustment",
        "unprojected_jet_max",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row.method,
                    "query_kind": row.query_kind,
                    "adapters": "+".join(row.adapters),
                    "predicted_score": row.predicted_score,
                    "heldout_score": row.heldout_score,
                    "feasible": int(row.feasible),
                    "predicted_weights": json.dumps(row.predicted_weights),
                    "heldout_weights": json.dumps(row.heldout_weights),
                    "coefficient_regret": row.coefficient_regret,
                    "margins": json.dumps(row.margins, sort_keys=True),
                    "obstructing_tasks": "+".join(row.obstructing_tasks),
                    "certificate_type": row.certificate_type,
                    "grid_points": row.grid_points,
                    "primal_upper": row.primal_upper,
                    "dual_lower": row.dual_lower,
                    "certificate_gap": row.certificate_gap,
                    "dual_weights": json.dumps(row.dual_weights or []),
                    "certificate_message": row.certificate_message,
                    "psd_projected": int(row.psd_projected),
                    "quadratic_min_eigenvalue": row.quadratic_min_eigenvalue,
                    "quadratic_psd_adjustment": row.quadratic_psd_adjustment,
                    "unprojected_jet_max": row.unprojected_jet_max,
                    "error": row.error,
                }
            )


def run_experiment(config: Dict[str, Any], out_dir: Path, max_pairs: Optional[int], max_triples: Optional[int], methods: Sequence[str]) -> None:
    adapters = adapters_from_config(config)
    rng = random.Random(stable_seed(config))
    shuffle = bool(config.get("shuffle_queries", False))
    pairs, triples = planned_queries(adapters, max_pairs, max_triples, rng, shuffle, config)
    selected = selection_summary(config, adapters, pairs, triples)
    prompt_sets = {}
    for adapter in adapters:
        calibration, heldout = split_prompts(config, adapter, rng)
        prompt_sets[adapter.name] = {"calibration": calibration, "heldout": heldout}

    model, tokenizer = load_peft_model(config, adapters)
    base_losses, single_losses = evaluate_base_and_singles(
        model,
        tokenizer,
        adapters,
        prompt_sets,
        int(config.get("max_length", 384)),
    )

    pair_grid = simplex_grid(
        2,
        float(config.get("pair_step", 0.25)),
        float(config.get("pair_min_weight", config.get("min_simplex_weight", 0.0))),
    )
    triple_grid = simplex_grid(
        3,
        float(config.get("triple_step", 0.5)),
        float(config.get("triple_min_weight", config.get("min_simplex_weight", 0.0))),
    )

    results: List[QueryResult] = []
    errors: List[QueryResult] = []
    serial = 0
    all_queries: List[Tuple[AdapterSpec, ...]] = list(pairs) + list(triples)
    dump_json(out_dir / "subset_selection.json", selected)
    for method in methods:
        for query in all_queries:
            grid = pair_grid if len(query) == 2 else triple_grid
            try:
                result, serial = run_query(
                    model,
                    tokenizer,
                    query,
                    method,
                    grid,
                    config,
                    prompt_sets,
                    base_losses,
                    single_losses,
                    serial,
                )
                results.append(result)
                write_results_csv(out_dir / "partial.results.csv", results)
                write_results_csv(out_dir / "partial.errors.csv", errors)
                print(
                    f"[atlas-lite] {method} {'+'.join(result.adapters)} "
                    f"score={result.heldout_score:.4f} feasible={int(result.feasible)}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - keep long GPU runs alive.
                errors.append(
                    QueryResult(
                        method=method,
                        query_kind="pair" if len(query) == 2 else "triple",
                        adapters=[adapter.name for adapter in query],
                        predicted_score=float("nan"),
                        heldout_score=float("nan"),
                        feasible=False,
                        predicted_weights=[],
                        heldout_weights=[],
                        coefficient_regret=float("nan"),
                        margins={},
                        obstructing_tasks=[],
                        error=repr(exc),
                    )
                )
                write_results_csv(out_dir / "partial.results.csv", results)
                write_results_csv(out_dir / "partial.errors.csv", errors)
                print(f"[atlas-lite] ERROR {method} {'+'.join(adapter.name for adapter in query)} {exc!r}", flush=True)

    summary = summarize_results(config, adapters, pairs, triples, results, errors)
    summary["query_selection"] = selected
    summary["base_losses"] = base_losses
    summary["single_losses"] = single_losses
    summary["evaluation_split_counts"] = {
        adapter.name: {
            "calibration_count": len(prompt_sets[adapter.name]["calibration"]),
            "heldout_count": len(prompt_sets[adapter.name]["heldout"]),
        }
        for adapter in adapters
    }
    summary["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    write_results_csv(out_dir / "results.csv", results)
    write_results_csv(out_dir / "errors.csv", errors)
    dump_json(out_dir / "subset_selection.json", selected)
    dump_json(out_dir / "summary.json", summary)
    write_summary_tex(out_dir / "summary.tex", summary)


def main() -> None:
    args = parse_args()
    configure_torch_runtime()
    config_path = Path(args.config)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = load_json(config_path)
    methods = [item.strip() for item in (args.methods.split(",") if args.methods else config.get("merge_methods", ["linear"])) if item.strip()]
    if not methods:
        raise ValueError("no merge methods configured")

    dump_json(out_dir / "manifest.lock.json", config)
    if args.dry_run:
        dry_run(config, out_dir, args.max_pairs, args.max_triples, methods)
    else:
        run_experiment(config, out_dir, args.max_pairs, args.max_triples, methods)


if __name__ == "__main__":
    main()
