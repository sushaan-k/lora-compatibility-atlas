#!/usr/bin/env python3
"""Loss-surface cartography (A1): is the surrogate real?"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "peft_atlas_lite"))
sys.path.insert(0, str(HERE))
from run_peft_atlas_lite import (fit_quadratic_model,  # noqa: E402
                                 hessian_from_quadratic_coeff)
from surfaces import jet_value, weights_to_coords  # noqa: E402

# three-adapter charts to map (indices into the 8-adapter panel)
CHARTS = [(0, 1, 2), (3, 4, 5), (0, 5, 7)]
STEP_DEN = 12  # barycentric grid step 1/12 -> 91 points


def bary_grid(step_den):
    pts = []
    for i in range(step_den + 1):
        for j in range(step_den + 1 - i):
            k = step_den - i - j
            pts.append((i / step_den, j / step_den, k / step_den))
    return pts


def cmd_run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=torch.bfloat16,
                                                 device_map=device)
    specs = cfg["adapters"]
    peft = None
    for s in specs:
        if peft is None:
            peft = PeftModel.from_pretrained(model, s["repo"], adapter_name=s["name"], is_trainable=False)
        else:
            peft.load_adapter(s["repo"], adapter_name=s["name"])
    peft.eval()
    names = [s["name"] for s in specs]
    tasks = {s["name"]: json.loads(Path(s["task_json"]).read_text()) for s in specs}
    n_items = int(cfg.get("n_items", 40)); max_length = int(cfg.get("max_length", 384))

    def prompt_of(defn, inp):
        msgs = [{"role": "user", "content": defn + "\n\n" + inp}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return defn + "\n\n" + inp + "\n"

    def ces(members, weight3):
        active = [(names[members[i]], weight3[i]) for i in range(3) if weight3[i] > 1e-9]
        if len(active) == 1:
            peft.set_adapter(active[0][0])
        else:
            peft.add_weighted_adapter([a for a, _ in active], [w for _, w in active],
                                      "mergeH", combination_type="linear")
            peft.set_adapter("mergeH")
        row = {}
        for mi in members:
            nm = names[mi]
            defn, inst = tasks[nm]["definition"], tasks[nm]["instances"][:n_items]
            ta = tp = 0.0
            for it in inst:
                p = prompt_of(defn, it["input"]); gold = it["output"][0]
                pids = tok(p, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
                full = tok(p + " " + gold, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
                lab = full.clone(); lab[:, :pids.shape[1]] = -100
                with torch.no_grad():
                    ta += peft(input_ids=full, labels=lab).loss.item()
                    tp += peft(input_ids=pids, labels=pids.clone()).loss.item()
            row[nm] = {"answer": ta / len(inst), "prompt": tp / len(inst)}
        if len(active) > 1:
            peft.delete_adapter("mergeH")
        return row

    grid = bary_grid(STEP_DEN)
    charts = [tuple(c) for c in cfg.get("charts", CHARTS)]
    ck = out / "carto_points.jsonl"; done = {}
    if ck.exists():
        for line in ck.read_text().splitlines():
            if line.strip():
                r = json.loads(line); done[r["key"]] = r["ce"]
    f = open(ck, "a"); t0 = time.time(); n = 0
    for ci, chart in enumerate(charts):
        for gi, w in enumerate(grid):
            key = f"{ci}:{gi}"
            if key in done:
                continue
            done[key] = ces(chart, w)
            f.write(json.dumps({"key": key, "chart": chart, "w": w, "ce": done[key]}) + "\n")
            f.flush(); n += 1
            if n % 10 == 0:
                print(f"  {n} points t={int(time.time()-t0)}s", flush=True)
    (out / "carto_raw.json").write_text(json.dumps(
        {"charts": charts, "step_den": STEP_DEN, "names": names,
         "grid": grid, "points": done}, indent=1))
    print("wrote", out / "carto_raw.json", flush=True)
    return 0


def cmd_analyze(args):
    raw = json.loads((Path(args.raw) / "carto_raw.json").read_text())
    charts = raw["charts"]; grid = raw["grid"]; names = raw["names"]; pts = raw["points"]
    step = raw["step_den"]
    report = {"charts": []}
    for ci, chart in enumerate(charts):
        cnames = [names[i] for i in chart]
        # poised singles+pairs design in barycentric coords (weights over 3 members)
        design_w = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                    (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
        # map barycentric weight to chart coords (drop-last): coord = (w0, w1)
        def coord(w):
            return np.array([w[0], w[1]])
        per_task = {}
        for mi in chart:
            nm = names[mi]
            for channel in ("answer", "prompt"):
                # fit on the six poised points nearest to the design weights
                def val_at(w):
                    # nearest grid point
                    best = min(range(len(grid)),
                               key=lambda g: sum((grid[g][a] - w[a]) ** 2 for a in range(3)))
                    return pts[f"{ci}:{best}"][nm][channel], grid[best]
                dvals, dws = [], []
                for w in design_w:
                    v, gw = val_at(w)
                    dvals.append(v); dws.append([gw[0], gw[1], gw[2]])
                coeff, info = fit_quadratic_model(dws, dvals, project_psd=False)
                H = np.asarray(hessian_from_quadratic_coeff(coeff, 2), float)
                eig = np.linalg.eigvalsh(H)
                a0, b0 = float(coeff[0]), np.array(coeff[1:3], float)
                # predict every grid point, R^2 against the true surface
                yy, pp = [], []
                for gi, w in enumerate(grid):
                    y = pts[f"{ci}:{gi}"][nm][channel]
                    c = np.array([w[0], w[1]])
                    p = jet_value(a0, b0, H, c)
                    yy.append(y); pp.append(p)
                yy, pp = np.array(yy), np.array(pp)
                ss_res = float(np.sum((yy - pp) ** 2)); ss_tot = float(np.sum((yy - yy.mean()) ** 2))
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
                per_task[f"{nm}:{channel}"] = {
                    "r2_full_surface": round(r2, 4) if r2 is not None else None,
                    "hessian_eigs": [round(float(x), 4) for x in eig],
                    "convex_psd": bool(eig.min() >= -1e-6),
                    "max_abs_residual": round(float(np.max(np.abs(yy - pp))), 4),
                    "surface_range": [round(float(yy.min()), 3), round(float(yy.max()), 3)]}
        report["charts"].append({"chart": cnames, "per_task": per_task})
        print(f"chart {cnames}:", {k: v["r2_full_surface"] for k, v in per_task.items()
                                    if k.endswith("answer")}, flush=True)
    # summary
    r2a = [v["r2_full_surface"] for c in report["charts"] for k, v in c["per_task"].items()
           if k.endswith("answer") and v["r2_full_surface"] is not None]
    convex = [v["convex_psd"] for c in report["charts"] for k, v in c["per_task"].items()
              if k.endswith("answer")]
    report["summary"] = {"answer_r2_median": round(float(np.median(r2a)), 4),
                         "answer_r2_min": round(float(np.min(r2a)), 4),
                         "answer_convex_fraction": round(float(np.mean(convex)), 3),
                         "n_task_surfaces": len(r2a)}
    print("summary:", report["summary"], flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "cartography_analysis.json").write_text(json.dumps(report, indent=1))
    print("wrote", out / "cartography_analysis.json")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--config", required=True)
    r.add_argument("--out", required=True); r.set_defaults(func=cmd_run)
    an = sub.add_parser("analyze"); an.add_argument("--raw", required=True)
    an.add_argument("--out", required=True); an.set_defaults(func=cmd_analyze)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
