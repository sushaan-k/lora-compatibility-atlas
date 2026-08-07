#!/usr/bin/env python3
"""Answer-token jets on the shared Mistral 5-adapter chart, validated against the measured accuracies of Experime"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "peft_atlas_lite"))


def weights_to_coords(w):
    return np.asarray(w[:-1], dtype=float)


def jet_value(alpha, b, H, c):
    c = np.asarray(c, dtype=float)
    return float(alpha + b @ c + 0.5 * c @ H @ c)


def fit_jets(grid, values_per_task, coord_dim):
    from run_peft_atlas_lite import fit_quadratic_model, hessian_from_quadratic_coeff
    jets = {}
    for name, vals in values_per_task.items():
        coeff, info = fit_quadratic_model(list(grid), vals, project_psd=True)
        H = np.asarray(hessian_from_quadratic_coeff(coeff, coord_dim), float)
        jets[name] = {"alpha": float(coeff[0]), "b": np.array(coeff[1:1 + coord_dim], float), "H": H,
                      "psd_projected": bool(info.get("psd_projected", False))}
    return jets


def shared_projector_residuals(jets, names):
    Hs = [jets[n]["H"] for n in names]
    H_bar = np.mean(np.stack(Hs), axis=0)
    evals, evecs = np.linalg.eigh(H_bar)
    V = evecs[:, np.argsort(evals)[::-1]]
    L_F = max(np.linalg.norm(H, "fro") for H in Hs)
    out = []
    for r in range(1, Hs[0].shape[0] + 1):
        Pi = V[:, :r] @ V[:, :r].T
        res = max(np.linalg.norm(H - Pi @ H @ Pi, "fro") for H in Hs) / L_F
        out.append({"r": r, "strict_max_residual": float(res)})
    return out


def auroc(labels, scores):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


# ---------------------------------------------------------------- fit (GPU)

def cmd_fit(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from run_peft_atlas_lite import simplex_grid

    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=torch.bfloat16, device_map=device)
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
    n_items = int(cfg.get("n_items", 40))
    max_length = int(cfg.get("max_length", 384))
    n = len(names)
    grid = simplex_grid(n, step=args.step, min_weight=0.0)
    n_d = 1 + (n - 1) + (n - 1) * n // 2
    print(f"n={n} d={n-1} grid={len(grid)} (N_d={n_d})", flush=True)

    def prompt_of(defn, inp):
        msgs = [{"role": "user", "content": defn + "\n\n" + inp}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return defn + "\n\n" + inp + "\n"

    def answer_ce(task):
        """Mean CE over answer tokens only, prompt masked, over the held-out items."""
        defn, inst = task["definition"], task["instances"][:n_items]
        tot = 0.0
        for it in inst:
            p = prompt_of(defn, it["input"]); gold = it["output"][0]
            pids = tok(p, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
            full = tok(p + " " + gold, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
            lab = full.clone(); lab[:, :pids.shape[1]] = -100
            with torch.no_grad():
                tot += peft(input_ids=full, labels=lab).loss.item()
        return tot / len(inst)

    ckpt_path = out / "fit_points.jsonl"
    done = 0
    values = {nm: [] for nm in names}
    if ckpt_path.exists():
        for line in ckpt_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                for nm in names:
                    values[nm].append(rec["values"][nm])
                done += 1
        print(f"resuming: {done}/{len(grid)}", flush=True)
    ckpt = open(ckpt_path, "a")
    t0 = time.time()
    for i, w in enumerate(grid):
        if i < done:
            continue
        active = [(nm, wt) for nm, wt in zip(names, w) if wt > 0]
        if len(active) == 1:
            peft.set_adapter(active[0][0])
        else:
            peft.add_weighted_adapter([nm for nm, _ in active], [wt for _, wt in active],
                                      "mergeC", combination_type="linear")
            peft.set_adapter("mergeC")
        rec = {nm: answer_ce(tasks[nm]) for nm in names}
        if len(active) > 1:
            peft.delete_adapter("mergeC")
        for nm in names:
            values[nm].append(rec[nm])
        ckpt.write(json.dumps({"i": i, "weights": list(w), "values": rec}) + "\n")
        ckpt.flush()
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(grid)} t={int(time.time()-t0)}s", flush=True)
    np.savez(out / "answer_jets_raw.npz", grid=np.array(grid),
             values=np.array([values[nm] for nm in names]), names=np.array(names))
    print("wrote", out / "answer_jets_raw.npz", flush=True)
    return 0


# ------------------------------------------------------------- analyze (CPU)

def cmd_analyze(args):
    z = np.load(args.raw, allow_pickle=True)
    grid = z["grid"]; names = [str(x) for x in z["names"]]; vals = z["values"]
    d = len(names) - 1
    jets = fit_jets([list(w) for w in grid], {nm: list(vals[i]) for i, nm in enumerate(names)}, d)
    residuals = shared_projector_residuals(jets, names)

    panel = json.loads(Path(args.panel).read_text())
    rows = panel["rows"]
    base = {r["task"]: r for r in rows if r["merge"] == "base"}
    alone = {r["task"]: r for r in rows if r["merge"] == "alone"}
    mg = [r for r in rows if r["merge"] not in ("base", "alone")]
    tau = args.tau_acc

    events = []
    for r in mg:
        members = r["merge"].split("+")
        host = r["task"]
        w = np.zeros(len(names))
        for mname in members:
            w[names.index(mname)] = 1.0 / len(members)
        c = weights_to_coords(w)
        pred_answer_loss = jet_value(jets[host]["alpha"], jets[host]["b"], jets[host]["H"], c)
        # accuracy-retention label from A's measured accuracies
        b_acc, a_acc = base[host]["accuracy"], alone[host]["accuracy"]
        retained = r["accuracy"] >= b_acc + tau * (a_acc - b_acc)
        events.append({"task": host, "merge": r["merge"], "pred_answer_loss": pred_answer_loss,
                       "measured_answer_loss": r["L_answer"], "accuracy": r["accuracy"],
                       "retained_acc": bool(retained)})

    # screening: higher predicted answer loss => less likely retained
    labels = [e["retained_acc"] for e in events]
    scores = [-e["pred_answer_loss"] for e in events]
    a_pred = auroc(labels, scores)
    a_meas = auroc(labels, [-e["measured_answer_loss"] for e in events])
    # jet prediction quality
    import math
    px = [e["pred_answer_loss"] for e in events]; mx = [e["measured_answer_loss"] for e in events]
    mp, mm = np.mean(px), np.mean(mx)
    r_pred = float(np.sum((np.array(px)-mp)*(np.array(mx)-mm)) /
                   (np.linalg.norm(np.array(px)-mp)*np.linalg.norm(np.array(mx)-mm)))
    result = {
        "n_events": len(events), "n_retained": int(sum(labels)),
        "tau_acc": tau,
        "auroc_predicted_jets_vs_acc_labels": a_pred,
        "auroc_measured_answer_loss_vs_acc_labels": a_meas,
        "r_predicted_vs_measured_answer_loss": r_pred,
        "psd_projected_tasks": [nm for nm in names if jets[nm]["psd_projected"]],
        "shared_projector_residuals": residuals,
        "events": events,
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "answer_jets_analysis.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "events"}, indent=2))
    print("wrote", out / "answer_jets_analysis.json")
    return 0


def cmd_selftest(_):
    # synthetic: quadratic answer-loss surfaces; verify jets recover them and the
    # AUROC pipeline orders retained/non-retained correctly
    rng = np.random.default_rng(0)
    names = [f"t{i}" for i in range(5)]
    d = 4
    from run_peft_atlas_lite import simplex_grid
    grid = simplex_grid(5, step=0.25, min_weight=0.0)
    true = {}
    for i, nm in enumerate(names):
        U = np.linalg.qr(rng.standard_normal((d, 2)))[0]
        H = U @ np.diag(rng.uniform(1, 3, 2)) @ U.T
        b = U @ (0.2 * rng.standard_normal(2))
        true[nm] = (float(rng.uniform(0.2, 0.6)), b, H)
    vals = {nm: [jet_value(a, b, H, weights_to_coords(w)) + rng.normal(0, 1e-4)
                 for w in grid] for nm, (a, b, H) in true.items()}
    jets = fit_jets([list(w) for w in grid], vals, d)
    errs = []
    for nm in names:
        a, b, H = true[nm]
        for w in grid[:20]:
            c = weights_to_coords(w)
            errs.append(abs(jet_value(a, b, H, c) - jet_value(jets[nm]["alpha"], jets[nm]["b"], jets[nm]["H"], c)))
    assert max(errs) < 1e-2, max(errs)
    labels = [True]*10 + [False]*10
    scores = [-1.0]*10 + [-2.0]*10  # retained events have lower loss => higher -loss
    assert auroc(labels, scores) == 1.0
    print("selftest OK: jets recover synthetic answer-loss surfaces; AUROC orientation correct")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("--config", required=True)
    f.add_argument("--out", required=True); f.add_argument("--step", type=float, default=0.25)
    f.set_defaults(func=cmd_fit)
    an = sub.add_parser("analyze"); an.add_argument("--raw", required=True)
    an.add_argument("--panel", required=True); an.add_argument("--out", required=True)
    an.add_argument("--tau-acc", type=float, default=0.5); an.set_defaults(func=cmd_analyze)
    st = sub.add_parser("selftest"); st.set_defaults(func=cmd_selftest)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
