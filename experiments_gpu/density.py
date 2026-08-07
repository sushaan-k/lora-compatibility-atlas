#!/usr/bin/env python3
"""The design-density law (A2): jet fidelity as a function of design redundancy."""
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

from surfaces import (fit_jets, jet_value,  # noqa: E402
                              weights_to_coords)


def simplex_singles_pairs(n):
    pts = []
    for i in range(n):
        w = [0.0] * n; w[i] = 1.0; pts.append(w)
    for i, j in itertools.combinations(range(n), 2):
        w = [0.0] * n; w[i] = w[j] = 0.5; pts.append(w)
    return pts


def equal_weight(members, n):
    w = [0.0] * n
    for m in members:
        w[m] = 1.0 / len(members)
    return w


def cmd_run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=dtype,
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
    n_items = int(cfg.get("n_items", 40)); n = len(names)
    max_length = int(cfg.get("max_length", 384))

    def prompt_of(defn, inp):
        msgs = [{"role": "user", "content": defn + "\n\n" + inp}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return defn + "\n\n" + inp + "\n"

    def answer_ce_all(weight):
        active = [(names[i], weight[i]) for i in range(n) if weight[i] > 1e-9]
        if len(active) == 1:
            peft.set_adapter(active[0][0])
        else:
            peft.add_weighted_adapter([a for a, _ in active], [w for _, w in active],
                                      "mergeE", combination_type="linear")
            peft.set_adapter("mergeE")
        row = {}
        for nm in names:
            defn, inst = tasks[nm]["definition"], tasks[nm]["instances"][:n_items]
            tot = 0.0
            for it in inst:
                p = prompt_of(defn, it["input"]); gold = it["output"][0]
                pids = tok(p, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
                full = tok(p + " " + gold, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
                lab = full.clone(); lab[:, :pids.shape[1]] = -100
                with torch.no_grad():
                    tot += peft(input_ids=full, labels=lab).loss.item()
            row[nm] = tot / len(inst)
        if len(active) > 1:
            peft.delete_adapter("mergeE")
        return row

    # design pool: singles+pairs (36) then all 56 triples, in a fixed order
    design = simplex_singles_pairs(n)
    triples = [equal_weight(list(t), n) for t in itertools.combinations(range(n), 3)]
    rng = np.random.default_rng(11)
    order = rng.permutation(len(triples)).tolist()
    design += [triples[i] for i in order]           # 36 + 56 = 92 pool points
    # held-out test set: 40 off-design 4- and 5-way merges
    quads = list(itertools.combinations(range(n), 4))
    fives = list(itertools.combinations(range(n), 5))
    rng2 = np.random.default_rng(23)
    test = ([equal_weight(list(quads[i]), n) for i in rng2.permutation(len(quads))[:25]]
            + [equal_weight(list(fives[i]), n) for i in rng2.permutation(len(fives))[:15]])

    ck = out / "density_points.jsonl"
    done = {}
    if ck.exists():
        for line in ck.read_text().splitlines():
            if line.strip():
                r = json.loads(line); done[r["key"]] = r["ce"]
    f = open(ck, "a")
    t0 = time.time()

    def evaluate(tag, pts):
        vals = []
        for i, w in enumerate(pts):
            key = f"{tag}:{i}"
            if key in done:
                vals.append(done[key]); continue
            ce = answer_ce_all(w)
            done[key] = ce; vals.append(ce)
            f.write(json.dumps({"key": key, "weight": w, "ce": ce}) + "\n"); f.flush()
            if (len(done)) % 5 == 0:
                print(f"  {len(done)} evals  t={int(time.time()-t0)}s", flush=True)
        return vals

    print(f"design pool {len(design)}, test {len(test)}", flush=True)
    design_ce = evaluate("design", design)
    test_ce = evaluate("test", test)
    np.savez(out / "density_raw.npz",
             design=np.array(design), design_ce=np.array([[c[nm] for nm in names] for c in design_ce]),
             test=np.array(test), test_ce=np.array([[c[nm] for nm in names] for c in test_ce]),
             names=np.array(names))
    print("wrote", out / "density_raw.npz", flush=True)
    return 0


def auroc(y, s):
    pos = [x for x, l in zip(s, y) if l]; neg = [x for x, l in zip(s, y) if not l]
    if not pos or not neg:
        return None
    return float(sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg)))


def cmd_analyze(args):
    z = np.load(Path(args.raw) / "density_raw.npz", allow_pickle=True)
    design = z["design"]; design_ce = z["design_ce"]
    test = z["test"]; test_ce = z["test_ce"]; names = [str(x) for x in z["names"]]
    n = len(names)
    n_sp = n + n * (n - 1) // 2  # 36
    # per test merge: for each member task, the measured answer loss and the
    # worst-member retention deficit (screening label + score analogue)
    test_members = [[i for i in range(n) if test[k][i] > 1e-9] for k in range(len(test))]

    levels = [n_sp, n_sp + 14, n_sp + 28, n_sp + 42, n_sp + 56]
    curve = []
    for L in levels:
        grid = [list(design[i]) for i in range(L)]
        vals = {nm: [design_ce[i][j] for i in range(L)] for j, nm in enumerate(names)}
        jets = fit_jets(grid, vals, n - 1)
        pred, meas = [], []
        for k in range(len(test)):
            for i in test_members[k]:
                c = weights_to_coords(test[k])
                pred.append(jet_value(jets[names[i]]["alpha"], jets[names[i]]["b"],
                                      jets[names[i]]["H"], c))
                meas.append(test_ce[k][i])
        r = float(np.corrcoef(pred, meas)[0, 1])
        redundancy = L / n_sp
        curve.append({"design_size": L, "redundancy": round(redundancy, 2),
                      "n_test_member_events": len(pred),
                      "r_pred_vs_measured": round(r, 4)})
        print(f"  design {L} ({redundancy:.2f}x): r(pred,measured) = {r:.4f} "
              f"over {len(pred)} held-out member events", flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "density_analysis.json").write_text(json.dumps(
        {"levels": curve, "n_singles_pairs": n_sp, "d": n - 1}, indent=2))
    print("wrote", out / "density_analysis.json")
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
