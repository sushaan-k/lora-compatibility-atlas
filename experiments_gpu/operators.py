#!/usr/bin/env python3
"""Answer-token screening under non-linear merge operators (C2)."""
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
from panel import extract_pred, masked_ce  # noqa: E402
from validate import all_merges  # noqa: E402


def merge_kwargs(operator, density):
    if operator == "linear":
        return dict(combination_type="linear")
    if operator == "ties":
        return dict(combination_type="ties", density=density)
    if operator == "dare_ties":
        return dict(combination_type="dare_ties", density=density)
    raise ValueError(operator)


def setup(cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
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
    regexes = {s["name"]: s.get("answer_regex", r"\b([A-Z])\b") for s in specs}
    return peft, tok, device, names, tasks, regexes


def prompt_of(tok, defn, inp):
    msgs = [{"role": "user", "content": defn + "\n\n" + inp}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return defn + "\n\n" + inp + "\n"


def materialize(peft, names, weights, mkw):
    active = [(names[i], weights[i]) for i in range(len(names)) if weights[i] > 1e-9]
    if len(active) == 1:
        peft.set_adapter(active[0][0]); return None
    peft.add_weighted_adapter([a for a, _ in active], [w for _, w in active], "mergeG", **mkw)
    peft.set_adapter("mergeG"); return "mergeG"


def cmd_fit(args):
    import torch
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    peft, tok, device, names, tasks, regexes = setup(cfg)
    n = len(names); n_items = int(cfg.get("n_items", 40)); max_length = int(cfg.get("max_length", 384))
    mkw = merge_kwargs(args.operator, args.density)

    # singles + pairs design (36)
    grid = []
    for i in range(n):
        w = [0.0] * n; w[i] = 1.0; grid.append(w)
    for i, j in itertools.combinations(range(n), 2):
        w = [0.0] * n; w[i] = w[j] = 0.5; grid.append(w)

    def ce_all(weight):
        tag = materialize(peft, names, weight, mkw)
        row = {}
        for nm in names:
            defn, inst = tasks[nm]["definition"], tasks[nm]["instances"][:n_items]
            tot = 0.0
            for it in inst:
                p = prompt_of(tok, defn, it["input"]); gold = it["output"][0]
                pids = tok(p, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
                full = tok(p + " " + gold, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
                lab = full.clone(); lab[:, :pids.shape[1]] = -100
                with torch.no_grad():
                    tot += peft(input_ids=full, labels=lab).loss.item()
            row[nm] = tot / len(inst)
        if tag:
            peft.delete_adapter(tag)
        return row

    ck = out / "fit_points.jsonl"; done = {}
    if ck.exists():
        for line in ck.read_text().splitlines():
            if line.strip():
                r = json.loads(line); done[r["i"]] = r["ce"]
    f = open(ck, "a"); t0 = time.time()
    vals = []
    for i, w in enumerate(grid):
        if i in done:
            vals.append(done[i]); continue
        ce = ce_all(w); vals.append(ce)
        f.write(json.dumps({"i": i, "ce": ce}) + "\n"); f.flush()
        if (i + 1) % 5 == 0:
            print(f"  fit {i+1}/{len(grid)} t={int(time.time()-t0)}s", flush=True)
    np.savez(out / "answer_jets_raw.npz", grid=np.array(grid),
             values=np.array([[c[nm] for nm in names] for c in vals]).T, names=np.array(names))
    print("wrote", out / "answer_jets_raw.npz", flush=True)
    return 0


def cmd_validate(args):
    import torch
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    peft, tok, device, names, tasks, regexes = setup(cfg)
    n = len(names); n_items = int(cfg.get("n_items", 40))
    max_new = int(cfg.get("max_new_tokens", 6)); max_length = int(cfg.get("max_length", 384))
    mkw = merge_kwargs(args.operator, args.density)

    def measure(task, regex):
        defn, inst = task["definition"], task["instances"][:n_items]
        corr = la = 0.0
        for it in inst:
            p = prompt_of(tok, defn, it["input"]); gold = it["output"][0]
            pids = tok(p, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
            with torch.no_grad():
                g = peft.generate(input_ids=pids, max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
            pred = extract_pred(tok.decode(g[0, pids.shape[1]:], skip_special_tokens=True), regex)
            gn = gold.strip().upper(); corr += float(pred == gn or pred == gn[:1])
            _, a = masked_ce(peft, tok, p, gold, device, max_length); la += a
        k = len(inst); return corr / k, la / k

    work = []
    for nm in names:
        work.append((nm, "base", {})); work.append((nm, "alone", {nm: 1.0}))
    for label, w in all_merges(names):
        for member in label.split("+"):
            work.append((member, label, w))

    rows, done = [], set()
    ck = out / "rows.jsonl"
    if ck.exists():
        for line in ck.read_text().splitlines():
            if line.strip():
                r = json.loads(line); rows.append(r); done.add((r["task"], r["merge"]))
    f = open(ck, "a"); t0 = time.time(); cur = None
    for idx, (host, label, w) in enumerate(work):
        if (host, label) in done:
            continue
        if not w:
            with peft.disable_adapter():
                acc, la = measure(tasks[host], regexes[host])
        else:
            key = tuple(sorted(w.items()))
            if key != cur:
                if cur is not None and len(dict(cur)) > 1:
                    try:
                        peft.delete_adapter("mergeG")
                    except Exception:
                        pass
                wv = [w.get(nm, 0.0) for nm in names]
                materialize(peft, names, wv, mkw); cur = key
            acc, la = measure(tasks[host], regexes[host])
        r = {"task": host, "merge": label, "accuracy": acc, "L_input": 0.0, "L_answer": la}
        rows.append(r); f.write(json.dumps(r) + "\n"); f.flush()
        if (idx + 1) % 10 == 0:
            print(f"  val [{idx+1}/{len(work)}] {host} {label} acc={acc:.2f} La={la:.3f} "
                  f"t={int(time.time()-t0)}s", flush=True)
    (out / "validation_rows.json").write_text(json.dumps({"rows": rows}, indent=1))
    print("wrote", out / "validation_rows.json", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("fit", cmd_fit), ("validate", cmd_validate)):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True); p.add_argument("--out", required=True)
        p.add_argument("--operator", default="ties")
        p.add_argument("--density", type=float, default=0.5)
        p.set_defaults(func=fn)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
