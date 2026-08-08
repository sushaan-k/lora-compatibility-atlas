#!/usr/bin/env python3
"""Fifteen-query deployed test of atlas coefficient selection (m=16)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

ORDERS = (4, 5, 6)
PER_ORDER = 5
SEED = 0


def cmd_plan(_):
    np.seterr(all="ignore")
    sys.path.insert(0, str(REPO / "scripts"))
    from bicore import load_jets, rho, w2c
    a, b, H = load_jets("m16")
    n = len(a)
    names = [s["name"] for s in json.loads(
        (REPO / "experiments_gpu" / "m16_panel.json").read_text())["adapters"]]

    def wvec(c):
        w = np.zeros(n); w[:n - 1] = c; w[n - 1] = 1.0 - c.sum(); return w

    def jmax(A, w):
        c = w2c(w)
        return max(float(a[t] + b[t] @ c + 0.5 * c @ H[t] @ c) for t in A)

    rng = np.random.default_rng(SEED)
    queries, seen = [], set()
    for order in ORDERS:
        for _ in range(PER_ORDER):
            while True:
                Q = tuple(sorted(rng.choice(n, size=order, replace=False).tolist()))
                if Q not in seen:
                    seen.add(Q); break
            queries.append(Q)

    merges, records = [], []
    for qi, Q in enumerate(queries):
        R, cstar = rho(a, b, H, Q, Q, n)
        w_opt = wvec(cstar)
        w_eq = np.zeros(n); w_eq[list(Q)] = 1.0 / len(Q)
        members = [names[i] for i in Q]
        tag = f"q{qi:02d}_o{len(Q)}"
        merges.append({"tag": tag + "_opt", "variant": "atlas_opt",
                       "members": members, "weights": [float(x) for x in w_opt]})
        merges.append({"tag": tag + "_eq", "variant": "equal",
                       "members": members, "weights": [float(x) for x in w_eq]})
        records.append({"tag": tag, "members": members, "order": len(Q),
                        "rho_opt": float(R), "fitted_worst_equal": jmax(Q, w_eq)})
        print(tag, members, "rho", round(R, 3), "eq", round(jmax(Q, w_eq), 3), flush=True)

    out = REPO / "results_public" / "intervention_15q"
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(
        {"panel": "mistral_m16_answer", "seed": SEED, "orders": list(ORDERS),
         "per_order": PER_ORDER, "names": names, "queries": records,
         "merges": merges}, indent=1))
    print("wrote", out / "plan.json")
    return 0


def cmd_run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    sys.path.insert(0, str(HERE))
    from panel import extract_pred, masked_ce

    cfg = json.loads(Path(args.config).read_text())
    plan = json.loads(Path(args.plan).read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=torch.bfloat16,
                                                 device_map=device)
    specs = {s["name"]: s for s in cfg["adapters"]}
    names = plan["names"]
    peft = None
    for nm in names:
        s = specs[nm]
        if peft is None:
            peft = PeftModel.from_pretrained(model, s["repo"], adapter_name=nm, is_trainable=False)
        else:
            peft.load_adapter(s["repo"], adapter_name=nm)
    peft.eval()
    n_items = int(cfg.get("n_items", 30)); max_new = int(cfg.get("max_new_tokens", 6))
    max_length = int(cfg.get("max_length", 384))

    def prompt_of(defn, inp):
        msgs = [{"role": "user", "content": defn + "\n\n" + inp}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return defn + "\n\n" + inp + "\n"

    def measure(nm):
        s = specs[nm]; task = json.loads(Path(s["task_json"]).read_text())
        regex = s.get("answer_regex", r"\b([A-Z])\b"); defn = task["definition"]
        held = task["instances"][n_items:2 * n_items]  # disjoint from the fit split
        corr = la = 0.0
        for it in held:
            p = prompt_of(defn, it["input"]); gold = it["output"][0]
            pids = tok(p, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
            with torch.no_grad():
                g = peft.generate(input_ids=pids, max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
            pred = extract_pred(tok.decode(g[0, pids.shape[1]:], skip_special_tokens=True), regex)
            gn = gold.strip().upper()
            corr += float(pred == gn or pred == gn[:1])
            _, ans = masked_ce(peft, tok, p, gold, device, max_length)
            la += ans
        k = len(held)
        return corr / k, la / k

    rows, done = [], set()
    ck = out / "deployed_rows.jsonl"
    if ck.exists():
        for line in ck.read_text().splitlines():
            if line.strip():
                r = json.loads(line); rows.append(r); done.add((r["tag"], r["task"]))
    f = open(ck, "a"); t0 = time.time()
    for mg in plan["merges"]:
        members, w, tag = mg["members"], mg["weights"], mg["tag"]
        active = [(names[i], w[i]) for i in range(len(names)) if w[i] > 1e-9]
        materialized = False
        for host in members:
            if (tag, host) in done:
                continue
            if not materialized:
                if len(active) == 1:
                    peft.set_adapter(active[0][0])
                else:
                    peft.add_weighted_adapter([x for x, _ in active], [v for _, v in active],
                                              "mergeI", combination_type="linear")
                    peft.set_adapter("mergeI")
                materialized = True
            acc, la = measure(host)
            r = {"tag": tag, "variant": mg["variant"], "task": host,
                 "accuracy": acc, "L_answer": la}
            rows.append(r); f.write(json.dumps(r) + "\n"); f.flush()
            print(f"  {tag:14s} {host:9s} acc={acc:.3f} La={la:.3f} t={int(time.time()-t0)}s",
                  flush=True)
        if materialized and len(active) > 1:
            peft.delete_adapter("mergeI")
    (out / "deployed_rows.json").write_text(json.dumps({"rows": rows}, indent=1))
    print("wrote", out / "deployed_rows.json", flush=True)
    return 0


def cmd_analyze(args):
    plan = json.loads(Path(args.plan).read_text())
    rows = json.loads(Path(args.rows).read_text())["rows"]
    by = {}
    for r in rows:
        base = r["tag"].rsplit("_", 1)[0]
        by.setdefault((base, r["variant"]), []).append(r)
    queries = []
    for q in plan["queries"]:
        opt = by[(q["tag"], "atlas_opt")]; eq = by[(q["tag"], "equal")]
        rec = {"tag": q["tag"], "order": q["order"], "members": q["members"],
               "rho_opt_fitted": q["rho_opt"], "fitted_worst_equal": q["fitted_worst_equal"],
               "deployed_worst_opt": max(r["L_answer"] for r in opt),
               "deployed_worst_eq": max(r["L_answer"] for r in eq),
               "mean_acc_opt": float(np.mean([r["accuracy"] for r in opt])),
               "mean_acc_eq": float(np.mean([r["accuracy"] for r in eq]))}
        rec["worst_improved"] = bool(rec["deployed_worst_opt"] < rec["deployed_worst_eq"])
        queries.append(rec)
    d_worst = [q["deployed_worst_eq"] - q["deployed_worst_opt"] for q in queries]
    d_acc = [q["mean_acc_opt"] - q["mean_acc_eq"] for q in queries]
    summary = {"n_queries": len(queries),
               "n_worst_improved": int(sum(q["worst_improved"] for q in queries)),
               "median_worst_delta_nats": float(np.median(d_worst)),
               "worst_delta_range": [float(min(d_worst)), float(max(d_worst))],
               "median_acc_delta": float(np.median(d_acc)),
               "queries": queries}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "queries"}, indent=1))
    print("wrote", out / "summary.json")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan").set_defaults(func=cmd_plan)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True); r.add_argument("--plan", required=True)
    r.add_argument("--out", required=True); r.set_defaults(func=cmd_run)
    an = sub.add_parser("analyze")
    an.add_argument("--plan", required=True); an.add_argument("--rows", required=True)
    an.add_argument("--out", required=True); an.set_defaults(func=cmd_analyze)
    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
