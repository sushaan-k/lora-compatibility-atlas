#!/usr/bin/env python3
"""Generative boundary test (C3): does answer-token loss track quality on
free-form generated answers, not just constrained classification labels?

Four short-answer Natural Instructions tasks (trivia, analogy, math, quartz),
scored by normalized exact-match and token-F1 on generated answers rather than a
fixed label regex. Fit the jets with expG_operators.py (answer-token CE, needs no
regex); this script validates every pair and triple by generating and scoring,
writing rows whose accuracy field is the exact-match rate so expD_answer_panel.py
analyze applies unchanged.

  python expG_operators.py fit --config gen_panel.json --operator linear --out results_gen
  python expK_generative.py validate --config gen_panel.json --out results_gen
"""
from __future__ import annotations

import argparse
import json
import re
import string
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from expA_answer_token_panel import masked_ce  # noqa: E402
from expD_answer_panel import all_merges  # noqa: E402


def norm(s):
    s = s.strip().lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred, gold):
    p, g = norm(pred).split(), norm(gold).split()
    if not p or not g:
        return float(p == g)
    common = {}
    for t in p:
        common[t] = min(p.count(t), g.count(t))
    ncommon = sum(common.values())
    if ncommon == 0:
        return 0.0
    prec = ncommon / len(p); rec = ncommon / len(g)
    return 2 * prec * rec / (prec + rec)


def cmd_validate(args):
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
    specs = cfg["adapters"]; names = [s["name"] for s in specs]
    tasks = {s["name"]: json.loads(Path(s["task_json"]).read_text()) for s in specs}
    peft = None
    for s in specs:
        if peft is None:
            peft = PeftModel.from_pretrained(model, s["repo"], adapter_name=s["name"], is_trainable=False)
        else:
            peft.load_adapter(s["repo"], adapter_name=s["name"])
    peft.eval()
    n_items = int(cfg.get("n_items", 40)); max_new = int(cfg.get("max_new_tokens", 10))
    max_length = int(cfg.get("max_length", 384))

    def prompt_of(defn, inp):
        try:
            return tok.apply_chat_template([{"role": "user", "content": defn + "\n\n" + inp}],
                                           tokenize=False, add_generation_prompt=True)
        except Exception:
            return defn + "\n\n" + inp + "\n"

    def measure(task):
        defn, inst = task["definition"], task["instances"][:n_items]
        em = f1 = la = 0.0
        for it in inst:
            p = prompt_of(defn, it["input"]); gold = it["output"][0]
            pids = tok(p, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
            with torch.no_grad():
                g = peft.generate(input_ids=pids, max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
            gen = tok.decode(g[0, pids.shape[1]:], skip_special_tokens=True).split("\n")[0]
            em += float(norm(gen) == norm(gold))
            f1 += token_f1(gen, gold)
            _, a = masked_ce(peft, tok, p, gold, device, max_length); la += a
        k = len(inst); return em / k, f1 / k, la / k

    work = []
    for nm in names:
        work.append((nm, "base", {})); work.append((nm, "alone", {nm: 1.0}))
    for label, w in all_merges(names):
        for member in label.split("+"):
            work.append((member, label, w))

    rows, done = [], set()
    ck = out / "gen_rows.jsonl"
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
                em, f1, la = measure(tasks[host])
        else:
            key = tuple(sorted(w.items()))
            if key != cur:
                if cur is not None and len(dict(cur)) > 1:
                    try:
                        peft.delete_adapter("mergeK")
                    except Exception:
                        pass
                active = [(nm, wt) for nm, wt in w.items()]
                if len(active) == 1:
                    peft.set_adapter(active[0][0])
                else:
                    peft.add_weighted_adapter([a for a, _ in active], [x for _, x in active],
                                              "mergeK", combination_type="linear")
                    peft.set_adapter("mergeK")
                cur = key
            em, f1, la = measure(tasks[host])
        r = {"task": host, "merge": label, "accuracy": em, "f1": f1,
             "L_input": 0.0, "L_answer": la}
        rows.append(r); f.write(json.dumps(r) + "\n"); f.flush()
        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(work)}] {host} {label} em={em:.2f} f1={f1:.2f} La={la:.3f} "
                  f"t={int(time.time()-t0)}s", flush=True)
    (out / "validation_rows.json").write_text(json.dumps({"rows": rows}, indent=1))
    print("wrote", out / "validation_rows.json", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate"); v.add_argument("--config", required=True)
    v.add_argument("--out", required=True); v.set_defaults(func=cmd_validate)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
