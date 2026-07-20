#!/usr/bin/env python3
"""Do the atlas coefficients help? (the actionable-verdict test).

Every accept returns not just a verdict but the merge coefficient that achieves
it. That coefficient has never been evaluated downstream: all validation merges
so far use equal weights. This evaluates each subset at both the atlas-optimal
coefficient (the minimax argmin over the answer-token jets) and equal weights,
on the same held-out items, and compares accuracy and answer-token loss.

Consumes a precomputed merge list (subset, weights, tag) so the coefficient math
stays on CPU and only the evaluation is on GPU.

Run:  python expF_coefficient.py --config expD_panel_m8.json --merges merges.json --out results_expF
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from expA_answer_token_panel import extract_pred, masked_ce  # noqa: E402


def cmd_run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    cfg = json.loads(Path(args.config).read_text())
    merges = json.loads(Path(args.merges).read_text())
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
    regexes = {s["name"]: s.get("answer_regex", r"\b([A-Z])\b") for s in specs}
    n_items = int(cfg.get("n_items", 40)); max_new = int(cfg.get("max_new_tokens", 6))
    max_length = int(cfg.get("max_length", 384))

    def prompt_of(defn, inp):
        msgs = [{"role": "user", "content": defn + "\n\n" + inp}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return defn + "\n\n" + inp + "\n"

    def measure(task, regex):
        defn, inst = task["definition"], task["instances"][:n_items]
        corr = la = 0.0
        for it in inst:
            p = prompt_of(defn, it["input"]); gold = it["output"][0]
            pids = tok(p, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
            with torch.no_grad():
                g = peft.generate(input_ids=pids, max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
            pred = extract_pred(tok.decode(g[0, pids.shape[1]:], skip_special_tokens=True), regex)
            gn = gold.strip().upper()
            corr += float(pred == gn or pred == gn[:1])
            _, a = masked_ce(peft, tok, p, gold, device, max_length)
            la += a
        k = len(inst)
        return corr / k, la / k

    rows, done = [], set()
    ck = out / "coeff_rows.jsonl"
    if ck.exists():
        for line in ck.read_text().splitlines():
            if line.strip():
                # Key on the full subset string, not the tag: distinct subsets
                # can share a tag (task1391 and task391 both shorten to "391").
                r = json.loads(line); rows.append(r); done.add((r["tag"], r["subset"], r["task"]))
    f = open(ck, "a"); t0 = time.time()
    for mg in merges:
        members = mg["members"]; w = mg["weights"]; tag = mg["tag"]
        active = [(names[i], w[i]) for i in range(len(names)) if w[i] > 1e-9]
        materialized = False
        for host in members:
            if (tag, "+".join(members), host) in done:
                continue
            if not materialized:
                if len(active) == 1:
                    peft.set_adapter(active[0][0])
                else:
                    peft.add_weighted_adapter([a for a, _ in active], [x for _, x in active],
                                              "mergeF", combination_type="linear")
                    peft.set_adapter("mergeF")
                materialized = True
            acc, la = measure(tasks[host], regexes[host])
            r = {"tag": tag, "variant": mg["variant"], "subset": "+".join(members),
                 "task": host, "accuracy": acc, "L_answer": la}
            rows.append(r); f.write(json.dumps(r) + "\n"); f.flush()
            print(f"  {tag:26s} {host:9s} acc={acc:.3f} La={la:.3f} t={int(time.time()-t0)}s", flush=True)
        if materialized and len(active) > 1:
            peft.delete_adapter("mergeF")
    (out / "coeff_rows.json").write_text(json.dumps({"rows": rows}, indent=1))
    print("wrote", out / "coeff_rows.json", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True); ap.add_argument("--merges", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
