#!/usr/bin/env python3
"""Answer-token retention across a panel (generalises scripts/mistral_probe_contrast.py).

Reviewer ask: the panels score input-prompt loss, which the paper's own single-task
probe shows does not track accuracy (Pearson +0.43) while answer-token loss does
(-0.96). Establish the answer-token fix on a PANEL of answer-bearing tasks, not one.

For every merge and every constituent task, measure on the same held-out items:
  L_input   mean CE over the input-prompt tokens (what the panels use),
  L_answer  mean CE over the answer tokens only, prompt masked (the fix),
  accuracy  exact-match generation.
Report per-task and pooled Pearson r(L_input, acc) and r(L_answer, acc). The claim
holds at panel scale iff L_answer tracks accuracy (strongly negative) while L_input
does not (near zero), replicating the single-task result across the panel.

Config JSON:
{
  "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
  "dtype": "bfloat16", "max_new_tokens": 6, "n_items": 40, "n_merges_per_task": 12,
  "answer_regex": "\\\\b([ECN])\\\\b",
  "adapters": [
    {"name": "task190", "repo": "org/mistral-snli-lora", "task_json": "/root/atlas/task190_natural.json"},
    ...
  ]
}
task_json = {"definition": str, "instances": [{"input": str, "output": [str, ...]}, ...]}.
Run:  python experiments_gpu/expA_answer_token_panel.py --config expA_panel.json --out results_public/answer_token_panel
Self-test (no GPU/data):  python experiments_gpu/expA_answer_token_panel.py --selftest
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path


# ---- GPU-free logic (self-tested) ------------------------------------------

def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (sx * sy) if sx * sy else float("nan")


def answer_label_mask(prompt_ids_len, full_ids_len):
    """Indices [prompt_ids_len, full_ids_len) are answer tokens; earlier are masked
    to -100. Returns a label mask list for the shifted-by-one CE convention used by
    the pipeline (per_prompt_nll shifts logits/labels by one)."""
    labels = [-100] * full_ids_len
    for i in range(prompt_ids_len, full_ids_len):
        labels[i] = 1  # keep
    return labels


def extract_pred(text, answer_regex):
    m = re.search(answer_regex, text.strip().upper())
    return m.group(1) if m else "?"


def retained(merged_loss, base_loss, single_loss, tau, min_gain):
    """Retention label under the paper's normalized-deficit rule: retained iff the
    merge keeps at least a tau fraction of the single-adapter gain over base."""
    gain = base_loss - single_loss
    if gain < min_gain:
        return None  # task has no headroom; excluded (reviewer's eligibility rule)
    return merged_loss <= base_loss - tau * gain


def summarize(rows, tau=0.5, min_gain=0.05):
    """Correlations over MERGES only (base and single-adapter reference rows are
    kept in the output but excluded here, matching the released single-task probe)."""
    mg = [r for r in rows if r["merge"] not in ("base", "alone")]
    tasks = sorted({r["task"] for r in mg})
    per_task = {}
    for t in tasks:
        rt = [r for r in mg if r["task"] == t]
        per_task[t] = {
            "n": len(rt),
            "r_Linput_acc": pearson([r["L_input"] for r in rt], [r["accuracy"] for r in rt]),
            "r_Lanswer_acc": pearson([r["L_answer"] for r in rt], [r["accuracy"] for r in rt]),
        }
    pooled = {
        "n": len(mg),
        "r_Linput_acc": pearson([r["L_input"] for r in mg], [r["accuracy"] for r in mg]),
        "r_Lanswer_acc": pearson([r["L_answer"] for r in mg], [r["accuracy"] for r in mg]),
    }
    return {"pooled": pooled, "per_task": per_task}


# ---- GPU path --------------------------------------------------------------

def masked_ce(model, tok, text, answer_text, device, max_length=384):
    """(L_input, L_answer): CE over the whole prompt, and CE over only the answer
    tokens appended after it. Mirrors run_peft_atlas_lite.per_prompt_nll (shift by
    one, mean over kept tokens)."""
    import torch
    import torch.nn.functional as F
    pids = tok(text, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
    full = tok(text + " " + answer_text, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
    with torch.no_grad():
        # L_input: score the prompt tokens
        li = model(input_ids=pids, labels=pids.clone()).loss.item()
        # L_answer: score only the answer tokens (mask the prompt span)
        lab = full.clone()
        lab[:, :pids.shape[1]] = -100
        la = model(input_ids=full, labels=lab).loss.item()
    return li, la


def cmd_run(args):
    import random
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(cfg.get("dtype", "bfloat16"), torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=dtype, device_map=device)

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
    regexes = {s["name"]: s.get("answer_regex", cfg.get("answer_regex", r"\b([A-Z])\b")) for s in specs}
    n_items = int(cfg.get("n_items", 40))
    n_merges = int(cfg.get("n_merges_per_task", 12))
    max_new = int(cfg.get("max_new_tokens", 6))
    rng = random.Random(cfg.get("seed", 5))

    def prompt_of(defn, inp):
        msgs = [{"role": "user", "content": defn + "\n\n" + inp}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return defn + "\n\n" + inp + "\n"

    def measure(task, regex):
        defn, inst = task["definition"], task["instances"][:n_items]
        corr = li_sum = la_sum = 0.0
        for it in inst:
            p = prompt_of(defn, it["input"]); gold = it["output"][0]
            pids = tok(p, return_tensors="pt").input_ids.to(device)
            with torch.no_grad():
                g = peft.generate(input_ids=pids, max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
            out_txt = tok.decode(g[0, pids.shape[1]:], skip_special_tokens=True)
            pred = extract_pred(out_txt, regex)
            gold_norm = gold.strip().upper()
            corr += float(pred == gold_norm or pred == gold_norm[:1])
            li, la = masked_ce(peft, tok, p, gold, device, int(cfg.get("max_length", 384)))
            li_sum += li; la_sum += la
        k = len(inst)
        return corr / k, li_sum / k, la_sum / k

    # Resume support: every completed measurement is appended to rows.jsonl, so a
    # preempted run (interruptible instances) restarts where it left off.
    rows = []
    done = set()
    rows_path = Path(args.out) / "rows.jsonl"
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows.append(row)
                done.add((row["task"], row["merge"]))
        print(f"resuming: {len(rows)} measurements already done", flush=True)
    ckpt = open(rows_path, "a")

    for host in specs:  # each task hosts merges that include its adapter
        hname = host["name"]
        others = [n for n in names if n != hname]
        merges = [("base", {}), ("alone", {hname: 1.0})]
        merges += [(f"{hname}+{o}", {hname: 0.5, o: 0.5}) for o in others]
        seen = set()
        if len(others) >= 2:
            for _ in range(n_merges):
                a, b = tuple(rng.sample(others, 2))
                key = frozenset((hname, a, b))
                if key in seen:
                    continue
                seen.add(key)
                merges.append((f"{hname}+{a}+{b}", {hname: 1/3, a: 1/3, b: 1/3}))
        for label, w in merges:
            if (hname, label) in done:
                continue
            if not w:  # base model, adapters disabled
                with peft.disable_adapter():
                    acc, li, la = measure(tasks[hname], regexes[hname])
            else:
                if len(w) == 1:
                    peft.set_adapter(hname)
                else:
                    peft.add_weighted_adapter(list(w), list(w.values()), "merge", combination_type="linear")
                    peft.set_adapter("merge")
                acc, li, la = measure(tasks[hname], regexes[hname])
                if len(w) > 1:
                    peft.delete_adapter("merge")
            row = {"task": hname, "merge": label, "accuracy": acc, "L_input": li, "L_answer": la}
            rows.append(row)
            ckpt.write(json.dumps(row) + "\n")
            ckpt.flush()
            print(f"  {hname:10s} {label:28s} acc={acc:.2f} Li={li:.3f} La={la:.3f}", flush=True)

    result = summarize(rows)
    (out / "answer_token_panel.json").write_text(json.dumps({"rows": rows, **result}, indent=2))
    print(json.dumps(result, indent=2))
    print("wrote", out / "answer_token_panel.json")
    return 0


def cmd_selftest(_args):
    # synthetic: L_answer anti-correlated with accuracy, L_input independent
    import random
    rng = random.Random(0)
    rows = []
    for t in ["taskA", "taskB", "taskC"]:
        for _ in range(20):
            acc = rng.uniform(0.4, 0.95)
            rows.append({"task": t, "merge": "m", "accuracy": acc,
                         "L_answer": 2.0 - 1.5 * acc + rng.gauss(0, 0.05),   # tracks accuracy
                         "L_input": 3.0 + rng.gauss(0, 0.3)})                 # flat
    s = summarize(rows)
    print("pooled r(L_answer,acc)=%.3f  r(L_input,acc)=%.3f" % (
        s["pooled"]["r_Lanswer_acc"], s["pooled"]["r_Linput_acc"]))
    for t, v in s["per_task"].items():
        print("  %s: r(La,acc)=%.3f r(Li,acc)=%.3f" % (t, v["r_Lanswer_acc"], v["r_Linput_acc"]))
    assert s["pooled"]["r_Lanswer_acc"] < -0.9, "answer-token loss should track accuracy"
    assert abs(s["pooled"]["r_Linput_acc"]) < 0.4, "input-prompt loss should be flat"
    # retention rule
    assert retained(1.0, 2.0, 0.5, tau=0.5, min_gain=0.05) is True   # keeps >50% of gain
    assert retained(1.9, 2.0, 0.5, tau=0.5, min_gain=0.05) is False  # keeps <50%
    assert retained(1.0, 2.0, 1.99, tau=0.5, min_gain=0.05) is None  # no headroom
    assert answer_label_mask(3, 5) == [-100, -100, -100, 1, 1]
    print("selftest OK: correlation, retention rule, and answer-mask logic verified")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config"); ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return cmd_selftest(args)
    if not (args.config and args.out):
        ap.error("--config and --out required unless --selftest")
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
