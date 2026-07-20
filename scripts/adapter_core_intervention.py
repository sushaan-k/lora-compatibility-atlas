#!/usr/bin/env python3
"""Real-model adapter-core intervention (reviewer follow-up).

Tests whether the sparse Caratheodory adapter core preserves the DEPLOYED
model's held-out answer loss and accuracy, not only the surrogate jet value.

For an accepted higher-way query on the m=8 Mistral answer-token panel it
compares three merges of the same adapters:
  (i)   equal weights over all 8 adapters (baseline),
  (ii)  the full minimax coefficient c* (supported on its adapter core, ~5),
  (iii) the rank-2 adapter core (3 adapters, reweighted to hold the latent
        optimum), the Caratheodory reduction with 2*delta slack.
If the theorem's adapter core carries the merge, (iii) should match (ii) on the
deployed answer loss and accuracy, and both should beat (i) on the worst task.

  plan : CPU, from the released jets -> results_public/adapter_core_intervention/merge_plan.json
  run  : GPU, loads mistralai/Mistral-7B-Instruct-v0.2 + the 8 adapters, evaluates each
         merge on held-out instances (reuses the expD merge/measure path)
  analyze : compares (i)/(ii)/(iii) and writes the summary

Needs the 8 Natural-Instructions task JSONs (task190, task1391, task280, task290,
task391, task363, task400, task379) referenced by experiments_gpu/expD_panel_m8.json.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in ("scripts", "experiments_gpu", "peft_atlas_lite"):
    sys.path.insert(0, str(REPO / p))

NAMES = ["task190", "task1391", "task280", "task290", "task391", "task363", "task400", "task379"]
OUT = REPO / "results_public" / "adapter_core_intervention"


def cmd_plan(_):
    np.seterr(all="ignore")
    from foundations_bicore import load_jets, rho, adapter_core
    from interval_certification import build_reduced_jets
    a, b, H = load_jets("m8")
    n = len(a)
    T = U = tuple(range(n))
    R, cstar = rho(a, b, H, T, U, n)
    Vfull = adapter_core(a, b, H, T, U, n, R, cstar)
    a2, b2, H2, _ = build_reduced_jets(a, b, H, 2)
    R2, cs2 = rho(a2, b2, H2, T, U, n)
    V2 = adapter_core(a2, b2, H2, T, U, n, R2, cs2)

    def wvec(c):
        w = np.zeros(n); w[:n - 1] = c; w[n - 1] = 1.0 - c.sum(); return w

    def as_dict(w, keep):
        return {NAMES[i]: float(w[i]) for i in range(n) if (i in keep and w[i] > 1e-4)}

    # reduced coefficient: re-solve the minimax restricted to V2 on the reduced jets,
    # then read the weights on those adapters (holds the rank-2 latent optimum).
    _, csV2 = rho(a2, b2, H2, T, V2, n)
    wV2 = wvec(csV2)
    plan = {
        "panel": "mistral_m8_answer", "query": "8-way", "names": NAMES,
        "rho_full": round(float(R), 4), "rho_rank2": round(float(R2), 4),
        "gap_full_rank2": round(float(abs(R - R2)), 4),
        "merges": {
            "equal8": {i: 0.125 for i in NAMES},
            "minimax_full": as_dict(wvec(cstar), set(range(n))),
            "adapter_core_rank2": as_dict(wV2, set(V2)),
        },
        "adapter_core_full_size": len(Vfull),
        "adapter_core_rank2_size": len(V2),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "merge_plan.json").write_text(json.dumps(plan, indent=2))
    print(json.dumps(plan, indent=2))
    print("wrote", OUT / "merge_plan.json")


def cmd_run(args):
    """GPU: evaluate each merge on held-out instances. Faithful to expD_answer_panel."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from expA_answer_token_panel import extract_pred, masked_ce

    cfg = json.load(open(REPO / "experiments_gpu" / "expD_panel_m8.json"))
    plan = json.load(open(OUT / "merge_plan.json"))
    device = "cuda"
    dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=dtype, device_map=device)
    specs = {s["name"]: s for s in cfg["adapters"]}
    peft = None
    for nm in NAMES:
        s = specs[nm]
        if peft is None:
            peft = PeftModel.from_pretrained(model, s["repo"], adapter_name=nm, is_trainable=False)
        else:
            peft.load_adapter(s["repo"], adapter_name=nm)
    n_items = int(cfg.get("n_items", 40))
    max_length = int(cfg.get("max_length", 384))
    max_new = int(cfg.get("max_new_tokens", 6))

    def task_data(nm):
        return json.load(open(specs[nm]["task_json"]))

    def prompt_of(defn, inp):
        msgs = [{"role": "user", "content": defn + "\n\n" + inp}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return defn + "\n\n" + inp + "\n"

    def measure(nm, w):
        s = specs[nm]; task = task_data(nm); regex = s["answer_regex"]
        defn = task["definition"]
        held = task["instances"][n_items:2 * n_items]  # held-out split (train split is [:n_items])
        corr = la = 0.0
        for it in held:
            p = prompt_of(defn, it["input"])
            gold = it["output"][0]
            pids = tok(p, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
            with torch.no_grad():
                g = peft.generate(input_ids=pids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
            out_txt = tok.decode(g[0, pids.shape[1]:], skip_special_tokens=True)
            pred = extract_pred(out_txt, regex)
            gnorm = gold.strip().upper()
            corr += float(pred == gnorm or pred == gnorm[:1])
            _, a = masked_ce(peft, tok, p, gold, device, max_length)
            la += a
        k = len(held)
        return corr / k, la / k

    results = {}
    for mname, w in plan["merges"].items():
        if "mergeX" in [ad for ad in (peft.peft_config or {})]:
            peft.delete_adapter("mergeX")
        peft.add_weighted_adapter(list(w), list(w.values()), "mergeX", combination_type="linear")
        peft.set_adapter("mergeX")
        rows = {}
        for nm in NAMES:
            acc, la = measure(nm, w)
            rows[nm] = {"accuracy": acc, "L_answer": la}
        peft.set_adapter(NAMES[0]); peft.delete_adapter("mergeX")
        results[mname] = rows
        print(mname, {k: round(v["L_answer"], 3) for k, v in rows.items()}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "deployed_results.json").write_text(json.dumps(results, indent=2))
    print("wrote", OUT / "deployed_results.json")


def cmd_analyze(_):
    res = json.load(open(OUT / "deployed_results.json"))
    def worst(m):
        return max(res[m][nm]["L_answer"] for nm in res[m])
    def mean(m):
        return float(np.mean([res[m][nm]["L_answer"] for nm in res[m]]))
    def acc(m):
        return float(np.mean([res[m][nm]["accuracy"] for nm in res[m]]))
    summ = {m: {"worst_L_answer": round(worst(m), 3), "mean_L_answer": round(mean(m), 3),
                "mean_acc": round(acc(m), 3)} for m in res}
    core_vs_full_worst = abs(worst("adapter_core_rank2") - worst("minimax_full"))
    summ["_core_preserves_deployed"] = {
        "worst_gap_core_vs_full": round(core_vs_full_worst, 3),
        "minimax_beats_equal_on_worst": bool(worst("minimax_full") < worst("equal8")),
    }
    (OUT / "intervention_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "analyze"])
    cmd = ap.parse_args()
    {"plan": cmd_plan, "run": cmd_run, "analyze": cmd_analyze}[cmd.cmd](cmd)
