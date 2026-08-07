#!/usr/bin/env python3
"""Gradient-alignment screener vs the atlas (E1)."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def auroc(y, s):
    pos = [x for x, l in zip(s, y) if l]; neg = [x for x, l in zip(s, y) if not l]
    if not pos or not neg:
        return None
    return float(sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True); ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
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
    n_items = int(cfg.get("n_items", 40)); max_length = int(cfg.get("max_length", 384))
    peft = None
    for s in specs:
        if peft is None:
            peft = PeftModel.from_pretrained(model, s["repo"], adapter_name=s["name"], is_trainable=False)
        else:
            peft.load_adapter(s["repo"], adapter_name=s["name"])
    peft.eval()

    # base q/v-proj weights to differentiate; fixed random projection per tensor
    targets = [(nm, p) for nm, p in peft.named_parameters()
               if ("q_proj.base_layer.weight" in nm or "v_proj.base_layer.weight" in nm)]
    if not targets:  # naming fallback
        targets = [(nm, p) for nm, p in peft.named_parameters()
                   if (nm.endswith("q_proj.weight") or nm.endswith("v_proj.weight"))]
    g = torch.Generator(device="cpu").manual_seed(0)
    randvecs = {nm: torch.randn(p.shape, generator=g).to(device=p.device, dtype=torch.float32)
                for nm, p in targets}
    for nm, p in targets:
        p.requires_grad_(True)

    def prompt_of(defn, inp):
        try:
            return tok.apply_chat_template([{"role": "user", "content": defn + "\n\n" + inp}],
                                           tokenize=False, add_generation_prompt=True)
        except Exception:
            return defn + "\n\n" + inp + "\n"

    def signature(task):
        peft.set_adapter(task)
        sig = {nm: 0.0 for nm, _ in targets}
        defn, inst = tasks[task]["definition"], tasks[task]["instances"][:n_items]
        for it in inst:
            p = prompt_of(defn, it["input"]); gold = it["output"][0]
            pids = tok(p, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
            full = tok(p + " " + gold, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
            lab = full.clone(); lab[:, :pids.shape[1]] = -100
            peft.zero_grad(set_to_none=True)
            loss = peft(input_ids=full, labels=lab).loss
            loss.backward()
            for nm, param in targets:
                if param.grad is not None:
                    sig[nm] += float((param.grad.float() * randvecs[nm]).sum().item())
        return np.array([sig[nm] for nm, _ in targets], dtype=float)

    sigs = {}
    for i, nm in enumerate(names):
        sigs[nm] = signature(nm)
        print(f"  signature {nm} ({i+1}/{len(names)}) norm {np.linalg.norm(sigs[nm]):.3g}", flush=True)

    def align(members):
        vs = [sigs[m] for m in members]
        cs = [float(np.dot(vs[a], vs[b]) / (np.linalg.norm(vs[a]) * np.linalg.norm(vs[b]) + 1e-12))
              for a, b in itertools.combinations(range(len(vs)), 2)]
        return float(np.mean(cs))

    # labels from the expD validation rows (answer-retention, tau=0.7), atlas AUROC for reference
    rows = json.loads(Path(args.rows).read_text())["rows"]
    base = {r["task"]: r for r in rows if r["merge"] == "base"}
    alone = {r["task"]: r for r in rows if r["merge"] == "alone"}
    meas = {(r["task"], r["merge"]): r for r in rows if r["merge"] not in ("base", "alone")}
    subsets = {}
    for (t, mg) in meas:
        subsets.setdefault(mg, mg.split("+"))
    tau = 0.7
    y, s_align = [], []
    for mg, members in subsets.items():
        if not all((m, mg) in meas for m in members):
            continue
        retained = all(meas[(m, mg)]["L_answer"] <= base[m]["L_answer"]
                       - tau * (base[m]["L_answer"] - alone[m]["L_answer"]) for m in members)
        y.append(retained); s_align.append(align(members))
    a_align = auroc(y, s_align)
    result = {"n_subsets": len(y), "n_retained": int(sum(y)),
              "gradient_alignment_auroc": a_align,
              "n_target_modules": len(targets),
              "note": "answer-retention labels at tau=0.7; compare against the atlas AUROC on the "
                      "same panel (0.95-0.98 central taus). Gradient signature is a per-module "
                      "random projection of the base q/v-proj gradient with each adapter active."}
    (out / "alignment.json").write_text(json.dumps(result, indent=1))
    print(f"gradient-alignment screener AUROC = {a_align} over {len(y)} subsets", flush=True)
    print("wrote", out / "alignment.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
