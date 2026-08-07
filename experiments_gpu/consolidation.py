#!/usr/bin/env python3
"""Consolidation drift (B3): how far do jets move when an accepted merge is baked into the base?"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from panel import masked_ce  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True); ap.add_argument("--pair", required=True)
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
    specs = cfg["adapters"]; names = [s["name"] for s in specs]
    tasks = {s["name"]: json.loads(Path(s["task_json"]).read_text()) for s in specs}
    n_items = int(cfg.get("n_items", 40)); max_length = int(cfg.get("max_length", 384))
    pair = args.pair.split("+")

    def prompt_of(defn, inp):
        try:
            return tok.apply_chat_template([{"role": "user", "content": defn + "\n\n" + inp}],
                                           tokenize=False, add_generation_prompt=True)
        except Exception:
            return defn + "\n\n" + inp + "\n"

    def alone_answer_ce(peft, host):
        defn, inst = tasks[host]["definition"], tasks[host]["instances"][:n_items]
        tot = 0.0
        for it in inst:
            p = prompt_of(defn, it["input"]); gold = it["output"][0]
            _, a = masked_ce(peft, tok, p, gold, device, max_length); tot += a
        return tot / len(inst)

    # 1) original base: each surviving adapter alone
    base = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=torch.bfloat16,
                                                device_map=device)
    peft = None
    for s in specs:
        if peft is None:
            peft = PeftModel.from_pretrained(base, s["repo"], adapter_name=s["name"], is_trainable=False)
        else:
            peft.load_adapter(s["repo"], adapter_name=s["name"])
    peft.eval()
    survivors = [nm for nm in names if nm not in pair]
    orig = {}
    t0 = time.time()
    for nm in survivors:
        peft.set_adapter(nm); orig[nm] = alone_answer_ce(peft, nm)
        print(f"  orig-base {nm}: {orig[nm]:.4f} t={int(time.time()-t0)}s", flush=True)

    # 2) consolidate the accepted pair into the base and unload
    peft.add_weighted_adapter(pair, [0.5, 0.5], "consolidated", combination_type="linear")
    peft.set_adapter("consolidated")
    consolidated_base = peft.merge_and_unload()   # bake the merge into base weights
    del peft, base
    torch.cuda.empty_cache()

    # 3) reload survivors on the consolidated base, re-measure alone loss
    peft2 = None
    for nm in survivors:
        s = next(x for x in specs if x["name"] == nm)
        if peft2 is None:
            peft2 = PeftModel.from_pretrained(consolidated_base, s["repo"], adapter_name=nm, is_trainable=False)
        else:
            peft2.load_adapter(s["repo"], adapter_name=nm)
    peft2.eval()
    drift = {}
    for nm in survivors:
        peft2.set_adapter(nm)
        after = alone_answer_ce(peft2, nm)
        drift[nm] = {"before": orig[nm], "after": after, "delta": after - orig[nm]}
        print(f"  consolidated-base {nm}: {after:.4f} (drift {after-orig[nm]:+.4f})", flush=True)

    deltas = [abs(v["delta"]) for v in drift.values()]
    import numpy as np
    result = {"consolidated_pair": pair, "survivors": survivors, "per_adapter": drift,
              "abs_drift_median": float(np.median(deltas)), "abs_drift_max": float(np.max(deltas)),
              "note": "drift is the change in each surviving adapter's alone answer-token loss "
                      "after the accepted pair is baked into the base; compare to jet fit noise "
                      "and certificate margins to set the refit schedule"}
    (out / "consolidation.json").write_text(json.dumps(result, indent=1))
    print("abs drift median %.4f max %.4f" % (result["abs_drift_median"], result["abs_drift_max"]), flush=True)
    print("wrote", out / "consolidation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
