#!/usr/bin/env python3
"""Answer-token screening panel at m=8 (Mistral, Lots-of-LoRAs): the primary-table
rerun on capability labels.

The Qwen m=50 community adapters have no recoverable task answers, so the
outstanding rerun named in the paper is realized here: eight answer-bearing
Natural Instructions classification tasks with Lots-of-LoRAs adapters on one
shared 7-simplex.

fit (GPU): use expC_answer_jets.py fit with the m=8 config and --step 0.5; the
step-1/2 grid is exactly the singles-and-pairs poised design (8 + 28 = 36 = N_7).

validate (GPU, this script): for base, each single, all 28 pairs and all 56
triples, measure accuracy, L_input and L_answer on each MEMBER task's 40
held-out items (240 rows total; per-row checkpointing for interruptible
instances).

analyze (CPU, this script): fit jets from the design, score every pair/triple
by the predicted worst member retention deficit at its merge coefficient, and
report subset-level AUROC against (a) answer-token retention labels over the
paper's tau grid and (b) accuracy-retention labels; per-task answer-token gains
(the degeneracy check against the prompt-loss gain clamp); decision preservation
of the (r+1)-core reduced jets on non-vacuous labels; shared-projector
residuals.

Self-test (CPU, no GPU/data): python expD_answer_panel.py selftest
"""
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

from expC_answer_jets import (auroc, fit_jets, jet_value,  # noqa: E402
                              shared_projector_residuals, weights_to_coords)

TAU_GRID = (0.55, 0.65, 0.70, 0.75, 0.85)
TAU_ACC = (0.5, 0.7, 0.9)


def all_merges(names, max_triples=None):
    """(label, weights dict) for all pairs and (sampled) triples, equal weights.
    max_triples caps the triple set with a fixed-seed sample for large panels."""
    out = []
    for a, b in itertools.combinations(names, 2):
        out.append((f"{a}+{b}", {a: 0.5, b: 0.5}))
    triples = list(itertools.combinations(names, 3))
    if max_triples is not None and len(triples) > max_triples:
        import random
        random.Random(5).shuffle(triples)
        triples = triples[:max_triples]
    for a, b, c in triples:
        out.append((f"{a}+{b}+{c}", {a: 1 / 3, b: 1 / 3, c: 1 / 3}))
    return out


# ------------------------------------------------------------- validate (GPU)

def cmd_validate(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from expA_answer_token_panel import extract_pred, masked_ce

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
    max_new = int(cfg.get("max_new_tokens", 6))
    max_length = int(cfg.get("max_length", 384))

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
            pids = tok(p, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
            with torch.no_grad():
                g = peft.generate(input_ids=pids, max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
            out_txt = tok.decode(g[0, pids.shape[1]:], skip_special_tokens=True)
            pred = extract_pred(out_txt, regex)
            gold_norm = gold.strip().upper()
            corr += float(pred == gold_norm or pred == gold_norm[:1])
            li, la = masked_ce(peft, tok, p, gold, device, max_length)
            li_sum += li; la_sum += la
        k = len(inst)
        return corr / k, li_sum / k, la_sum / k

    # work list: base and alone per task, then every pair/triple on its members
    max_triples = cfg.get("max_triples")
    work = []
    for nm in names:
        work.append((nm, "base", {}))
        work.append((nm, "alone", {nm: 1.0}))
    for label, w in all_merges(names, max_triples):
        for member in label.split("+"):
            work.append((member, label, w))

    rows, done = [], set()
    rows_path = out / "rows.jsonl"
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows.append(row)
                done.add((row["task"], row["merge"]))
        print(f"resuming: {len(rows)}/{len(work)} rows done", flush=True)
    ckpt = open(rows_path, "a")
    t0 = time.time()
    current = None  # (frozen weights) of the adapter currently materialized
    for i, (host, label, w) in enumerate(work):
        if (host, label) in done:
            continue
        if not w:
            with peft.disable_adapter():
                acc, li, la = measure(tasks[host], regexes[host])
        else:
            key = tuple(sorted(w.items()))
            if key != current:
                if current is not None and len(dict(current)) > 1:
                    peft.delete_adapter("mergeD")
                if len(w) == 1:
                    peft.set_adapter(next(iter(w)))
                else:
                    peft.add_weighted_adapter(list(w), list(w.values()), "mergeD",
                                              combination_type="linear")
                    peft.set_adapter("mergeD")
                current = key
            acc, li, la = measure(tasks[host], regexes[host])
        row = {"task": host, "merge": label, "accuracy": acc, "L_input": li, "L_answer": la}
        rows.append(row)
        ckpt.write(json.dumps(row) + "\n")
        ckpt.flush()
        print(f"  [{i+1}/{len(work)}] {host:10s} {label:32s} acc={acc:.2f} La={la:.3f} t={int(time.time()-t0)}s",
              flush=True)
    (out / "validation_rows.json").write_text(json.dumps({"rows": rows}, indent=1))
    print("wrote", out / "validation_rows.json", flush=True)
    return 0


# -------------------------------------------------------------- analyze (CPU)

def build_events(names, jets, rows):
    """Per (merge, member) and per subset: predicted and measured deficits."""
    base = {r["task"]: r for r in rows if r["merge"] == "base"}
    alone = {r["task"]: r for r in rows if r["merge"] == "alone"}
    meas = {(r["task"], r["merge"]): r for r in rows if r["merge"] not in ("base", "alone")}
    gains = {nm: base[nm]["L_answer"] - alone[nm]["L_answer"] for nm in names}

    per_member, per_subset = [], []
    for label, w in all_merges(names):
        members = label.split("+")
        wv = np.zeros(len(names))
        for nm, wt in w.items():
            wv[names.index(nm)] = wt
        c = weights_to_coords(wv)
        mem_rows = []
        for nm in members:
            r = meas.get((nm, label))
            if r is None:
                continue
            g = gains[nm]
            pred_la = jet_value(jets[nm]["alpha"], jets[nm]["b"], jets[nm]["H"], c)
            rec = {"task": nm, "merge": label, "order": len(members), "gain": g,
                   "pred_L_answer": pred_la, "L_answer": r["L_answer"],
                   "accuracy": r["accuracy"],
                   "base_acc": base[nm]["accuracy"], "alone_acc": alone[nm]["accuracy"],
                   "base_La": base[nm]["L_answer"], "alone_La": alone[nm]["L_answer"]}
            mem_rows.append(rec)
            per_member.append(rec)
        if len(mem_rows) == len(members):
            per_subset.append({"merge": label, "order": len(members), "members": mem_rows})
    return gains, per_member, per_subset


def retained_answer(rec, tau):
    """Merge keeps at least tau of the task's answer-token gain."""
    return rec["L_answer"] <= rec["base_La"] - tau * rec["gain"]


def pred_deficit(rec, tau):
    return rec["pred_L_answer"] - (rec["base_La"] - tau * rec["gain"])


def retained_acc(rec, tau_acc):
    return rec["accuracy"] >= rec["base_acc"] + tau_acc * (rec["alone_acc"] - rec["base_acc"])


def cmd_analyze(args):
    z = np.load(args.raw, allow_pickle=True)
    grid = z["grid"]; names = [str(x) for x in z["names"]]; vals = z["values"]
    # chart dimension comes from the coefficient grid, not the task count
    # (they differ when scored tasks are a subset of the merge adapters)
    d = len(grid[0]) - 1
    jets = fit_jets([list(w) for w in grid], {nm: list(vals[i]) for i, nm in enumerate(names)}, d)
    residuals = shared_projector_residuals(jets, names)

    rows = json.loads(Path(args.rows).read_text())["rows"]
    gains, per_member, per_subset = build_events(names, jets, rows)

    # prediction quality of the jets on validation merges
    pm = np.array([e["pred_L_answer"] for e in per_member])
    mm = np.array([e["L_answer"] for e in per_member])
    r_pred = float(np.corrcoef(pm, mm)[0, 1])
    r_pred_triples = float(np.corrcoef(pm[[e["order"] == 3 for e in per_member]],
                                       mm[[e["order"] == 3 for e in per_member]])[0, 1])

    result = {"n_tasks": len(names), "names": names,
              "answer_token_gains": {nm: gains[nm] for nm in names},
              "n_positive_gains": int(sum(g > 0 for g in gains.values())),
              "n_member_events": len(per_member), "n_subsets": len(per_subset),
              "r_pred_vs_measured_L_answer": r_pred,
              "r_pred_vs_measured_L_answer_triples_only": r_pred_triples,
              "shared_projector_residuals": residuals,
              "psd_projected_tasks": [nm for nm in names if jets[nm]["psd_projected"]],
              "by_answer_tau": {}, "by_acc_tau": {}}

    # subset-level screening: retained iff EVERY member retained; score = worst
    # member predicted deficit at the merge coefficient (higher => reject)
    for tau in TAU_GRID:
        labels = [all(retained_answer(m, tau) for m in s["members"]) for s in per_subset]
        scores = [-max(pred_deficit(m, tau) for m in s["members"]) for s in per_subset]
        pairs = [i for i, s in enumerate(per_subset) if s["order"] == 2]
        trips = [i for i, s in enumerate(per_subset) if s["order"] == 3]
        result["by_answer_tau"][str(tau)] = {
            "n_retained": int(sum(labels)), "n": len(labels),
            "auroc_pooled": auroc(labels, scores),
            "auroc_pairs": auroc([labels[i] for i in pairs], [scores[i] for i in pairs]),
            "auroc_triples": auroc([labels[i] for i in trips], [scores[i] for i in trips]),
        }
    for ta in TAU_ACC:
        labels = [all(retained_acc(m, ta) for m in s["members"]) for s in per_subset]
        # accuracy labels are threshold-free on the prediction side: rank by the
        # worst member predicted answer loss relative to its own adapter's
        scores = [-max((m["pred_L_answer"] - m["alone_La"]) / max(m["gain"], 1e-6)
                       for m in s["members"]) for s in per_subset]
        pairs = [i for i, s in enumerate(per_subset) if s["order"] == 2]
        trips = [i for i, s in enumerate(per_subset) if s["order"] == 3]
        meas_scores = [-max((m["L_answer"] - m["alone_La"]) / max(m["gain"], 1e-6)
                            for m in s["members"]) for s in per_subset]
        result["by_acc_tau"][str(ta)] = {
            "n_retained": int(sum(labels)), "n": len(labels),
            "auroc_pooled_predicted": auroc(labels, scores),
            "auroc_pooled_measured": auroc(labels, meas_scores),
            "auroc_pairs_predicted": auroc([labels[i] for i in pairs], [scores[i] for i in pairs]),
            "auroc_triples_predicted": auroc([labels[i] for i in trips], [scores[i] for i in trips]),
        }

    # decision preservation of rank-r reduced jets on the answer-token labels
    # (non-vacuous whenever both decision classes occur at the tau)
    try:
        sys.path.insert(0, str(REPO / "scripts"))
        from interval_certification import build_reduced_jets
        alphas = np.array([jets[nm]["alpha"] for nm in names])
        bs = [jets[nm]["b"] for nm in names]
        Hs = [jets[nm]["H"] for nm in names]
        preservation = {}
        for r in (1, 2, 3):
            a_r, b_r, H_r, _ = build_reduced_jets(alphas, bs, Hs, r)
            agree = {}
            for tau in TAU_GRID:
                full_dec, red_dec = [], []
                for s in per_subset:
                    fd = max(pred_deficit(m, tau) for m in s["members"]) <= 0
                    wv2 = np.zeros(len(names))
                    for nm2 in s["merge"].split("+"):
                        wv2[names.index(nm2)] = 1.0 / s["order"]
                    c2 = weights_to_coords(wv2)
                    rd = []
                    for m in s["members"]:
                        i = names.index(m["task"])
                        pl = jet_value(a_r[i], b_r[i], H_r[i], c2)
                        rd.append(pl - (m["base_La"] - tau * m["gain"]))
                    full_dec.append(fd)
                    red_dec.append(max(rd) <= 0)
                agree[str(tau)] = float(np.mean([f == r2 for f, r2 in zip(full_dec, red_dec)]))
            preservation[f"r{r}"] = agree
        result["reduced_jet_decision_agreement"] = preservation
    except Exception as e:  # keep analysis usable if scripts/ is absent
        result["reduced_jet_decision_agreement"] = f"skipped: {e}"

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    result_full = dict(result)
    result_full["per_member"] = per_member
    (out / "answer_panel_analysis.json").write_text(json.dumps(result_full, indent=1))
    print(json.dumps(result, indent=1))
    print("wrote", out / "answer_panel_analysis.json")
    return 0


def cmd_selftest(_):
    rng = np.random.default_rng(1)
    names = [f"t{i}" for i in range(8)]
    d = 7
    from run_peft_atlas_lite import simplex_grid
    grid = simplex_grid(8, step=0.5, min_weight=0.0)
    assert len(grid) == 36, len(grid)  # singles + pairs = N_7
    # synthetic truth: each task's answer CE rises away from its own vertex
    true = {}
    for i, nm in enumerate(names):
        H = np.eye(d) * rng.uniform(1.5, 2.5)
        b = rng.normal(0, 0.05, d)
        true[nm] = (float(rng.uniform(0.5, 0.9)), b, H)
    vals = {nm: [jet_value(a, b, H, weights_to_coords(w)) for w in grid]
            for nm, (a, b, H) in true.items()}
    jets = fit_jets([list(w) for w in grid], vals, d)
    for nm in names:
        a, b, H = true[nm]
        for w in grid[:10]:
            c = weights_to_coords(w)
            assert abs(jet_value(a, b, H, c)
                       - jet_value(jets[nm]["alpha"], jets[nm]["b"], jets[nm]["H"], c)) < 1e-6
    # synthetic rows: base high CE, alone low, merges in between; accuracy tied to CE
    rows = []
    for nm in names:
        rows.append({"task": nm, "merge": "base", "accuracy": 0.4, "L_input": 3.0,
                     "L_answer": 2.0})
        rows.append({"task": nm, "merge": "alone", "accuracy": 0.9, "L_input": 3.0,
                     "L_answer": 0.5})
    for label, w in all_merges(names):
        for member in label.split("+"):
            la = 0.5 + 0.8 * (1 - w[member])  # worse dilution -> higher CE
            rows.append({"task": member, "merge": label, "accuracy": 0.9 - 0.3 * (1 - w[member]),
                         "L_input": 3.0, "L_answer": la})
    gains, per_member, per_subset = build_events(names, jets, rows)
    assert len(per_subset) == 28 + 56
    assert all(g > 0 for g in gains.values())
    # pairs dilute less than triples, so at a mid tau pairs are retained and
    # triples are not; a correct pipeline separates them perfectly BY THE
    # MEASURED deficit (labels)
    tau = 0.70
    labels = [all(retained_answer(m, tau) for m in s["members"]) for s in per_subset]
    assert any(labels) and not all(labels), (sum(labels), len(labels))
    meas_scores = [-max(m["L_answer"] - (m["base_La"] - tau * m["gain"]) for m in s["members"])
                   for s in per_subset]
    assert auroc(labels, meas_scores) == 1.0
    print("selftest OK: 36-point poised design, jet recovery, subset labels and "
          "orientation correct")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate"); v.add_argument("--config", required=True)
    v.add_argument("--out", required=True); v.set_defaults(func=cmd_validate)
    an = sub.add_parser("analyze"); an.add_argument("--raw", required=True)
    an.add_argument("--rows", required=True); an.add_argument("--out", required=True)
    an.set_defaults(func=cmd_analyze)
    st = sub.add_parser("selftest"); st.set_defaults(func=cmd_selftest)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
