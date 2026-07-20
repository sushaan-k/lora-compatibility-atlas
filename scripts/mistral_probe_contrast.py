#!/usr/bin/env python3
"""Does the atlas's retained-loss predict task ACCURACY, and what would fix it?
For ~20 equal-weight merges containing the SNLI (task190) adapter on Mistral-7B,
measure on the same held-out SNLI instances three things:
  L_input  = mean CE over the task INPUT tokens (what the pipeline's retained-loss
             actually measures -- input-only probes),
  L_answer = mean CE over the ANSWER token(s) only, input masked (the proposed fix),
  accuracy = exact-match label accuracy (generation).
Then correlate each loss with accuracy. Prediction: L_answer tracks accuracy while
L_input does not -- i.e. the screen predicts capability iff it scores the answer.
"""
import json, re, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "mistralai/Mistral-7B-Instruct-v0.2"
D = json.load(open("/root/atlas/mistral_merges.json"))
n2r = D["name2repo"]
dat = json.load(open("/root/atlas/task190_natural.json"))
DEFN, INST = dat["definition"], dat["instances"][:40]
others = [a for a in n2r if a != "task190"]

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
peft = None
for name in n2r:
    if peft is None:
        peft = PeftModel.from_pretrained(model, n2r[name], adapter_name=name)
    else:
        peft.load_adapter(n2r[name], adapter_name=name)
peft.eval()

random.seed(5)
MERGES = ([("base", {})] + [("t190-alone", {"task190": 1.0})]
          + [(f"task190+{o}", {"task190": 0.5, o: 0.5}) for o in others]
          + [(f"task190+{a}+{b}", {"task190": 1/3, a: 1/3, b: 1/3})
             for a, b in [random.sample(others, 2) for _ in range(10)]])

def prompt_of(inp):
    return tok.apply_chat_template([{"role": "user", "content": DEFN + "\n\n" + inp}],
                                   tokenize=False, add_generation_prompt=True)

def pred(out):
    m = re.search(r"\b([ECN])\b", out.strip().upper())
    return m.group(1) if m else "?"

def measure():
    corr, li, la = 0, 0.0, 0.0
    for i in INST:
        p = prompt_of(i["input"]); gold = i["output"][0]
        pids = tok(p, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            g = peft.generate(input_ids=pids, max_new_tokens=6, do_sample=False, pad_token_id=tok.eos_token_id)
        corr += pred(tok.decode(g[0, pids.shape[1]:], skip_special_tokens=True)) == gold
        # L_input: CE over the input prompt tokens
        with torch.no_grad():
            lab = pids.clone()
            li += peft(input_ids=pids, labels=lab).loss.item()
        # L_answer: CE over the answer tokens only (prompt masked)
        enc = tok(p + " " + gold, return_tensors="pt").to("cuda")
        lab = enc.input_ids.clone(); lab[:, :pids.shape[1]] = -100
        with torch.no_grad():
            la += peft(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=lab).loss.item()
    n = len(INST)
    return corr / n, li / n, la / n

rows = []
print("acc    L_input  L_answer  merge", flush=True)
for label, w in MERGES:
    if not w:
        with peft.disable_adapter():
            a, li, la = measure()
    else:
        if len(w) == 1:
            peft.set_adapter("task190")
        else:
            peft.add_weighted_adapter(list(w), list(w.values()), "merge", combination_type="linear")
            peft.set_adapter("merge")
        a, li, la = measure()
        if len(w) > 1:
            peft.delete_adapter("merge")
    rows.append((label, a, li, la))
    print(f"{a:.2f}   {li:.3f}   {la:.3f}    {label}", flush=True)

def r(xs, ys):
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    sx = sum((x-mx)**2 for x in xs)**0.5; sy = sum((y-my)**2 for y in ys)**0.5
    return cov/(sx*sy) if sx*sy else float("nan")

# correlations over the actual merges (exclude base/alone reference rows)
mg = [x for x in rows if x[0] not in ("base", "t190-alone")]
ac = [x[1] for x in mg]; li = [x[2] for x in mg]; la = [x[3] for x in mg]
print(f"\nn_merges={len(mg)}", flush=True)
print(f"r(L_input,  accuracy) = {r(li, ac):+.3f}   [what the pipeline's retained-loss measures]", flush=True)
print(f"r(L_answer, accuracy) = {r(la, ac):+.3f}   [the fix: score the answer tokens]", flush=True)
