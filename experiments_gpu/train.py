#!/usr/bin/env python3
"""Train answer-bearing LoRA adapters on a second base (C1)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cmd_train(args):
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              TrainingArguments, Trainer, DataCollatorForLanguageModeling)
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset

    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    base = cfg["base_model"]
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    max_length = int(cfg.get("max_length", 384))
    n_train = int(cfg.get("n_train", 400))

    def prompt_of(defn, inp):
        msgs = [{"role": "user", "content": defn + "\n\n" + inp}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return defn + "\n\n" + inp + "\n"

    for spec in cfg["adapters"]:
        name = spec["name"]
        adir = out / name
        if (adir / "adapter_config.json").exists():
            print(f"  {name}: already trained, skipping", flush=True)
            continue
        task = json.loads(Path(spec["task_json"]).read_text())
        defn = task["definition"]
        # train on items AFTER the 40 held-out eval items to avoid leakage
        train_items = task["instances"][40:40 + n_train]
        texts = []
        for it in train_items:
            p = prompt_of(defn, it["input"]); gold = it["output"][0]
            texts.append(p + " " + gold + tok.eos_token)

        def tokenize(batch):
            enc = tok(batch["text"], truncation=True, max_length=max_length, padding=False)
            return enc
        ds = Dataset.from_dict({"text": texts}).map(tokenize, batched=True,
                                                    remove_columns=["text"])

        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                                     device_map="cuda")
        lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        model = get_peft_model(model, lcfg)
        targs = TrainingArguments(
            output_dir=str(adir / "_ckpt"), per_device_train_batch_size=4,
            gradient_accumulation_steps=2, num_train_epochs=3, learning_rate=2e-4,
            bf16=True, logging_steps=25, save_strategy="no", report_to=[],
            warmup_ratio=0.05, lr_scheduler_type="cosine")
        collator = DataCollatorForLanguageModeling(tok, mlm=False)
        Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator).train()
        model.save_pretrained(str(adir))
        print(f"  {name}: trained on {len(texts)} examples, saved to {adir}", flush=True)
        del model
        torch.cuda.empty_cache()
    print("all adapters trained", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    return cmd_train(args)


if __name__ == "__main__":
    sys.exit(main())
