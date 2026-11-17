#!/usr/bin/env python3
"""Fine-tune Qwen2.5-0.5B (base) with LoRA on the book-request -> JSON task.

Usage (from task1-lora/):
    ../../.venv/bin/python train.py                     # full run, 3 epochs
    ../../.venv/bin/python train.py --max-examples 20 --epochs 1 --smoke
                                                        # timed smoke run

What it does:
* Loads Qwen/Qwen2.5-0.5B (base, NOT Instruct -- the base-vs-fine-tuned
  comparison must be fair) and wraps q_proj + v_proj with LoRA
  (r=16, alpha=32, dropout 0.05).
* Trains only on the completion tokens: prompt tokens are masked
  (label -100), so the loss is the JSON output, not the request text.
* Auto-detects device (cuda if present else cpu); saves ONLY the LoRA
  adapter to adapter/ (weights are a run artefact, never committed)
  and writes lora_config.json with the exact config used.

Hyperparameters are the ones in the task brief; the justification for
each is in README.md.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
MAX_LEN = 192


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class JsonDataset(torch.utils.data.Dataset):
    """Prompt-masked examples: input_ids = prompt + response + EOS,
    labels = -100 on the prompt, real ids on the response."""

    def __init__(self, rows, tokenizer):
        self.examples = []
        for r in rows:
            prompt_ids = tokenizer(r["prompt"], add_special_tokens=False)["input_ids"]
            resp_ids = tokenizer(r["response"], add_special_tokens=False)["input_ids"]
            resp_ids = resp_ids + [tokenizer.eos_token_id]
            ids = (prompt_ids + resp_ids)[:MAX_LEN]
            labels = ([-100] * len(prompt_ids) + resp_ids)[:MAX_LEN]
            self.examples.append({"input_ids": ids, "labels": labels})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = maxlen - n
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * n + [0] * pad)
    return {"input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--train-file", default=str(BASE_DIR / "data" / "train.jsonl"))
    ap.add_argument("--output-dir", default=str(BASE_DIR / "adapter"))
    ap.add_argument("--config-out", default=str(BASE_DIR / "lora_config.json"))
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-examples", type=int, default=None,
                    help="use only the first N train examples (smoke run)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  torch {torch.__version__}  "
          f"threads: {torch.get_num_threads()}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32)
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    config_record = {
        "base_model": args.model,
        "model_params_b": 0.494,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha,
                 "dropout": 0.05, "target_modules": ["q_proj", "v_proj"],
                 "bias": "none", "task_type": "CAUSAL_LM"},
        "training": {"learning_rate": args.lr, "epochs": args.epochs,
                     "per_device_train_batch_size": args.batch_size,
                     "gradient_accumulation_steps": args.grad_accum,
                     "effective_batch_size": args.batch_size * args.grad_accum,
                     "max_len": MAX_LEN, "seed": args.seed,
                     "loss": "cross-entropy on completion tokens only "
                             "(prompt tokens masked with -100)",
                     "optimizer": "adamw_torch (Trainer default)",
                     "lr_scheduler": "linear (Trainer default)",
                     "warmup_steps": 5},
        "device": device,
        "train_file": args.train_file,
        "max_examples": args.max_examples,
    }
    with open(args.config_out, "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)
    print(f"config written to {args.config_out}")

    rows = load_jsonl(args.train_file)
    if args.max_examples:
        rows = rows[: args.max_examples]
    print(f"train examples: {len(rows)}")
    dataset = JsonDataset(rows, tokenizer)

    train_args = TrainingArguments(
        output_dir=str(BASE_DIR / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        # transformers 5.x removed warmup_ratio; 5 steps ≈ 3% of the
        # 150 total steps of a full run (50 steps/epoch x 3 epochs)
        warmup_steps=5,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=0,
        bf16=False, fp16=False,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=train_args, train_dataset=dataset,
        data_collator=lambda b: collate(b, tokenizer.pad_token_id))
    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    elapsed = time.time() - t0
    config_record["wall_clock_seconds"] = round(elapsed, 1)
    config_record["wall_clock"] = f"{elapsed / 60:.1f} min"
    with open(args.config_out, "w", encoding="utf-8") as f:
        json.dump(config_record, f, indent=2)
    print(f"adapter saved to {args.output_dir}")
    print(f"TOTAL wall clock: {elapsed / 60:.1f} min ({elapsed:.0f} s)")


if __name__ == "__main__":
    main()
