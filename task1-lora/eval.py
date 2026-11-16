#!/usr/bin/env python3
"""Evaluate base vs LoRA-fine-tuned Qwen2.5-0.5B on the held-out test set.

Usage (from task1-lora/):
    ../../.venv/bin/python eval.py                       # both models, n=100
    ../../.venv/bin/python eval.py --base-only           # base model only
    ../../.venv/bin/python eval.py --limit 10            # quick check

Decoding is identical for both models: greedy (do_sample=False),
max_new_tokens=80, same prompt. Metrics (n = test-set size):
* exact_match: parsed JSON dict equals the gold dict exactly
  (key order / whitespace ignored -- this is a semantic exact match on
  the JSON object, which is what the task requires).
* valid_json: output contains a parseable JSON object.
* per-field accuracy: fraction of examples where that field equals gold
  (year compared numerically, so "2016" counts as 2016).

Writes results/results_base.json, results/results_finetuned.json,
results/predictions_base.jsonl, results/predictions_finetuned.jsonl and
results/failures.md (up to 3 fine-tuned failures, with analysis).
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
FIELDS = ["title", "author", "genre", "year"]
MAX_NEW_TOKENS = 80

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_json(text):
    """Return the first {...} substring parsed as a dict, else None."""
    m = JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def field_equal(gold, pred, field):
    if field not in pred:
        return False
    g, p = gold[field], pred[field]
    if field == "year":
        try:
            return int(p) == int(g)
        except (TypeError, ValueError):
            return False
    return p == g


def evaluate(model, tokenizer, rows, device):
    preds = []
    n_exact = n_valid = 0
    field_ok = {f: 0 for f in FIELDS}
    t0 = time.time()
    for r in rows:
        inputs = tokenizer(r["prompt"], return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()
        parsed = extract_json(raw)
        gold = {f: r[f] for f in FIELDS}
        valid = parsed is not None
        exact = bool(valid and all(field_equal(gold, parsed, f) for f in FIELDS)
                     and set(parsed.keys()) == set(FIELDS))
        if valid:
            n_valid += 1
            for f in FIELDS:
                if field_equal(gold, parsed, f):
                    field_ok[f] += 1
        if exact:
            n_exact += 1
        preds.append({"request": r["request"], "expected": gold,
                      "raw_output": raw, "parsed": parsed,
                      "valid_json": valid, "exact_match": exact})
    n = len(rows)
    results = {
        "n": n,
        "exact_match_accuracy": round(n_exact / n, 4) if n else 0.0,
        "valid_json_rate": round(n_valid / n, 4) if n else 0.0,
        "per_field_accuracy": {f: round(field_ok[f] / n, 4) if n else 0.0
                               for f in FIELDS},
        "exact_match_count": n_exact,
        "valid_json_count": n_valid,
        "per_field_counts": field_ok,
        "decoding": {"greedy": True, "do_sample": False,
                     "max_new_tokens": MAX_NEW_TOKENS},
        "eval_seconds": round(time.time() - t0, 1),
    }
    return results, preds


def analyse_failure(pred):
    """Heuristic diagnosis of why one fine-tuned prediction failed."""
    gold, parsed = pred["expected"], pred["parsed"]
    if parsed is None:
        return ("Output is not parseable JSON: the model produced free text / "
                "a truncated or malformed object instead of a single JSON "
                "object, so no field could be scored. This is the classic "
                "base-model failure mode (continuing the prompt as prose); "
                "after LoRA it should be rare and usually means the "
                "generation was cut off or the model fell back to prose.")
    wrong = [f for f in FIELDS if not field_equal(gold, parsed, f)]
    missing = [f for f in FIELDS if f not in parsed]
    extra = [k for k in parsed if k not in FIELDS]
    bits = []
    if missing:
        bits.append(f"missing key(s): {', '.join(missing)}")
    if extra:
        bits.append(f"unexpected extra key(s): {', '.join(extra)}")
    for f in wrong:
        if f in parsed:
            bits.append(f"{f}: got {parsed[f]!r}, expected {gold[f]!r}")
    if "title" in wrong or "author" in wrong:
        bits.append("Likely cause: the title/author are unseen invented "
                    "names (test names are disjoint from train), so the "
                    "model must copy them from the request verbatim; a "
                    "copy error (dropped/changed word) breaks that field "
                    "while the format stays correct.")
    if "year" in wrong:
        bits.append("Likely cause: the year is a 4-digit number that must "
                    "be copied exactly; digit-level copying is the "
                    "hardest sub-task for a small model.")
    if "genre" in wrong:
        bits.append("Likely cause: genre must be one of the 12 fixed "
                    "labels; the model paraphrased it or picked a "
                    "near-synonym instead of the exact label.")
    return " ".join(bits) if bits else "Unknown failure."


def write_failures(preds, path):
    fails = [p for p in preds if not p["exact_match"]][:3]
    lines = ["# Fine-tuned model failures (held-out test set)\n",
             f"{len(fails)} failure(s) shown (first 3 of "
             f"{sum(1 for p in preds if not p['exact_match'])} total).\n"]
    for i, p in enumerate(fails, 1):
        lines += [f"## {i}. Request: {p['request']}\n",
                  f"Expected: `{json.dumps(p['expected'], ensure_ascii=False)}`\n",
                  f"Got (raw): `{p['raw_output']}`\n",
                  f"Parsed: `{json.dumps(p['parsed'], ensure_ascii=False)}`\n",
                  f"**Why:** {analyse_failure(p)}\n"]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=str(BASE_DIR / "adapter"))
    ap.add_argument("--test-file", default=str(BASE_DIR / "data" / "test.jsonl"))
    ap.add_argument("--results-dir", default=str(BASE_DIR / "results"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--base-only", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    rows = load_jsonl(args.test_file)
    if args.limit:
        rows = rows[: args.limit]
    print(f"test examples: {len(rows)}")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32).to(device)
    base.eval()

    results, preds = evaluate(base, tokenizer, rows, device)
    results["model"] = f"{args.model} (base)"
    (results_dir / "results_base.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    with open(results_dir / "predictions_base.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("BASE:", json.dumps(results, indent=2))

    if args.base_only:
        print("base-only run: fine-tuned eval skipped (no adapter yet).")
        return

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        raise SystemExit(f"adapter not found at {adapter_path} -- "
                         f"run train.py first, or use --base-only")
    ft = PeftModel.from_pretrained(base, str(adapter_path)).to(device)
    ft.eval()
    results_ft, preds_ft = evaluate(ft, tokenizer, rows, device)
    results_ft["model"] = f"{args.model} + LoRA adapter ({args.adapter})"
    (results_dir / "results_finetuned.json").write_text(
        json.dumps(results_ft, indent=2), encoding="utf-8")
    with open(results_dir / "predictions_finetuned.jsonl", "w", encoding="utf-8") as f:
        for p in preds_ft:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("FINE-TUNED:", json.dumps(results_ft, indent=2))
    fails = write_failures(preds_ft, results_dir / "failures.md")
    print(f"failures.md: {len(fails)} failure(s) documented")


if __name__ == "__main__":
    main()
