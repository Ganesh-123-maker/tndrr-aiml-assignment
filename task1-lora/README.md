# Task 1 — LoRA fine-tuning (Qwen2.5-0.5B, base)

> **STATUS — 16 Nov 2026, ~23:30 IST: FINE-TUNED NUMBERS PENDING.**
> The code, data and eval harness are complete and the data is generated,
> but **no model has been trained or evaluated on this VM yet**: the
> download of `Qwen/Qwen2.5-0.5B` from Hugging Face failed after
> ~19 minutes with a 403 from the Hugging Face Xet (CAS) download
> server, and the run was not retried (see "Compute path" below).
> Because the base model could not be downloaded, the base-model eval
> could not be run here either — there are **no results JSON files yet**.
> The fine-tuned numbers require **one Colab run by the user**
> (`colab_finetune.ipynb`, T4 GPU, ~15–30 min); the exact steps are in
> "Colab steps" below. Nothing in this README is a measured result
> until `results/results_base.json` / `results/results_finetuned.json`
> exist — the table in "Results" is a placeholder to be filled in from
> those files.

## Task definition

Fine-tune a small open-weight model (≤ 1.5B params) with LoRA on a
narrow, verifiable task: **plain-English book request → fixed JSON**.

Input (example):

> Add the book The Silent River by Ana Petrova, a mystery book from 1987

Required output (exact keys, `year` an integer):

```json
{"title": "The Silent River", "author": "Ana Petrova", "genre": "mystery", "year": 1987}
```

The model must learn the *mapping* (extract the four fields, emit
valid JSON with exactly these keys), not memorise one phrasing.

## Data generation

`data/gen_data.py` (seed fixed at 42, so regeneration is identical)
synthesises the data from templates — no external dataset, no manual
labelling.

* **10 distinct surface templates** (e.g. `Add the book {title} by
  {author}, a {genre} book from {year}`; `New entry: {title} —
  {author}, {genre}, published {year}.`; `Book: {title}; Author:
  {author}; Genre: {genre}; Year: {year}. Please add it.`) — the model
  cannot pass by matching one string pattern.
* **Values:** titles/authors composed from word/name pools (invented,
  e.g. "The Falling Ember" by "Arjun Rossi"); genres from a fixed list
  of 12 (`fiction, mystery, science fiction, fantasy, romance,
  thriller, historical fiction, biography, horror, poetry, adventure,
  non-fiction`); years uniform in 1950–2025.
* **Held-out test is genuinely held out:** first names, last names and
  title words are *disjoint* across splits, so every test title/author
  is unseen in training (verified: title overlap train/test = 0).
  The test therefore measures generalisation / verbatim copying of
  unseen names, not recall.
* Splits: **train 800, dev 100, test 100** → `data/train.jsonl`,
  `data/dev.jsonl`, `data/test.jsonl`. Each line has `prompt`
  (full model input), `request`, `response` (target JSON string) and
  the gold fields. The prompt is fixed for train and eval:
  `Convert the following book request into JSON with keys title,
  author, genre, year.\nRequest: <request>\nJSON:`

Regenerate with:

```bash
cd task1-lora/data
~/workspace/tndrr-aiml-assignment/.venv/bin/python gen_data.py
```

## Model

**Qwen/Qwen2.5-0.5B (base, 0.494B params) — NOT the Instruct version.**
Using the base model is deliberate: the comparison "base vs
fine-tuned" is only fair if the starting point has not already been
instruction-tuned on tasks like this. Fallback if Qwen2.5-0.5B cannot
be obtained: `HuggingFaceTB/SmolLM2-360M` (SmolLM2-1.7B is out — over
the 1.5B cap). The fallback was **not** used: no alternative download
was attempted after the Qwen download was blocked/denied.

## LoRA settings — and why

Recorded exactly in `lora_config.json` (train.py rewrites it with the
real run config + wall-clock time when it runs).

| Setting | Value | Why |
|---|---|---|
| Rank `r` | 16 | The task is narrow (copy 4 fields, emit a fixed schema). r=16 gives ~1.1M trainable params on a 0.5B model (computed from the config: hidden 896, 24 layers, q_out 896, v_out 128) — enough capacity to learn the format + copying, small enough to train fast and not overfit 800 examples. r=4 would risk underfitting the copy behaviour; r=64 would add parameters (and time) for no visible gain on a task this small. |
| Alpha | 32 | Scaling factor alpha/r = 2: the LoRA update is applied at 2× strength, the standard pairing with r=16 — a moderate update magnitude that lets the adapter actually move the base model's behaviour in 3 epochs without destabilising it. |
| Target modules | `q_proj` + `v_proj` | The two attention projections that control *what* the model attends to and *what* it passes on — the right place to teach "attend to the name/year tokens and copy them". Restricting to these two (rather than all linear layers) keeps trainable params and step time down on CPU/Colab; `q+v` is the classic LoRA recipe and sufficient for a format/copying task. |
| Learning rate | 2e-4 | Standard LoRA LR for a small model: high enough that the adapter learns in 3 epochs, low enough (with the alpha/r=2 scaling) to avoid wrecking the base weights. Full-FT LRs (1e-5–5e-5) would be far too small for training only the adapter. |
| Epochs | 3 | 800 examples is small; 1 epoch underfits the JSON format, 3 epochs is the usual sweet spot, and beyond ~5 epochs the model would start memorising the training titles/authors (which are disjoint from test anyway, so extra epochs buy nothing). |
| Batch | 4, grad accum 4 (effective 16) | Batch 4 is what fits a 0.5B model in fp32 on a ~7 GB CPU VM; accumulation to an effective 16 stabilises the gradient without the memory cost of a true batch-16 step. On a Colab T4 the same settings are kept so the run is identical. |
| Loss | cross-entropy on completion tokens only | Prompt tokens are masked (`-100`), so the model is scored only on the JSON it produces — it is not rewarded for "predicting" the request it was given. |
| Max length | 192 tokens | Prompts are ~60 tokens, responses ~35 — 192 leaves comfortable headroom with no truncation. |

## Run commands

All commands run from `task1-lora/` using the shared venv
(`~/workspace/tndrr-aiml-assignment/.venv/bin/python`). `train.py`
auto-detects device (CUDA if present, else CPU). The adapter is saved
to `adapter/` — a run artefact, **never committed** (root `.gitignore`
ignores it, as it does `*.safetensors` / checkpoints).

```bash
cd ~/workspace/tndrr-aiml-assignment/task1-lora

# timed smoke run (20 examples, 1 epoch) — extrapolate before the full run
../../.venv/bin/python train.py --max-examples 20 --epochs 1

# full run (3 epochs, 800 examples)
../../.venv/bin/python train.py

# evaluate BOTH models on the held-out test set (n=100, greedy decoding)
../../.venv/bin/python eval.py

# base model only (no adapter needed)
../../.venv/bin/python eval.py --base-only

# quick check on 10 test examples
../../.venv/bin/python eval.py --limit 10
```

## Compute path — what actually happened

1. **Smoke run: NOT COMPLETED.** The smoke run requires the model
   weights, and the download failed first (see below). No training
   step has ever run on this VM, so there is no measured per-step
   time and no valid extrapolation to a full CPU run. The planned
   smoke run (20 examples, 1 epoch) is the first cell after install
   in the Colab notebook.
2. **Model download FAILED, not retried.** `AutoModelForCausalLM.
   from_pretrained("Qwen/Qwen2.5-0.5B")` ran ~19 minutes (tokenizer
   downloaded fine, vocab 151,665) and then failed loading the model
   weights with a 403 from the Hugging Face Xet/CAS server
   (`cas-server.xethub.hf.co`, `File reconstruction error … HTTP
   status client error (403 Forbidden)`), and the action was
   declined — so it was **not retried and no fallback model was
   downloaded**. The failure may be transient (Xet/HF-side) or a
   network restriction on this VM; it is not evidence the model
   does not exist.
3. **Base eval: NOT RUN here** — it needs the same weights, so it
   could not run either. It will run in Colab (the notebook's eval
   cell does base + fine-tuned in one go).
4. **Decision: Colab path.** Because no local training time could be
   measured, the honest course is the notebook, not a guessed CPU
   estimate. A 0.5B model in fp32 on 4 CPU cores would in any case
   be expected to take hours, not minutes, for 150 optimiser steps.

## Colab steps (for the user — one run, ~15–30 min)

1. Open `colab_finetune.ipynb` in Google Colab.
2. **Runtime → Change runtime type → T4 GPU.** Run the GPU-check
   cell: it must print a Tesla T4. If it says CPU, stop and change
   the runtime.
3. Get the code: run the clone cell — **fill in the real repo URL
   first** (the notebook has a `<USER>/<REPO>` placeholder; do not
   guess it). If cloning fails, upload the `task1-lora/` folder via
   the Files panel instead and `%cd` into it (the notebook says how).
4. Run the install cell (~2–4 min), then the data sanity cell.
5. Run the smoke cell (20 examples, ~1–2 min). If it fails, stop
   and report the error — do not proceed to the full run.
6. Run the full training cell (~10–20 min on a T4), then the eval
   cell (~2–5 min).
7. Download **`results/results_base.json`**,
   **`results/results_finetuned.json`**, **`results/failures.md`**
   and **`lora_config.json`** from the Files panel and send them
   back. Do not push or commit anything from Colab.

## Results — PENDING the Colab run

| Metric (n=100, held-out test) | Base Qwen2.5-0.5B | + LoRA fine-tuned |
|---|---|---|
| Exact-match accuracy | _pending_ | _pending_ |
| Valid-JSON rate | _pending_ | _pending_ |
| Per-field: title | _pending_ | _pending_ |
| Per-field: author | _pending_ | _pending_ |
| Per-field: genre | _pending_ | _pending_ |
| Per-field: year | _pending_ | _pending_ |

Definitions (implemented in `eval.py`, identical decoding for both
models — greedy, `do_sample=False`, `max_new_tokens=80`):
* **Exact-match** = parsed JSON dict equals the gold dict exactly
  (key order / whitespace ignored; year compared numerically).
* **Valid-JSON** = the output contains a parseable JSON object.
* **Per-field** = fraction of the 100 examples where that field
  equals gold (year numerically, so `"2016"` counts as 2016).

Source of truth when the run exists: `results/results_base.json`
and `results/results_finetuned.json` (plus `results/predictions_*.jsonl`
with every raw output). Expectation (not a result): the base model
should score near-zero exact-match — a base model continues the
prompt as prose rather than emitting a single JSON object — and the
fine-tuned model should score high, with the residual errors
concentrated in title/author copying of unseen names.

## Failure cases — PENDING

`eval.py` writes `results/failures.md` with the first 3 fine-tuned
failures and a per-failure diagnosis (not-parseable JSON vs wrong /
missing field, and which field). No failures exist yet because no
model has been evaluated. This section will be filled in from that
file after the Colab run; the expected failure modes are verbatim
copy errors on unseen titles/authors and digit errors on years.

## Memory calculation — fine-tuning a 7B model

Full working in `memory_calculation.md`. Assumptions: bf16 weights
2 B/param, fp32 master weights 4 B/param, AdamW m+v 8 B/param, bf16
gradients 2 B/param, NF4 0.5 B/param; activations ~1–4 GB depending
on checkpointing.

| Method | Total (7B) |
|---|---|
| Full fine-tuning | **≈ 115 GB** (14 weights + 28 master + 14 grads + 56 AdamW + ~2–4 activations) |
| LoRA (bf16 base, ~8.4M trainable) | **≈ 16 GB** (14 frozen weights + ~0.13 trainable/optimiser + ~2 activations) |
| QLoRA (NF4 base) | **≈ 5 GB** (3.5 quantised weights + ~0.13 + ~1 activations with checkpointing) |

So a 7B model needs multi-GPU / 80 GB-class hardware for full FT,
fits on one 24 GB card with LoRA, and fits on a free Colab T4 with
QLoRA.

## Files

```
task1-lora/
├── README.md                  this file
├── train.py                   LoRA training (auto device, prompt-masked loss)
├── eval.py                    base vs fine-tuned eval, greedy, n=100
├── lora_config.json           exact LoRA + training config (PLANNED until a run overwrites it)
├── memory_calculation.md      7B memory working: full FT / LoRA / QLoRA
├── colab_finetune.ipynb       ready-to-run Colab notebook (T4 GPU)
├── data/
│   ├── gen_data.py            synthetic data generator (seed 42)
│   ├── train.jsonl            800 examples
│   ├── dev.jsonl              100 examples
│   └── test.jsonl             100 examples (held out, disjoint names)
├── adapter/                   LoRA adapter — RUN ARTEFACT, created by train.py, never committed
├── checkpoints/               Trainer output dir (save disabled; empty)
└── results/                   (empty until eval.py runs)
    ├── results_base.json
    ├── results_finetuned.json
    ├── predictions_base.jsonl
    ├── predictions_finetuned.jsonl
    └── failures.md
```

## Environment

Shared venv `~/workspace/tndrr-aiml-assignment/.venv`:
torch 2.15.0+cpu, transformers 5.23.0, peft 0.21.3. No GPU on this
VM (4 CPUs, ~7 GB RAM). No new venv was created and torch was not
reinstalled.
