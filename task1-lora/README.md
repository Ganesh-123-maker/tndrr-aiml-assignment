# Task 1 — LoRA fine-tuning (Qwen2.5-0.5B, base)

> **STATUS — 17 Nov 2026, 08:15 IST: DONE, with measured numbers.**
> Trained and evaluated on this CPU-only VM (no Colab run was needed):
> **base exact-match 0.52 (52/100) vs fine-tuned 1.00 (100/100)** on the
> 100-example held-out test set, greedy decoding. The fine-tuned model
> made **0 errors**, so the "3 fine-tuned failure examples" the task
> asks for do not exist — the failure analysis below instead uses 3
> base-model failures and explains why fine-tuning fixed them. Full
> run: 3 epochs, 800 examples, 150 optimiser steps, **157.7 min wall
> clock**, final train loss 0.01554. Source of truth:
> `results/results_base.json` / `results/results_finetuned.json`.

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

1. **Model download FAILED, then recovered.** `AutoModelForCausalLM.
   from_pretrained("Qwen/Qwen2.5-0.5B")` ran ~19 minutes and failed
   loading the weights with a 403 from the Hugging Face Xet/CAS server
   (`cas-server.xethub.hf.co`); the retry was declined, so no fallback
   was attempted by the builder. On 17 Nov ~00:15 IST I retried the
   download a different way — a direct `curl` of the `resolve/main`
   URL, which bypasses Xet — and it worked: `model.safetensors`
   (988,097,824 bytes) downloaded in ~3 min and verified with
   `safetensors.safe_open` (290 tensors, Qwen2, hidden 896, 24 layers).
   The weights live in `model_qwen/` (local only, never committed).
2. **One code fix.** The smoke run crashed before any training step:
   `TrainingArguments` got `warmup_ratio`, which transformers 5.23.0
   removed. Fixed in `train.py`: `warmup_steps=5` (≈3% of the 150
   total steps) and the same change in the recorded config.
3. **Smoke run PASSED.** 20 examples, 1 epoch, 2 optimiser steps:
   91 s of training (~46 s/step), 0.219 samples/s, train loss 0.38 —
   extrapolating to 150 steps gave ~2–3 h for the full run, so CPU
   training was feasible and the Colab path was not needed.
4. **First full run KILLED by a VM restart.** Training reached step
   22/150 (~00:43 IST) when the VM was replaced; the process and its
   watcher died with it. Log preserved as `train_full.log.1`.
5. **Restarted from scratch ~04:54 IST and completed.** 150 steps,
   9,434 s of training (0.254 samples/s), average train loss 0.01554
   over all steps (per-step loss fell to ~0.00017 by epoch 3), **total wall clock 157.7 min**. A watcher script then ran
   `eval.py` automatically (base eval 1,400.8 s, fine-tuned 1,236.3 s).
   A 30-minute monitor cron watched the pipeline and reported
   completion; it is removed once this README is pushed.

## Colab steps (fallback — NOT needed now; CPU training succeeded)

The notebook (`colab_finetune.ipynb`, T4 GPU, ~15–30 min) remains as
a reproducible alternative path — it runs the same `train.py` +
`eval.py`. It is no longer needed for this submission.

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

## Results — measured 17 Nov 2026 (CPU, greedy, n=100 held-out test)

| Metric (n=100, held-out test) | Base Qwen2.5-0.5B | + LoRA fine-tuned |
|---|---|---|
| Exact-match accuracy | **0.52 (52/100)** | **1.00 (100/100)** |
| Valid-JSON rate | 1.00 (100/100) | 1.00 (100/100) |
| Per-field: title | 0.95 | 1.00 |
| Per-field: author | 1.00 | 1.00 |
| Per-field: genre | 0.57 | 1.00 |
| Per-field: year | 1.00 | 1.00 |
| Eval time | 1,400.8 s | 1,236.3 s |

Definitions (implemented in `eval.py`, identical decoding for both
models — greedy, `do_sample=False`, `max_new_tokens=80`):
* **Exact-match** = parsed JSON dict equals the gold dict exactly
  (key order / whitespace ignored; year compared numerically).
* **Valid-JSON** = the output contains a parseable JSON object.
* **Per-field** = fraction of the 100 examples where that field
  equals gold (year numerically, so `"2016"` counts as 2016).

Source of truth: `results/results_base.json` and
`results/results_finetuned.json` (plus `results/predictions_*.jsonl`
with every raw output).

Reading the numbers honestly: the base model already emits valid JSON
every time (1.00) and copies author/year perfectly — what it gets
wrong is the *convention*, not the content (see Failure cases). The
fine-tuned 1.00 is real but narrow: this is a templated synthetic
task, so it measures "learn the output format + copy unseen names
verbatim", not general book knowledge. The disjoint title/author
pools rule out memorisation of test answers, but the 10 templates
and 12 genres are shared across splits — a harder test would use
unseen phrasings and genres.

## Failure cases

`results/failures.md` is the full record. The fine-tuned model made
**0 errors on the 100-example test set, so there are no fine-tuned
failure examples to show** — the task's "3 failure examples" are
therefore 3 base-model failures, with the diagnosis of why
fine-tuning fixed each one:

1. **Genre casing (42 of the base model's 48 errors).**
   Request: *"Add Arjun Rossi's science fiction book The Falling Ember
   (2016)."* Base output: `"genre": "Science Fiction"` — expected
   `"science fiction"`. The base model title-cases the genre, as in
   ordinary prose; exact-match is case-sensitive, so all 42 such
   examples fail even though the content is right. Fine-tuning fixed
   it by teaching the exact lowercase label set.
2. **Whole-request copied into the title (all 5 title errors).**
   Request: *"Catalogue The Sleeping Tide by Lena Kaplan — non-fiction
   — 1985."* Base output: `"title": "Catalogue The Sleeping Tide by
   Lena Kaplan — non-fiction — 1985"` — the whole request, not just
   the title. This template ("Catalogue … — … — …") has no "by" cue
   separating title from author in the way the other templates do, so
   the base model falls back to copying everything. Fine-tuning fixed
   it: the adapter learned the extraction boundary for this template
   too (year was also a string `"1985"` in 3 of these 5; fine-tuned
   outputs an integer year in all 100).
3. **Genre + year together (3 errors).** Same two causes compounding:
   a title-cased genre plus a string year in the "Catalogue" template.
   Both are format/convention errors, not content errors — the base
   model never invented a wrong title, author or year out of thin air.

Why fine-tuning fixed all of them: every base failure is a format
convention (lowercase genre labels, extract-only-the-title, integer
year), and 3 epochs on 800 examples is exactly enough to teach those
conventions for 10 known templates. The residual risk is the reverse:
on an *unseen* template the fine-tuned model may be worse than base,
because it has overfit the 10 training phrasings — that is the honest
limit of this result, and the reason the 1.00 should not be read as
"the model understands books".

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
├── lora_config.json           exact LoRA + training config from the real run (wall clock 157.7 min)
├── memory_calculation.md      7B memory working: full FT / LoRA / QLoRA
├── colab_finetune.ipynb       ready-to-run Colab notebook (T4 GPU)
├── data/
│   ├── gen_data.py            synthetic data generator (seed 42)
│   ├── train.jsonl            800 examples
│   ├── dev.jsonl              100 examples
│   └── test.jsonl             100 examples (held out, disjoint names)
├── model_qwen/                Qwen2.5-0.5B weights, downloaded via direct CDN link
│                              (HF Xet 403 bypass) — LOCAL ONLY, never committed
├── train_full.log             full training log (157.7 min run); .1 = killed run
├── eval_full.log              eval log (base + fine-tuned)
├── adapter/                   LoRA adapter — RUN ARTEFACT, created by train.py, never committed
├── checkpoints/               Trainer output dir (save disabled; empty)
└── results/                   measured results (committed)
    ├── results_base.json      base: exact-match 0.52, valid JSON 1.00
    ├── results_finetuned.json fine-tuned: exact-match 1.00, valid JSON 1.00
    ├── predictions_base.jsonl
    ├── predictions_finetuned.jsonl
    └── failures.md            0 fine-tuned failures; 3 base failures analysed
```

## Environment

Shared venv `~/workspace/tndrr-aiml-assignment/.venv`:
torch 2.15.0+cpu, transformers 5.23.0, peft 0.21.3. No GPU on this
VM (4 CPUs, ~7 GB RAM). No new venv was created and torch was not
reinstalled.
