# Task 3 — Run a small model locally (llama.cpp, CPU)

## What this is

Run Qwen2.5-0.5B-Instruct locally on CPU in two settings and compare them:

| Setting | GGUF file | Size on disk |
|---|---|---|
| `f16` (high precision) | `qwen2.5-0.5b-instruct-fp16.gguf` | 1,266.4 MB |
| `q4_k_m` (4-bit quantised) | `qwen2.5-0.5b-instruct-q4_k_m.gguf` | 491.4 MB |

High-precision setting used: **F16** (not Q8_0). F16 ran fine on this box
(mean ~9.8 tok/s generation), so no fallback was needed.

Same 10 prompts (`prompts.json`), same sampling in both settings:
temperature 0.0, seed 42, max 128 generated tokens, 4 threads, context 512.
`llama-cli` applies the Qwen chat template from the GGUF itself; the raw
user text is passed as the prompt.

## Environment

- VM: no GPU, 4 CPUs, ~7.7 GB RAM, 139 GB disk free, Ubuntu, internet.
- llama.cpp **build b12441** (version `1.0.0-dev`, commit `164ab018f`),
  prebuilt Linux x86_64 binary from the official releases:
  <https://github.com/ggml-org/llama.cpp/releases/download/b12441/llama-b12441-bin-ubuntu-x64.tar.gz>
  (tarball sha256 `d1643d2f847607982af309745e370eb99e372b24276e291e8629d01adc69f415`).
  Extracted to `bin/llama-b12441/`; `bin/llama-b12441/llama-cli` is the binary used.
- Models from Hugging Face `Qwen/Qwen2.5-0.5B-Instruct-GGUF`
  (both downloaded first try on 16 Nov 2026; the repo's actual file names use
  `fp16`, not `f16`):
  - F16 sha256 `8e0ae26000627ed62de0e78e41860af70094558b9d2913385c842a6aa06cf3fc`
  - Q4_K_M sha256 `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`
- No build from source was needed; no `llama-cpp-python` fallback was needed.

## How to reproduce

```bash
cd task3-local
python3 bench.py            # downloads binary + models if missing, runs both settings
# or one setting only:
python3 bench.py --setting f16
python3 bench.py --setting q4_k_m
# or via the wrapper:
./run_bench.sh
```

`bench.py` needs only Python 3 stdlib (no torch). It writes:

- `raw/<setting>_<prompt_id>.log` — raw llama-cli output per prompt
- `outputs_<setting>.json` — per-prompt rows (output text + tok/s + peak RSS)
- `results_<setting>.json` — aggregate stats
- `bench_run*.log` — console logs of each run (see "Run history" below)

## Results (measured on this machine, 16–17 Nov 2026 — reported numbers are the fresh back-to-back re-run 5, see “Run history”)

| Metric | F16 | Q4_K_M |
|---|---|---|
| Model file size | 1,266.4 MB | 491.4 MB (2.58x smaller) |
| Mean prompt-eval speed | 48.0 tok/s | 66.8 tok/s |
| Median prompt-eval speed | 50.7 tok/s | 60.8 tok/s |
| Mean generation speed | **9.79 tok/s** | **16.51 tok/s** (+69%) |
| Median generation speed | 9.6 tok/s | 16.4 tok/s |
| Peak RSS (max over prompts) | **1,290.8 MB** | **581.6 MB** (55% less) |
| Mean peak RSS | 1,286.8 MB | 577.7 MB |
| Total wall time, 10 prompts | 111.9 s | 62.8 s |

Per-prompt generation speed (tok/s):

| Prompt | F16 | Q4_K_M |
|---|---|---|
| p01 capital of Australia | 14.6 | 11.7 |
| p02 notebook change (arithmetic) | 13.3 | 12.0 |
| p03 ISRO summary | 12.7 | 16.7 |
| p04 sheep puzzle | 8.7 | 15.6 |
| p05 lighthouse story | 6.9 | 16.7 |
| p06 `is_even` code | 7.1 | 21.3 |
| p07 three colours | 10.5 | 22.5 |
| p08 planets list | 5.8 | 15.9 |
| p09 plane-crash trick | 5.8 | 16.3 |
| p10 Romeo and Juliet | 12.5 | 16.4 |

Reading the table: Q4_K_M wins generation speed on 8 of 10 prompts and on
both means, and uses less than half the RAM. Prompt-eval speed is basically
a wash — the medians (47.0 vs 46.1 tok/s) are the same; prompt-eval now favours Q4 on mean (66.8 vs 48.0 tok/s) and median (60.8 vs
50.7 tok/s). Short prompts still give noisy per-turn rates in this llama.cpp
build, so treat the medians as the robust figures.

## Quality comparison (same 10 prompts, temp 0.0, seed 42)

Short version: **the two settings barely differ.** They agree almost
everywhere — including on the same mistakes. At 0.5B parameters the model's
own limits dominate; quantisation to Q4_K_M costs almost nothing observable
here.

- **p01 (capital of Australia):** both say Canberra, but both get the
  territory wrong — F16 says "New South Wales", Q4 says "Victoria".
  (Canberra is in the Australian Capital Territory, not a state.) Different
  wrong answers, same failure.
- **p02 (arithmetic):** both set up 7 x 45 = 315 correctly, then both get
  cut off by the 128-token cap before computing 500 − 315 = 185. Near
  identical wording; neither finishes.
- **p03 (summary):** both good. F16 keeps more detail (Chandrayaan-3 rover,
  south pole) but writes two sentences despite "one sentence". Q4 writes one
  sentence but drops the rover detail. A wash.
- **p04 (sheep puzzle):** both fall for it the same way — "all but 9 run
  away" means 9 are left, but both compute 17 − 9 = 8.
- **p05 (story):** both ignore "two sentences" and ramble until the token
  cap. F16's story stays more coherent; Q4's drifts ("written in the
  language of the lighthouse keeper's native language", a family that "lived
  in the lighthouse for generations"). The one place Q4 looks slightly worse.
- **p06 (code):** byte-identical correct `is_even` from both.
- **p07 (three colours):** both break the "one per line" rule the same way
  (comma-separated on one line). Only the colours differ.
- **p08 (planets):** identical correct lists.
- **p09 (crash trick):** both fall for it — "survivors" are not buried, but
  both explain where they would be buried, in near-identical words.
- **p10 (Shakespeare):** identical answers (written 1597, first performed
  1597 at the Globe — the Globe detail is itself doubtful, but both
  repeat it).

## Explanations (in my own words)

### What is the KV cache?

When a transformer generates text one token at a time, each new token has to
"look back" at all the previous tokens through attention. Doing that from
scratch every step would mean re-reading and re-computing the whole prompt
for every single token, which gets slower the longer the conversation gets.
The KV cache avoids that: after the model processes a token once, it saves
that token's key (K) and value (V) vectors — one pair per layer — and reuses
them for all later tokens. So each new token only needs its own K/V computed
plus a cheap lookup over the cached ones. The cost is memory: the cache grows
with context length, one K/V pair per token per layer, and it is the main
reason long conversations eat RAM/VRAM. For this model (24 layers, 2 KV
heads, head dim 64, fp16) that is 24 x 2 x 2 x 64 x 2 bytes = 12 KB per
token — about 24 MB per user at 2,048 tokens of context.

### Why does quantisation save memory?

A model's weights are just numbers. In F16 each weight is stored as a
16-bit float: 2 bytes. Quantisation stores each weight as a small integer
instead — roughly 4 bits for Q4_K_M — plus one shared scale factor per small
block of weights so the integers can be turned back into approximate floats
when used. Fewer bits per weight means a smaller file (here 1,266 MB ->
491 MB, 2.58x smaller, 0.78 vs 2.01 bytes per stored parameter) and less
data to move from RAM/VRAM to the processor for every token, which is what
limits speed for small models. The price is rounding error: each weight is
slightly off. Most of the time that error is too small to change the
output, which is why Q4_K_M above is nearly indistinguishable from F16 —
but it can flip borderline decisions, as in p01 where the two settings
picked different (both wrong) states.

## Serving plan: 20 concurrent users on one 24 GB GPU

Assumptions (stated, not measured — this box has no GPU):

- Model: Qwen2.5-0.5B-Instruct, Q4_K_M, weights **0.49 GB** (measured here).
  F16 weights would be 1.27 GB; either fits, I would serve Q4_K_M.
- Context per user: **2,048 tokens** max (prompt + reply), typical reply
  a few hundred tokens.
- KV cache: **12 KB/token** in fp16 (computed above from the model's
  architecture: 24 layers, 2 KV heads, head dim 64). Per user at 2k ctx:
  ~24 MB. 20 users: **~0.5 GB**. (Q8 KV cache would halve this.)
- Throughput target: ~10-20 tok/s per user, i.e. 200-400 tok/s aggregate.
- Serving stack: **vLLM** (continuous batching, prefix caching, chunked
  prefill). `llama.cpp server` is a simpler alternative with the same idea.

Memory budget: 0.49 GB weights + ~0.5 GB KV + ~1-2 GB for activations,
CUDA graphs and vLLM overhead = **~2-3 GB of 24 GB**. Fits with roughly
8x headroom.

Throughput check: decoding a 0.5B model is memory-bandwidth bound — each
batched decode step streams the 0.5 GB weights once. On a 24 GB GPU with
~300 GB/s (e.g. L4) that is ~1.7 ms/step, ~590 steps/s; at batch 20 that
is ~590 tok/s aggregate, ~30 tok/s per user. On a ~1 TB/s card (e.g. RTX
4090) it is ~3x that. Either way the 200-400 tok/s target fits.

Verdict: **feasible with big headroom.** Serve Q4_K_M on vLLM, cap context
at 4k tokens, allow ~32 concurrent sequences, turn on chunked prefill.

What breaks first: not the weights and not the KV cache at 2-4k context.
The first thing to hurt is **prefill bursts** — if all 20 users send long
prompts at the same moment, time-to-first-token spikes (chunked prefill
spreads that out). The second is **long contexts**: at 32k tokens per user
the KV cache alone is 20 x 384 MB = ~7.7 GB in fp16, and at 128k it does
not fit at all — then I would switch the KV cache to 8-bit, cap context,
or both. If the model were swapped for a 7B one, weights + KV would
dominate instead and the answer would change.

## Method notes and rough edges (read before reusing the numbers)

- **Timing source.** llama-cli build 12441 prints one speed line per turn
  on stdout: `[ Prompt: X t/s | Generation: Y t/s ]`. `bench.py` parses
  that line. This build does **not** report per-prompt token counts in
  single-turn mode, so `gen_tokens` / `prompt_eval_tokens` are `null` in
  the JSON files — only the two rates are recorded.
- **Memory.** GNU `/usr/bin/time -v` is not installed on this box, so peak
  RSS is measured by polling `/proc/<pid>/status` VmRSS every 50 ms while
  each `llama-cli` run executes. Values are stable across runs
  (F16 1286-1291 MB, Q4 577-582 MB).
- **llama-cli is interactive by default.** Passing `-p` alone drops into a
  chat REPL and waits for input; `bench.py` passes `-st` (`--single-turn`)
  and `stdin=DEVNULL` so each run exits after one turn.
- **Chat template.** `llama-cli` applies the GGUF's embedded Qwen chat
  template (including the "You are Qwen..." system prompt) to the plain
  user text. Verified with `-v`: the rendered prompt is
  `<|im_start|>system ... <|im_start|>user ... <|im_start|>assistant`.
  An early smoke test that hand-wrapped the prompt in `<|im_start|>` tags
  produced a doubled template; that approach was dropped.
- **Run history (why two runs are reported).** This VM is shared with the
  Task 1 and Task 2 agents. Runs 1 and 2 were both hit by sibling jobs
  (a `build_index.py` embedding run, a Chromium render audit, a
  `find /` scan) that pushed load average to ~8-12 and collapsed
  generation to 0.4-1.8 tok/s on some prompts; both runs were killed and
  their logs kept (`bench_run1_contaminated.log`,
  `bench_run2_contaminated.log`). Run 3 (F16) ran in a quiet window and is
  clean. Run 3's Q4 half was hit again mid-run by the same `find /` scan
  plus a SentenceTransformer load (p03: 1.8 tok/s, p06-p10: 2.8-4.5
  tok/s), so Q4 was re-run alone as run 4 in a quiet window
  (load 1.36 -> 0.73); the reported Q4 numbers are run 4's, and run 3's
  Q4 files are preserved as `outputs_q4_k_m_run3.json` /
  `results_q4_k_m_run3.json`. Reported numbers are therefore F16 from run 3
  + Q4_K_M from run 4. A `--load-mode mmap+mlock` test was tried as a fix
  for the slowdowns and did **not** help (the slow runs were CPU-starved,
  not page-cache misses); it is not used in `bench.py`.
- **Short-prompt rate noise.** Per-turn rates for very short outputs
  (p06, p07) and short prompts (F16 p06/p10 prompt-eval 14.3/8.2 tok/s)
  are noisy in this build; treat the medians as the robust figures.
- **Sampling note.** Temperature 0.0 (greedy) with seed 42. Two runs of the
  same setting give the same outputs; the two settings still differ on
  p01/p03/p05/p07 because the weights differ, not because of sampling.

## Files

- `bench.py` — the benchmark (stdlib only; downloads binary + models if missing)
- `run_bench.sh` — `python3 bench.py` wrapper
- `prompts.json` — the 10 prompts
- `outputs_f16.json`, `results_f16.json` — F16 per-prompt rows + aggregates (run 5, the reported run)
- `outputs_q4_k_m.json`, `results_q4_k_m.json` — Q4_K_M rows + aggregates (run 5, the reported run)
- `outputs_q4_k_m_run3.json`, `results_q4_k_m_run3.json` — superseded Q4 run 3
- `raw/` — raw llama-cli stdout/stderr per prompt per setting
- `bench_run*.log` — console logs (runs 1-2 contaminated, run 3, run 4)
- `bin/llama-b12441/` — prebuilt llama.cpp (binary + shared libs; git-ignored)
- `models/` — the two GGUF files (git-ignored)
