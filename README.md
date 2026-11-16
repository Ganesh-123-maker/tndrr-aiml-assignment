# tndrr AI/ML Intern — Assignment

Candidate: **Barmavath Ganesh** · B.Tech CSE, IIIT Guwahati
Submission: this repo + the 4-page report in [`report/tndrr-aiml-report.pdf`](report/tndrr-aiml-report.pdf)
Deadline: Thu 19 Nov 2026, 11:59 PM IST

Three tasks, one folder each. Every number in the report comes from my own
runs on a CPU-only VM (4 CPUs, ~7 GB RAM, no GPU); raw results JSON files
sit next to the code that produced them so any number can be traced.

## Task 1 — LoRA fine-tuning ([`task1-lora/`](task1-lora/))

Plain-English book request → fixed JSON (`title`, `author`, `genre`, `year`).
Qwen2.5-0.5B base, LoRA r=16, α=32, `q_proj`+`v_proj`, lr 2e-4, 3 epochs.
Data synthesised from templates (seed 42): 800 train / 100 dev / **100 test**.

> **Status, 17 Nov 2026: fine-tuned numbers are PENDING.** The Qwen2.5-0.5B
> download from Hugging Face failed on this VM (403 from HF's Xet server)
> and was not retried. The code, data, eval harness and a ready-to-run
> Colab notebook ([`task1-lora/colab_finetune.ipynb`](task1-lora/colab_finetune.ipynb),
> T4 GPU, ~15–30 min) are all in place — one Colab run produces
> `results/results_base.json` and `results/results_finetuned.json`, and the
> report will be updated with the real base-vs-fine-tuned numbers. Nothing
> in the report's Task 1 results table is a measured number yet; the 7B
> memory calculation (full FT ≈115 GB, LoRA ≈16 GB, QLoRA ≈5 GB, working
> shown) is complete in [`task1-lora/memory_calculation.md`](task1-lora/memory_calculation.md).

## Task 2 — RAG from scratch ([`task2-rag/`](task2-rag/))

No LangChain/LlamaIndex/FAISS — `sentence-transformers/all-MiniLM-L6-v2`
+ NumPy cosine similarity over **38 Wikipedia pages** (history of
computing, 264,705 words). 15 self-written questions, each with a verbatim
gold passage.

| Chunk size | recall@5 (n=15) | recall@1 |
|---|---|---|
| 256 tokens (50 overlap, 1,711 chunks) | **0.800 (12/15)** | 0.333 |
| 512 tokens (50 overlap, 772 chunks) | 0.733 (11/15) | 0.200 |

256 tokens won: large chunks dilute the answer (q10/q11 hit at 256, miss at
512), and 727/772 of the 512-token chunks re-tokenize past the model's
512-token limit, so their tails get silently truncated. Full analysis,
2 failure cases + fixes, and BM25 / re-ranker explanations in
[`task2-rag/README.md`](task2-rag/README.md).

## Task 3 — Local model & optimisation ([`task3-local/`](task3-local/))

Qwen2.5-0.5B-Instruct on CPU with llama.cpp (build b12441), same 10 prompts
both settings (temp 0.0, seed 42, 128 tokens):

| Setting | Size | Generation | Prompt-eval | Peak RSS |
|---|---|---|---|---|
| F16 | 1,266.4 MB | 7.95 tok/s (median 7.9) | 40.5 tok/s (median 47.0) | 1,290.8 MB |
| Q4_K_M | 491.4 MB | **10.03 tok/s** (median 9.6, +26%) | 52.3 tok/s (median 46.1) | **581.6 MB** |

Quality barely differs — the two settings agree almost everywhere,
including on the same mistakes. Serving plan (20 users, one 24 GB GPU):
**feasible with big headroom** — Q4_K_M + vLLM, ~2–3 GB of 24 GB; prefill
bursts break first. Details in [`task3-local/README.md`](task3-local/README.md).

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install transformers peft accelerate datasets sentence-transformers
# Task 1 (needs the model download; Colab notebook if it fails here)
cd task1-lora && python data/gen_data.py && python train.py && python eval.py
# Task 2
cd ../task2-rag && python download_corpus.py && python build_index.py --chunk-size 256 \
  && python build_index.py --chunk-size 512 && python eval.py
# Task 3 (stdlib only; downloads llama.cpp binary + GGUFs itself)
cd ../task3-local && python3 bench.py
```

Model weights are downloaded at run time, never committed. See each task's
README for the full method, results, and rough edges.
