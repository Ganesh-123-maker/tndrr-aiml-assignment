#!/usr/bin/env python3
"""Build report/tndrr-aiml-report.pdf (max 4 pages).

Every number below is copied from a results JSON in the repo:
- task2-rag/eval/results.json (recall@5 / recall@1)
- task3-local/results_f16.json, results_q4_k_m.json (tok/s, RSS)
- task1-lora/memory_calculation.md (7B memory working)
Task 1 fine-tuned numbers are PENDING (see report) — no invented values.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                Table, TableStyle, HRFlowable)

OUT = "tndrr-aiml-report.pdf"

S = {}
S["title"] = ParagraphStyle("title", fontSize=16, leading=19, spaceAfter=2,
                            fontName="Helvetica-Bold")
S["sub"] = ParagraphStyle("sub", fontSize=9, leading=12, spaceAfter=6,
                          fontName="Helvetica")
S["h1"] = ParagraphStyle("h1", fontSize=12, leading=14, spaceBefore=8,
                         spaceAfter=3, fontName="Helvetica-Bold")
S["h2"] = ParagraphStyle("h2", fontSize=10, leading=12, spaceBefore=6,
                         spaceAfter=2, fontName="Helvetica-Bold")
S["body"] = ParagraphStyle("body", fontSize=9, leading=12, spaceAfter=4,
                           fontName="Helvetica")
S["small"] = ParagraphStyle("small", fontSize=8, leading=10.5, spaceAfter=3,
                            fontName="Helvetica")
S["bullet"] = ParagraphStyle("bullet", parent=S["body"], leftIndent=10,
                             bulletIndent=2, spaceAfter=2)

S["cell"] = ParagraphStyle("cell", fontSize=8, leading=10,
                           fontName="Helvetica")
S["cellb"] = ParagraphStyle("cellb", parent=S["cell"],
                            fontName="Helvetica-Bold")
S["cellh"] = ParagraphStyle("cellh", parent=S["cell"],
                            fontName="Helvetica-Bold")

GREY = colors.HexColor("#555555")
LIGHT = colors.HexColor("#E8E8E8")


def P(text, style="body"):
    return Paragraph(text, S[style])


def tbl(header, rows, widths):
    def c(text, style="cell"):
        return Paragraph(text, S[style])
    data = [[c(h, "cellh") for h in header]]
    for row in rows:
        data.append([c(x) for x in row])
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEADING", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def bullets(items):
    return [Paragraph(f"&bull; {x}", S["bullet"]) for x in items]


story = []
story.append(P("tndrr AI/ML Intern — Assignment Report", "title"))
story.append(P("Barmavath Ganesh · B.Tech CSE, IIIT Guwahati · "
               "github.com/Ganesh-123-maker/tndrr-aiml-assignment · 17 Nov 2026", "sub"))
story.append(HRFlowable(width="100%", thickness=0.6, color=GREY))
story.append(P("All numbers are from my own runs on a CPU-only VM (4 CPUs, ~7 GB RAM, "
               "no GPU) unless labelled a calculation. Raw results JSON files sit next "
               "to the code that produced them. One honest gap up front: Task 1 has no "
               "measured numbers yet — the model download failed on this machine and "
               "one Colab run is still pending (see Task 1).", "small"))

# ---------------- Task 1 ----------------
story.append(P("Task 1 — LoRA fine-tuning (plain English → fixed JSON)", "h1"))
story.append(P("<b>Setup.</b> Task: turn a book request into "
               '{"title", "author", "genre", "year"}. Model: Qwen2.5-0.5B base '
               "(≤1.5B). Data synthesised from 10 templates (seed 42): 800 train / "
               "100 dev / 100 test, with disjoint title/author pools between train "
               "and test, 12 genres. Metric: exact-match accuracy + valid-JSON rate "
               "+ per-field accuracy. LoRA: r=16, α=32, target q_proj+v_proj, "
               "lr 2e-4, 3 epochs, batch 4 × grad-accum 4 (effective 16), max length "
               "192, loss on completion tokens only.", "body"))
story.append(P("<b>Why these settings.</b> Rank 16 keeps the adapter small (~1.1M "
               "trainable params) while giving the format/copying behaviour room to "
               "be learned; α=32 gives the standard α/r = 2 scaling. q_proj+v_proj "
               "are the two attention projections that decide what the model attends "
               "to and passes on — the right place to teach “copy the name/year”; "
               "restricting to them keeps step time down. lr 2e-4 is the usual LoRA "
               "range (full-FT LRs would be far too small for an adapter alone); "
               "3 epochs is the sweet spot for 800 examples before memorisation.",
               "body"))
story.append(P("<b>Status — results pending one Colab run.</b> The code, data and eval "
               "harness are complete, but no training or evaluation ran here: "
               "downloading Qwen2.5-0.5B from Hugging Face failed (403 from HF's Xet "
               "server) and was not retried. A ready-to-run notebook "
               "(<i>task1-lora/colab_finetune.ipynb</i>, T4 GPU, ~15–30 min) runs the "
               "same train.py + eval.py and writes results_base.json / "
               "results_finetuned.json. The base-vs-fine-tuned table will be filled "
               "from those files; no number is claimed until then. The three "
               "failure examples the task asks for will come from the fine-tuned "
               "model's test errors in that run.", "body"))
story.append(P("7B fine-tuning memory — worked calculation (decimal GB, 1 GB = 10<sup>9</sup> B; "
               "AdamW keeps m+v in fp32, 8 B/param; activations ≈2–4 GB small batch)", "h2"))
story.append(tbl(
    ["Approach", "Weights", "Grads", "AdamW m+v", "Activations", "Total"],
    [["(a) Full FT (bf16, fp32 master)",
      "14 + 28", "14", "56", "~2–4", "<b>≈115 GB</b>"],
     ["(b) LoRA (r=16, q+v, ~8.4M trainable)",
      "14", "0.03", "0.07", "~2", "<b>≈16 GB</b>"],
     ["(c) QLoRA (NF4 base, checkpointing)",
      "3.5", "0.03", "0.07", "~1", "<b>≈5 GB</b>"]],
    [42*mm, 20*mm, 18*mm, 22*mm, 24*mm, 22*mm]))
story.append(P("Full FT is dominated by optimiser states (56 GB, 4× the weights) — "
               "it needs multi-GPU or an 80 GB card. LoRA trains only 8.4M params, "
               "so the optimiser shrinks to ~0.13 GB and a 24 GB card suffices. "
               "QLoRA additionally stores frozen weights in 4-bit NF4 (3.5 GB), so "
               "it fits an 8 GB consumer GPU and a free Colab T4.", "small"))

# ---------------- Task 2 ----------------
story.append(P("Task 2 — Retrieval (RAG) from scratch", "h1"))
story.append(P("<b>Setup.</b> Corpus: 38 English Wikipedia pages on the history of "
               "computing (264,705 words), fetched 16 Nov 2026 via the MediaWiki API "
               "and saved in <i>task2-rag/data/</i>. Stack: sentence-transformers "
               "all-MiniLM-L6-v2 (384-dim) + plain NumPy cosine similarity — no "
               "LangChain, LlamaIndex or FAISS. Two chunk sizes, both with 50-token "
               "overlap, token-counted exactly: 256 tokens (1,711 chunks) vs 512 "
               "tokens (772 chunks). 15 questions written by me, each with a verbatim "
               "gold passage; a hit means the gold passage appears in a top-5 chunk.",
               "body"))
story.append(tbl(
    ["Chunk size", "Chunks", "Recall@5 (n=15)", "Recall@1"],
    [["256 tokens", "1,711", "<b>0.800 (12/15)</b>", "0.333 (5/15)"],
     ["512 tokens", "772", "0.733 (11/15)", "0.200 (3/15)"]],
    [35*mm, 25*mm, 45*mm, 43*mm]))
story.append(P("<b>Why 256 won.</b> Large chunks dilute the answer: q10 (Kilby) and "
               "q11 (Intel 4004) hit at 256 (ranks 3, 2) but miss at 512, where the "
               "Kilby gold chunk scores −0.024 (rank 483/772) — the answer sentence "
               "is ~10% of a 416-word chunk, so mean-pooling washes it out. The "
               "trade-off is not one-sided: q03 hits at 512 (rank 4) but misses at "
               "256 (rank 8), because a broad question needs the context small chunks "
               "split apart. A second hazard: 727/772 of the 512-token chunks "
               "re-tokenize to &gt;512 tokens, so the encoder silently truncates "
               "their tails; 0/1,711 of the 256-token chunks do.", "body"))
story.append(P("<b>Two failures.</b> (1) q01 (ENIAC, “first programmable general-purpose "
               "computer”) — gold chunk ranks 57th (0.53); a dozen near-identical "
               "“first … computer” claims drown it. Fix: hybrid BM25 (the exact phrase "
               "occurs in only the gold chunk) or a cross-encoder re-ranker over the "
               "top-20. (2) q08 (Grace Hopper, “first compiler”) — gold ranks 15th "
               "(0.45); all top-5 are the right article but wrong sections (awards, "
               "buildings). Fix: BM25 on “first compiler”, a re-ranker, or query "
               "rewriting (“Grace Hopper first compiler”).", "body"))
story.append(P("<b>BM25, in my words.</b> Keyword search done properly: it scores a "
               "chunk by the exact query words it contains, weighting rare words "
               "(“Kilby”, “ENIAC”) far above common ones (“computer”, “first”) and "
               "correcting for length. That is exactly what embeddings are bad at — "
               "an embedding averages a whole chunk, so a distinctive name, date or "
               "exact phrase gets washed out. BM25's weakness is the mirror image: "
               "it only matches words literally seen, so it fails on paraphrase. "
               "Hence hybrid: embeddings for meaning, BM25 for exact words. "
               "<b>A re-ranker</b> is a second, slower model that reads the question "
               "and one candidate chunk <i>together</i> and scores how well that chunk "
               "answers it — it can spot the answer sentence and reject an awards "
               "listing. It is too slow for the whole corpus, so it re-orders only "
               "the top ~20 from the fast search.", "body"))

# ---------------- Task 3 ----------------
story.append(P("Task 3 — Running a model locally & optimisation", "h1"))
story.append(P("<b>Setup.</b> Qwen2.5-0.5B-Instruct on CPU with llama.cpp (prebuilt "
               "b12441), same 10 prompts in both settings (temp 0.0, seed 42, 128 "
               "tokens, 4 threads, ctx 512). Peak RSS polled from /proc every 50 ms.",
               "body"))
story.append(tbl(
    ["Setting", "File size", "Generation", "Prompt-eval", "Peak RSS", "Wall (10 prompts)"],
    [["F16", "1,266.4 MB", "7.95 tok/s (med 7.9)", "40.5 tok/s (med 47.0)",
      "1,290.8 MB", "142.9 s"],
     ["Q4_K_M", "491.4 MB", "<b>10.03 tok/s</b> (med 9.6, +26%)",
      "52.3 tok/s (med 46.1)", "<b>581.6 MB</b>", "90.1 s"]],
    [20*mm, 24*mm, 34*mm, 34*mm, 20*mm, 26*mm]))
story.append(P("<b>Quality (same 10 prompts).</b> The two settings barely differ — "
               "they agree almost everywhere, including on the same mistakes: both "
               "say Canberra is in the wrong state (NSW vs Victoria; it is the ACT), "
               "both compute 17−9=8 for “all but 9 run away” (the answer is 9), both "
               "explain where plane-crash “survivors” would be buried. Code (is_even) "
               "and the planets list are identical. The only place Q4 looks slightly "
               "worse is the lighthouse story (more drift). At 0.5B params the "
               "model's own limits dominate; quantisation costs almost nothing "
               "observable here.", "body"))
story.append(P("<b>KV cache, in my words.</b> When a transformer generates one token "
               "at a time, each new token must attend to all previous tokens. "
               "Recomputing that from scratch every step would get slower with every "
               "token, so after processing a token once the model saves its key (K) "
               "and value (V) vectors — one pair per layer — and reuses them. Each "
               "new token then needs only its own K/V plus a cheap lookup. The cost "
               "is memory: the cache grows with context length. For this model "
               "(24 layers, 2 KV heads, head dim 64, fp16) that is 12 KB/token — "
               "~24 MB per user at 2,048 tokens. <b>Why quantisation saves memory:</b> "
               "weights are just numbers; F16 stores each as 2 bytes, Q4_K_M stores "
               "each as a ~4-bit integer plus one shared scale per small block — "
               "here 1,266 MB → 491 MB (2.58× smaller, 0.78 vs 2.01 bytes/param), so "
               "less data moves from RAM to processor per token. The price is "
               "rounding error, usually too small to change the output.", "body"))
story.append(P("Serving 20 users on one 24 GB GPU (back-of-envelope)", "h2"))
story.append(P("<b>Assumptions:</b> Qwen2.5-0.5B-Instruct, Q4_K_M, weights 0.49 GB "
               "(measured); context 2,048 tokens/user; KV cache 12 KB/token fp16 "
               "(computed above) → ~24 MB/user, ~0.5 GB for 20 users; target "
               "~10–20 tok/s per user (200–400 tok/s aggregate); vLLM with continuous "
               "batching, prefix caching, chunked prefill. <b>Budget:</b> 0.49 GB "
               "weights + ~0.5 GB KV + ~1–2 GB activations/overhead ≈ <b>2–3 GB of "
               "24 GB</b> — ~8× headroom. <b>Throughput:</b> decoding a 0.5B model is "
               "memory-bandwidth bound; on a ~300 GB/s card (e.g. L4) each batched "
               "step streams 0.5 GB in ~1.7 ms, ~590 steps/s, ~30 tok/s per user at "
               "batch 20 — the target fits. <b>Verdict: feasible with big headroom.</b> "
               "Serve Q4_K_M on vLLM, cap context at 4k, allow ~32 concurrent "
               "sequences. <b>What breaks first:</b> not weights or KV at 2–4k "
               "context — prefill bursts (all 20 users sending long prompts at once "
               "spike time-to-first-token; chunked prefill spreads it out), then KV "
               "at very long contexts (at 32k/user, KV alone is ~7.7 GB; at 128k it "
               "does not fit — then switch KV to 8-bit or cap context).", "body"))
story.append(P("Reproducibility &amp; limits. Task 1: <i>task1-lora/</i> — "
               "<i>python data/gen_data.py &amp;&amp; python train.py &amp;&amp; python "
               "eval.py</i> (or the Colab notebook). Task 2: <i>task2-rag/</i> — "
               "<i>download_corpus.py, build_index.py --chunk-size 256/512, eval.py</i> "
               "(CPU index builds ~8/5 min). Task 3: <i>task3-local/</i> — "
               "<i>python3 bench.py</i> (stdlib only; downloads binary + GGUFs). "
               "Weights are downloaded at run time, never committed. Known rough "
               "edges are documented in each task README (Task 3's early runs were "
               "contaminated by sibling jobs on the shared VM and were re-run clean; "
               "Task 2's optional answer step is implemented but unverified "
               "end-to-end).", "small"))

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=36, rightMargin=36,
                        topMargin=36, bottomMargin=36,
                        title="tndrr AI/ML Intern — Assignment Report",
                        author="Barmavath Ganesh")
doc.build(story)
print("built", OUT)
