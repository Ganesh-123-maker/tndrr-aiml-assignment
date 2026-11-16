# Task 2 — Question-answering retrieval from scratch

A dense retrieval system over Wikipedia, built with **no LangChain, no
LlamaIndex, no FAISS**: one embedding model + NumPy cosine similarity.
Retrieval numbers below depend only on embeddings — no LLM is involved in
retrieval or in the evaluation. (An optional answer step exists in
`answer.py`, but it is separate and its output never feeds the numbers.)

## Corpus

- **Source:** English Wikipedia, fetched 2026-11-16 through the public
  MediaWiki API (`https://en.wikipedia.org/w/api.php`, `prop=extracts`,
  plain text). Reproducible: re-run `download_corpus.py`.
- **Topic:** one coherent area — the history of computing.
- **Size: 38 pages, 264,705 words, 1,733,862 characters.** One file per
  article in `data/`; titles + URLs in `corpus_manifest.json`.
- The 38 titles: History of computing, Computer, Central processing unit,
  Transistor, Integrated circuit, Microprocessor, Personal computer,
  Internet, World Wide Web, Operating system, Programming language,
  Artificial intelligence, Machine learning, Alan Turing, Charles Babbage,
  Ada Lovelace, John von Neumann, ENIAC, UNIVAC, IBM, Apple Inc.,
  Microsoft, Linux, Unix, ARPANET, Tim Berners-Lee, Grace Hopper, Moore's
  law, Computer memory, Hard disk drive, Computer network, Database,
  Compiler, Algorithm, Data structure, Software engineering, Cybersecurity
  (redirects to "Computer security"), Cloud computing.
- Only the article body up to the "See also" heading is kept; references /
  external-links sections are dropped as noise.

## Stack

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim,
  L2-normalised embeddings), run on CPU in the shared venv.
- Search: plain NumPy — cosine similarity is a dot product of normalised
  vectors (`retrieve.py`). FAISS is not installed in the shared venv, and
  for ~1,700 chunks NumPy is fast enough (the search itself is
  milliseconds once the model is loaded; loading the model takes ~20 s).
- Chunking: token-based, exact token counts from the same model's
  tokenizer (`chunking.py`). No characters/4 guessing.

## Chunking — the two sizes compared

| | chunk size | overlap | chunks | mean words / chunk |
|---|---|---|---|---|
| A | **256 tokens** | 50 tokens | 1,711 | 211 |
| B | **512 tokens** | 50 tokens | 772 | 416 |

Sliding windows over each article's token ids, step = chunk − overlap, so
each window shares 50 tokens with the previous one. A trailing window under
20 tokens is dropped (its text is already covered by the overlap).
Windows can start mid-word: the decoded text occasionally starts with a
stray subword fragment such as `##cessor` — cosmetic, noted as a rough edge.

## How to reproduce

All commands run from this directory (`task2-rag/`), using the shared venv
(`../.venv/bin/python`). Index builds take a while on CPU with no GPU
(256-token index ≈ 8 min, 512-token index ≈ 5 min on this 4-CPU VM).

```bash
../.venv/bin/python download_corpus.py        # fetch 38 pages -> data/ + corpus_manifest.json
../.venv/bin/python build_index.py --chunk-size 256
../.venv/bin/python build_index.py --chunk-size 512
../.venv/bin/python eval.py                   # -> eval/results.json
../.venv/bin/python retrieve.py --chunk-size 256 --query "Who invented the World Wide Web?" --k 5
../.venv/bin/python answer.py --query "What does Moore's law observe?"   # optional, see below
```

## Evaluation

15 questions I wrote myself (`eval/questions.json`), each answerable from a
single passage. Each question stores the gold passage **verbatim from the
corpus** plus its source article. Gold passages are 1–2 sentences
(14–46 words).

**Gold-matching rule.** A question is a hit at k if the gold passage —
after normalisation (below) — appears as a substring of any of the top-k
retrieved chunks. The gold chunk is therefore any chunk containing the gold
passage. Before scoring, `eval.py` verifies that every gold passage is
fully contained in at least one chunk **at each chunk size**, and aborts
rather than report a misleading recall if one is not.

**Straddling / normalisation.** Gold passages are short (≤46 words), and
the 50-token overlap means a passage cut by one window boundary is whole in
the neighbouring window — verified: all 15 passages are contained in ≥1
chunk at both sizes. Matching is done on normalised text because chunk text
comes from the (uncased) BERT tokenizer's `decode`, which differs
cosmetically from the source: lower-cased, spaces around punctuation
(`( ;`, `general - purpose`, `moore ' s`, `10 : 30`), and accents stripped
(`née` decodes as `nee`). `normalize()` in `chunking.py` applies the same
lower-casing, accent folding, and punctuation-spacing rules to both sides,
so a passage matches iff the same words occur in the same order.

## Results (n = 15)

| chunk size | recall@5 | recall@1 |
|---|---|---|
| **256 tokens** | **0.800 (12/15)** | **0.333 (5/15)** |
| 512 tokens | 0.733 (11/15) | 0.200 (3/15) |

Per-question (hit rank in top-5, "—" = miss):

| id | question (short) | 256 | 512 |
|---|---|---|---|
| q01 | First programmable general-purpose computer? | — | — |
| q02 | What did Turing's 1936 paper introduce? | 1 | 1 |
| q03 | What is Ada Lovelace chiefly known for? | — | 4 |
| q04 | Who invented the first mechanical computer? | 4 | 2 |
| q05 | What does Moore's law observe? | 1 | 1 |
| q06 | First ARPANET host-to-host connection? | 3 | 5 |
| q07 | Who invented the World Wide Web? | 1 | 1 |
| q08 | What is Grace Hopper credited with creating? | — | — |
| q09 | What did Bardeen, Brattain & Shockley invent in 1947? | 1 | 5 |
| q10 | What did Kilby demonstrate on 12 Sep 1958? | 3 | — |
| q11 | First commercially available microprocessor? | 2 | — |
| q12 | When was the Linux kernel first released? | 3 | 2 |
| q13 | What was the UNIVAC I famous for predicting? | 1 | 4 |
| q14 | When did IBM introduce the first disk drive? | 5 | 2 |
| q15 | What does a compiler do? | 2 | 2 |

Full per-question top-5 lists with scores are in `eval/results.json`.

## Which chunk size worked better, and why

**256 tokens won** — recall@5 0.800 vs 0.733, recall@1 0.333 vs 0.200.
What the data actually shows:

- **Large chunks dilute the answer.** q10 and q11 hit at 256 (ranks 3 and 2)
  but miss at 512. At 512 tokens the Kilby gold chunk scores **−0.024**
  (rank 483 of 772) — the answer sentence is ~10% of a 416-word chunk, so
  mean-pooling washes it out, and the best chunk for that query anywhere
  scores only 0.295 (a references list from the Alan Turing article): the
  ranking is near-noise. q11's gold chunk scores 0.476 at 512 (rank 32)
  against a 0.678 top score from a Personal computer chunk.
- **The trade-off is not one-sided.** q03 is the counter-example: it hits
  at 512 (rank 4) but misses at 256, landing just outside the top-5 at
  rank 8. The question is broad ("chiefly known for"), and at 256 tokens
  the many same-article chunks about Lovelace (awards, lectures, books)
  crowd out the lead paragraph. Small chunks can split the context a broad
  question needs.
- **A second, real hazard of 512-token chunks:** they sit exactly at this
  model's 512-token limit. Decoded chunk text re-tokenizes *longer* than
  the token window it came from (spaces around punctuation become extra
  tokens), so **727 of 772** 512-token chunks re-tokenize to >512 tokens
  and the encoder silently truncates their tails; **0 of 1,711**
  256-token chunks do. In q10's case only ~2 tokens were clipped, so
  dilution — not truncation — is the main cause of that miss, but chunking
  at the model's maximum length is a genuine defect: the safe choice is a
  chunk size comfortably below the limit.

## Failure cases

Two questions miss at **both** chunk sizes — full detail in
[`failures.md`](failures.md):

1. **q01 (ENIAC)** — gold chunk ranks 57th at 256 tokens (score 0.53).
   The corpus is full of near-identical "first … computer" claims, so the
   dense vectors of a dozen chunks cluster and the true one drowns.
   Fix: hybrid BM25 — the exact phrase occurs verbatim in only the gold
   chunk — or a cross-encoder re-ranker over the top-20.
2. **q08 (Grace Hopper)** — gold chunk ranks 15th at 256 tokens (0.45).
   All top-5 chunks are from the right article but the wrong sections
   (awards, buildings named after her); the lead paragraph's embedding is
   diluted by biographical detail. Fix: BM25 on "first compiler", a
   re-ranker, or query rewriting ("Grace Hopper first compiler").

## What BM25 adds to embedding search, and what a re-ranker does

(in my own words)

**BM25** is keyword search done properly. It scores a chunk by the exact
words of the query it contains, and it gives rare words much more weight
than common ones: "Kilby" or "ENIAC" count for far more than "computer" or
"first", and it corrects for length so a long chunk doesn't win just for
containing more words. That is exactly what embeddings are bad at — an
embedding turns a whole chunk into one blurry average, so a distinctive
name, date, or exact phrase ("12 September 1958", "first compiler") gets
washed out by everything else in the chunk. BM25 catches those exact
matches. Its weakness is the mirror image: it only understands words it
has literally seen, so it can't tell that "car" and "automobile" are the
same thing, and it fails on paraphrases. That is why the two are combined
rather than one replacing the other: embeddings handle meaning,
BM25 handles exact words, and a simple fusion of the two scores is better
than either alone — as q01 shows, where the exact phrase exists in exactly
one chunk.

**A re-ranker** is a second, slower model that reads the question and one
candidate chunk *together* and scores how well that chunk actually answers
that question. The fast embedding search only ever compares two
pre-computed vectors, so it can never notice that a chunk says "credited
with creating the first compiler" — it just knows the chunk is vaguely
about Grace Hopper. A re-ranker (typically a cross-encoder) sees the words
side by side, so it can spot the exact answer sentence inside a chunk and
also spot that an awards listing is not an answer. You can't run it on the
whole corpus — reading every question/chunk pair is far too slow — so it
sits after the fast search and re-orders only the top ~20 candidates. It
fixes "right theme, wrong passage" failures like q08, at the cost of one
extra model call per candidate.

## Optional answer step (separate from retrieval)

`answer.py` retrieves the top-5 chunks and asks the Task 3 local model
(Qwen2.5-0.5B-Instruct, GGUF `q4_k_m`, run with llama.cpp `llama-cli`) to
answer from those passages only. Its output never affects recall numbers.
**Status: implemented but NOT verified end-to-end in this session** — when
run, `llama-cli` could not get enough CPU/RAM on the shared VM (only
~395 MB RAM available at the time; the process accumulated ~4 s of CPU in
4 minutes, i.e. it was starved/thrashing), so no answer was produced. The
script itself is wired up and ready to run when the machine is quieter.

## Notes / rough edges

- First run of `build_index.py` hit one transient Hugging Face Xet
  disconnect (`Server disconnected without sending a response`); the
  library retried automatically and the download completed. The model
  cache is Xet-backed, so the hub `blobs/` directory looks small (79-byte
  pointer files) — the decisive check is that `SentenceTransformer` loads
  and encodes, which was verified (both indexes built from it; all 2,483
  chunk embeddings are distinct unit-norm vectors).
- Chunk text is the tokenizer's `decode` of token windows, so it is
  lower-cased with spaced-out punctuation (see normalisation above) and can
  start mid-word (`##cessor`). Retrieval is unaffected, but the stored
  chunk text is not pretty print — `answer.py` shows it as-is.
- `eval/results.json` (70 KB) stores the full top-5 per question, so the
  failures can be re-inspected without re-running the model.
