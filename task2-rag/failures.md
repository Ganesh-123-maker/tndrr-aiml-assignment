# Retrieval failures — 2 cases (both fail at BOTH chunk sizes)

These are the two questions that miss the top-5 at 256 tokens AND at 512
tokens, so they are not chunk-size artefacts — they are retrieval failures
worth fixing. Ranks/scores below are from `eval/results.json` (reproduced
with `retrieve.py`).

## Case 1 — q01: the "first computer" question (gold rank 57/1711 at 256 tokens)

**Question:** "Which computer was the first programmable, electronic,
general-purpose digital computer, and when was it completed?"

**Gold passage** (from `ENIAC`, lead paragraph): "ENIAC (; Electronic
Numerical Integrator and Computer) was the first programmable, electronic,
general-purpose digital computer, completed in 1945."

**What was retrieved instead (256 tokens, top-5):**

| rank | score | doc / chunk | snippet |
|---|---|---|---|
| 1 | 0.665 | History of computing / c12 | "intended to solve systems of linear equations, though it was not programmable. the computer was never truly completed due to atanasoff's departure …" |
| 2 | 0.651 | Personal computer / c4 | "…cessor, the intel 4004, in 1971. the first microcomputers, based on microprocessors, were developed during the early 1970s …" |
| 3 | 0.641 | History of computing / c11 | "… the first digital electronic computer was d…" |
| 4 | 0.637 | Computer / c20 | "… the theoretical basis for the stored-program computer was laid out by turing in his 1936 paper. in 1945, turing joined the national physic…" |
| 5 | 0.615 | Computer / c0 | "a computer is a machine that can be programmed to automatically carry out sequences of arithmetic or logical operations (computation) …" |

The gold chunk (`eniac_c0`) scores only 0.53 and ranks **57th**.
The question is phrased almost identically to the gold sentence, yet the
embedding cannot tell it apart from the many near-identical "first … 
computer" claims in this corpus: `History of computing`, `Computer` and
`Personal computer` all discuss firsts (Atanasoff, Turing, 4004), so the
dense vectors of a dozen chunks cluster together and the true one drowns.

**What would fix it:** hybrid search (BM25 + embeddings). The exact phrase
"first programmable, electronic, general-purpose digital computer" occurs
verbatim in exactly one chunk — the gold one. A keyword score would put it
at rank 1, and a simple score fusion (or BM25 pre-filter + embedding
re-rank) would fix this case without touching the model. A cross-encoder
re-ranker over the top-20 would also work: it reads the query and each
candidate together, so it can notice the exact phrase match instead of
comparing two pooled vectors.

## Case 2 — q08: right document, wrong passage (gold rank 15/1711 at 256 tokens)

**Question:** "What is Grace Hopper often credited with creating?"

**Gold passage** (from `Grace Hopper`, lead): "She is credited with writing
the first computer manual, "A Manual of Operation for the Automatic Sequence
Controlled Calculator", and is often credited with creating the first
compiler."

**What was retrieved instead (256 tokens, top-5 — ALL from the Grace Hopper
article, none of them the lead):**

| rank | score | chunk | snippet |
|---|---|---|---|
| 1 | 0.646 | grace_hopper_c18 | "… she became the first computer history museum fellow award recipient "for contributions to the development of programming languages, for standardization efforts …" |
| 2 | 0.637 | grace_hopper_c29 | "… chapter 17. grace hopper. women laureates 1970–2007 : national medal of science …" |
| 3 | 0.585 | grace_hopper_c22 | "named the grace hopper building, and also contains the history of naval communications museum …" |
| 4 | 0.568 | grace_hopper_c23 | "the former orlando naval training center, is named for her. programs women at microsoft corporation formed an employee group called hoppers …" |
| 5 | 0.561 | grace_hopper_c19 | "… the department of energy's national energy research scientific computing center named its flagship system "hopper" …" |

The gold chunk scores 0.45 and ranks **15th**. The retriever found the right
person but the wrong part of her article: the question's words ("credited",
"first", "creating") match the awards/legacy sections — full of "first … 
recipient", "named for her", "contributions to … programming languages" —
better than they match the lead paragraph, whose embedding is diluted by
biographical detail (birth/death dates, the manual's long title) sitting in
the same chunk as the answer sentence.

**What would fix it:** again, hybrid search first — BM25 on "first compiler"
hits the gold chunk directly, because the phrase appears there and almost
nowhere else in the article. Failing that, a cross-encoder re-ranker over the
top-20: it would read "often credited with creating" against each candidate
and prefer the sentence that actually says "credited with creating the first
compiler" over award listings. A cheaper fix in the same direction is query
rewriting: "Grace Hopper first compiler" (dropping the "what is … credited
with creating" framing) moves the query vector toward the answer's nouns.
