#!/usr/bin/env python3
"""Evaluate retrieval: recall@k for each chunk size over eval/questions.json.

Gold-matching rule (per chunk size): a question is a HIT at k if the
question's gold passage -- after whitespace normalisation -- appears as a
substring of ANY of the top-k retrieved chunks' (normalised) text.
The gold chunk for a question is therefore any chunk containing the gold
passage verbatim. Gold passages are 1-2 sentences (~25-50 tokens), short
enough that the 50-token overlap guarantees each one is fully contained in
at least one chunk at both chunk sizes; this is verified by
verify_gold_containment() below, which fails loudly if a passage straddles
every chunk boundary instead of silently counting a miss.

Retrieval numbers here depend ONLY on embeddings + cosine similarity.
No LLM is used.

Usage:
    ../.venv/bin/python eval.py [--k 5] [--chunk-sizes 256 512]
Writes eval/results.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from chunking import normalize
from retrieve import get_model, load_index

BASE = Path(__file__).resolve().parent


def verify_gold_containment(questions, chunk_sizes):
    """Check every gold passage is fully inside >= 1 chunk per chunk size.

    Returns {chunk_size: [question ids NOT contained]}. A non-empty list
    means the eval would be unfair for that question/size, so eval.py
    aborts instead of reporting a misleading recall number.
    """
    bad = {}
    for size in chunk_sizes:
        _, chunks, _ = load_index(size)
        norm_chunks = [normalize(c["text"]) for c in chunks]
        missing = []
        for q in questions:
            gold = normalize(q["gold_passage"])
            if not any(gold in c for c in norm_chunks):
                missing.append(q["id"])
        bad[size] = missing
    return bad


def evaluate(questions, chunk_size, k, model):
    emb, chunks, _ = load_index(chunk_size)
    norm_chunks = [normalize(c["text"]) for c in chunks]
    per_question = []
    for q in questions:
        gold = normalize(q["gold_passage"])
        qvec = model.encode([q["question"]], normalize_embeddings=True,
                            convert_to_numpy=True).astype(np.float32)[0]
        scores = emb @ qvec
        top = np.argsort(-scores)[:k]
        hit = False
        hit_rank = None
        retrieved = []
        for rank, i in enumerate(top, 1):
            retrieved.append({
                "rank": rank,
                "score": round(float(scores[i]), 4),
                "doc_title": chunks[i]["doc_title"],
                "chunk_id": chunks[i]["chunk_id"],
                "snippet": chunks[i]["text"][:200],
            })
            if gold in norm_chunks[i]:
                hit = True
                if hit_rank is None:
                    hit_rank = rank
        per_question.append({
            "id": q["id"],
            "question": q["question"],
            "source_doc": q["source_doc"],
            "hit_at_k": hit,
            "hit_rank": hit_rank,
            "top1_doc": retrieved[0]["doc_title"],
            "top1_score": retrieved[0]["score"],
            "retrieved": retrieved,
        })
    hits = sum(1 for r in per_question if r["hit_at_k"])
    recall_at_1 = sum(1 for r in per_question if r["hit_rank"] == 1) / len(per_question)
    return {
        "chunk_size": chunk_size,
        "k": k,
        "n": len(per_question),
        "hits": hits,
        "recall_at_k": round(hits / len(per_question), 4),
        "recall_at_1": round(recall_at_1, 4),
        "per_question": per_question,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[256, 512])
    args = ap.parse_args()

    questions = json.loads((BASE / "eval" / "questions.json").read_text())
    print(f"{len(questions)} eval questions")
    bad = verify_gold_containment(questions, args.chunk_sizes)
    for size, missing in bad.items():
        if missing:
            raise SystemExit(
                f"ABORT: gold passage(s) not contained in any {size}-token "
                f"chunk: {missing}. Shorten the passage or raise the overlap.")
    print("gold-containment check passed for all chunk sizes")

    model = get_model()
    results = {"k": args.k, "n": len(questions), "by_chunk_size": {}}
    for size in args.chunk_sizes:
        r = evaluate(questions, size, args.k, model)
        results["by_chunk_size"][str(size)] = r
        print(f"chunk_size={size}: recall@{args.k} = {r['recall_at_k']:.3f} "
              f"({r['hits']}/{r['n']}), recall@1 = {r['recall_at_1']:.3f}")

    (BASE / "eval" / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote eval/results.json")


if __name__ == "__main__":
    main()
