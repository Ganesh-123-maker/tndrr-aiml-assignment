#!/usr/bin/env python3
"""Retrieve top-k chunks for a question (NumPy cosine similarity).

The query is embedded with the same model as the index; scores are cosine
similarities against the L2-normalised chunk embeddings (plain dot
product). No LLM is involved anywhere in retrieval.

Usage:
    ../.venv/bin/python retrieve.py --chunk-size 256 --query "Who built the first programmable computer?" --k 5
"""
import argparse
import json
from pathlib import Path

import numpy as np

from chunking import MODEL_NAME

BASE = Path(__file__).resolve().parent

_model = None


def load_index(chunk_size: int):
    """Return (embeddings, chunk records, meta) for a chunk size."""
    stem = f"chunks_{chunk_size}"
    data = json.loads((BASE / "index" / f"{stem}.json").read_text())
    emb = np.load(BASE / "index" / f"{stem}.npz")["embeddings"]
    return emb, data["chunks"], data


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def search(query: str, chunk_size: int, k: int = 5, model=None):
    """Return the top-k chunks for ``query`` as (score, chunk) pairs."""
    emb, chunks, _ = load_index(chunk_size)
    model = model or get_model()
    q = model.encode([query], normalize_embeddings=True,
                     convert_to_numpy=True).astype(np.float32)[0]
    scores = emb @ q
    top = np.argsort(-scores)[:k]
    return [(float(scores[i]), chunks[int(i)]) for i in top]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    for rank, (score, c) in enumerate(search(args.query, args.chunk_size, args.k), 1):
        print(f"\n--- #{rank}  score={score:.4f}  [{c['doc_title']} | {c['chunk_id']}] ---")
        print(c["text"][:600])


if __name__ == "__main__":
    main()
