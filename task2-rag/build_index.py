#!/usr/bin/env python3
"""Build a dense retrieval index for one chunk size (NumPy cosine search).

No FAISS here (not installed in the shared venv) and no LangChain /
LlamaIndex: embeddings come from sentence-transformers only, and search is
plain L2-normalised dot product (= cosine similarity) in NumPy, which is
fast enough for a few thousand chunks.

Usage:
    ../.venv/bin/python build_index.py --chunk-size 256
    ../.venv/bin/python build_index.py --chunk-size 512
    ../.venv/bin/python build_index.py --chunk-size 256 --overlap 50

Outputs (under index/):
    chunks_<size>.npz   embeddings, float32, L2-normalised, shape (n, 384)
    chunks_<size>.json  chunk records: doc title, chunk id, text, token span
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from chunking import MODEL_NAME, chunk_document, get_tokenizer

BASE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    manifest = json.loads((BASE / "corpus_manifest.json").read_text())
    tokenizer = get_tokenizer()

    chunks = []
    for doc in manifest["documents"]:
        text = (BASE / "data" / doc["filename"]).read_text(encoding="utf-8")
        chunks.extend(chunk_document(doc["title"], doc["filename"], text,
                                     args.chunk_size, args.overlap, tokenizer))
    print(f"{len(chunks)} chunks at {args.chunk_size} tokens "
          f"(overlap {args.overlap}) from {len(manifest['documents'])} docs")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    t0 = time.time()
    texts = [c["text"] for c in chunks]
    emb = model.encode(texts, batch_size=args.batch_size,
                       normalize_embeddings=True, show_progress_bar=True,
                       convert_to_numpy=True)
    emb = np.asarray(emb, dtype=np.float32)
    print(f"embedded in {time.time() - t0:.1f}s; shape {emb.shape}")

    out_dir = BASE / "index"
    out_dir.mkdir(exist_ok=True)
    stem = f"chunks_{args.chunk_size}"
    np.savez_compressed(out_dir / f"{stem}.npz", embeddings=emb)
    (out_dir / f"{stem}.json").write_text(
        json.dumps({"model": args.model, "chunk_size": args.chunk_size,
                    "overlap": args.overlap, "n_chunks": len(chunks),
                    "chunks": chunks}, indent=1, ensure_ascii=False),
        encoding="utf-8")
    print(f"wrote index/{stem}.npz + index/{stem}.json")


if __name__ == "__main__":
    main()
