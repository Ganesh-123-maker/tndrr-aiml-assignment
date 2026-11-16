#!/usr/bin/env python3
"""Optional answer step: generate an answer from the top-k retrieved chunks.

This is SEPARATE from retrieval evaluation: it runs AFTER retrieval, uses
the Task 3 local model (Qwen2.5-0.5B-Instruct, GGUF, run with llama.cpp's
llama-cli — no new download), and never feeds back into recall numbers.
The model is only shown the retrieved chunks and told to answer from them.

Usage:
    ../.venv/bin/python answer.py --query "What does Moore's law observe?" [--k 5] [--chunk-size 256]
"""
import argparse
import subprocess
import sys
from pathlib import Path

from retrieve import search

BASE = Path(__file__).resolve().parent
LLAMA_CLI = BASE.parent / "task3-local" / "bin" / "llama-b12441" / "llama-cli"
MODEL = BASE.parent / "task3-local" / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"

SYSTEM = ("You answer questions using ONLY the passages below. "
          "If the passages do not contain the answer, say so. "
          "Keep the answer to 1-3 sentences.")


def build_prompt(query: str, chunks):
    context = "\n\n".join(
        f"[Passage {i} — {c['doc_title']}]\n{c['text']}"
        for i, (_, c) in enumerate(chunks, 1))
    user = f"Passages:\n{context}\n\nQuestion: {query}\nAnswer:"
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()

    if not LLAMA_CLI.exists():
        sys.exit(f"llama-cli not found at {LLAMA_CLI} (Task 3 binary)")
    if not MODEL.exists():
        sys.exit(f"model not found at {MODEL} (Task 3 model)")

    chunks = search(args.query, args.chunk_size, args.k)
    print(f"Retrieved top-{args.k} (chunk size {args.chunk_size}):")
    for rank, (score, c) in enumerate(chunks, 1):
        print(f"  #{rank} {score:.3f} [{c['doc_title']} | {c['chunk_id']}]")

    prompt = build_prompt(args.query, chunks)
    cmd = [str(LLAMA_CLI), "-m", str(MODEL), "-p", prompt,
           "-n", str(args.max_tokens), "-t", "4", "--temp", "0.0",
           "-c", "4096", "--no-display-prompt", "-s", "42"]
    print("\n--- answer (Qwen2.5-0.5B-Instruct, local, from retrieved passages only) ---")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    out = (proc.stdout or "").strip()
    print(out if out else f"[no output; stderr: {proc.stderr[-500:]}]")


if __name__ == "__main__":
    main()
