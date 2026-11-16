"""Shared chunking + normalisation helpers for Task 2.

Chunking is TOKEN-based, not character-based: text is encoded with the
all-MiniLM-L6-v2 tokenizer and cut into sliding windows of ``chunk_tokens``
tokens with ``overlap_tokens`` tokens of overlap between consecutive
windows. Token counts are therefore exact for the embedding model used,
not a characters/4 guess. (For reference, on this corpus 1 token ~= 4.0
characters of English prose.)
"""
import re

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_tokenizer = None


def get_tokenizer():
    """Lazily load the embedding model's tokenizer (small, ~100 KB)."""
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


def normalize(text: str) -> str:
    """Normalise text for gold-passage containment checks.

    Chunk text comes from the (uncased) BERT tokenizer's ``decode`` of
    token windows, so it differs cosmetically from the source article:
    everything is lower-cased and spaces appear around punctuation
    (``( ;``, ``general - purpose``, ``moore ' s``, ``10 : 30``).
    The same normalisation is applied to the gold passage, so a passage
    matches iff the same words/punctuation occur in the same order.
    Steps: lower-case, fold accents (the uncased tokenizer strips them:
    ``née`` decodes as ``nee``), collapse whitespace, join hyphen/dash
    splits, drop spaces adjacent to punctuation.
    """
    import unicodedata
    t = unicodedata.normalize("NFKD", text)
    t = t.encode("ascii", "ignore").decode("ascii")
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s*[-–—]\s*", "-", t)          # "general - purpose" -> "general-purpose"
    t = re.sub(r"\s+([.,;:!?%)\]}'\"”’])", r"\1", t)  # space before punctuation
    t = re.sub(r"([(\[{'\"“‘])\s+", r"\1", t)   # space after opening punctuation
    t = re.sub(r"\s+'\s*", "'", t)              # "moore ' s" -> "moore's"
    t = re.sub(r"([:.,])\s+", r"\1", t)         # "10: 30" -> "10:30"
    return t.strip()


def chunk_text(text: str, chunk_tokens: int, overlap_tokens: int,
               tokenizer=None):
    """Split ``text`` into token windows.

    Returns a list of dicts: {text, token_start, token_end}.
    Windows start every (chunk_tokens - overlap_tokens) tokens, so each
    window shares ``overlap_tokens`` tokens with the previous one. A final
    short window is kept as long as it holds at least 20 tokens; a shorter
    tail is merged into the previous window's text is NOT done -- instead a
    tail under 20 tokens is dropped only if a previous window already covers
    it, which the overlap guarantees. In practice tails are kept whenever
    they add new tokens.
    """
    tok = tokenizer or get_tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    if not ids:
        return []
    step = chunk_tokens - overlap_tokens
    chunks = []
    for start in range(0, len(ids), step):
        end = min(start + chunk_tokens, len(ids))
        if end - start < 20 and chunks:
            # Tail shorter than 20 tokens: skip, it is already covered by
            # the previous window's overlap.
            break
        chunks.append({
            "text": tok.decode(ids[start:end], skip_special_tokens=True).strip(),
            "token_start": start,
            "token_end": end,
        })
        if end == len(ids):
            break
    return chunks


def chunk_document(title: str, filename: str, text: str,
                   chunk_tokens: int, overlap_tokens: int, tokenizer=None):
    """Chunk one document; adds doc title / chunk index to each chunk."""
    out = []
    for i, c in enumerate(chunk_text(text, chunk_tokens, overlap_tokens, tokenizer)):
        out.append({
            "doc_title": title,
            "filename": filename,
            "chunk_id": f"{filename[:-4]}_c{i}",
            "chunk_index": i,
            "token_start": c["token_start"],
            "token_end": c["token_end"],
            "text": c["text"],
        })
    return out
