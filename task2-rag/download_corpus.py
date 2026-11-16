#!/usr/bin/env python3
"""Download the Task 2 corpus: Wikipedia articles on the history of computing.

Uses only the Python standard library (urllib) + the public Wikipedia API.
Reproducible offline: re-running this script re-fetches the same 38 titles
and rebuilds data/ + corpus_manifest.json from scratch.

Usage:
    ../../.venv/bin/python download_corpus.py [--out data]
"""
import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "tndrr-aiml-assignment-task2/1.0 (student project; contact: ganeshbarmavath24@gmail.com)"

# 38 Wikipedia articles, one coherent topic area: the history of computing.
TITLES = [
    "History of computing",
    "Computer",
    "Central processing unit",
    "Transistor",
    "Integrated circuit",
    "Microprocessor",
    "Personal computer",
    "Internet",
    "World Wide Web",
    "Operating system",
    "Programming language",
    "Artificial intelligence",
    "Machine learning",
    "Alan Turing",
    "Charles Babbage",
    "Ada Lovelace",
    "John von Neumann",
    "ENIAC",
    "UNIVAC",
    "IBM",
    "Apple Inc.",
    "Microsoft",
    "Linux",
    "Unix",
    "ARPANET",
    "Tim Berners-Lee",
    "Grace Hopper",
    "Moore's law",
    "Computer memory",
    "Hard disk drive",
    "Computer network",
    "Database",
    "Compiler",
    "Algorithm",
    "Data structure",
    "Software engineering",
    "Cybersecurity",
    "Cloud computing",
]


def slugify(title: str) -> str:
    s = title.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def fetch_batch(titles, retries=3):
    """Fetch plain-text extracts for a batch of titles from the Wikipedia API."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "exlimit": "max",
        "exsectionformat": "plain",
        "redirects": "1",
        "format": "json",
        "formatversion": "2",
        "titles": "|".join(titles),
    }
    url = API_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001 - retry transient network errors
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Wikipedia API request failed after {retries} tries: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    out_dir = base / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"source": "Wikipedia (en.wikipedia.org) via the MediaWiki API",
                "topic": "History of computing",
                "fetched": time.strftime("%Y-%m-%d"),
                "api": API_URL,
                "documents": []}
    fetched = {}
    # Whole-article extracts are capped at 1 page per API call (exlimit is
    # lowered to 1), so fetch one title at a time: 38 quick calls.
    for title in TITLES:
        data = fetch_batch([title])
        page = data["query"]["pages"][0]
        if "missing" in page or not page.get("extract", "").strip():
            raise RuntimeError(f"Wikipedia page missing/empty: {title}")
        # Store under the REQUESTED title: redirects (e.g. Cybersecurity ->
        # Computer security) return the target's title, which we don't want
        # as the key. The manifest records the requested title + its URL.
        fetched[title] = page["extract"]
        time.sleep(0.5)  # be polite to the API

    missing = [t for t in TITLES if t not in fetched]
    if missing:
        # Redirects normalise some titles (e.g. case); match case-insensitively.
        lower = {k.lower(): v for k, v in fetched.items()}
        for t in list(missing):
            if t.lower() in lower:
                fetched[t] = lower[t.lower()]
                missing.remove(t)
    if missing:
        raise RuntimeError(f"Could not fetch: {missing}")

    total_chars = 0
    total_words = 0
    for title in TITLES:
        text = fetched[title].strip()
        # Drop the "== See also ==" tail and below: references/external links add
        # noise and are not article prose. Keep everything up to that heading.
        m = re.search(r"\n==\s*See also\s*==", text)
        if m:
            text = text[:m.start()].strip()
        fname = slugify(title) + ".txt"
        (out_dir / fname).write_text(text, encoding="utf-8")
        n_chars = len(text)
        n_words = len(text.split())
        total_chars += n_chars
        total_words += n_words
        manifest["documents"].append({
            "title": title,
            "filename": fname,
            "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
            "chars": n_chars,
            "words": n_words,
        })
        print(f"{title}: {n_words:,} words")

    manifest["page_count"] = len(manifest["documents"])
    manifest["total_chars"] = total_chars
    manifest["total_words"] = total_words
    (base / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {manifest['page_count']} pages, {total_words:,} words total -> {out_dir}/")


if __name__ == "__main__":
    main()
