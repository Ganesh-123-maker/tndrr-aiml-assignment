#!/usr/bin/env python3
"""Generate synthetic data for Task 1: plain-English book request -> fixed JSON.

Task: given a request such as
    "Add the book The Silent River by Ana Petrova, a mystery book from 1987"
produce exactly:
    {"title": "The Silent River", "author": "Ana Petrova",
     "genre": "mystery", "year": 1987}

Design choices (see README.md for the full rationale):
* 10 distinct surface templates, so the model must learn the *mapping*,
  not memorise one string pattern.
* Titles / authors are composed from word pools and are DISJOINT across
  train / dev / test (different first-name, last-name and title-word pools
  per split). The test set therefore measures generalisation to unseen
  titles/authors, not recall of training examples.
* Genres come from a fixed list of 12; years are uniform in 1950-2025.
* Fixed seed (42) so regeneration is byte-identical.

Output: data/train.jsonl (800), data/dev.jsonl (100), data/test.jsonl (100).
Each line: {"prompt", "request", "response", "title", "author", "genre", "year"}
where `prompt` is the full model input and `response` is the target JSON
string (keys in fixed order title, author, genre, year; year is an int).
"""
import json
import random
from pathlib import Path

SEED = 42
GENRES = [
    "fiction", "mystery", "science fiction", "fantasy", "romance",
    "thriller", "historical fiction", "biography", "horror", "poetry",
    "adventure", "non-fiction",
]

TEMPLATES = [
    "Add the book {title} by {author}, a {genre} book from {year}",
    'Please add "{title}" by {author} ({genre}, {year}) to the catalogue.',
    "I want to catalogue a {genre} novel called {title}, written by {author} in {year}.",
    "New entry: {title} — {author}, {genre}, published {year}.",
    "Add {author}'s {genre} book {title} ({year}).",
    "Could you register the book titled {title}, author {author}, genre {genre}, year {year}?",
    "Book: {title}; Author: {author}; Genre: {genre}; Year: {year}. Please add it.",
    'Add to the library: "{title}" ({year}) by {author}, genre {genre}.',
    "I'd like to add {title} by {author}. It's a {genre} book from {year}.",
    "Catalogue {title} by {author} — {genre} — {year}.",
]

PROMPT_PREFIX = (
    "Convert the following book request into JSON with keys "
    "title, author, genre, year.\nRequest: "
)
PROMPT_SUFFIX = "\nJSON:"

# Disjoint pools per split: no first name, last name or title word is
# shared between train / dev / test, so test titles/authors are unseen.
FIRST_NAMES = {
    "train": ["Ana", "Marcus", "Priya", "Tomas", "Elena", "David", "Sofia", "Kenji",
              "Marta", "Liam", "Zara", "Felix", "Ines", "Omar", "Nina", "Ravi"],
    "dev": ["Clara", "Hugo", "Meera", "Jonas", "Aisha", "Bruno"],
    "test": ["Lena", "Arjun", "Maya", "Diego", "Freya", "Kofi"],
}
LAST_NAMES = {
    "train": ["Petrova", "Chen", "Nair", "Novak", "Garcia", "Kim", "Silva", "Tanaka",
              "Kowalski", "O'Brien", "Haddad", "Berg", "Costa", "Singh", "Meyer", "Ali"],
    "dev": ["Dubois", "Larsen", "Iyer", "Moreau", "Fischer", "Diallo"],
    "test": ["Rossi", "Kaplan", "Mensah", "Vargas", "Lindqvist", "Okafor"],
}
TITLE_FIRST = {
    "train": ["Silent", "Golden", "Hidden", "Broken", "Distant", "Crimson", "Quiet",
              "Endless", "Forgotten", "Burning", "Hollow", "Velvet", "Iron", "Silver"],
    "dev": ["Lonely", "Frozen", "Ancient", "Wild", "Gentle", "Dark"],
    "test": ["Bright", "Sleeping", "Wandering", "Secret", "Falling", "Emerald"],
}
TITLE_SECOND = {
    "train": ["River", "Mountain", "Garden", "City", "Ocean", "Forest", "Bridge",
              "Mirror", "Storm", "Valley", "Harbour", "Desert", "Island", "Winter"],
    "dev": ["Meadow", "Castle", "Library", "Road", "Sky", "Lantern"],
    "test": ["Compass", "Journey", "Horizon", "Willow", "Ember", "Tide"],
}
# Optional third word, used for ~25% of titles, e.g. "The Silent River of Dawn".
TITLE_THIRD = ["of Dawn", "of Shadows", "of Tomorrow", "of Stone", "of Light",
               "of Winter", "of Stars", "of Ashes"]


def make_title(rng, split):
    t = f"The {rng.choice(TITLE_FIRST[split])} {rng.choice(TITLE_SECOND[split])}"
    if rng.random() < 0.25:
        t += " " + rng.choice(TITLE_THIRD)
    return t


def make_example(rng, split, used):
    while True:
        title = make_title(rng, split)
        author = f"{rng.choice(FIRST_NAMES[split])} {rng.choice(LAST_NAMES[split])}"
        genre = rng.choice(GENRES)
        year = rng.randint(1950, 2025)
        key = (title, author)
        if key not in used:
            used.add(key)
            break
    request = rng.choice(TEMPLATES).format(
        title=title, author=author, genre=genre, year=year)
    response = json.dumps(
        {"title": title, "author": author, "genre": genre, "year": year},
        ensure_ascii=False)
    prompt = PROMPT_PREFIX + request + PROMPT_SUFFIX
    return {"prompt": prompt, "request": request, "response": response,
            "title": title, "author": author, "genre": genre, "year": year}


def main():
    out_dir = Path(__file__).resolve().parent
    rng = random.Random(SEED)
    sizes = {"train": 800, "dev": 100, "test": 100}
    for split, n in sizes.items():
        used = set()
        rows = [make_example(rng, split, used) for _ in range(n)]
        path = out_dir / f"{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # sanity: templates / genres actually varied
        n_templates = len({r["request"].split()[0] for r in rows})
        print(f"{split}: {n} rows -> {path} "
              f"(unique titles {len({r['title'] for r in rows})}, "
              f"genres {len({r['genre'] for r in rows})}/12)")
        # exact-match target is well-formed JSON by construction
        assert all(json.loads(r["response"]) == {
            "title": r["title"], "author": r["author"],
            "genre": r["genre"], "year": r["year"]} for r in rows)


if __name__ == "__main__":
    main()
