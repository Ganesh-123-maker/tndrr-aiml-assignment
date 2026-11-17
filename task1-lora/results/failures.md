# Fine-tuned model failures (held-out test set, n=100)

**0 failure(s).** The fine-tuned model got 100/100 exact-match on the
held-out test set (greedy decoding), so there are no fine-tuned
failure examples to show. The task asks for 3 failure examples — those
are the base model's, below, with the diagnosis of why fine-tuning
fixed each one.

## Base model: 48 failures / 100

Breakdown by field (from `predictions_base.jsonl`):
- genre only: 40 · genre + year: 3 · title + year: 4 · title only: 1
- 42 of the 43 genre mismatches are case-only (e.g. "Science Fiction"
  vs "science fiction"); all 5 title errors are the same template.

### 1. Genre casing (42 errors)

Request: *Add Arjun Rossi's science fiction book The Falling Ember (2016).*

- Expected genre: `"science fiction"` — base output: `"Science Fiction"`.
- Same pattern for fantasy → "Fantasy", non-fiction → "Non-fiction".
- Why: the base model title-cases the genre, as in ordinary prose.
  Exact-match is case-sensitive, so the example fails even though the
  content is right. Fine-tuning taught the exact lowercase label set.

### 2. Whole request copied into the title (5 errors)

Request: *Catalogue The Sleeping Tide by Lena Kaplan — non-fiction — 1985.*

- Expected title: `"The Sleeping Tide"` — base output:
  `"Catalogue The Sleeping Tide by Lena Kaplan — non-fiction — 1985"`.
- Why: this template ("Catalogue … — … — …") gives no "by" cue
  separating title from author the way the other templates do, so the
  base model falls back to copying the whole request into the title
  field. In 3 of these 5 the year was also a string (`"1985"`)
  instead of an integer. Fine-tuning learned the extraction boundary
  for this template too.

### 3. Genre + year compounding (3 errors)

Same two causes together: a title-cased genre plus a string year in
the "Catalogue" template. Both are format/convention errors, not
content errors — the base model never invented a wrong title, author
or year out of thin air.

## Why fine-tuning fixed all of them

Every base failure is a format convention (lowercase genre labels,
extract-only-the-title, integer year), and 3 epochs on 800 examples
is exactly enough to teach those conventions for 10 known templates.
The honest limit: on an *unseen* template the fine-tuned model may be
worse than base, because it has overfit the 10 training phrasings —
the 1.00 should not be read as "the model understands books".
