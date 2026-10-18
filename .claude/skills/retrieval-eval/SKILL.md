---
name: retrieval-eval
description: Protocol for evaluating search retrieval (Recall@k, MRR, nDCG) and comparing runs. Use it ALWAYS before and after any change that may affect which chunks are retrieved or in what order — chunker, embedding model, BM25/FTS5, RRF fusion, reranker, top_n/k parameters, query assembly — and when the user asks whether search "improved", "got worse" or why a question failed.
---

# Retrieval evaluation

Rule 3 of CLAUDE.md: no retrieval change is accepted without measuring it. Intuition about what "should"
improve search is unreliable; data replaces it.

## The evaluation set

`eval/questions.jsonl`, one question per line (format in `eval/questions.example.jsonl`):

- `type`: `lookup` (paraphrase of something in a section), `exact_term` (acronym, proper name, identifier),
  `synthesis` (answer spread over 2+ sections or guides), `absent` (not in the library).
- `expected`: list of `{path, heading}`. `heading` is the last element of the expected `heading_path`.
  Each item may carry `"alternatives": [{path, heading}, ...]`: other sections that answer the same
  thing equally well. Finding the primary section **or** any alternative satisfies the item, which
  still counts once (recall and nDCG cannot be inflated by listing many alternatives).
- Byte-identical copies of an expected file are accepted automatically (same text, same answer);
  every run records `identical_copies_accepted`.

**The questions are written by the user.** If Claude generates questions from the chunks, the set
inherits the literal vocabulary of the text and overestimates performance. Claude may suggest which types
are missing or review the format, but must not write questions from the corpus.

**Alternatives are judged by the user too.** `biblio candidates <tag>` writes `eval/review/<tag>.md`
with what the search returned for each missed question; the user copies into `alternatives` only the
sections that really answer the question. Claude never adds an alternative on its own: an alternative
that merely mentions the topic makes the evaluation lenient, not fair. Mechanical fixes to the file
(paths, ids, types that contradict the user's own notes) are fine with the user's agreement.

## Metrics

Compute them at two granularities, **file** and **section**, and report them separately:

- **Recall@k** (k = 1, 5, 8, 20): fraction of `expected` present in the top-k. For `synthesis`, also
  report the fraction of questions with *all* expected items in the top-k.
- **MRR**: mean of 1/rank of the first hit.
- **nDCG@8**: binary relevance.
- `absent` questions do not enter these metrics. They are used in Phase 5 to measure the tutor's abstention.

Always report the breakdown by `type`. A change that raises the average but sinks `exact_term` is
usually a real regression in disguise.

## Protocol for a change

1. Make sure the tree is clean or note the diff. Run `biblio eval --tag before-<change>`.
2. Apply the change. If it affects the chunker or embeddings, reindex with `--full`.
3. Run `biblio eval --tag after-<change>`.
4. Compare the two runs by metric and by type, and list the questions that went from hit to miss and vice versa.
5. **Decide:** accepted if it improves the target metric without a regression > 2 points in any type. With
   40–60 questions, a difference of 1–2 questions is noise. Do not claim an improvement because of it; say so explicitly.
6. Record the decision in the PLAN.md log. If it changes the architecture, write an ADR.

## Results format

Each run appends one line to `eval/results/runs.jsonl`:

```json
{"ts": "2026-10-05T12:00:00", "tag": "baseline-dense", "commit": "abc123",
 "config": {"dense_top_n": 50, "rerank": false, "chunker_version": 1, "embedding_model": "BAAI/bge-m3"},
 "n_questions": 48,
 "metrics": {"section": {"recall@8": 0.71, "mrr": 0.52}, "file": {"recall@8": 0.85}},
 "by_type": {"lookup": {...}, "exact_term": {...}, "synthesis": {...}},
 "failures": ["q007", "q019"]}
```

## Error analysis

For each failed question, classify the cause before proposing solutions:

- **chunking**: the answer is split, merged with something else or without a useful heading.
- **vocabulary**: the question uses terms that do not appear (synonym, cross-language).
- **ambiguity**: several sections legitimately answer; the user adds them as `alternatives`
  (`biblio candidates`), then the run is repeated.
- **wrong eval**: the `expected` is badly written or points to a section that changed.

Fix evaluation errors first; otherwise you will be optimizing against noise.
