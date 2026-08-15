# Retrieval baselines

Four `rag eval` runs over `eval/questions.yaml`, one per retrieval configuration, so a
change to chunking, prompts or fusion has something to regress against. The JSON next to
this file is the full report per run; each one records the settings it used, so a baseline
is self-describing and does not depend on this page staying accurate.

Committed here rather than in `eval/reports/`, which stays git-ignored: a baseline is a
reference point that should change only deliberately, an ordinary run is scratch output.

## Runs

| | `RETRIEVAL_MODE` | `RERANK_PROVIDER` |
| --- | --- | --- |
| [`dense.json`](dense.json) | `dense` | `none` |
| [`hybrid.json`](hybrid.json) | `hybrid` | `none` |
| [`dense-rerank.json`](dense-rerank.json) | `dense` | `fastembed` |
| [`hybrid-rerank.json`](hybrid-rerank.json) | `hybrid` | `fastembed` |

Recorded 2026-08-15 · answer `claude-haiku-4-5-20251001` · judge `claude-sonnet-5` ·
`top_k=5` · `chunk=900/150` · embeddings `fastembed:BAAI/bge-small-en-v1.5` ·
reranker `Xenova/ms-marco-MiniLM-L-6-v2` · `RERANK_CANDIDATES=20` · `RRF_K=60`.

The models are the ones in the developer `.env`, not the `claude-opus-5` default in
`config.py` — worth re-recording against the defaults before treating these as the
project's headline numbers.

## Results

| Metric | dense | hybrid | dense+rerank | hybrid+rerank |
| --- | --- | --- | --- | --- |
| `mean_groundedness` | 5.0 | 5.0 | 5.0 | 5.0 |
| `grounded_rate_pct` | 100.0 | 100.0 | 100.0 | 100.0 |
| `hallucination_rate_pct` | 0.0 | 0.0 | 0.0 | 0.0 |
| `citation_validity_pct` | 100.0 | 100.0 | 100.0 | 100.0 |
| `dangling_citation_count` | 0 | 0 | 0 | 0 |
| `abstention_accuracy_pct` | 100.0 | 100.0 | 100.0 | 100.0 |
| `retrieval_recall_pct` | 100.0 | 100.0 | 100.0 | 100.0 |
| `must_contain_pass_pct` | 100.0 | 100.0 | 100.0 | 100.0 |

`citation_span_integrity_pct` is omitted: it is a native-mode metric and these runs are in
marker mode, where the summary reports it as `0.0` because nothing was checked, not because
a check failed.

## What these numbers can and cannot be used for

**They are a regression floor, not an A/B comparison.** Every configuration scores
identically on every metric, so as it stands the evaluation set cannot tell the four apart.
That is a property of the corpus, not evidence that the retrieval work is inert: the sample
corpus is 3 documents and 11 chunks, and `top_k=5` returns nearly half of it for every
question. Recall is satisfied before retrieval quality has a chance to matter.

The configurations do retrieve differently — comparing per-question output against the dense
run:

- **hybrid** differs on 4 of 15 questions (`hb-expense-approval`, `atlas-dlq-retention`,
  `sec-incident-report`, `sec-prod-credentials`)
- **rerank** (both modes, identically) differs on 4 of 15 (`atlas-batch-limit`,
  `atlas-retries`, `atlas-rate-limit`, `near-miss-vacation-carryover-deadline`)

but the differences are reorderings within a passage set that already contained the right
answer, and the correct source is cited in every case. So the pipeline behaves as designed
and the metrics have no room to move.

**Using them.** As a floor they are still worth having: any future change that drops a
number below these has broken something, and `--fail-under` has a real value to check
against. What they cannot do is justify turning hybrid or reranking on — that needs an
evaluation set where dense retrieval measurably fails, which means either a corpus large
enough that `top_k` is selective, or questions built around exact tokens that embeddings
miss. Until then, treat a tie here as "no regression", never as "no difference".
