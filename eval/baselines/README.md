# Retrieval baselines

Two question sets, each run through all four retrieval configurations. The JSON next to
this file is the full report per run; each one records the settings it used, so a baseline
is self-describing and does not depend on this page staying accurate.

- **`eval/questions.yaml`** — the regression set. Files without a prefix below. Every
  configuration scores 100%, so it is a floor to hold, not a comparison.
- **`eval/questions-hard.yaml`** — the discrimination set, built to separate the
  configurations. Files prefixed `hard-`. [Results below](#discrimination-set).

Committed here rather than in `eval/reports/`, which stays git-ignored: a baseline is a
reference point that should change only deliberately, an ordinary run is scratch output.

## Regression set

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

### What the regression set can and cannot be used for

**It is a regression floor, not an A/B comparison.** Every configuration scores
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

**Using it.** As a floor it is still worth having: any future change that drops a number
below these has broken something, and `--fail-under` has a real value to check against.
What it cannot do is justify turning hybrid or reranking on. Treat a tie here as "no
regression", never as "no difference" — `questions-hard.yaml` exists to answer the second
question.

---

## Discrimination set

`eval/questions-hard.yaml`, 20 questions over the same corpus, written to break dense
retrieval rather than to be passed: bare identifiers (`X-Atlas-Key`, `atlas.dlq.v1`,
`SEC-001`), figures that recur across documents with different meanings (14 days, 5
minutes, "two days"), cross-document chains, and unanswerable questions whose retrieved
passage nearly contains the answer.

| | `RETRIEVAL_MODE` | `RERANK_PROVIDER` |
| --- | --- | --- |
| [`hard-dense.json`](hard-dense.json) | `dense` | `none` |
| [`hard-hybrid.json`](hard-hybrid.json) | `hybrid` | `none` |
| [`hard-dense-rerank.json`](hard-dense-rerank.json) | `dense` | `fastembed` |
| [`hard-hybrid-rerank.json`](hard-hybrid-rerank.json) | `hybrid` | `fastembed` |

Same models and settings as the regression runs above.

| Metric | dense | hybrid | dense+rerank | hybrid+rerank |
| --- | --- | --- | --- | --- |
| `mean_groundedness` | 4.95 | 4.95 | 4.9 | **5.0** |
| `grounded_rate_pct` | 100.0 | 100.0 | 95.0 | 100.0 |
| `hallucination_rate_pct` | 0.0 | 0.0 | 0.0 | 0.0 |
| `citation_validity_pct` | 95.0 | 100.0 | 100.0 | 100.0 |
| `abstention_accuracy_pct` | 95.0 | 100.0 | 100.0 | 100.0 |
| `retrieval_recall_pct` | 100.0 | 100.0 | 100.0 | 100.0 |
| `must_contain_pass_pct` | 100.0 | 100.0 | 100.0 | 100.0 |

The metrics move, which is the point — but only two questions are responsible, and the two
are not equally meaningful.

### `trap-who-approves` — a real retrieval failure, verified without the judge

*"Who must approve access to Restricted data, and who approves a request to work from
another country?"* Dense-only **abstains on a question the corpus answers**. Hybrid, and
either reranked configuration, answers it correctly.

This is not judge variance. Running the retriever directly on the question, top-5 chunks:

```
dense                                   hybrid
[1] security-policy.md::1               [1] security-policy.md::1     both
[2] security-policy.md::0               [2] security-policy.md::0     both
[3] security-policy.md::2               [3] security-policy.md::2     both
[4] employee-handbook.md::2             [4] employee-handbook.md::0   lexical  <- "People Operations"
[5] security-policy.md::3               [5] employee-handbook.md::2   dense
```

Dense spends four of five slots on the security policy — every one of them a good semantic
match for "who must approve" — and the one handbook chunk it does return is the wrong one.
The chunk containing "approved by People Operations" enters the hybrid ranking at position
4 **through the lexical channel alone**. The model then abstained on the half it could not
see, which is correct behaviour on a context that genuinely lacked the fact: the failure
was retrieval, and generation reported it honestly.

Note what `retrieval_recall_pct` did with this: **100% for both**. It checks whether the
expected source *file* appeared, and `employee-handbook.md` appeared either way — chunk
`::2` instead of `::0`. Source-level recall is structurally blind to the failure this
question exists to catch. `must_contain_pass_pct` also passed, because the dense run
abstained rather than answering wrongly, and an abstention has no missing substring to
find. `abstention_accuracy_pct` is the metric that caught it.

### `unans-dlq-replicas` — judge variance, not a retrieval difference

All four configurations **abstained correctly**. Only the wording differed, and the judge
scored `dense+fastembed`'s phrasing 3/5 and `partially_grounded` for leading with the
14-day retention fact before declining. Retrieved sources are identical across all four.
The `cross-vendor-atlas-telemetry` 4-vs-5 difference is the same kind of noise.

### What this does and does not establish

**Established, deterministically:** dense-only retrieval loses a chunk on this corpus that
the lexical channel recovers, and the project's headline retrieval metric cannot see it.
That is a concrete argument for `RETRIEVAL_MODE=hybrid`, and it is reproducible without
calling a judge at all.

**Not established:** any ranking of the four configurations by score. One question out of
20 moves a percentage by 5 points, each configuration ran once, and the judge is an LLM.
`hybrid+fastembed` being the only configuration with a clean sheet is one question's worth
of evidence, not a result.

**Also worth noting:** the `lex-*` family — the exact-token questions this set was
originally built around — did **not** discriminate. All four configurations answered every
one of them. On an 11-chunk corpus, `top_k=5` catches the right chunk even when the ranking
is poor. The question that did discriminate was a cross-document one, where four
competing passages crowded the second document out of the top-5 entirely. That is the
shape of query worth writing more of.
