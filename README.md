# RAG Document Assistant — Grounded LLM Q&A with Citations

Python · LangChain · Anthropic API · Chroma

An end-to-end retrieval-augmented generation pipeline: documents are ingested and
chunked, embedded into a Chroma vector database, retrieved by semantic search, and
answered by Claude **strictly from the retrieved passages**, with inline `[n]`
citations resolved back to source files. A second stage scores every answer for
groundedness against its own retrieved context and flags hallucinations across a
curated test-question set.

---

## What it does

**RAG pipeline**

- **Ingestion & chunking** — Markdown/text/PDF loading, recursive splitting on
  structural boundaries, deterministic chunk IDs (`<source>::<index>`) so re-ingestion
  upserts instead of duplicating.
- **Semantic search** — embeddings stored in a persistent Chroma collection; query and
  passage sides encoded asymmetrically where the model supports it.
- **Hybrid retrieval (optional)** — a lexical BM25 channel fused with vector search by
  reciprocal rank fusion, so exact tokens (figures, codes, product names) that embeddings
  squash away still surface.
- **Two-stage retrieval (optional)** — a wide candidate set is fetched by vector search,
  then a cross-encoder reranker reads query and passage *together* and keeps the best
  `TOP_K`. Off by default, so it can be A/B'd against the single-stage baseline.
- **Grounded generation** — Claude answers only from the numbered context block, cites
  with inline markers, and abstains (`INSUFFICIENT_CONTEXT: …`) rather than guessing.
- **Citation resolution** — markers are mapped back to chunks, sources and relevance
  scores; markers pointing outside the context block are reported as fabricated.
  Optionally the Anthropic **Citations API** replaces marker parsing entirely: the API
  returns the exact character span behind each claim, so a citation is checkable against
  the source instead of being the model's own claim about what it used.

**Evaluation**

- **LLM-as-judge groundedness** — the answer is decomposed into atomic claims, each
  marked `supported` / `partially_supported` / `unsupported` against the retrieved
  passages only, producing a 1–5 groundedness score and a verdict.
- **Deterministic checks** that need no model — retrieval recall against expected
  sources, dangling citation markers, required-content coverage, and abstention
  accuracy on questions the corpus deliberately cannot answer.
- **Reports** — per-run JSON + Markdown in `eval/reports/`, plus `--fail-under` for CI.

**HTTP API**

- **FastAPI service** over the same pipeline — `POST /ask`, `POST /ingest`, `GET /status`,
  `GET /health` — with an OpenAPI schema and Swagger UI at `/docs`.

---

## Quick start

Dependencies are managed with [uv](https://docs.astral.sh/uv/). One command creates the
environment, resolves the locked dependency set and installs the project:

```bash
uv sync
```

Copy `.env.example` to `.env` and put your Anthropic key in it, then:

```bash
uv run rag ingest
```

```bash
uv run rag ask "Who has to approve an expense claim of 350 pounds?"
```

```bash
uv run rag eval
```

`uv run` uses the project environment without activating it. If you prefer an activated
shell, `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (macOS/Linux)
works too, after which `rag ask "…"` and `python -m rag.cli ask "…"` are equivalent.

Sample output for `ask`:

```
┌────────────────────────────── Answer ──────────────────────────────┐
│ An expense claim of £350 requires approval from the employee's     │
│ direct manager and additional sign-off from the department head,   │
│ because claims at or above £200 need department-head approval [1]. │
└────────────────────────────────────────────────────────────────────┘
                              Citations
  #    Source                  Score   Snippet
 [1]   employee-handbook.md    0.712   ## Expenses Expenses under £200 are approved by…
```

---

## Commands

| Command | Purpose |
| --- | --- |
| `uv run rag ingest [--reset]` | Load, chunk, embed and persist every document in `data/docs` |
| `uv run rag ask "<question>" [-k N] [--show-context]` | Grounded answer with a citation table |
| `uv run rag eval [--questions …] [--fail-under 4.5]` | Run the evaluation loop and write a report |
| `uv run rag status` | Show resolved configuration and index size |
| `uv run rag serve [--host …] [--port …] [--reload]` | Serve the HTTP API, Swagger UI at `/docs` |

Add your own documents by dropping `.md`, `.txt` or `.pdf` files into `data/docs/`
(subdirectories are walked) and re-running `ingest`.

---

## HTTP API

```bash
uv run rag serve
```

Swagger UI on <http://127.0.0.1:8000/docs>, raw schema on `/openapi.json`.

| Endpoint | Purpose |
| --- | --- |
| `POST /ask` | `{"question": …, "k": 5, "include_context": false}` → answer, citations, `dangling_citations`, `abstained`, token usage |
| `POST /ingest` | `{"reset": false}` → files, chunks and resulting collection size |
| `GET /status` | Resolved configuration and index size |
| `GET /health` | Liveness — touches neither the index nor the API key |

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question":"Who has to approve an expense claim of £350?"}'
```

The pipeline is built on first use, not at startup, so the service boots even with an
empty index or a missing key and reports the problem where it matters: `409` when
nothing is indexed, `503` when `ANTHROPIC_API_KEY` is unset. Everything the CLI returns
is on the wire too — citations keep their source, chunk ID and relevance score, and
fabricated markers stay in a separate `dangling_citations` field rather than being
silently dropped.

The default bind is loopback; `--host 0.0.0.0` exposes the service, which has no
authentication and spends API credits per request — put it behind something.

---

## Configuration

Everything is environment-driven — see `.env.example` for the full list.

| Variable | Default | Notes |
| --- | --- | --- |
| `ANSWER_MODEL` / `JUDGE_MODEL` | `claude-opus-5` | Adaptive thinking left on; it improves grounding and citation accuracy |
| `EMBEDDING_PROVIDER` | `fastembed` | `fastembed` (local ONNX, no extra key) · `voyage` · `huggingface` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Provider-appropriate default is used if unset |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `900` / `150` | Characters |
| `TOP_K` | `5` | Passages shown to the model per question |
| `SCORE_THRESHOLD` | `0.0` | Drop weak matches; `0.0` disables. Applied to vector scores *before* reranking |
| `RERANK_PROVIDER` | `none` | `none` · `fastembed` (local, no extra key) · `voyage` · `huggingface` · `llm` (Claude; not for eval runs — non-deterministic and one extra call per question) |
| `RERANK_MODEL` | `Xenova/ms-marco-MiniLM-L-6-v2` | Provider-appropriate default is used if unset |
| `RERANK_CANDIDATES` | `20` | Passages fetched for the reranker to narrow to `TOP_K` |
| `RETRIEVAL_MODE` | `dense` | `dense` (embeddings only) · `hybrid` (embeddings + BM25 fused by RRF) |
| `RRF_K` | `60` | Fusion damping — higher makes cross-channel agreement count for more |
| `BM25_K1` / `BM25_B` | `1.5` / `0.75` | Okapi BM25 term saturation and length normalisation |
| `CITATION_MODE` | `markers` | `markers` (inline `[n]`, parsed back) · `native` (Citations API character spans) |

Anthropic does not serve an embeddings endpoint, so that half of the pipeline is
pluggable. The default runs locally with no second API key and no torch dependency.
Alternative backends are declared as optional extras. For Voyage AI (Anthropic's
recommended embedding partner):

```bash
uv sync --extra voyage
```

then set `EMBEDDING_PROVIDER=voyage`, `EMBEDDING_MODEL=voyage-3` and `VOYAGE_API_KEY`
in `.env`. For local sentence-transformers instead:

```bash
uv sync --extra huggingface
```

and set `EMBEDDING_PROVIDER=huggingface`. Switching embedding model changes the vector
space — re-run `uv run rag ingest --reset` afterwards.

### Hybrid retrieval

Vector search matches on meaning, which is exactly why it misses exact tokens: a query for
`£350`, a product name, or an error code has to survive being squashed into one dense
vector. BM25 cannot generalise at all, but a rare term appearing verbatim pins its chunk to
the top. The two fail on different queries, so both run and the rankings are fused:

```bash
RETRIEVAL_MODE=hybrid
```

Fusion is by **reciprocal rank**, not score — a cosine similarity and a BM25 score share no
scale, so any weighted sum of the two would be tuning a meaningless constant. RRF asks each
channel only for an ordering, which is why it needs no calibration. BM25 is implemented in
`rag/hybrid.py` rather than pulled in as a dependency: it is one formula, and owning the
tokenizer is what makes `£200` and `200` match the same term. Tokenization is Unicode-aware,
so accented and non-Latin words survive intact.

There is no stemming: the lexical channel matches surface forms only. On heavily inflected
languages a query in one grammatical case will not match another (`wydatków` does not find
`wydatki`), so the dense channel carries more of the load there. Matching nothing is the
intended behaviour — a lexical channel that guesses is worse than one that abstains.

Hybrid composes with reranking — retrieve wide on both channels, fuse, then let the
cross-encoder narrow to `TOP_K`. The index is built from the chunks already in Chroma, so
both channels always range over exactly the same corpus, and no re-ingestion is needed.

### Native citations

By default the model writes `[n]` into its prose and the pipeline parses it back out — the
citation is the model's own claim about what it used. `CITATION_MODE=native` sends each
passage as a citation-enabled document block and lets the API report citations as
structured data instead:

```bash
CITATION_MODE=native
```

Each citation comes back with `cited_text` and a `start_char_index`/`end_char_index` span
produced by the serving layer, not written by the model — so `chunk.text[start:end]` equals
`cited_text` exactly, and a citation cannot point at text that isn't there. Fabricated
references stop being expressible rather than being detected after the fact, which is why
`dangling_citations` is always empty in this mode. The tradeoff: answers carry no inline
`[n]`, so the mapping lives in the citation table rather than in the prose.

Because there are no markers to check, the evaluation loop swaps in a judge prompt that
scores support only, and verifies attribution in code instead —
`citation_span_integrity_pct` re-derives every span from the source and needs no model at
all. Asking an LLM whether a citation was correct is strictly weaker than looking it up.

### Reranking

Reranking reuses the same extras, so no new dependency is involved either way: the default
`fastembed` backend ships a cross-encoder in the package that is already required, and the
`voyage` / `huggingface` extras above already carry theirs. Turn it on with

```bash
RERANK_PROVIDER=fastembed
```

The first question after enabling it downloads ~90 MB, then caches. Unlike an embedding
change, reranking does **not** touch the vector space — no re-ingestion needed, which is
what makes it cheap to A/B: run `uv run rag eval` with it off, then on, and compare
`retrieval_recall_pct` and `mean_groundedness` in the two reports.

---

## Evaluation set

`eval/questions.yaml` is a curated set covering single-fact lookups, rule application,
multi-part questions, a cross-document question, and — most importantly — questions the
corpus **cannot** answer:

```yaml
- id: hb-expense-approval
  question: Who has to approve an expense claim of £350?
  answerable: true
  expected_sources: [employee-handbook.md]
  must_contain: ["department head"]

- id: unanswerable-atlas-pricing
  question: How much does Atlas cost per million ingested records?
  answerable: false
  note: Pricing is never stated in the spec.
```

Metrics reported per run:

| Metric | Meaning |
| --- | --- |
| `mean_groundedness` | Mean judge score, 1–5 |
| `grounded_rate_pct` | Answers with every claim supported and correctly cited |
| `hallucination_rate_pct` | Answers containing at least one unsupported claim |
| `citation_validity_pct` | Cited passages actually back the sentence they are attached to (judge; marker mode) |
| `citation_span_integrity_pct` | Every reported span matches its source text exactly (deterministic; native mode) |
| `dangling_citation_count` | `[n]` markers pointing outside the retrieved context |
| `abstention_accuracy_pct` | Abstained exactly when the corpus lacks the answer |
| `retrieval_recall_pct` | Expected source appeared in the top-k |
| `must_contain_pass_pct` | Required facts present in the answer |

Any question that trips a check is listed in the report with its flags, the judge's
reasoning, and the specific unsupported claims — so a regression points at a cause,
not just a lower number.

---

## Layout

```
rag/
  config.py      environment-driven settings
  loaders.py     markdown / text / PDF loading with source metadata
  ingest.py      chunking + deterministic IDs + upsert into Chroma
  embeddings.py  pluggable embedding backends
  store.py       persistent Chroma collection
  retriever.py   semantic search, numbered citation-ready passages
  hybrid.py      BM25 lexical channel + reciprocal rank fusion
  rerank.py      pluggable second-stage cross-encoder reranking
  citations.py   Citations API document blocks and span extraction
  prompts.py     grounded-answer, rerank and judge prompts
  llm.py         Anthropic chat model + tolerant JSON parsing
  pipeline.py    retrieve → generate → resolve citations
  evaluate.py    groundedness judge, deterministic checks, reporting
  api.py         FastAPI service over the pipeline (OpenAPI + Swagger UI)
  cli.py         ingest / ask / eval / status / serve
data/docs/       sample corpus (fictional company documents)
eval/            questions.yaml + generated reports
tests/           unit tests for chunking, citation resolution, parsing, HTTP layer
```

Run the tests (no API calls, no network):

```bash
uv run pytest
```

`pyproject.toml` is the single source of truth for dependencies and `uv.lock` pins the
exact resolved set — commit both. If a consumer needs a `requirements.txt`, generate one
with `uv export --no-dev --format requirements-txt > requirements.txt`.

---

## Design notes

- **Abstention is a first-class outcome.** The system prompt defines an exact
  `INSUFFICIENT_CONTEXT:` response, the pipeline surfaces it as `abstained`, and the
  eval set contains questions where abstaining is the only correct answer. A RAG system
  that never declines is not grounded — it is lucky.
- **Two independent hallucination signals.** The judge is an LLM and can be wrong;
  dangling citation markers, retrieval misses and missing required facts are checked in
  code and cannot be talked around.
- **The judge sees only the retrieved context.** A claim that is true in the world but
  absent from the passages is scored `unsupported` — that is the property being measured.
- **Sampling parameters are omitted.** `temperature` / `top_p` / `top_k` are rejected by
  Claude Opus 5; behaviour is steered through the prompt instead.
- **Every retrieval stage records its own score.** A cosine similarity, a BM25 score, an
  RRF score and a cross-encoder logit share no scale, so collapsing them into one number
  destroys all four. A passage carries whichever of them applied, and `channel` says which
  side found it — the citation table can then show both where a passage came from and why
  it was promoted.
- **Reranking never overwrites the vector score.** A cross-encoder emits unbounded logits
  that are comparable only within one query; the 0..1 relevance score is what
  `SCORE_THRESHOLD` and the citation table are calibrated against. Both are kept, so the
  citation table can show where a passage came from *and* why it was promoted.
- **Re-ingestion is idempotent.** Chunk IDs are derived from source path and index, so
  editing one document and re-running `ingest` updates its chunks in place.
