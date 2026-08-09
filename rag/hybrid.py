"""Hybrid retrieval: a lexical BM25 channel fused with dense vector search.

Dense embeddings match on meaning, which is exactly what makes them miss exact
tokens. A query for `£350` or `Atlas` or an error code has to survive being
squashed into a 384-dimensional vector alongside everything else in the chunk —
and often doesn't. BM25 is the opposite: it cannot generalise at all, but a rare
term that appears verbatim in one chunk pins that chunk to the top.

The two fail on different queries, so the fusion is the point. Reciprocal rank
fusion combines them on *rank* rather than score, which sidesteps the fact that
a cosine similarity and a BM25 score share no scale and no calibration.

BM25 is implemented here rather than pulled in as a dependency: it is one
formula, and owning the tokenizer matters — the default splitter keeps digits,
so `£200` and `200` match the same token.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric terms.

    Currency symbols and punctuation are dropped rather than kept, so `£200`,
    `200`, and `(200)` all reduce to the same token — the whole reason to run a
    lexical channel is to match figures like that verbatim.
    """
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Index:
    """Okapi BM25 over an in-memory corpus, keyed by chunk ID.

    Built from the chunks already in Chroma, so the lexical and dense channels
    are guaranteed to range over exactly the same corpus.
    """

    ids: list[str]
    _docs: list[list[str]]
    _doc_freq: dict[str, int]
    _avg_len: float
    k1: float
    b: float

    @classmethod
    def build(
        cls,
        ids: Sequence[str],
        texts: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Index":
        docs = [tokenize(text) for text in texts]
        doc_freq: Counter[str] = Counter()
        for tokens in docs:
            doc_freq.update(set(tokens))
        avg_len = (sum(len(d) for d in docs) / len(docs)) if docs else 0.0
        return cls(
            ids=list(ids),
            _docs=docs,
            _doc_freq=dict(doc_freq),
            _avg_len=avg_len,
            k1=k1,
            b=b,
        )

    def __len__(self) -> int:
        return len(self.ids)

    def _idf(self, term: str) -> float:
        # Smoothed IDF: always positive, so a term present in every document
        # contributes ~0 rather than dragging the score negative.
        n = len(self._docs)
        df = self._doc_freq.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return up to `k` (chunk_id, score) pairs, best first.

        Chunks scoring zero — no query term present at all — are dropped rather
        than padded in: a lexical channel with nothing to say should contribute
        nothing to the fusion, not an arbitrary tail.
        """
        terms = tokenize(query)
        if not terms or not self._docs:
            return []

        idf = {term: self._idf(term) for term in set(terms)}
        scored: list[tuple[str, float]] = []
        for chunk_id, tokens in zip(self.ids, self._docs):
            if not tokens:
                continue
            counts = Counter(tokens)
            norm = self.k1 * (1 - self.b + self.b * len(tokens) / (self._avg_len or 1))
            score = 0.0
            for term in idf:
                freq = counts.get(term)
                if not freq:
                    continue
                score += idf[term] * (freq * (self.k1 + 1)) / (freq + norm)
            if score > 0:
                scored.append((chunk_id, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]


def reciprocal_rank_fusion(
    rankings: Iterable[Sequence[str]], k: int = 60
) -> dict[str, float]:
    """Fuse ranked ID lists into one score per ID: sum of 1 / (k + rank).

    Rank-based rather than score-based on purpose — a cosine similarity and a
    BM25 score have no common scale, so any weighted sum of the two would be
    tuning a meaningless constant. RRF only asks each channel for an ordering.

    `k` damps the head of each list: at the default 60 the gap between rank 1
    and rank 2 is small, so a chunk both channels rank highly beats one that a
    single channel ranks first. Lower `k` sharpens the winner-takes-most effect.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def fuse_rankings(
    rankings: Iterable[Sequence[str]], k: int = 60
) -> list[tuple[str, float]]:
    """`reciprocal_rank_fusion`, returned as a ranked list instead of a mapping.

    Ties break on the order IDs were first seen, which makes the dense channel —
    passed first by the retriever — the tiebreaker.
    """
    rankings = [list(r) for r in rankings]
    order: dict[str, int] = {}
    for doc_id in (doc_id for r in rankings for doc_id in r):
        order.setdefault(doc_id, len(order))  # first sighting wins
    scores = reciprocal_rank_fusion(rankings, k=k)
    return sorted(scores.items(), key=lambda pair: (-pair[1], order[pair[0]]))
