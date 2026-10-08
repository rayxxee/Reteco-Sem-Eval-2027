#!/usr/bin/env python3
"""
src/bm25_retriever.py
Thin wrapper around the starter-kit's pure-Python BM25.
Copies bm25.py logic here so we don't need to import from starter_kit dir.
k1=0.9, b=0.4 (same as official baseline).
"""
import sys
import math
import re
from pathlib import Path

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


class BM25Retriever:
    """
    Okapi BM25 retriever (verbatim port of starter_kit/bm25.py).
    k1=0.9, b=0.4  (RETECO official defaults).
    """

    def __init__(self, doc_ids: list[str], doc_texts: list[str], k1=0.9, b=0.4):
        self.k1, self.b = k1, b
        self.doc_ids = list(doc_ids)
        self.N = len(doc_ids)

        # tokenize
        self.docs = [tokenize(t) for t in doc_texts]
        self.doc_len = [len(d) for d in self.docs]
        self.avgdl = sum(self.doc_len) / self.N if self.N else 0.0

        # tf and df
        self.tf: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for d in self.docs:
            counts: dict[str, int] = {}
            for w in d:
                counts[w] = counts.get(w, 0) + 1
            self.tf.append(counts)
            for w in counts:
                df[w] = df.get(w, 0) + 1

        # IDF (BM25+ style, non-negative)
        self.idf = {
            w: math.log(1 + (self.N - n + 0.5) / (n + 0.5))
            for w, n in df.items()
        }

    def _score(self, q_terms: list[str], i: int) -> float:
        score, dl, k1, b = 0.0, self.doc_len[i], self.k1, self.b
        tf_i = self.tf[i]
        avgdl = self.avgdl or 1.0
        for w in q_terms:
            if w not in tf_i:
                continue
            f = tf_i[w]
            denom = f + k1 * (1 - b + b * dl / avgdl)
            score += self.idf.get(w, 0.0) * (f * (k1 + 1)) / denom
        return score

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Return [(doc_id, score), ...] sorted descending."""
        q = tokenize(query)
        scored = [(self.doc_ids[i], self._score(q, i)) for i in range(self.N)]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_k]

    def batch_search(
        self, queries: list[tuple[str, str]], top_k: int = 100
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Parameters
        ----------
        queries : [(qid, query_text), ...]

        Returns
        -------
        {qid: [(doc_id, score), ...]}
        """
        return {qid: self.search(q, top_k) for qid, q in queries}
