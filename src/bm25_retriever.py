#!/usr/bin/env python3
"""
src/bm25_retriever.py
Wrapper around rank_bm25.BM25Okapi for high-speed CPU retrieval.
Uses k1=0.9, b=0.4 (same as official baseline).
"""
import sys
import re
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[A-Za-z0-9]+")

def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())

class BM25Retriever:
    """
    Okapi BM25 retriever (optimized via rank_bm25).
    k1=0.9, b=0.4 (RETECO official defaults).
    """

    def __init__(self, doc_ids: list[str], doc_texts: list[str], k1=0.9, b=0.4):
        self.doc_ids = list(doc_ids)
        self.N = len(doc_ids)
        
        # tokenize
        docs = [tokenize(t) for t in doc_texts]
        
        # build rank_bm25 index
        self.bm25 = BM25Okapi(docs, k1=k1, b=b)

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """Return [(doc_id, score), ...] sorted descending."""
        q = tokenize(query)
        scores = self.bm25.get_scores(q)
        
        # Optimized top-k sort
        if self.N == 0:
            return []
        
        k = min(top_k, self.N)
        # argpartition is O(N) instead of O(N log N)
        top_idx = np.argpartition(scores, -k)[-k:]
        # Sort just the top-k subset
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        
        return [(self.doc_ids[i], float(scores[i])) for i in top_idx]

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
