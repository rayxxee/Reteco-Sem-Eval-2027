#!/usr/bin/env python3
"""
src/features.py
Feature engineering for the BM25 top-50 reranker (Phase 3).

All features are derived from data we actually have:
  - BM25 score (from BM25Retriever.search)
  - TF-IDF cosine similarity (from sklearn TfidfVectorizer)
  - Query-doc term overlap (Jaccard on token sets)
  - Normalized document length
  - Normalized query length
  - Reciprocal rank position from BM25
  - Temporal match features:
      * year overlap (intersection of years in query vs doc)
      * date overlap (any date string in common)
      * doc has any temporal cue (bool)
      * query has any temporal cue (bool)
"""
import math
import re
from typing import Optional

import numpy as np


# ── Token overlap helpers ────────────────────────────────────────────────────

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> set[str]:
    return set(_TOKEN.findall((text or "").lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── Temporal helpers (re-use data_utils patterns) ────────────────────────────

_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")
_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},?\s+\d{4}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b",
    re.IGNORECASE,
)


def years_in(text: str) -> set[str]:
    return set(_YEAR_RE.findall(text or ""))


def has_date(text: str) -> bool:
    return bool(_DATE_RE.search(text or ""))


# ── Feature vector builder ───────────────────────────────────────────────────

FEATURE_NAMES = [
    "bm25_score",           # raw BM25 score
    "bm25_score_norm",      # bm25 / max_bm25_in_top50
    "tfidf_cosine",         # TF-IDF cosine (set during batch_build)
    "term_overlap_jaccard", # Jaccard of query tokens vs doc tokens
    "doc_len_log",          # log(1 + n_doc_tokens)
    "query_len_log",        # log(1 + n_query_tokens)
    "rank_recip",           # 1 / rank (rank from BM25 order)
    "year_overlap",         # |years_query ∩ years_doc|
    "year_overlap_frac",    # year_overlap / (|years_query| + 1e-6)
    "query_has_year",       # bool: query contains any year
    "doc_has_year",         # bool: doc contains any year
    "query_has_date",       # bool: query contains date expression
    "doc_has_date",         # bool: doc contains date expression
]

N_FEATURES = len(FEATURE_NAMES)


def build_features(
    query: str,
    query_id: str,
    bm25_results: list[tuple[str, float]],   # [(doc_id, bm25_score), ...]
    doc_texts: dict[str, str],                # doc_id -> content
    tfidf_scores: Optional[dict[str, float]] = None,  # doc_id -> cosine
    max_bm25: Optional[float] = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Build feature matrix for one query over its top-k BM25 candidates.

    Returns
    -------
    X : np.ndarray  shape (k, N_FEATURES)
    doc_ids : list[str]  same order as rows of X
    """
    q_toks = tokenize(query)
    q_years = years_in(query)
    q_has_year = float(bool(q_years))
    q_has_date = float(has_date(query))
    q_len = math.log1p(len(q_toks))

    if max_bm25 is None:
        scores_only = [s for _, s in bm25_results]
        max_bm25 = max(scores_only) if scores_only else 1.0
    if max_bm25 == 0:
        max_bm25 = 1.0

    rows = []
    doc_ids_out = []
    for rank, (doc_id, bm25_score) in enumerate(bm25_results, start=1):
        doc_text = doc_texts.get(doc_id, "")
        d_toks = tokenize(doc_text)
        d_years = years_in(doc_text)
        d_has_year = float(bool(d_years))
        d_has_date = float(has_date(doc_text))
        d_len = math.log1p(len(d_toks))

        year_ov = float(len(q_years & d_years))
        year_frac = year_ov / (len(q_years) + 1e-6)
        jaccard_ov = jaccard(q_toks, d_toks)
        tfidf_cos = tfidf_scores.get(doc_id, 0.0) if tfidf_scores else 0.0

        row = [
            bm25_score,
            bm25_score / max_bm25,
            tfidf_cos,
            jaccard_ov,
            d_len,
            q_len,
            1.0 / rank,
            year_ov,
            year_frac,
            q_has_year,
            d_has_year,
            q_has_date,
            d_has_date,
        ]
        rows.append(row)
        doc_ids_out.append(doc_id)

    return np.array(rows, dtype=np.float32), doc_ids_out


def build_training_pairs(
    queries: list[dict],
    qrels: dict[str, set[str]],
    bm25_results: dict[str, list[tuple[str, float]]],
    doc_texts: dict[str, str],
    tfidf_scores_by_query: Optional[dict[str, dict[str, float]]] = None,
    top_k: int = 50,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build (X, y) training data for the reranker.

    For each query:
      - positives: docs in qrels (label=1)
      - negatives: BM25 top-k docs not in qrels (label=0)

    Returns
    -------
    X : (N, N_FEATURES)
    y : (N,) binary labels
    qids : list of query IDs (for group-wise ranking)
    """
    X_parts, y_parts, qids_out = [], [], []

    for q_rec in queries:
        qid = q_rec["id"]
        query = q_rec["query"]
        relevant = qrels.get(qid, set())
        candidates = bm25_results.get(qid, [])[:top_k]
        if not candidates:
            continue

        max_bm25 = max(s for _, s in candidates) if candidates else 1.0
        tfidf = (tfidf_scores_by_query or {}).get(qid, {})
        X, doc_ids = build_features(query, qid, candidates, doc_texts, tfidf, max_bm25)
        y = np.array([1.0 if d in relevant else 0.0 for d in doc_ids])

        X_parts.append(X)
        y_parts.append(y)
        qids_out.extend([qid] * len(doc_ids))

    if not X_parts:
        return np.zeros((0, N_FEATURES)), np.zeros(0), []

    return np.vstack(X_parts), np.concatenate(y_parts), qids_out
