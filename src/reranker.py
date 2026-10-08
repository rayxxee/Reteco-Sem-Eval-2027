#!/usr/bin/env python3
"""
src/reranker.py
LightGBM reranker trained on BM25 top-50 features (Phase 3).
Train on TRAIN split only; evaluate on DEV.
"""
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np


SEED = 42


class LGBMReranker:
    """
    LightGBM binary classifier used as a reranker.
    Trained on (feature_vector, binary_relevance_label) pairs.
    At inference, sort by predicted probability descending.
    """

    def __init__(self, n_estimators=200, learning_rate=0.05, max_depth=6):
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=4,
            verbosity=-1,
        )
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LGBMReranker":
        import lightgbm as lgb
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(relevant) for each row."""
        if self.model is None:
            raise RuntimeError("Model not fitted yet")
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("Model not fitted yet")
        imp = self.model.feature_importances_
        return dict(sorted(zip(feature_names, imp), key=lambda x: -x[1]))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: Path) -> "LGBMReranker":
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        return self


def rerank_run(
    queries: list[dict],
    bm25_results: dict[str, list[tuple[str, float]]],
    doc_texts: dict[str, str],
    reranker: LGBMReranker,
    tfidf_scores_by_query: Optional[dict[str, dict[str, float]]] = None,
    top_k: int = 50,
) -> dict[str, list[tuple[str, float]]]:
    """
    Rerank BM25 top-k results for each query.

    Returns
    -------
    dict  qid -> [(doc_id, score), ...]  sorted by score desc
    """
    from features import build_features, FEATURE_NAMES

    reranked = {}
    for q_rec in queries:
        qid = q_rec["id"]
        query = q_rec["query"]
        candidates = bm25_results.get(qid, [])[:top_k]
        if not candidates:
            reranked[qid] = []
            continue

        max_bm25 = max(s for _, s in candidates) if candidates else 1.0
        tfidf = (tfidf_scores_by_query or {}).get(qid, {})
        X, doc_ids = build_features(query, qid, candidates, doc_texts, tfidf, max_bm25)

        probs = reranker.predict_proba(X)
        ranked = sorted(zip(doc_ids, probs.tolist()), key=lambda x: -x[1])
        reranked[qid] = ranked

    return reranked
