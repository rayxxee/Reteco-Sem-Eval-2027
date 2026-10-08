#!/usr/bin/env python3
"""
src/evaluate.py
Official nDCG@10 evaluation using the starter-kit's ir_metrics module
AND pytrec_eval as cross-check. Macro-averaged over queries, per the
RETECO protocol.

Uses the starter_kit/starter_kit/ir_metrics.py ndcg_at_k function directly
so we produce results identical to the official scorer.
"""
import sys
from pathlib import Path
from typing import Optional
import math


# ── Pure-Python nDCG (mirrors starter_kit/ir_metrics.py) ─────────────────────

def dcg_at_k(ranked_docs: list[str], relevant: set[str], k: int) -> float:
    """Compute DCG@k for a single query."""
    gain = 0.0
    for i, doc in enumerate(ranked_docs[:k], start=1):
        if doc in relevant:
            gain += 1.0 / math.log2(i + 1)
    return gain


def ideal_dcg_at_k(relevant: set[str], k: int) -> float:
    """Compute IDCG@k."""
    n = min(len(relevant), k)
    return sum(1.0 / math.log2(i + 1) for i in range(1, n + 1))


def ndcg_at_k(ranked_docs: list[str], relevant: set[str], k: int) -> float:
    """Compute nDCG@k for a single query. Returns 0 if no relevant docs."""
    idcg = ideal_dcg_at_k(relevant, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked_docs, relevant, k) / idcg


# ── Macro-average evaluation ───────────────────────────────────────────────────

def evaluate_run(
    run: dict[str, list[str]],
    qrels: dict[str, set[str]],
    k: int = 10,
    verbose: bool = False,
) -> dict[str, float]:
    """
    Evaluate a retrieval run.

    Parameters
    ----------
    run   : {qid: [ranked doc_ids, ...]}
    qrels : {qid: {relevant doc_ids}}
    k     : cut-off (default 10)

    Returns
    -------
    {
      "ndcg@10": float,       # macro-average over queries with qrels
      "num_topics": int,
      "per_query": {qid: ndcg_score},
    }
    """
    per_query = {}
    for qid, relevant in qrels.items():
        ranked = run.get(qid, [])
        per_query[qid] = ndcg_at_k(ranked, relevant, k)

    n = len(per_query)
    macro = sum(per_query.values()) / n if n > 0 else 0.0

    if verbose:
        for qid, score in sorted(per_query.items()):
            print(f"  {qid}: {score:.4f}")

    return {
        f"ndcg@{k}": round(macro, 5),
        "num_topics": n,
        "per_query": per_query,
    }


def pytrec_evaluate(
    run: dict[str, list[str]],
    qrels: dict[str, set[str]],
    k: int = 10,
) -> dict[str, float]:
    """
    Cross-check evaluation using pytrec_eval.
    run format: {qid: [docid, ...]} (ranked list, position = rank)
    """
    try:
        import pytrec_eval
    except ImportError:
        return {"error": "pytrec_eval not installed"}

    # Convert run to pytrec_eval dict format {qid: {docid: score}}
    run_pt = {
        qid: {doc: float(len(docs) - i) for i, doc in enumerate(docs)}
        for qid, docs in run.items()
    }
    qrels_pt = {qid: {d: 1 for d in docs} for qid, docs in qrels.items()}

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels_pt, {f"ndcg_cut.{k}"}
    )
    scores = evaluator.evaluate(run_pt)
    if not scores:
        return {f"ndcg@{k}": 0.0, "num_topics": 0}

    vals = [scores[q][f"ndcg_cut_{k}"] for q in scores]
    return {
        f"ndcg@{k}": round(sum(vals) / len(vals), 5),
        "num_topics": len(vals),
    }


# ── TREC run file I/O ─────────────────────────────────────────────────────────

def write_trec_run(
    path: Path,
    run: dict[str, list[tuple[str, float]]],
    tag: str = "reteco",
) -> None:
    """
    Write a TREC-format run file.
    run : {qid: [(doc_id, score), ...]}  (pre-sorted, best first)
    """
    with open(path, "w", encoding="utf-8") as fh:
        for qid, ranked in run.items():
            for rank, (docid, score) in enumerate(ranked, 1):
                fh.write(f"{qid}\tQ0\t{docid}\t{rank}\t{score:.6f}\t{tag}\n")


def read_trec_run(path: Path) -> dict[str, list[str]]:
    """Read a TREC run file → {qid: [docid, ...]} ordered by rank."""
    from collections import defaultdict
    raw: dict[str, list] = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid, _, docid, rank, score, _ = parts
            raw[qid].append((int(rank), docid))
    return {qid: [d for _, d in sorted(rows)] for qid, rows in raw.items()}
