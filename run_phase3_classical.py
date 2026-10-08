#!/usr/bin/env python3
"""
run_phase3_classical.py
Phase 3: Classical baselines — BM25, TF-IDF cosine, LightGBM reranker.

Produces:
  - outputs/tables/phase3_results.csv
  - outputs/tables/feature_importance.csv
  - outputs/runs/tfidf_{domain}_{split}.trec
  - outputs/runs/reranker_{domain}_dev.trec
  - outputs/models/lgbm_{domain}.pkl

Usage:
  py -3.12 run_phase3_classical.py
"""
import sys, json, time, csv, os, pickle, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import lightgbm as lgb

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "data" / "raw" / "track1_tempo"
OUT = ROOT / "outputs"
(OUT / "runs").mkdir(parents=True, exist_ok=True)
(OUT / "tables").mkdir(exist_ok=True)
(OUT / "models").mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "starter_kit" / "starter_kit"))
from data_utils import (load_corpus, load_queries_1a, load_qrels,
                         extract_years, extract_dates, has_temporal_cue)
from evaluate import evaluate_run, pytrec_evaluate, write_trec_run
import bm25 as official_bm25

import re
_TOK = re.compile(r"[A-Za-z0-9]+")
def tokenize(text):
    return _TOK.findall((text or "").lower())

# ── Load chosen domains ──────────────────────────────────────────────────────
chosen_path = OUT / "runs" / "chosen_domains.json"
if chosen_path.exists():
    CHOSEN = json.load(open(chosen_path))["chosen_domains"]
else:
    CHOSEN = ["quant", "law", "workplace"]
print(f"Phase 3 domains: {CHOSEN}")

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    "bm25_score", "bm25_score_norm",
    "tfidf_cosine",
    "term_overlap_jaccard",
    "doc_len_log", "query_len_log",
    "rank_recip",
    "year_overlap", "year_overlap_frac",
    "query_has_year", "doc_has_year",
    "query_has_date", "doc_has_date",
]

def build_features(query_text, bm25_results, doc_map, tfidf_scores=None):
    """Build feature matrix for one query's BM25 candidates."""
    q_toks = set(tokenize(query_text))
    q_years = set(extract_years(query_text))
    q_has_year = 1.0 if q_years else 0.0
    q_has_date = 1.0 if extract_dates(query_text) else 0.0
    q_len = np.log1p(len(q_toks))

    max_bm25 = max((s for _, s in bm25_results), default=1.0)
    if max_bm25 == 0:
        max_bm25 = 1.0

    rows = []
    doc_ids_out = []
    for rank, (did, bm25_score) in enumerate(bm25_results, 1):
        doc_text = doc_map.get(did, "")
        d_toks = set(tokenize(doc_text))
        d_years = set(extract_years(doc_text))

        jac = len(q_toks & d_toks) / len(q_toks | d_toks) if (q_toks | d_toks) else 0
        yr_overlap = len(q_years & d_years)
        yr_frac = yr_overlap / (len(q_years) + 1e-6)

        tfidf_cos = tfidf_scores.get(did, 0.0) if tfidf_scores else 0.0

        row = [
            bm25_score,
            bm25_score / max_bm25,
            tfidf_cos,
            jac,
            np.log1p(len(d_toks)),
            q_len,
            1.0 / rank,
            float(yr_overlap),
            yr_frac,
            q_has_year,
            1.0 if d_years else 0.0,
            q_has_date,
            1.0 if extract_dates(doc_text) else 0.0,
        ]
        rows.append(row)
        doc_ids_out.append(did)

    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32), doc_ids_out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

results_all = {}  # method -> domain -> split -> ndcg@10

for domain in CHOSEN:
    print(f"\n{'='*60}")
    print(f"DOMAIN: {domain}")
    print(f"{'='*60}")

    d_dir = DATA_ROOT / domain
    t0 = time.time()

    # Load corpus
    doc_ids, doc_texts = load_corpus(d_dir)
    doc_map = dict(zip(doc_ids, doc_texts))
    print(f"  Corpus: {len(doc_ids):,} docs loaded in {time.time()-t0:.1f}s")

    # ── BM25 ──────────────────────────────────────────────────────────────────
    print("  Building BM25 index...")
    bm25 = official_bm25.BM25(doc_ids, doc_texts, k1=0.9, b=0.4)

    # ── TF-IDF ────────────────────────────────────────────────────────────────
    print("  Building TF-IDF index...")
    # Use pre-tokenized strings for TF-IDF
    tfidf_vec = TfidfVectorizer(
        analyzer=lambda x: tokenize(x),
        max_features=100_000,
        sublinear_tf=True,
    )
    tfidf_matrix = tfidf_vec.fit_transform(doc_texts)
    print(f"  TF-IDF matrix: {tfidf_matrix.shape}")

    # ── Run BM25 + TF-IDF on train and dev ────────────────────────────────────
    bm25_runs = {}
    tfidf_runs = {}

    for split in ("train", "dev"):
        q_recs = load_queries_1a(d_dir, split)
        queries = [(q["id"], q["query"]) for q in q_recs]
        qrels = load_qrels(d_dir, split)

        # BM25 retrieval
        bm25_run = {}
        for qid, qtxt in queries:
            ranked = bm25.search(qtxt, top_k=100)
            bm25_run[qid] = ranked
        bm25_runs[split] = bm25_run

        write_trec_run(OUT / "runs" / f"bm25_official_{domain}_{split}.trec", bm25_run, tag="bm25_official")

        run_lists_bm25 = {qid: [d for d, _ in r] for qid, r in bm25_run.items()}
        res_bm25 = evaluate_run(run_lists_bm25, qrels)
        results_all.setdefault("BM25", {}).setdefault(domain, {})[split] = res_bm25["ndcg@10"]
        print(f"  BM25 {split}: nDCG@10={res_bm25['ndcg@10']:.4f}")

        # TF-IDF retrieval (cosine similarity)
        tfidf_run = {}
        for qid, qtxt in queries:
            q_vec = tfidf_vec.transform([qtxt])
            sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
            top_k = min(100, len(sims))
            top_idx = np.argpartition(sims, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
            tfidf_run[qid] = [(doc_ids[i], float(sims[i])) for i in top_idx]
        tfidf_runs[split] = tfidf_run

        write_trec_run(OUT / "runs" / f"tfidf_{domain}_{split}.trec", tfidf_run, tag="tfidf")
        run_lists_tfidf = {qid: [d for d, _ in r] for qid, r in tfidf_run.items()}
        res_tfidf = evaluate_run(run_lists_tfidf, qrels)
        results_all.setdefault("TF-IDF", {}).setdefault(domain, {})[split] = res_tfidf["ndcg@10"]
        print(f"  TF-IDF {split}: nDCG@10={res_tfidf['ndcg@10']:.4f}")

    # ── Precompute TF-IDF scores for BM25 top-50 candidates ──────────────────
    def get_tfidf_scores(query_text, bm25_candidates):
        """Get TF-IDF cosine similarity for specific doc candidates."""
        q_vec = tfidf_vec.transform([query_text])
        scores = {}
        for did, _ in bm25_candidates:
            idx = doc_ids.index(did) if did in doc_ids else -1
            if idx >= 0:
                scores[did] = float(cosine_similarity(q_vec, tfidf_matrix[idx:idx+1]).flatten()[0])
            else:
                scores[did] = 0.0
        return scores

    # Build doc_id -> index mapping for fast lookup
    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}

    def get_tfidf_scores_fast(query_text, bm25_candidates):
        q_vec = tfidf_vec.transform([query_text])
        scores = {}
        for did, _ in bm25_candidates:
            idx = doc_id_to_idx.get(did, -1)
            if idx >= 0:
                scores[did] = float(cosine_similarity(q_vec, tfidf_matrix[idx:idx+1]).flatten()[0])
            else:
                scores[did] = 0.0
        return scores

    # ── LightGBM Reranker — Train on TRAIN split ──────────────────────────────
    print("  Building reranker training data (BM25 top-50)...")
    train_q = load_queries_1a(d_dir, "train")
    train_qrels = load_qrels(d_dir, "train")

    X_train, y_train, qids_train = [], [], []
    for q in train_q:
        qid = q["id"]
        bm25_cands = bm25_runs["train"].get(qid, [])[:50]
        if not bm25_cands:
            continue
        gold = train_qrels.get(qid, set())
        tfidf_sc = get_tfidf_scores_fast(q["query"], bm25_cands)
        feats, did_list = build_features(q["query"], bm25_cands, doc_map, tfidf_sc)
        labels = np.array([1.0 if did in gold else 0.0 for did in did_list])
        X_train.append(feats)
        y_train.append(labels)
        qids_train.extend([qid] * len(did_list))

    X_train = np.vstack(X_train) if X_train else np.zeros((0, len(FEATURE_NAMES)))
    y_train = np.concatenate(y_train) if y_train else np.array([])
    print(f"  Training data: {X_train.shape[0]} pairs, {y_train.sum():.0f} positive")

    # Train LightGBM
    print("  Training LightGBM reranker...")
    lgbm_model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        force_col_wise=True,
    )
    if len(X_train) > 0 and y_train.sum() > 0:
        lgbm_model.fit(X_train, y_train)
        # Save model
        with open(OUT / "models" / f"lgbm_{domain}.pkl", "wb") as f:
            pickle.dump(lgbm_model, f)

        # Feature importance
        importances = dict(zip(FEATURE_NAMES, lgbm_model.feature_importances_))
        imp_sorted = sorted(importances.items(), key=lambda x: -x[1])
        print("  Feature importance (top 5):")
        for fname, imp in imp_sorted[:5]:
            print(f"    {fname}: {imp}")
    else:
        print("  WARNING: No positive training pairs, skipping reranker training")
        imp_sorted = [(f, 0) for f in FEATURE_NAMES]

    # ── Evaluate reranker on DEV ──────────────────────────────────────────────
    print("  Reranking BM25 top-50 on dev...")
    dev_q = load_queries_1a(d_dir, "dev")
    dev_qrels = load_qrels(d_dir, "dev")

    reranked_run = {}
    for q in dev_q:
        qid = q["id"]
        bm25_cands = bm25_runs["dev"].get(qid, [])[:50]
        if not bm25_cands:
            reranked_run[qid] = []
            continue
        tfidf_sc = get_tfidf_scores_fast(q["query"], bm25_cands)
        feats, did_list = build_features(q["query"], bm25_cands, doc_map, tfidf_sc)
        if len(feats) > 0 and y_train.sum() > 0:
            probs = lgbm_model.predict_proba(feats)[:, 1]
            ranked = sorted(zip(did_list, probs), key=lambda x: -x[1])
            reranked_run[qid] = [(d, float(p)) for d, p in ranked]
        else:
            reranked_run[qid] = bm25_cands

    write_trec_run(OUT / "runs" / f"reranker_{domain}_dev.trec", reranked_run, tag="lgbm_rerank")
    run_lists_rerank = {qid: [d for d, _ in r] for qid, r in reranked_run.items()}
    res_rerank = evaluate_run(run_lists_rerank, dev_qrels)
    results_all.setdefault("LightGBM-Reranker", {}).setdefault(domain, {})["dev"] = res_rerank["ndcg@10"]
    print(f"  LightGBM reranker dev: nDCG@10={res_rerank['ndcg@10']:.4f}")

    # Save feature importance
    with open(OUT / "tables" / f"feature_importance_{domain}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "importance"])
        for fname, imp in imp_sorted:
            w.writerow([fname, imp])

    # Free memory
    del tfidf_matrix, tfidf_vec, bm25, doc_ids, doc_texts, doc_map

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("PHASE 3 RESULTS — nDCG@10")
print("="*80)

header = f"{'Method':<25} " + " ".join(f"{d:>12}" for d in CHOSEN) + f" {'Macro':>12}"
print(header)
print("-" * len(header))

table_rows = []
for method in ["BM25", "TF-IDF", "LightGBM-Reranker"]:
    vals = []
    for d in CHOSEN:
        v = results_all.get(method, {}).get(d, {}).get("dev", -1)
        vals.append(v)
    macro = np.mean([v for v in vals if v >= 0])
    row = f"{method:<25} " + " ".join(f"{v:>12.4f}" if v >= 0 else f"{'N/A':>12}" for v in vals) + f" {macro:>12.4f}"
    print(row)
    table_rows.append({"method": method, **{d: vals[i] for i, d in enumerate(CHOSEN)}, "macro": macro})

# Save results CSV
df_results = pd.DataFrame(table_rows)
df_results.to_csv(OUT / "tables" / "phase3_results.csv", index=False)

# Save full results JSON
json.dump(results_all, open(OUT / "runs" / "phase3_results.json", "w"), indent=2)

print(f"\nResults saved to {OUT / 'tables' / 'phase3_results.csv'}")

print("\n" + "="*60)
print("CHECKPOINT 3 — PHASE 3 COMPLETE")
print("="*60)
