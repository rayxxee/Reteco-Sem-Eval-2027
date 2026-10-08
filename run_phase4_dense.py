#!/usr/bin/env python3
"""
run_phase4_dense.py
Phase 4: Pretrained Baseline (Frozen Dense Bi-Encoder)

Uses BAAI/bge-small-en-v1.5 in frozen mode (no fine-tuning).
Embeddings are cached to disk to save compute.
Computes dense baseline and a simple hybrid (RRF of BM25 + Dense).
"""
import sys
import json
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "data" / "raw" / "track1_tempo"
OUT = ROOT / "outputs"
(OUT / "runs").mkdir(parents=True, exist_ok=True)
(OUT / "tables").mkdir(exist_ok=True)
CACHE_DIR = OUT / "embeddings_cache"
CACHE_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))
from data_utils import load_corpus, load_queries_1a, load_qrels
from evaluate import evaluate_run, write_trec_run, read_trec_run
from dense_retriever import FrozenBiEncoder

# ── 1. Load configuration ─────────────────────────────────────────────────────
chosen_path = OUT / "runs" / "chosen_domains.json"
if chosen_path.exists():
    CHOSEN = json.load(open(chosen_path))["chosen_domains"]
else:
    CHOSEN = ["quant", "law", "workplace"]

print(f"Phase 4 domains: {CHOSEN}")

# Load BM25 results from Phase 3 for Hybrid Fusion
bm25_results_path = OUT / "runs" / "phase3_results.json"
if bm25_results_path.exists():
    bm25_res = json.load(open(bm25_results_path)).get("BM25", {})
else:
    bm25_res = {}

# We also need the actual runs for RRF. Let's load the TREC files.
def load_bm25_run(domain, split):
    p = OUT / "runs" / f"bm25_official_{domain}_{split}.trec"
    if p.exists():
        # read_trec_run returns {qid: [docid, ...]} but we need ranks
        # Actually read_trec_run returns them ordered by rank, so index is rank-1
        run_lists = read_trec_run(p)
        return {qid: {d: i+1 for i, d in enumerate(docs)} for qid, docs in run_lists.items()}
    return {}

# ── 2. Run Dense Retrieval ────────────────────────────────────────────────────
encoder = FrozenBiEncoder()
dense_res = {}
hybrid_res = {}

for domain in CHOSEN:
    print(f"\n{'='*60}")
    print(f"DOMAIN: {domain}")
    print(f"{'='*60}")
    
    d_dir = DATA_ROOT / domain
    
    t0 = time.time()
    doc_ids, doc_texts = load_corpus(d_dir)
    print(f"  Corpus: {len(doc_ids):,} docs loaded in {time.time()-t0:.1f}s")
    
    # Encode/load corpus
    corpus_embs = encoder.encode_corpus_cached(doc_ids, doc_texts, domain, CACHE_DIR)
    
    dense_res[domain] = {}
    hybrid_res[domain] = {}
    
    for split in ["train", "dev"]:
        queries = load_queries_1a(d_dir, split)
        qrels = load_qrels(d_dir, split)
        
        q_list = [(q["id"], q["query"]) for q in queries]
        
        # Dense search
        print(f"  Running dense search for {split} ({len(q_list)} queries)...")
        results_dense = encoder.batch_search(q_list, corpus_embs, doc_ids, top_k=100)
        
        # Save dense run
        write_trec_run(OUT / "runs" / f"dense_{domain}_{split}.trec", results_dense, tag="dense_bge_small")
        
        run_lists_dense = {qid: [d for d, _ in r] for qid, r in results_dense.items()}
        res = evaluate_run(run_lists_dense, qrels)
        dense_res[domain][split] = res["ndcg@10"]
        print(f"  Dense {split} nDCG@10: {res['ndcg@10']:.4f}")
        
        # Hybrid search (RRF)
        bm25_run_dict = load_bm25_run(domain, split)
        if bm25_run_dict:
            k_rrf = 60
            hybrid_run = {}
            for qid in results_dense.keys():
                dense_ranked = run_lists_dense.get(qid, [])
                bm25_ranks = bm25_run_dict.get(qid, {})
                
                scores = {}
                for i, d in enumerate(dense_ranked):
                    scores[d] = scores.get(d, 0) + 1.0 / (k_rrf + i + 1)
                for d, rank in bm25_ranks.items():
                    scores[d] = scores.get(d, 0) + 1.0 / (k_rrf + rank)
                    
                sorted_docs = sorted(scores.items(), key=lambda x: -x[1])
                hybrid_run[qid] = sorted_docs[:100]
                
            write_trec_run(OUT / "runs" / f"hybrid_{domain}_{split}.trec", hybrid_run, tag="rrf_bm25_dense")
            run_lists_hybrid = {qid: [d for d, _ in r] for qid, r in hybrid_run.items()}
            res_hyb = evaluate_run(run_lists_hybrid, qrels)
            hybrid_res[domain][split] = res_hyb["ndcg@10"]
            print(f"  Hybrid {split} nDCG@10: {res_hyb['ndcg@10']:.4f}")

# ── 3. Consolidated Results Table ─────────────────────────────────────────────
print("\n" + "="*80)
print("CONSOLIDATED RESULTS (nDCG@10 on DEV)")
print("="*80)

# Load Phase 3 results if available
phase3_path = OUT / "tables" / "phase3_results.csv"
if phase3_path.exists():
    df_p3 = pd.read_csv(phase3_path)
else:
    df_p3 = pd.DataFrame()

methods = []
results_table = []

# Collect BM25
row_bm25 = {"Method": "BM25"}
for d in CHOSEN:
    v = bm25_res.get(d, {}).get("dev", {}).get("ndcg@10", -1)
    row_bm25[d] = v
row_bm25["Macro"] = np.mean([row_bm25[d] for d in CHOSEN if row_bm25[d] >= 0])
results_table.append(row_bm25)

# Collect TF-IDF & LightGBM from Phase 3 if they exist
if not df_p3.empty:
    for _, r in df_p3.iterrows():
        if r["method"] == "BM25": continue # already got it
        row = {"Method": r["method"]}
        for d in CHOSEN:
            row[d] = r[d]
        row["Macro"] = r["macro"]
        results_table.append(row)

# Collect Dense
row_dense = {"Method": "Dense (bge-small frozen)"}
for d in CHOSEN:
    row_dense[d] = dense_res.get(d, {}).get("dev", -1)
row_dense["Macro"] = np.mean([row_dense[d] for d in CHOSEN if row_dense[d] >= 0])
results_table.append(row_dense)

# Collect Hybrid
row_hyb = {"Method": "Hybrid (RRF BM25+Dense)"}
for d in CHOSEN:
    row_hyb[d] = hybrid_res.get(d, {}).get("dev", -1)
row_hyb["Macro"] = np.mean([row_hyb[d] for d in CHOSEN if row_hyb[d] >= 0])
results_table.append(row_hyb)

df_all = pd.DataFrame(results_table)
print(df_all.to_string(index=False))

df_all.to_csv(OUT / "tables" / "consolidated_results.csv", index=False)
print(f"\nSaved consolidated results to {OUT / 'tables' / 'consolidated_results.csv'}")

print("\n" + "="*60)
print("CHECKPOINT 4 — PHASE 4 COMPLETE")
print("="*60)
