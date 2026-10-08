#!/usr/bin/env python3
"""
run_phase5_error_analysis.py
Phase 5: Error Analysis

- Compares BM25 vs Dense baseline.
- Programmatically auto-categorizes 30 failed queries (nDCG@10 = 0).
- Saves failure analysis to outputs/tables/failed_queries_analysis.csv
- Produces gap analysis summary.
"""
import sys, json, os, csv
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "data" / "raw" / "track1_tempo"
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "src"))

from data_utils import load_corpus, load_queries_1a, load_qrels, extract_years
from evaluate import read_trec_run, ndcg_at_k
import re
_TOK = re.compile(r"[A-Za-z0-9]+")
def tokenize(text): return _TOK.findall((text or "").lower())

chosen_path = OUT / "runs" / "chosen_domains.json"
if chosen_path.exists():
    CHOSEN = json.load(open(chosen_path))["chosen_domains"]
else:
    CHOSEN = ["quant", "law", "workplace"]

bm25_res_file = OUT / "runs" / "bm25_official_results.json"
has_bm25 = bm25_res_file.exists()

print("="*80)
print("PHASE 5: ERROR ANALYSIS")
print("="*80)

# 1. Load runs and compute per-query nDCG@10
query_stats = []

for domain in CHOSEN:
    d_dir = DATA_ROOT / domain
    dev_q = load_queries_1a(d_dir, "dev")
    qrels = load_qrels(d_dir, "dev")
    
    # Load steps for multi-step reasoning check
    steps_path = d_dir / "steps_dev.jsonl"
    step_counts = {}
    if steps_path.exists():
        with open(steps_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    step_counts[obj["id"]] = len(obj.get("steps", []))
                    
    doc_ids, doc_texts = load_corpus(d_dir)
    doc_map = dict(zip(doc_ids, doc_texts))
    
    bm25_run = read_trec_run(OUT / "runs" / f"bm25_official_{domain}_dev.trec") if (OUT / "runs" / f"bm25_official_{domain}_dev.trec").exists() else {}
    dense_run = read_trec_run(OUT / "runs" / f"dense_{domain}_dev.trec") if (OUT / "runs" / f"dense_{domain}_dev.trec").exists() else {}
    
    for q in dev_q:
        qid = q["id"]
        gold = qrels.get(qid, set())
        if not gold: continue
        
        bm25_docs = bm25_run.get(qid, [])
        dense_docs = dense_run.get(qid, [])
        
        bm25_ndcg = ndcg_at_k(bm25_docs, gold, 10)
        dense_ndcg = ndcg_at_k(dense_docs, gold, 10)
        
        # Get top retrieved content
        bm25_top = doc_map.get(bm25_docs[0], "") if bm25_docs else ""
        dense_top = doc_map.get(dense_docs[0], "") if dense_docs else ""
        
        gold_list = list(gold)
        gold_top = doc_map.get(gold_list[0], "") if gold_list else ""
        
        query_stats.append({
            "domain": domain,
            "qid": qid,
            "query": q["query"],
            "bm25_ndcg": bm25_ndcg,
            "dense_ndcg": dense_ndcg,
            "gold_docs": len(gold),
            "step_count": step_counts.get(qid, 0),
            "bm25_top1": bm25_top,
            "dense_top1": dense_top,
            "gold_top1": gold_top
        })

df = pd.DataFrame(query_stats)
if df.empty:
    print("No run data found. Run phase 3 & 4 first.")
    sys.exit(0)

# 2. Compare BM25 vs Dense
bm25_beats_dense = df[df.bm25_ndcg > df.dense_ndcg + 0.1]
dense_beats_bm25 = df[df.dense_ndcg > df.bm25_ndcg + 0.1]

print("\n--- BM25 vs Dense Gap Analysis ---")
print(f"Total dev queries evaluated: {len(df)}")
print(f"BM25 substantially beats Dense (delta > 0.1): {len(bm25_beats_dense)} queries")
print(f"Dense substantially beats BM25 (delta > 0.1): {len(dense_beats_bm25)} queries")

print("\nPer-domain breakdown (Average nDCG@10):")
print(df.groupby("domain")[["bm25_ndcg", "dense_ndcg"]].mean().to_string())

# 3. Categorize 30 failed queries (where both failed, or max ndcg is very low)
failed = df[(df.bm25_ndcg == 0) & (df.dense_ndcg == 0)].copy()

# Heuristics for categorization:
def categorize(row):
    q = row["query"]
    g = row["gold_top1"]
    
    q_toks = set(tokenize(q))
    g_toks = set(tokenize(g))
    jac = len(q_toks & g_toks) / (len(q_toks | g_toks) + 1e-6)
    
    q_years = set(extract_years(q))
    g_years = set(extract_years(g))
    
    if row["step_count"] > 2:
        return "Multi-step reasoning needed", f"Query requires {row['step_count']} steps to resolve."
    elif q_years and not (q_years & g_years):
        return "Temporal mismatch", f"Query mentions years {q_years}, gold mentions {g_years}."
    elif jac < 0.05:
        return "Lexical gap", f"Jaccard overlap is very low ({jac:.3f}), exact terms missing."
    else:
        return "Domain effect / Other", "Complex semantic or structural mismatch."

categories = []
reasons = []
for _, row in failed.iterrows():
    c, r = categorize(row)
    categories.append(c)
    reasons.append(r)

failed["category"] = categories
failed["reason"] = reasons

sample_failed = failed.sample(min(30, len(failed)), random_state=42)

out_cols = ["domain", "qid", "query", "category", "reason", "gold_top1", "bm25_top1", "dense_top1"]
sample_failed[out_cols].to_csv(OUT / "tables" / "failed_queries_analysis.csv", index=False)
print(f"\nSaved analysis of {len(sample_failed)} failed queries to outputs/tables/failed_queries_analysis.csv")

print("\nFailure Categories Distribution:")
print(sample_failed["category"].value_counts().to_string())

print("\n" + "="*60)
print("CHECKPOINT 5 — ERROR ANALYSIS COMPLETE")
print("="*60)
