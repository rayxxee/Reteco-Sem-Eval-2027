#!/usr/bin/env python3
"""
run_phase1.py
Phase 1: data summary, BM25 baseline reproduction, domain selection.

Uses the EXACT same pure-Python BM25 as the starter kit (bm25.py) to
reproduce official baseline numbers. Also runs our rank_bm25 variant
and cross-checks with pytrec_eval.

Usage:
  py -3.12 run_phase1.py
"""
import sys, json, time, csv, os, importlib
from pathlib import Path
from collections import defaultdict

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "data" / "raw"
OUT_ROOT  = ROOT / "outputs"
(OUT_ROOT / "runs").mkdir(parents=True, exist_ok=True)
(OUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "starter_kit" / "starter_kit"))

from data_utils import load_corpus, load_queries_1a, load_qrels, TRACK1_DOMAINS
from evaluate import evaluate_run, pytrec_evaluate, write_trec_run

# Import OFFICIAL starter kit BM25
import bm25 as official_bm25

# Also import our rank_bm25 wrapper
from bm25_retriever import BM25Retriever

import numpy as np
np.random.seed(42)

# ── 1. Discover domains & count everything ────────────────────────────────────
track1_root = DATA_ROOT / "track1_tempo"
domains_found = sorted([d.name for d in track1_root.iterdir() if d.is_dir()])
print(f"Track-1 domains found ({len(domains_found)}): {domains_found}")

print("\n" + "="*80)
print("DOMAIN STATISTICS")
print("="*80)
header = f"{'Domain':<15} {'Docs':>8} {'Train Q':>8} {'Dev Q':>8} {'Train qrels':>12} {'Dev qrels':>12} {'Corpus MB':>10}"
print(header)
print("-"*80)

domain_stats = []
for domain in domains_found:
    d = track1_root / domain
    n_docs  = sum(1 for l in open(d/"documents.jsonl", encoding="utf-8") if l.strip())
    n_train = sum(1 for l in open(d/"examples_train.jsonl", encoding="utf-8") if l.strip())
    n_dev   = sum(1 for l in open(d/"examples_dev.jsonl", encoding="utf-8") if l.strip())
    n_qrels_train = sum(1 for l in open(d/"qrels_train.txt", encoding="utf-8") if l.strip())
    n_qrels_dev   = sum(1 for l in open(d/"qrels_dev.txt", encoding="utf-8") if l.strip())
    mb = os.path.getsize(d / "documents.jsonl") / 1e6
    domain_stats.append({
        "domain": domain, "n_docs": n_docs,
        "n_train_q": n_train, "n_dev_q": n_dev,
        "n_qrels_train": n_qrels_train, "n_qrels_dev": n_qrels_dev,
        "corpus_mb": round(mb, 1),
    })
    print(f"{domain:<15} {n_docs:>8,} {n_train:>8} {n_dev:>8} {n_qrels_train:>12} {n_qrels_dev:>12} {mb:>10.1f}")

# Save stats CSV
with open(OUT_ROOT / "tables" / "domain_counts.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=domain_stats[0].keys())
    w.writeheader()
    w.writerows(domain_stats)

total_docs = sum(r["n_docs"] for r in domain_stats)
total_train = sum(r["n_train_q"] for r in domain_stats)
total_dev = sum(r["n_dev_q"] for r in domain_stats)
print(f"\nTotal: {total_docs:,} docs | {total_train} train queries | {total_dev} dev queries")

# ── 2. Record format examples ────────────────────────────────────────────────
print("\n" + "="*80)
print("RECORD FORMAT VERIFICATION")
print("="*80)

sample_domain = "law"
d = track1_root / sample_domain
print(f"\nUsing domain '{sample_domain}' for format examples:")

# Documents
with open(d / "documents.jsonl", encoding="utf-8") as fh:
    doc = json.loads(fh.readline())
print(f"\ndocuments.jsonl keys: {list(doc.keys())}")
print(f"  id: {doc['id'][:60]}...")
print(f"  content: {doc['content'][:100]}...")

# Examples (queries)
with open(d / "examples_dev.jsonl", encoding="utf-8") as fh:
    ex = json.loads(fh.readline())
print(f"\nexamples_dev.jsonl keys: {list(ex.keys())}")
print(f"  id: {ex['id']}")
print(f"  query: {ex['query'][:100]}...")
if "gold_ids" in ex:
    print(f"  gold_ids: {ex['gold_ids'][:3]}...")
if "gold_answers" in ex:
    print(f"  gold_answers: {str(ex['gold_answers'])[:100]}...")

# Qrels
with open(d / "qrels_dev.txt", encoding="utf-8") as fh:
    line = fh.readline().strip()
print(f"\nqrels_dev.txt format: {line}")
print(f"  (columns: qid  iter  docid  rel)")

# Steps
with open(d / "steps_dev.jsonl", encoding="utf-8") as fh:
    step_rec = json.loads(fh.readline())
print(f"\nsteps_dev.jsonl keys: {list(step_rec.keys())}")
if "steps" in step_rec:
    print(f"  steps[0] keys: {list(step_rec['steps'][0].keys())}")

# Guidance
with open(d / "guidance_dev.jsonl", encoding="utf-8") as fh:
    guid = json.loads(fh.readline())
print(f"\nguidance_dev.jsonl keys: {list(guid.keys())}")

# ── 3. BM25 baseline — OFFICIAL implementation (all 13 domains) ──────────────
print("\n" + "="*80)
print("BM25 BASELINE (official starter-kit implementation, k1=0.9, b=0.4)")
print("="*80)

# Official reference numbers from BASELINE_RESULTS.md
OFFICIAL_REF = {
    "bitcoin":   {"train": 0.0856, "dev": 0.0370},
    "cardano":   {"train": 0.1848, "dev": 0.1027},
    "economics": {"train": 0.0382, "dev": 0.0467},
    "genealogy": {"train": 0.1242, "dev": 0.1949},
    "history":   {"train": 0.0654, "dev": 0.0873},
    "hsm":       {"train": 0.1736, "dev": 0.2489},
    "iota":      {"train": 0.1563, "dev": 0.3309},
    "law":       {"train": 0.0946, "dev": 0.0574},
    "monero":    {"train": 0.0382, "dev": 0.0320},
    "politics":  {"train": 0.2810, "dev": 0.2625},
    "quant":     {"train": 0.0255, "dev": 0.0237},
    "travel":    {"train": 0.0506, "dev": 0.0442},
    "workplace": {"train": 0.0798, "dev": 0.0230},
}

results_official = {}  # domain -> split -> {ndcg, pytrec, num_topics}
results_rankbm25 = {}  # our rank_bm25 variant

for domain in domains_found:
    d_dir = track1_root / domain
    print(f"\n{'='*60}")
    print(f"DOMAIN: {domain}")
    print(f"{'='*60}")
    
    t0 = time.time()
    
    # Load corpus
    doc_ids, doc_texts = load_corpus(d_dir)
    print(f"  Corpus: {len(doc_ids):,} docs loaded in {time.time()-t0:.1f}s")
    
    # Build OFFICIAL BM25 index
    t1 = time.time()
    bm25_off = official_bm25.BM25(doc_ids, doc_texts, k1=0.9, b=0.4)
    print(f"  Official BM25 index built in {time.time()-t1:.1f}s")
    
    # Build rank_bm25 index
    t2 = time.time()
    bm25_ours = BM25Retriever(doc_ids, doc_texts, k1=0.9, b=0.4)
    print(f"  rank_bm25 index built in {time.time()-t2:.1f}s")
    
    results_official[domain] = {}
    results_rankbm25[domain] = {}
    
    for split in ("train", "dev"):
        q_recs = load_queries_1a(d_dir, split)
        queries = [(q["id"], q["query"]) for q in q_recs]
        qrels = load_qrels(d_dir, split)
        
        # --- OFFICIAL BM25 ---
        run_off = {}
        for qid, qtxt in queries:
            ranked = bm25_off.search(qtxt, top_k=100)
            run_off[qid] = ranked  # [(docid, score), ...]
        
        # Save TREC run
        write_trec_run(
            OUT_ROOT / "runs" / f"bm25_official_{domain}_{split}.trec",
            run_off, tag="bm25_official",
        )
        
        # Evaluate
        run_lists_off = {qid: [d for d, _ in ranked] for qid, ranked in run_off.items()}
        res_off = evaluate_run(run_lists_off, qrels)
        res_pt_off = pytrec_evaluate(run_lists_off, qrels)
        
        ref = OFFICIAL_REF.get(domain, {}).get(split, -1)
        diff = abs(res_off["ndcg@10"] - ref) if ref >= 0 else -1
        flag = "OK MATCH" if diff < 0.001 else f"diff={diff:.4f}"
        
        results_official[domain][split] = {
            "ndcg@10": res_off["ndcg@10"],
            "ndcg@10_pytrec": res_pt_off.get("ndcg@10", -1),
            "reference": ref,
            "match": flag,
            "num_topics": res_off["num_topics"],
        }
        
        print(f"  Official BM25 {split:5s}: nDCG@10={res_off['ndcg@10']:.4f}  "
              f"pytrec={res_pt_off.get('ndcg@10', -1):.4f}  "
              f"ref={ref:.4f}  {flag}  [{res_off['num_topics']} q]")
        
        # --- RANK_BM25 (our implementation) ---
        run_ours = bm25_ours.batch_search(queries, top_k=100)
        write_trec_run(
            OUT_ROOT / "runs" / f"bm25_rankbm25_{domain}_{split}.trec",
            run_ours, tag="bm25_rankbm25",
        )
        run_lists_ours = {qid: [d for d, _ in ranked] for qid, ranked in run_ours.items()}
        res_ours = evaluate_run(run_lists_ours, qrels)
        
        results_rankbm25[domain][split] = {
            "ndcg@10": res_ours["ndcg@10"],
            "num_topics": res_ours["num_topics"],
        }
        print(f"  rank_bm25    {split:5s}: nDCG@10={res_ours['ndcg@10']:.4f}")
    
    # Free memory
    del bm25_off, bm25_ours, doc_ids, doc_texts

# ── 4. Macro-averages ─────────────────────────────────────────────────────────
print("\n" + "="*80)
print("MACRO-AVERAGE nDCG@10 (over all 13 domains)")
print("="*80)

ref_macro = {"train": 0.1075, "dev": 0.1147}
for split in ("train", "dev"):
    macro_off = sum(results_official[d][split]["ndcg@10"] for d in domains_found) / len(domains_found)
    macro_ours = sum(results_rankbm25[d][split]["ndcg@10"] for d in domains_found) / len(domains_found)
    ref = ref_macro[split]
    diff = abs(macro_off - ref)
    flag = "PASS OK" if diff < 0.005 else f"MISMATCH (diff={diff:.4f})"
    print(f"  Official BM25 {split:5s}: {macro_off:.4f}  (reference: {ref})  {flag}")
    print(f"  rank_bm25     {split:5s}: {macro_ours:.4f}")

# ── 5. Domain selection ───────────────────────────────────────────────────────
print("\n" + "="*80)
print("DOMAIN SELECTION")
print("="*80)

# Strategy: pick 3 smallest domains with ≥10 dev queries (CPU feasibility).
# This gives quant (25K), law (36K), workplace (43K).
candidates = sorted(domain_stats, key=lambda x: x["n_docs"])
eligible = [c for c in candidates if c["n_dev_q"] >= 10]

print("\nAll domains sorted by corpus size:")
for c in candidates:
    elig = "OK" if c["n_dev_q"] >= 10 else "X (<10 dev q)"
    dev_ndcg = results_official[c["domain"]]["dev"]["ndcg@10"]
    print(f"  {c['domain']:<15} {c['n_docs']:>8,} docs  {c['n_dev_q']:>3} dev q  "
          f"BM25 dev={dev_ndcg:.4f}  {elig}")

CHOSEN = [c["domain"] for c in eligible[:3]]
print(f"\nCHOSEN DOMAINS: {CHOSEN}")
print("Justification: 3 smallest corpora with ≥10 dev queries → fast CPU indexing")
print("  + dense retrieval embedding feasible without GPU")

# ── 6. Save all results ───────────────────────────────────────────────────────
json.dump(results_official, open(OUT_ROOT / "runs" / "bm25_official_results.json", "w"), indent=2)
json.dump(results_rankbm25, open(OUT_ROOT / "runs" / "bm25_rankbm25_results.json", "w"), indent=2)
json.dump(
    {"chosen_domains": CHOSEN,
     "criterion": "3 smallest corpora with ≥10 dev queries",
     "domain_stats": {c["domain"]: c for c in eligible[:3]}},
    open(OUT_ROOT / "runs" / "chosen_domains.json", "w"),
    indent=2,
)

# Per-domain results table CSV
with open(OUT_ROOT / "tables" / "bm25_all_domains.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["domain", "n_docs", "n_dev_q",
                "bm25_official_train", "bm25_official_dev", "bm25_official_dev_ref",
                "bm25_rankbm25_train", "bm25_rankbm25_dev", "match"])
    for d in domains_found:
        s = next(x for x in domain_stats if x["domain"] == d)
        w.writerow([
            d, s["n_docs"], s["n_dev_q"],
            f"{results_official[d]['train']['ndcg@10']:.4f}",
            f"{results_official[d]['dev']['ndcg@10']:.4f}",
            f"{OFFICIAL_REF[d]['dev']:.4f}",
            f"{results_rankbm25[d]['train']['ndcg@10']:.4f}",
            f"{results_rankbm25[d]['dev']['ndcg@10']:.4f}",
            results_official[d]["dev"]["match"],
        ])

# ── CHECKPOINT 1 SUMMARY ─────────────────────────────────────────────────────
print("\n" + "="*80)
print("CHECKPOINT 1 — PHASE 1 COMPLETE")
print("="*80)
print(f"Data version      : v1.1 (deduplicated, 25 Sep 2026)")
print(f"Domains found     : {len(domains_found)}")
print(f"Total corpus docs : {total_docs:,}")
print(f"Total train queries: {total_train}")
print(f"Total dev queries  : {total_dev}")
print()
print("BM25 macro-average nDCG@10 (official implementation):")
for split in ("train", "dev"):
    macro = sum(results_official[d][split]["ndcg@10"] for d in domains_found) / len(domains_found)
    ref = ref_macro[split]
    diff = abs(macro - ref)
    flag = "PASS OK" if diff < 0.005 else f"MISMATCH (diff={diff:.4f})"
    print(f"  {split:5s}: {macro:.4f}  (reference: {ref})  {flag}")
print()
print("pytrec_eval cross-check:")
for split in ("train", "dev"):
    diffs = []
    for d in domains_found:
        ours = results_official[d][split]["ndcg@10"]
        pt = results_official[d][split]["ndcg@10_pytrec"]
        if pt >= 0:
            diffs.append(abs(ours - pt))
    max_diff = max(diffs) if diffs else -1
    print(f"  {split}: max |official - pytrec| = {max_diff:.6f}")
print()
print(f"CHOSEN DOMAINS: {CHOSEN}")
for d in CHOSEN:
    s = next(x for x in domain_stats if x["domain"] == d)
    dev_ndcg = results_official[d]["dev"]["ndcg@10"]
    print(f"  {d:<15} {s['n_docs']:>8,} docs  {s['n_dev_q']:>3} dev queries  BM25 dev nDCG@10={dev_ndcg:.4f}")
print()
print("Hardware: CPU-only (no GPU), 16.9 GB RAM, 8 cores")
print("Java: not installed → using starter-kit pure-Python BM25")
print(f"All outputs in: {OUT_ROOT}")
