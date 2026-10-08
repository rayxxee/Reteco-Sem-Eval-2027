# RETECO SemEval-2027 – Sub-track 1a Experiment

**Assignment 1 — NLP Shared Task Project**  
SemEval-2027 Task 1: RETECO (Reasoning-Oriented Retrieval)  
Sub-track: **1a** (Temporal Retrieval, whole-query ranking)  
Data version: **v1.1** (deduplicated, released 25 Sep 2026)

---

## Hardware

| Item | Value |
|------|-------|
| OS | Windows 11 |
| CPU | 8 logical cores |
| RAM | 16.9 GB |
| GPU | None (CPU-only) |
| Python | 3.12 (packages) |

> No GPU. All bi-encoder embeddings computed on CPU with reduced batch size and
> truncated max sequence length. See Phase 4 notebook for runtime details.

---

## Setup (exact reproduction steps)

```powershell
# 1. Install packages (Python 3.12)
py -3.12 -m pip install -r requirements.txt

# 2. Download spaCy model
py -3.12 -m spacy download en_core_web_sm

# 3. Download dataset (v1.1, Track-1 only, ~1.5 GB)
py -3.12 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'DataScience-UIBK/RETECO-SemEval2027',
    repo_type='dataset',
    local_dir='data/raw',
    ignore_patterns=['track2_recor/*'],
)
print('Done')
"

# 4. Run all phases (Jupyter nbconvert)
#    Phase 1: Setup & BM25 baseline
py -3.12 -m jupyter nbconvert --to notebook --execute notebooks/01_phase1_setup.ipynb
#    Phase 2: EDA
py -3.12 -m jupyter nbconvert --to notebook --execute notebooks/02_phase2_eda.ipynb
#    Phase 3: Classical baselines
py -3.12 -m jupyter nbconvert --to notebook --execute notebooks/03_phase3_classical.ipynb
#    Phase 4: Dense retrieval
py -3.12 -m jupyter nbconvert --to notebook --execute notebooks/04_phase4_dense.ipynb
#    Phase 5: Error analysis
py -3.12 -m jupyter nbconvert --to notebook --execute notebooks/05_phase5_error_analysis.ipynb
```

---

## Project Layout

```
RETECO semeval 2027/
├── data/
│   ├── raw/                  # Downloaded from HuggingFace (v1.1)
│   │   └── track1_tempo/     # 13 domains, each: documents.jsonl, examples_{train,dev}.jsonl, qrels_{train,dev}.txt
│   └── processed/            # Cached embeddings, feature matrices
├── src/
│   ├── data_utils.py         # Schema-correct loaders (verified against HF card)
│   ├── bm25_retriever.py     # Pure-Python BM25 (k1=0.9, b=0.4), mirrors starter_kit
│   ├── evaluate.py           # nDCG@10 + pytrec_eval cross-check
│   ├── features.py           # Feature engineering for reranker
│   ├── reranker.py           # LightGBM / LogReg reranker
│   └── dense_retriever.py    # Frozen bi-encoder (sentence-transformers)
├── notebooks/
│   ├── 01_phase1_setup.ipynb
│   ├── 02_phase2_eda.ipynb
│   ├── 03_phase3_classical.ipynb
│   ├── 04_phase4_dense.ipynb
│   └── 05_phase5_error_analysis.ipynb
├── outputs/
│   ├── runs/                 # TREC run files + results JSON
│   ├── tables/               # CSV result tables
│   └── figures/              # EDA plots + eda_figures.pdf
├── report/
│   ├── main.tex
│   ├── refs.bib
│   └── sections/             # sec_intro.tex, sec_data.tex, etc.
├── starter_kit/              # Cloned from github.com/DataScienceUIBK/RETECO
├── requirements.txt
├── run_all.sh                # Full reproducible run script
└── README.md
```

---

## Domain Selection (Sub-track 1a)

We use **3 domains** chosen to be the smallest by corpus size (enabling
fast CPU-only experimentation) while each having ≥30 dev queries for
reliable nDCG@10 estimation. Exact domains confirmed from data after download.

---

## Results Summary

See `outputs/tables/` for full results. Key numbers:

| Method | Overall nDCG@10 (dev) |
|--------|----------------------|
| BM25 (official baseline, all 13 domains) | 0.1147 |
| BM25 (our 3 domains) | TBD after download |
| TF-IDF cosine | TBD |
| BM25 + LightGBM reranker | TBD |
| Frozen bi-encoder (sentence-transformers) | TBD |

---

## Random Seeds

All experiments fix `SEED = 42`. Documented in each notebook cell.

---

## Citation

```bibtex
@inproceedings{abdallah2027reteco,
  title  = {{RETECO}: Reasoning-Oriented Retrieval -- {SemEval}-2027 Task 1},
  author = {Abdallah, Abdelrahman and Ali, Mohammed and Abdul-Mageed, Muhammad
            and Duh, Kevin and Jatowt, Adam},
  booktitle = {Proceedings of the 21st International Workshop on Semantic Evaluation},
  year   = {2027},
}
```
