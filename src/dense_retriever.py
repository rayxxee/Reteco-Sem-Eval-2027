#!/usr/bin/env python3
"""
src/dense_retriever.py
Frozen bi-encoder dense retrieval (Phase 4).

Model: BAAI/bge-small-en-v1.5  (33M params, 512 max tokens, CPU-friendly)
Representations: frozen (no fine-tuning). Cosine similarity.

Hardware adaptation (no GPU):
  - batch_size = 32
  - max_seq_length = 256 (truncated from default 512 for speed)
  - Embeddings cached to disk (numpy .npy files)

State clearly: FROZEN representations / feature extraction, NO fine-tuning.
"""
import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np


MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 32          # CPU-safe
MAX_SEQ_LEN = 256        # Truncated for speed (documented in report)
EMBED_DIM = 384          # bge-small embedding dimension


def _get_cache_path(cache_dir: Path, domain: str, split: str) -> Path:
    """Return path to cached embedding file."""
    return cache_dir / f"{domain}_{split}_embeddings.npy"


def _get_meta_path(cache_dir: Path, domain: str, split: str) -> Path:
    return cache_dir / f"{domain}_{split}_meta.json"


class FrozenBiEncoder:
    """
    Sentence-transformer bi-encoder in frozen (inference-only) mode.
    No gradient computation, no weight updates.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        batch_size: int = BATCH_SIZE,
        max_seq_length: int = MAX_SEQ_LEN,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self._model = None
        self.query_instruction = "Represent this sentence for searching relevant passages: "

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(self.model_name, device=device)
            self._model.max_seq_length = self.max_seq_length
            # Ensure frozen: no gradients
            for param in self._model.parameters():
                param.requires_grad = False
            print(f"Loaded {self.model_name} on {device.upper()} (frozen, max_seq={self.max_seq_length})")
        return self._model

    def encode(self, texts: list[str], show_progress: bool = True, is_query: bool = False) -> np.ndarray:
        """Encode texts L2-normalized embeddings. No fine-tuning."""
        model = self._load_model()
        t0 = time.time()
        
        # Add instruction for queries for BGE models
        if is_query and "bge" in self.model_name.lower():
            texts = [self.query_instruction + t for t in texts]
            
        embs = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # L2-norm => cosine = dot product
            convert_to_numpy=True,
        )
        elapsed = time.time() - t0
        print(f"  Encoded {len(texts):,} texts in {elapsed:.1f}s "
              f"({len(texts)/elapsed:.0f} texts/sec)")
        return embs

    def encode_corpus_cached(
        self,
        doc_ids: list[str],
        doc_texts: list[str],
        domain: str,
        cache_dir: Path,
        force_recompute: bool = False,
    ) -> np.ndarray:
        """Encode corpus with caching. Reuse on subsequent runs."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        emb_path = _get_cache_path(cache_dir, domain, "corpus")
        meta_path = _get_meta_path(cache_dir, domain, "corpus")

        if emb_path.exists() and meta_path.exists() and not force_recompute:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if (meta.get("model") == self.model_name
                    and meta.get("n_docs") == len(doc_ids)
                    and meta.get("max_seq_length") == self.max_seq_length):
                print(f"  Loading cached corpus embeddings for {domain} "
                      f"from {emb_path}")
                return np.load(emb_path)

        print(f"  Computing corpus embeddings for {domain} "
              f"({len(doc_ids):,} docs)...")
        embs = self.encode(doc_texts, show_progress=True, is_query=False)
        np.save(emb_path, embs)
        with open(meta_path, "w") as f:
            json.dump({
                "model": self.model_name,
                "n_docs": len(doc_ids),
                "max_seq_length": self.max_seq_length,
                "batch_size": self.batch_size,
                "embed_dim": embs.shape[1],
            }, f, indent=2)
        return embs

    def search(
        self,
        query: str,
        corpus_embs: np.ndarray,
        doc_ids: list[str],
        top_k: int = 100,
    ) -> list[tuple[str, float]]:
        """Retrieve top-k docs by cosine similarity (dot product on L2-normed embs)."""
        q_emb = self.encode([query], show_progress=False, is_query=True)  # (1, D)
        scores = (corpus_embs @ q_emb.T).reshape(-1)                      # (N,)
        
        k = min(top_k, len(scores))
        if k == 0:
            return []
            
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        
        return [(doc_ids[i], float(scores[i])) for i in top_idx]

    def batch_search(
        self,
        queries: list[tuple[str, str]],  # [(qid, query_text), ...]
        corpus_embs: np.ndarray,
        doc_ids: list[str],
        top_k: int = 100,
    ) -> dict[str, list[tuple[str, float]]]:
        """Batch retrieve; encode all queries at once then matrix multiply."""
        qids = [q[0] for q in queries]
        qtexts = [q[1] for q in queries]
        q_embs = self.encode(qtexts, show_progress=True, is_query=True)   # (Q, D)
        scores = q_embs @ corpus_embs.T                    # (Q, N)
        results = {}
        
        k = min(top_k, scores.shape[1])
        if k == 0:
            return {qid: [] for qid in qids}
            
        for i, qid in enumerate(qids):
            row_scores = scores[i]
            top_idx = np.argpartition(row_scores, -k)[-k:]
            top_idx = top_idx[np.argsort(-row_scores[top_idx])]
            results[qid] = [(doc_ids[j], float(row_scores[j])) for j in top_idx]
            
        return results
