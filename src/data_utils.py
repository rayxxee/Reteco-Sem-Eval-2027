#!/usr/bin/env python3
"""
src/data_utils.py
Utilities for loading RETECO Track-1 Sub-track 1a data.
No field names are invented – all match the verified schema from the
HuggingFace card and starter-kit docs.

Track-1 record formats (confirmed):
  documents.jsonl      : {id: str, content: str}
  examples_{split}.jsonl : {id: str, query: str, gold_ids: list[str], gold_answers: list[str]}
  qrels_{split}.txt    : TREC 4-col  qid  0  docid  1
"""
import json
import re
from pathlib import Path
from typing import Iterator


# ── Helpers ──────────────────────────────────────────────────────────────────

def iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield each non-empty line of a JSONL file as a parsed dict."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    return list(iter_jsonl(path))


def load_json(path: Path):
    """Load a JSON file (may be a list or dict)."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ── Track-1 loaders ───────────────────────────────────────────────────────────

def load_corpus(domain_dir: Path) -> tuple[list[str], list[str]]:
    """
    Load Track-1 corpus for one domain.

    Returns
    -------
    doc_ids   : list of str   (field: 'id')
    doc_texts : list of str   (field: 'content')
    """
    records = load_jsonl(domain_dir / "documents.jsonl")
    doc_ids = [r["id"] for r in records]
    doc_texts = [r.get("content", "") for r in records]
    return doc_ids, doc_texts


def load_queries_1a(domain_dir: Path, split: str) -> list[dict]:
    """
    Load Sub-track 1a queries.

    Parameters
    ----------
    split : 'train' | 'dev'

    Returns
    -------
    list of {id, query, gold_ids, gold_answers}
    """
    return load_jsonl(domain_dir / f"examples_{split}.jsonl")


def load_qrels(domain_dir: Path, split: str) -> dict[str, set[str]]:
    """
    Load TREC-format qrels (binary relevance, rel=1).

    Returns
    -------
    dict  qid -> set of relevant doc_ids
    """
    path = domain_dir / f"qrels_{split}.txt"
    qrels: dict[str, set] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            qid, _, docid, rel = parts[0], parts[1], parts[2], parts[3]
            if int(rel) > 0:
                qrels.setdefault(qid, set()).add(docid)
    return qrels


def qrels_to_pytrec(qrels: dict[str, set[str]]) -> dict[str, dict[str, int]]:
    """Convert {qid: {docid}} to pytrec_eval format {qid: {docid: rel}}."""
    return {qid: {d: 1 for d in docs} for qid, docs in qrels.items()}


# ── Domain discovery ──────────────────────────────────────────────────────────

TRACK1_DOMAINS = [
    "bitcoin", "cardano", "economics", "genealogy", "history",
    "hsm", "iota", "law", "monero", "politics", "quant",
    "travel", "workplace",
]


def get_domain_dir(data_root: Path, domain: str) -> Path:
    """Return the path to a Track-1 domain directory."""
    return data_root / "track1_tempo" / domain


def corpus_size(data_root: Path, domain: str) -> int:
    """Count documents in a domain corpus (line count of documents.jsonl)."""
    path = get_domain_dir(data_root, domain) / "documents.jsonl"
    return sum(1 for line in open(path, encoding="utf-8") if line.strip())


def query_counts(data_root: Path, domain: str) -> dict[str, int]:
    """Return {split: n_queries} for a domain."""
    counts = {}
    d = get_domain_dir(data_root, domain)
    for split in ("train", "dev"):
        p = d / f"examples_{split}.jsonl"
        counts[split] = sum(1 for line in open(p, encoding="utf-8") if line.strip())
    return counts


# ── Temporal utilities ────────────────────────────────────────────────────────

_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")
_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},?\s+\d{4}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
    r"|\b\d{4}-\d{2}-\d{2}\b",
    re.IGNORECASE,
)


def extract_years(text: str) -> list[str]:
    """Return all 4-digit year strings found in text (1000–2029)."""
    return _YEAR_RE.findall(text or "")


def extract_dates(text: str) -> list[str]:
    """Return all date-expression matches found in text."""
    return _DATE_RE.findall(text or "")


def has_temporal_cue(text: str) -> bool:
    """True if text contains at least one year or date expression."""
    return bool(_YEAR_RE.search(text or "") or _DATE_RE.search(text or ""))
