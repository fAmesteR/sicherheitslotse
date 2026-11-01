"""The search index on disk: chunks, embeddings and BM25, checked against the code at load."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import bm25s
import numpy as np

from lotse.config import EMBEDDING_MODEL, INDEX_DIR
from lotse.retrieval.bm25 import TOKEN_PATTERN


@dataclass(frozen=True)
class Index:
    chunks: list[dict]
    embeddings: np.ndarray
    bm25: bm25s.BM25


def expected_meta(chunks_file: Path) -> dict:
    return {
        "embedding_model": EMBEDDING_MODEL,
        "token_pattern": TOKEN_PATTERN,
        "chunks_sha256": hashlib.sha256(chunks_file.read_bytes()).hexdigest(),
    }


def load_index(path: Path = INDEX_DIR) -> Index:
    """Stops with an error if the index was built from other chunks or with other settings."""
    chunks_file = path / "chunks.jsonl"
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    for key, value in expected_meta(chunks_file).items():
        if meta.get(key) != value:
            raise SystemExit(f"Index mismatch on {key}: run scripts/build_index.py again")

    chunks = [json.loads(line) for line in chunks_file.open(encoding="utf-8")]
    embeddings = np.load(path / "embeddings.npy")
    if len(embeddings) != len(chunks):
        raise SystemExit("Index mismatch: embeddings and chunks differ in length")
    return Index(chunks, embeddings, bm25s.BM25.load(path / "bm25"))
