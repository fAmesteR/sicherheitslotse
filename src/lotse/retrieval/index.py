"""The search index on disk: chunks, embeddings and BM25, checked against the code at load."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import bm25s
import numpy as np
from huggingface_hub import snapshot_download

from lotse.chunking import passage
from lotse.config import EMBEDDING_MODEL, INDEX_DIR, INDEX_REPO, INDEX_REVISION, MODEL_REVISIONS
from lotse.retrieval import bm25
from lotse.retrieval.bm25 import TOKEN_PATTERN


@dataclass(frozen=True)
class Index:
    chunks: list[dict]
    embeddings: np.ndarray
    bm25: bm25s.BM25


def sha256(*files: Path) -> str:
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.read_bytes())
    return digest.hexdigest()


def expected_meta(path: Path) -> dict:
    """What meta.json must say for the index in path to be used: the settings it was built with
    and a hash of every file, so a half-copied or mixed index stops at load."""
    return {
        "embedding_model": EMBEDDING_MODEL,
        "embedding_revision": MODEL_REVISIONS[EMBEDDING_MODEL],
        "token_pattern": TOKEN_PATTERN,
        "chunks_sha256": sha256(path / "chunks.jsonl"),
        "embeddings_sha256": sha256(path / "embeddings.npy"),
        "bm25_sha256": {file.name: sha256(file) for file in sorted((path / "bm25").iterdir())},
    }


def load_index(path: Path = INDEX_DIR) -> Index:
    """Stops with an error if the index was built from other chunks or with other settings."""
    chunks_file = path / "chunks.jsonl"
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    for key, value in expected_meta(path).items():
        if meta.get(key) != value:
            raise SystemExit(f"Index mismatch on {key}: run scripts/build_index.py again")

    chunks = [json.loads(line) for line in chunks_file.open(encoding="utf-8")]
    embeddings = np.load(path / "embeddings.npy")
    if len(embeddings) != len(chunks):
        raise SystemExit("Index mismatch: embeddings and chunks differ in length")
    return Index(chunks, embeddings, bm25s.BM25.load(path / "bm25"))


def save_subset(index: Index, docs: frozenset[str], path: Path) -> None:
    """Write the index for `docs` alone, so that no other document's text is published with it.
    BM25 is built again, because its word counts belong to the documents it covers."""
    keep = [at for at, chunk in enumerate(index.chunks) if chunk["doc"] in docs]
    chunks = [index.chunks[at] for at in keep]
    path.mkdir(parents=True, exist_ok=True)
    (path / "meta.json").unlink(missing_ok=True)  # a save that dies must not look finished
    with (path / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as file:
        file.writelines(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks)
    np.save(path / "embeddings.npy", index.embeddings[keep])
    bm25.build([passage(chunk) for chunk in chunks]).save(path / "bm25", show_progress=False)
    (path / "meta.json").write_text(json.dumps(expected_meta(path), indent=2), encoding="utf-8")


def fetch_index(path: Path = INDEX_DIR) -> None:
    """Download the published index unless there is one here already. The Space starts with this.
    The dataset is private, so it needs a read token in HF_TOKEN."""
    if not (path / "meta.json").exists():
        snapshot_download(INDEX_REPO, repo_type="dataset", revision=INDEX_REVISION, local_dir=path)
