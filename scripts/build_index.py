"""Build index/ from data/chunks.jsonl: BM25, bge-m3 embeddings and meta.json."""

import json
import shutil
import time

import numpy as np

from lotse.chunking import passage
from lotse.config import CHUNKS, INDEX_DIR
from lotse.retrieval import bm25, dense
from lotse.retrieval.index import expected_meta


def main() -> None:
    chunks = [json.loads(line) for line in CHUNKS.open(encoding="utf-8")]
    texts = [passage(chunk) for chunk in chunks]

    INDEX_DIR.mkdir(exist_ok=True)
    (INDEX_DIR / "meta.json").unlink(missing_ok=True)  # a build that dies must not look finished
    shutil.copy(CHUNKS, INDEX_DIR / "chunks.jsonl")

    bm25.build(texts).save(INDEX_DIR / "bm25", show_progress=False)

    start = time.perf_counter()
    embeddings = dense.embed(texts)
    np.save(INDEX_DIR / "embeddings.npy", embeddings)
    print(f"{embeddings.shape} embedded in {time.perf_counter() - start:.0f} s")

    meta = expected_meta(INDEX_DIR)
    (INDEX_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
