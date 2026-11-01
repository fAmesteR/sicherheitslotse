"""Parse and chunk the sources into data/chunks.jsonl. --sample N prints N chunks to read."""

import argparse
import json
import random
import sys
from collections import Counter

from lotse.chunking import chunk_section, context_prefix, count_tokens
from lotse.config import (
    CHUNKS,
    LAWS,
    MANIFEST,
    MAX_CHUNK_TOKENS,
    PARSED_DIR,
    PDF_DIR,
    PDFS,
    RAW_DIR,
)
from lotse.parsing.laws_xml import Section, load_root, parse_law
from lotse.parsing.pdf_docling import parse_pdf


def sections_of(entry: dict) -> list[Section]:
    doc = entry["doc"]
    if doc in LAWS:
        return parse_law(load_root(RAW_DIR / f"{LAWS[doc]}.zip"), LAWS[doc])
    stem, title, version = PDFS[doc]
    path, cache = PDF_DIR / f"{stem}.pdf", PARSED_DIR / f"{stem}.json"
    return parse_pdf(path, cache, doc, entry["url"], title, version)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    chunks = []
    for entry in json.loads(MANIFEST.read_text(encoding="utf-8")):
        for section in sections_of(entry):
            chunks += chunk_section(entry["doc"], section, entry["version_date"])

    duplicates = [key for key, n in Counter(c["id"] for c in chunks).items() if n > 1]
    if duplicates:
        raise SystemExit(f"Duplicate chunk ids: {duplicates[:10]}")

    with CHUNKS.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    sizes = sorted(
        count_tokens(context_prefix(c["doc"], c["section"], c["heading"]) + c["text"])
        for c in chunks
    )
    over = sum(size > MAX_CHUNK_TOKENS for size in sizes)
    print(f"{len(chunks)} chunks, tokens median {sizes[len(sizes) // 2]}, max {sizes[-1]}")
    print(f"over {MAX_CHUNK_TOKENS}: {over}, under 30: {sum(size < 30 for size in sizes)}")
    for doc, n in Counter(c["doc"] for c in chunks).items():
        print(f"  {doc:24} {n}")

    for chunk in random.Random(args.seed).sample(chunks, args.sample):
        print(f"\n--- {chunk['id']}\n{chunk['heading']}\n{chunk['url']}\n{chunk['text']}")


if __name__ == "__main__":
    main()
