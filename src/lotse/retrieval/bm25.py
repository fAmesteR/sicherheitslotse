"""Keyword search: BM25 over stemmed German tokens."""

import re

import bm25s
import Stemmer

# The bm25s default turns "A2.3" into "a2" and drops the 5 in "§ 5".
TOKEN_PATTERN = r"(?u)§?\w+(?:\.\w+)*"
# "§ 5" becomes the one token "§5". Otherwise a search for it also matches every "Abs. 5".
PARAGRAPH_SIGN = re.compile(r"§+\s*(?=\d)")


def tokenize(texts: list[str]) -> list[list[str]]:
    return bm25s.tokenize(
        [PARAGRAPH_SIGN.sub("§", text) for text in texts],
        token_pattern=TOKEN_PATTERN,
        stopwords="german",  # the bm25s default is English, which removes the wrong words
        stemmer=Stemmer.Stemmer("german"),
        return_ids=False,
        show_progress=False,
    )


def build(texts: list[str]) -> bm25s.BM25:
    index = bm25s.BM25()
    index.index(tokenize(texts), show_progress=False)
    return index


def search(index: bm25s.BM25, question: str, k: int) -> list[int]:
    """Chunk positions, best first. Chunks that share no word with the question are left out."""
    positions, scores = index.retrieve(tokenize([question]), k=k, show_progress=False)
    return [int(p) for p, score in zip(positions[0], scores[0]) if score > 0]
