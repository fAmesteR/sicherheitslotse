"""Turn parsed sections into chunks of at most MAX_CHUNK_TOKENS, context prefix included."""

import re
from collections.abc import Callable
from functools import cache

from tokenizers import Tokenizer

from lotse.config import EMBEDDING_MODEL, MAX_CHUNK_TOKENS, MODEL_REVISIONS
from lotse.parsing.laws_xml import Section

ABSATZ = re.compile(r"\((\d+[a-z]?)\)")
SENTENCE_END = re.compile(r"(?<=[.;:])\s+")

Counter = Callable[[str], int]


@cache
def tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained(EMBEDDING_MODEL, revision=MODEL_REVISIONS[EMBEDDING_MODEL])


def count_tokens(text: str) -> int:
    return len(tokenizer().encode(text, add_special_tokens=False).ids)


def context_prefix(doc: str, section: str, heading: str) -> str:
    """Goes in front of the text for embedding and BM25: a paragraph rarely names its law."""
    return " ".join(filter(None, [doc, section, heading])) + ": "


def passage(chunk: dict) -> str:
    """A chunk as search reads it."""
    return context_prefix(chunk["doc"], chunk["section"], chunk["heading"]) + chunk["text"]


def rerank_text(chunk: dict, prefix: bool = False) -> str:
    """A chunk as the reranker reads it: without the prefix, unless asked (see RERANK_PREFIX)."""
    return passage(chunk) if prefix else chunk["text"]


def split_absaetze(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Group lines by Absatz. Text ahead of "(1)" comes first, with no number."""
    groups: list[tuple[str, list[str]]] = [("", [])]
    for line in lines:
        if match := ABSATZ.match(line):
            groups.append((match[1], []))
        groups[-1][1].append(line)
    return [group for group in groups if group[1]]


def shorten(line: str, limit: int, count: Counter) -> list[str]:
    """Cut a line longer than limit at sentence ends, then between words."""
    if count(line) <= limit:
        return [line]
    pieces, current = [], ""
    sentences = SENTENCE_END.split(line)
    words = len(sentences) == 1
    for piece in line.split() if words else sentences:
        candidate = f"{current} {piece}".strip()
        if current and count(candidate) > limit:
            pieces.append(current)
            current = piece
        else:
            current = candidate
    pieces.append(current)
    return [
        short
        for piece in pieces
        for short in (shorten(piece, limit, count) if not words else [piece])
    ]


def pack(lines: list[str], limit: int, count: Counter) -> list[str]:
    """Fill parts line by line. Each later part repeats the first line if that is short,
    so a table keeps its header row and a list its lead-in."""
    head = lines[0] if count(lines[0]) < limit // 2 else ""
    room = limit - count(head) - 1
    parts: list[list[str]] = [[]]
    for line in lines:
        for piece in shorten(line, room, count):
            if parts[-1] and count("\n".join([*parts[-1], piece])) > limit:
                parts.append([head] if head else [])
            parts[-1].append(piece)
    return ["\n".join(part) for part in parts]


def fit(section: Section, doc: str, count: Counter) -> list[tuple[str, str]]:
    """(label, text) pairs: the whole section if it fits, else one per Absatz, else cut further."""
    whole = "\n".join(section.lines)
    limit = MAX_CHUNK_TOKENS - count(context_prefix(doc, section.label, section.heading))
    if count(whole) <= limit:
        return [(section.label, whole)]

    pieces = []
    for number, lines in split_absaetze(section.lines):
        label = f"{section.label} Abs. {number}" if number else section.label
        room = MAX_CHUNK_TOKENS - count(context_prefix(doc, label, section.heading))
        pieces += [(label, text) for text in pack(lines, room, count)]
    return pieces


def chunk_section(
    doc: str, section: Section, version: str, count: Counter = count_tokens
) -> list[dict]:
    chunks = []
    seen: dict[str, int] = {}
    for label, text in fit(section, doc, count):
        seen[label] = seen.get(label, 0) + 1
        key = f"{doc} {label}".lower().replace("§ ", "p").replace("abs. ", "a")
        key = re.sub(r"[^\w.]+", "-", key)
        chunks.append(
            {
                "id": key if seen[label] == 1 else f"{key}-{seen[label]}",
                "doc": doc,
                "section": label,
                "heading": section.heading,
                "text": text,
                "url": section.url,
                "page": section.page,
                "version_date": version,
            }
        )
    return chunks
