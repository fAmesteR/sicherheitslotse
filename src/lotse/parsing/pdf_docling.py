"""Turn a BAuA PDF into citable sections, with Docling's layout and table models.

Docling is a dev dependency and only runs here, offline. Nothing the app imports needs it.

Known limit: a footnote row under a table can land inside its last row. The numbers stay right, and
scripts/check_pdfs.py lists these cells. A section's page is the page it starts on.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import takewhile
from pathlib import Path
from typing import Any

from lotse.chunking import ABSATZ
from lotse.parsing.laws_xml import Section

# Docling files most page furniture separately, but a few banners still land in the body.
FURNITURE = re.compile(
    r"(?:- *)?(?:inoffizieller Text - maßgeblich ist der im GMBl bekanntgemachte ASR-Text"
    r"|Ausschuss für Arbeitsstätten [-–] ASTA-Geschäftsführung [-–] BAuA [-–] www\.baua\.de)"
    r"(?: *-)?"
)
PAGE_NUMBER = re.compile(r"(?:- *)?\d+ *-")
NUMBERED = re.compile(r"(\d+(?:\.\d+)*[a-z]?)\.? +(\S.*)")
ANNEX = re.compile(r"Anhang (\d+) *(.*)")
CAPTION = re.compile(r"(?:Tabelle|Tab\.) *(\d+(?:\.\d+)?\w?) *:? *(.*)")

# Words after a real hyphen: "Be- und Entladung" must stay as it is.
CONNECTOR = r"(?:und|oder|bzw|sowie|wie|als|bis)\b"
SOFT_BEFORE_CONNECTOR = re.compile(rf"­\s*(?={CONNECTOR})")
SOFT_HYPHEN = re.compile(r"­\s*")
LINE_BREAK = re.compile(rf"(?<=\w)- (?!{CONNECTOR})(?=[a-zäöüß])")
# Docling also rejoins "Flucht-\nund" across a line end, which gives "Fluchtund".
GLUED = re.compile(r"\b(\w{3,}?)(und|oder)\b")
REAL_WORD = re.compile(r"(?:grund|befund|verbund|gesund|schlund|schwund)$", re.IGNORECASE)


@dataclass(frozen=True)
class Block:
    kind: str  # "heading", "text" or "table"
    text: str = ""
    page: int = 0
    rows: list[list[str]] = field(default_factory=list)
    caption: str = ""


def clean(text: str) -> str:
    """Undo line-end hyphenation and nothing else, because the wording must not change."""
    text = SOFT_BEFORE_CONNECTOR.sub("- ", text)
    text = SOFT_HYPHEN.sub("", text)
    text = LINE_BREAK.sub("", text)
    text = GLUED.sub(lambda m: m[0] if REAL_WORD.search(m[0]) else f"{m[1]}- {m[2]}", text)
    return " ".join(text.split())


def strip_furniture(line: str) -> str:
    """Cut, never drop: Docling can glue a footer to the sentence running on over the page."""
    line = FURNITURE.sub("", line).strip()
    return "" if PAGE_NUMBER.fullmatch(line) else line


def refuse_excerpt(texts: Iterable[str], doc: str) -> None:
    """A change sheet prints deleted wording as strikethrough, and a parser reads it as current."""
    if any("Textauszug" in text for text in texts):
        raise SystemExit(f"{doc} is a Textauszug with deleted text in it. Use the full PDF.")


def refuse_other_version(texts: Iterable[str], doc: str, version: str) -> None:
    """A newer file from BAuA would be cited under the old version without this."""
    if not any(version in text for text in texts):
        raise SystemExit(f"{doc}: the first page does not say {version!r}. Update lotse.config.")


def table_rows(grid: Sequence[Sequence[Any]]) -> list[list[str]]:
    """Header rows merge per column. Below them a spanned cell repeats in Docling's grid, so
    compare positions: it keeps its text in the first column, and equal values are real."""
    heads = list(takewhile(lambda row: any(cell.column_header for cell in row), grid))
    rows = []
    if len(heads) > 1:
        names = [[clean(cell.text) for cell in column] for column in zip(*heads)]
        rows.append([" ".join(dict.fromkeys(filter(None, column))) for column in names])
    for row in grid[len(heads) if rows else 0 :]:
        seen = set()
        values = []
        for cell in row:
            position = cell.start_row_offset_idx, cell.start_col_offset_idx
            values.append("" if position in seen else clean(cell.text))
            seen.add(position)
        while values and not values[-1]:
            values.pop()
        if values:
            rows.append(values)
    return rows


def parse_blocks(blocks: Iterable[Block], url: str, title: str) -> list[Section]:
    """Sections start at chapter 1, so the preamble and the table of contents stay out."""
    sections: list[Section] = []
    current: Section | None = None  # None before chapter 1 and inside Literaturhinweise
    chapter = annex = ""
    titles: dict[str, str] = {}  # section number -> title, for the headings below it
    table: Section | None = None  # notes and a second part below a table belong to it
    group: Section | None = None  # the part of a long table that is being filled
    header = ""

    def add(label: str, heading: str, page: int, lines: list[str] | None = None) -> Section:
        section = Section(label, heading, lines or [], url, page)
        sections.append(section)
        return section

    def add_titled(number: str, name: str, page: int) -> Section:
        parts = number.split(".")
        titles[number] = name
        names = [titles.get(".".join(parts[: i + 1])) for i in range(len(parts))]
        return add(number, ", ".join(filter(None, [title, *names])), page)

    for block in blocks:
        numbered = NUMBERED.fullmatch(block.text)

        if block.kind == "heading":
            group = None
            before = current
            found_annex = ANNEX.fullmatch(block.text)
            if numbered and block.text.startswith(f"{numbered[1]}."):
                numbered = None  # a list item Docling took for a heading: chapters have no dot
            if block.text.startswith("Literaturhinweise"):
                current = None
            elif found_annex:
                annex = f"Anhang {found_annex[1]}"
                heading = f"{title}, {found_annex[2]}" if found_annex[2] else title
                current = add(annex, heading, block.page)
            elif numbered and annex and current:
                heading = f"{title}, {numbered[2]}"
                current = add(f"{annex} Nr. {numbered[1]}", heading, block.page)
            elif numbered and (current or (not sections and numbered[1] == "1")):
                chapter = numbered[1].split(".")[0]
                current = add_titled(numbered[1], numbered[2], block.page)
            elif current and current.label == annex and current.heading == title:
                current.heading = f"{title}, {block.text}"
            elif current:
                (table or current).lines.append(block.text)
            if current is not before:
                table = None

        elif current is None:
            continue

        elif block.kind == "table":
            rows = [" | ".join(row) for row in block.rows]
            caption = CAPTION.fullmatch(block.caption)
            has_groups = any(len(row) == 1 and NUMBERED.fullmatch(row[0]) for row in block.rows)
            if caption:
                heading = f"{title}, {caption[2]}" if caption[2] else current.heading
                table = add(f"{current.label} Tabelle {caption[1]}", heading, block.page, rows)
            elif has_groups or (group and rows[0] == header):
                header = rows[0]
                for row, line in zip(block.rows[1:], rows[1:]):
                    name = NUMBERED.fullmatch(row[0]) if len(row) == 1 else None
                    if name:
                        label, heading = f"{current.label} Nr. {name[1]}", name[2]
                        group = add(label, f"{current.heading}, {heading}", block.page, [header])
                    else:
                        (group or current).lines.append(line)
            else:
                (table or current).lines += rows

        elif table and not numbered and not ABSATZ.match(block.text):
            table.lines.append(block.text)

        else:
            table = None
            is_entry = numbered and "." in numbered[1] and numbered[1].split(".")[0] == chapter
            if is_entry and not annex and block.text[-1] not in ".,;:":
                current = add_titled(numbered[1], numbered[2], block.page)  # a title read as text
                continue
            if is_entry and not annex:  # definitions are numbered text: "3.2 Die Lufttemperatur"
                current = add(numbered[1], current.heading, block.page)
            current.lines.append(block.text)

    return [section for section in sections if section.lines]


def read_blocks(document: Any) -> list[Block]:
    """Docling's document as plain blocks, which keeps the parsing rules testable without it."""
    blocks = []
    captions = set()  # Docling repeats a table's caption as an item below it
    pending = ""  # a caption it did not tie to the table that follows
    for item, _ in document.iterate_items():
        label = str(item.label)
        page = item.prov[0].page_no if item.prov else 0
        if hasattr(item, "data"):
            caption = clean(item.caption_text(document)) or pending
            captions.add(caption)
            pending = ""
            rows = table_rows(item.data.grid)
            if rows:
                blocks.append(Block("table", page=page, rows=rows, caption=caption))
            continue
        text = strip_furniture(clean(getattr(item, "text", "") or getattr(item, "orig", "")))
        marker = getattr(item, "marker", "") or ""
        if label == "list_item" and marker and not NUMBERED.fullmatch(text):
            text = f"{marker} {text}"  # item.text drops the "(1)" a citation needs
        if label == "caption":  # also where it files some annex titles and notes
            if ANNEX.fullmatch(text):
                blocks.append(Block("heading", text, page))
            elif not CAPTION.fullmatch(text):
                if not text.startswith("Abb"):
                    blocks.append(Block("text", text, page))
            elif text not in captions:
                pending = text
        elif text:
            if label in ("section_header", "title"):
                blocks.append(Block("heading", text, page))
            elif label in ("text", "list_item", "footnote", "formula"):
                blocks.append(Block("text", text, page))
    return blocks


def furniture_texts(document: Any) -> list[str]:
    layers = {"furniture"}
    items = document.iterate_items(included_content_layers=layers)
    return [item.text for item, _ in items if hasattr(item, "text")]


def convert(path: Path, cache: Path) -> Any:
    """The slow step. Docling reads the pages once and data/parsed keeps the result."""
    from docling_core.types.doc import DoclingDocument

    if cache.exists():
        return DoclingDocument.load_from_json(cache)

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_HERON_101
    from docling.datamodel.pipeline_options import (
        LayoutOptions,
        PdfPipelineOptions,
        TableFormerMode,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
    options.table_structure_options.mode = TableFormerMode.ACCURATE
    # Of three layout models on these files, only this one left no table values loose.
    options.layout_options = LayoutOptions(model_spec=DOCLING_LAYOUT_HERON_101)
    formats = {InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    document = DocumentConverter(format_options=formats).convert(path).document
    cache.parent.mkdir(parents=True, exist_ok=True)
    document.save_as_json(cache)
    return document


def parse_pdf(
    path: Path, cache: Path, doc: str, url: str, title: str, version: str
) -> list[Section]:
    document = convert(path, cache)
    blocks = read_blocks(document)
    refuse_excerpt(furniture_texts(document), doc)
    refuse_other_version([b.text for b in blocks if b.page == 1], doc, version)
    return parse_blocks(blocks, url, title)
