"""Compare the parsed PDFs with their own text layer. Run it after parsing a new PDF.

Lists every line of the PDF that did not reach a section, and every table cell with words in a
column of numbers. Figure captions, drawing labels and Literaturhinweise are meant to be missing,
and so are headings of chapters with no text of their own. Anything else is a parsing bug.
"""

import re
import sys

import pypdfium2

from lotse.config import PARSED_DIR, PDF_DIR, PDFS
from lotse.parsing.pdf_docling import convert, parse_blocks, read_blocks, strip_furniture

NUMERIC = re.compile(r"[<>≥≤+~-]?\s*\d[\d,.]*\s*(?:\*\)|m|cm|mm|lx|%|°C|m²)?(?: bis \d+)?")


def word_triples(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.replace("￾", "").replace("­", "").lower())
    return {tuple(words[i : i + 3]) for i in range(len(words) - 2)}


def lost_lines(stem: str, parsed: str) -> list[str]:
    known = word_triples(parsed)
    pdf = pypdfium2.PdfDocument(PDF_DIR / f"{stem}.pdf")
    lost = []
    for number in range(1, len(pdf)):  # the first page is the title and the contents
        for line in pdf[number].get_textpage().get_text_range().splitlines():
            found = word_triples(strip_furniture(line.strip()))
            if len(found) >= 3 and len(found & known) / len(found) < 0.5:
                lost.append(f"p{number + 1}: {line.strip()}")
    return lost


def mixed_cells(rows: list[list[str]]) -> list[str]:
    """Words in a column of numbers: a table model pulled a neighbouring line into the row."""
    width = max(map(len, rows))
    full = [row for row in rows[1:] if len(row) == width]
    mixed = []
    for column in range(1, width):
        cells = [row[column] for row in full if row[column]]
        numeric = [cell for cell in cells if NUMERIC.fullmatch(cell)]
        if len(cells) >= 3 and len(numeric) >= 0.6 * len(cells):
            mixed += [cell for cell in cells if cell not in numeric]
    return mixed


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    for doc, (stem, title, _) in PDFS.items():
        blocks = read_blocks(convert(PDF_DIR / f"{stem}.pdf", PARSED_DIR / f"{stem}.json"))
        sections = parse_blocks(blocks, "", title)
        parsed = " ".join(f"{s.label} {s.heading} {' '.join(s.lines)}" for s in sections)
        print(f"=== {doc}: {len(sections)} sections")
        for line in lost_lines(stem, parsed):
            print(f"  lost {line}")
        for block in blocks:
            if block.kind == "table":
                for cell in mixed_cells(block.rows):
                    print(f"  mixed p{block.page}: {cell}")


if __name__ == "__main__":
    main()
