"""Read the official gesetze-im-internet.de XML into citable sections."""

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from zipfile import ZipFile

from lxml import etree

from lotse.config import GII_URL

GERMAN_DATE = re.compile(r"\bv(?:om)?\.? *(\d{1,2})\.(\d{1,2})\.(\d{4})")
NUMBERED = re.compile(r"(\d+(?:\.\d+)*)\.? +([^|]+)")  # "3.5 Raumtemperatur", never a table row
NUMMER = re.compile(r"Nummer (\d+)")
PART = re.compile(r"(?:Abschnitt|Teil) \d+")
TABLE_CAPTION = re.compile(r"Tabelle (\d+) ?(.*)")
TABLE_ROW = re.compile(r"(\d+\.\d+) \| (.*)")  # a device row in Tabelle 12
REPEALED = re.compile(r"(?:\(\w+\)|\d+(?:\.\d+)*\.?)?\s*\(weggefallen\)\W*")
TITLE_NOISE = re.compile(r"^\(zu .*?\)\s*|\s*Inhaltsübersicht$")

DROPPED_TAGS = {"kommentar", "noindex", "TOC", "FnR", "FnArea", "IMG"}
STRONG_TAGS = {"B", "SP", "Title", "Ident"}
BLOCK_TAGS = {"P", "DL", "Title", "Ident"}


@dataclass(frozen=True)
class Line:
    text: str
    strong: bool = False  # fully bold or a Title: how annexes mark their headings
    table: bool = False


@dataclass
class Section:
    label: str  # "§ 5", "Anhang Nr. 3.5", "Anhang 2 Abschnitt 4 Nr. 2.1"
    heading: str
    lines: list[str] = field(default_factory=list)
    url: str = ""
    page: int | None = None  # PDFs only: a citation can open this page


def parse_xml(data: bytes) -> etree._Element:
    """XML from the web, without entities, network access, comments or processing instructions."""
    parser = etree.XMLParser(
        resolve_entities=False, no_network=True, remove_comments=True, remove_pis=True
    )
    return etree.fromstring(data, parser)


def load_root(path: Path) -> etree._Element:
    """Parse the one XML file in a law's zip, read in memory so nothing is unpacked to disk."""
    with ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".xml"))
        return parse_xml(archive.read(name))


def version_date(root: etree._Element) -> str:
    """The latest amendment in the header's standangabe, as ISO date.

    Only Stand and Neuf describe the text as published. Sonst and Hinweis may name
    amendments that were never applied or are not worked in yet.
    """
    entries = root.xpath(
        "norm[1]/metadaten/standangabe[standtyp='Stand' or standtyp='Neuf']/standkommentar"
    )
    dates = [
        date(int(y), int(m), int(d))
        for entry in entries
        for d, m, y in GERMAN_DATE.findall("".join(entry.itertext()))
    ]
    if not dates:
        return root.xpath("string(norm[1]/metadaten/ausfertigung-datum)")
    return max(dates).isoformat()


def flatten(content: etree._Element) -> list[Line]:
    """Turn a Content element into lines: one per paragraph, list item or table row."""
    lines: list[Line] = []
    buffer: list[tuple[str, bool]] = []

    def flush() -> None:
        text = " ".join("".join(part for part, _ in buffer).split())
        if text:
            lines.append(Line(text, all(strong for part, strong in buffer if part.strip())))
        buffer.clear()

    def walk(element: etree._Element, strong: bool) -> None:
        buffer.append((element.text or "", strong))
        for child in element:
            tag = child.tag
            if tag == "BR":
                flush()
            elif tag == "table":
                flush()
                lines.extend(table_lines(child))
            elif tag == "DT":
                walk(child, strong)
                buffer.append((" ", strong))
            elif tag not in DROPPED_TAGS:
                if tag in BLOCK_TAGS:
                    flush()
                walk(child, strong or tag in STRONG_TAGS)
                if tag in BLOCK_TAGS or tag in ("DD", "LA"):
                    flush()
            buffer.append((child.tail or "", strong))

    walk(content, False)
    flush()
    return lines


def expand_rows(rows: list[etree._Element], columns: dict[str, int]) -> list[list[str]]:
    """Resolve merged cells: a cell spanning rows repeats in each, so every row stands alone.
    A cell spanning columns keeps its text in the first one, so columns stay aligned."""
    expanded: list[dict[int, str]] = []
    carried: dict[int, tuple[str, int]] = {}  # column -> (text, rows still to fill)
    for row in rows:
        cells = {column: text for column, (text, _) in carried.items()}
        carried = {c: (text, left - 1) for c, (text, left) in carried.items() if left > 1}
        for entry in row.findall("entry"):
            start = columns.get(entry.get("namest") or entry.get("colname"))
            if start is None:
                start = next(i for i in range(len(cells) + 1) if i not in cells)
            end = columns.get(entry.get("nameend"), start)
            text = " ".join(line.text for line in flatten(entry))
            for column in range(start, end + 1):
                cell = text if column == start else ""
                cells[column] = cell
                if entry.get("morerows"):
                    carried[column] = (cell, int(entry.get("morerows")))
        expanded.append(cells)
    width = max((max(cells, default=-1) + 1 for cells in expanded), default=0)
    return [[cells.get(column, "") for column in range(width)] for cells in expanded]


def table_lines(table: etree._Element) -> list[Line]:
    """One line per row, with the header rows merged per column so the header is one line."""
    columns = {spec.get("colname"): i for i, spec in enumerate(table.iter("colspec"))}
    head = expand_rows(table.xpath("tgroup/thead/row"), columns)
    body = expand_rows(table.xpath("tgroup/tbody/row"), columns)
    header = [" ".join(dict.fromkeys(filter(None, cells))) for cells in zip(*head)]
    rows = [row for row in [header, *body] if any(row)]
    return [Line(" | ".join(row).rstrip(" |"), table=True) for row in rows]


def annex_sections(name: str, title: str, lines: list[Line]) -> list[Section]:
    """Split an annex by its own numbering, or keep it whole if it has none.

    Annexes number their entries in several ways: <Title>Nummer 2</Title>, a bold
    "1.1 Anwendungsbereich", or a DT "1.1" in front of the rule text. Plain "1." is a list
    item inside an entry. Text ahead of the first entry is the annex's own table of contents.
    Numbers restart in each Abschnitt, so the label keeps it. A short numbered line with no
    sentence end is a title, not an entry: its text goes to the headings below it. A captioned
    table can cover several numbers, so it is cited by its own number, and text after it goes
    back to the entry it interrupted. Where its rows are numbered ("Tabelle 12 Ziffer 7.2"),
    each number becomes a section with the header row, so a limit keeps its device and column.
    """
    sections: list[Section] = []
    titles: dict[str, str] = {}  # entry number -> title, "" for the current Abschnitt or Teil
    part = ""
    awaits_title = None  # the key in titles that the next bold lines name
    entry = caption_table = table = None

    def heading(number: str = "") -> str:
        parents = [number[:i] for i, char in enumerate(number) if char == "."]
        names = [title, *(titles.get(n, "") for n in dict.fromkeys(["", *parents, number]))]
        return " – ".join(filter(None, names))

    def open_section(label: str, section_heading: str) -> Section:
        section = Section(" ".join(filter(None, [name, part, label])), section_heading)
        sections.append(section)
        return section

    for line in lines:
        numbered = NUMBERED.fullmatch(line.text)
        nummer = NUMMER.fullmatch(line.text)
        caption = TABLE_CAPTION.fullmatch(line.text)
        if table and table.lines and not line.table:
            caption_table = table = None

        if line.strong and PART.fullmatch(line.text):
            part, awaits_title = line.text, ""
            titles.clear()
            entry = open_section("", heading())
        elif line.strong and nummer:
            awaits_title = nummer[1]
            entry = open_section(f"Nr. {awaits_title}", heading(awaits_title))
        elif numbered and (line.strong or "." in numbered[1]):
            number, rest = numbered.groups()
            is_title = line.strong or (len(rest) < 100 and rest[-1] not in ".,;:")
            awaits_title = None
            titles[number] = rest if is_title else ""
            entry = open_section(f"Nr. {number}", heading(number))
            if not is_title:
                entry.lines.append(rest)
        elif caption:
            awaits_title = None
            names = filter(None, [heading(), caption[2]])
            caption_table = table = open_section(f"Tabelle {caption[1]}", " – ".join(names))
        elif table and line.table:
            row = TABLE_ROW.fullmatch(line.text)
            if row and caption_table.lines:
                if table is caption_table and len(table.lines) == 1:
                    sections.remove(table)  # a header row alone answers nothing
                table = Section(
                    f"{caption_table.label} Ziffer {row[1]}",
                    f"{caption_table.heading} – {row[2]}",
                    caption_table.lines[:1],
                )
                sections.append(table)
            table.lines.append(line.text)
        elif entry is None:
            continue
        elif line.strong and awaits_title is not None and not line.text.startswith("("):
            titles[awaits_title] = f"{titles.get(awaits_title, '')} {line.text}".strip()
            entry.heading = heading(awaits_title)
        else:
            awaits_title = None
            entry.lines.append(line.text)

    return sections or [Section(name, title, [line.text for line in lines])]


def page_name(name: str) -> str:
    """The HTML page of a norm: "§ 24a" -> __24a.html, "Anhang II" -> anhang_ii.html."""
    if name.startswith("§"):
        return f"__{name.removeprefix('§').strip()}.html"
    return name.lower().replace(" ", "_") + ".html"


def parse_law(root: etree._Element, slug: str) -> list[Section]:
    """The citable sections of a law, without header, contents pages and repealed entries."""
    sections: list[Section] = []
    for norm in root.findall("norm"):
        name = norm.findtext("metadaten/enbez") or ""
        content = norm.find("textdaten/text/Content")
        title_element = norm.find("metadaten/titel")
        title = (
            "" if title_element is None else " ".join(" ".join(title_element.itertext()).split())
        )
        title = TITLE_NOISE.sub("", title)
        if content is None or not name.startswith(("§", "Anhang")) or "weggefallen" in title:
            continue

        lines = [line for line in flatten(content) if not REPEALED.fullmatch(line.text)]
        if name.startswith("Anhang"):
            found = annex_sections(name, title, lines)
        else:
            found = [Section(name, title, [line.text for line in lines])]

        for section in found:
            if any(char.isalpha() for line in section.lines for char in line):
                section.url = f"{GII_URL}/{slug}/{page_name(name)}"
                sections.append(section)
    return sections
