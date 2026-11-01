"""The text here is invented in the shape of the real files: ASR text stays private for now."""

from types import SimpleNamespace

import pytest

from lotse.chunking import chunk_section
from lotse.parsing.laws_xml import Section
from lotse.parsing.pdf_docling import (
    Block,
    clean,
    parse_blocks,
    read_blocks,
    refuse_excerpt,
    refuse_other_version,
    strip_furniture,
    table_rows,
)

URL = "https://example.org/regel.pdf"


def heading(text: str, page: int = 1) -> Block:
    return Block("heading", text, page)


def text(value: str, page: int = 1) -> Block:
    return Block("text", value, page)


def table(rows: list[list[str]], caption: str = "", page: int = 1) -> Block:
    return Block("table", page=page, rows=rows, caption=caption)


def parse(*blocks: Block) -> dict[str, Section]:
    return {s.label: s for s in parse_blocks([heading("1 Ziel"), *blocks], URL, "Regel")}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("die Ar­ beitszeit", "die Arbeitszeit"),
        ("Hauptflucht­wege", "Hauptfluchtwege"),
        ("Blend- schutz", "Blendschutz"),
        ("Be­ und Entladung", "Be- und Entladung"),
        ("Be- und Entladung", "Be- und Entladung"),
        ("Arm- oder Beinarbeit", "Arm- oder Beinarbeit"),
        ("Fluchtund Rettungspläne", "Flucht- und Rettungspläne"),
        ("Pausenund Kantinenräume", "Pausen- und Kantinenräume"),
        ("Sicherheitsund Gesundheitsschutz", "Sicherheits- und Gesundheitsschutz"),
        ("einoder zweiseitig", "ein- oder zweiseitig"),
        ("aufgrund der Hintergrund", "aufgrund der Hintergrund"),
        ("gesund und Befund oder Verbund", "gesund und Befund oder Verbund"),
        ("zwei  Leerzeichen", "zwei Leerzeichen"),
    ],
)
def test_clean(raw, expected):
    assert clean(raw) == expected


@pytest.mark.parametrize(
    "line",
    [
        "- 8 -",
        "- inoffizieller Text - maßgeblich ist der im GMBl bekanntgemachte ASR-Text -",
        "- Ausschuss für Arbeitsstätten - ASTA-Geschäftsführung - BAuA - www.baua.de -",
        "Ausschuss für Arbeitsstätten - ASTA-Geschäftsführung - BAuA - www.baua.de -",
    ],
)
def test_furniture_is_removed(line):
    assert strip_furniture(line) == ""


@pytest.mark.parametrize("line", ["- arbeitsplatzbezogene Maßnahmen", "8 Gänge", "0,60"])
def test_real_lines_are_not_furniture(line):
    assert strip_furniture(line) == line


def test_a_sentence_glued_to_a_footer_survives():
    line = "- Ausschuss für Arbeitsstätten - ASTA-Geschäftsführung - BAuA - www.baua.de - sind zu"
    assert strip_furniture(line) == "sind zu"


def test_an_excerpt_is_refused():
    with pytest.raises(SystemExit, match="Textauszug"):
        refuse_excerpt(["- inoffizieller Textauszug - maßgeblich ist der GMBl-Text -"], "ASR A1.3")
    refuse_excerpt(["- inoffizieller Text - maßgeblich ist der GMBl-Text -"], "ASR A3.5")


def test_a_file_of_another_version_is_refused():
    with pytest.raises(SystemExit, match="GMBl 2022, S. 198"):
        refuse_other_version(["zuletzt geändert GMBl 2024, S. 1"], "ASR A3.5", "GMBl 2022, S. 198")
    refuse_other_version(["zuletzt geändert GMBl 2022, S. 198"], "ASR A3.5", "GMBl 2022, S. 198")


def cell(value: str, row: int, column: int, header: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        text=value, start_row_offset_idx=row, start_col_offset_idx=column, column_header=header
    )


def test_table_rows_merge_spanned_cells_but_keep_equal_neighbours():
    spanned = cell("Hinweis über alle Spalten", 1, 0)
    grid = [
        [cell("bis 5", 0, 0), cell("0,90", 0, 1), cell("0,90", 0, 2)],
        [spanned, spanned, spanned],
        [cell("bis 20", 2, 0), cell("", 2, 1), cell("", 2, 2)],
    ]
    assert table_rows(grid) == [
        ["bis 5", "0,90", "0,90"],
        ["Hinweis über alle Spalten"],
        ["bis 20"],
    ]


def test_table_rows_keep_an_empty_cell_in_the_middle():
    grid = [[cell("Sitzen", 0, 0), cell("", 0, 1), cell("+19 °C", 0, 2)]]
    assert table_rows(grid) == [["Sitzen", "", "+19 °C"]]


def test_a_cell_spanning_some_columns_keeps_the_others_aligned():
    spanned = cell("12.1 Friseur", 0, 0)
    assert table_rows([[spanned, spanned, cell("500", 0, 2), cell("90", 0, 3)]]) == [
        ["12.1 Friseur", "", "500", "90"]
    ]


def test_header_rows_merge_so_each_column_keeps_its_full_name():
    length = cell("Länge", 0, 0, True)
    width = cell("Breite", 0, 1, True)
    grid = [
        [length, width, width],
        [length, cell("innen", 1, 1, True), cell("außen", 1, 2, True)],
        [cell("< 50", 2, 0), cell("0,80", 2, 1), cell("0,60", 2, 2)],
    ]
    assert table_rows(grid) == [
        ["Länge", "Breite innen", "Breite außen"],
        ["< 50", "0,80", "0,60"],
    ]


def test_one_header_row_stays_as_it_is():
    grid = [
        [cell("", 0, 0, True), cell("A", 0, 1, True)],
        [cell("Nr.", 1, 0), cell("Anzahl", 1, 1)],
    ]
    assert table_rows(grid) == [["", "A"], ["Nr.", "Anzahl"]]


def fake_document(*items: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(iterate_items=lambda: [(item, 0) for item in items])


def item(label: str, value: str = "", orig: str = "", page: int = 1) -> SimpleNamespace:
    prov = [SimpleNamespace(page_no=page)]
    return SimpleNamespace(label=label, text=value, orig=orig, prov=prov, marker="")


def table_item(caption: str, value: str, page: int = 1) -> SimpleNamespace:
    data = SimpleNamespace(grid=[[cell(value, 0, 0)]])
    prov = [SimpleNamespace(page_no=page)]
    return SimpleNamespace(
        label="table", data=data, prov=prov, caption_text=lambda document: caption
    )


def test_a_caption_docling_did_not_tie_to_its_table_reaches_the_next_table():
    blocks = read_blocks(
        fake_document(item("caption", "Tab. 2: Treppen"), table_item("", "bis 30"))
    )
    assert [(b.kind, b.caption) for b in blocks] == [("table", "Tab. 2: Treppen")]


def test_a_tied_caption_is_not_handed_to_the_next_table():
    blocks = read_blocks(
        fake_document(
            table_item("Tab. 1: Wege", "bis 5"),
            item("caption", "Tab. 1: Wege"),
            table_item("", "bis 30"),
        )
    )
    assert [b.caption for b in blocks] == ["Tab. 1: Wege", ""]


def test_a_picture_caption_is_not_a_table_caption():
    blocks = read_blocks(fake_document(item("caption", "Abb. 1: Treppe"), table_item("", "a")))
    assert [b.caption for b in blocks] == [""]


def test_an_annex_title_filed_as_a_caption_is_a_heading():
    blocks = read_blocks(fake_document(item("caption", "Anhang 4")))
    assert [(b.kind, b.text) for b in blocks] == [("heading", "Anhang 4")]


def test_a_note_filed_as_a_caption_stays_in_the_text():
    blocks = read_blocks(fake_document(item("caption", "Hinweis: Türen öffnen nach außen.")))
    assert [(b.kind, b.text) for b in blocks] == [("text", "Hinweis: Türen öffnen nach außen.")]


def test_a_formula_keeps_its_text():
    blocks = read_blocks(fake_document(item("formula", "", "D = E / F · 100 %")))
    assert [(b.kind, b.text) for b in blocks] == [("text", "D = E / F · 100 %")]


def test_text_before_the_first_chapter_is_skipped():
    sections = parse_blocks(
        [
            heading("Technische Regeln"),
            text("Die Technischen Regeln geben den Stand der Technik wieder."),
            text("1 Ziel"),
            heading("1 Ziel"),
            text("Diese Regel konkretisiert etwas."),
        ],
        URL,
        "Regel",
    )
    assert [(s.label, s.lines) for s in sections] == [("1", ["Diese Regel konkretisiert etwas."])]


def test_numbered_headings_start_sections_and_keep_their_page():
    found = parse(
        text("Einleitung."),
        heading("4 Oberkapitel", 3),
        heading("4.2 Unterkapitel", 4),
        text("(1) Inhalt.", 4),
    )
    assert "4" not in found
    assert found["4.2"].heading == "Regel, Oberkapitel, Unterkapitel"
    assert found["4.2"].lines == ["(1) Inhalt."]
    assert found["4.2"].page == 4
    assert found["4.2"].url == URL


def test_a_false_heading_stays_in_the_text():
    found = parse(heading("2 Regel"), text("(1) Satz."), heading("Hinweis:"), text("Ein Hinweis."))
    assert found["2"].lines == ["(1) Satz.", "Hinweis:", "Ein Hinweis."]


def test_a_list_item_taken_for_a_heading_stays_in_the_text():
    found = parse(
        heading("4.2 Regel"),
        heading("(3) Die Größe beträgt:"),
        heading("1. 7,75 m² in kleinen Hallen und"),
        heading("2. 8,25 m² in großen Hallen"),
        text("betragen."),
    )
    assert list(found) == ["4.2"]
    assert found["4.2"].lines == [
        "(3) Die Größe beträgt:",
        "1. 7,75 m² in kleinen Hallen und",
        "2. 8,25 m² in großen Hallen",
        "betragen.",
    ]


def test_a_numbered_line_starts_a_section_only_inside_its_chapter():
    found = parse(
        heading("3 Begriffe"),
        text("3.1 Ein Fluchtweg ist ein Weg."),
        text("3.2 Ein Notausgang ist ein Ausgang."),
        heading("4 Regeln"),
        text("3.9 Punkte stehen in der anderen Regel."),
    )
    assert found["3.1"].lines == ["3.1 Ein Fluchtweg ist ein Weg."]
    assert found["3.2"].heading == "Regel, Begriffe"
    assert found["4"].lines == ["3.9 Punkte stehen in der anderen Regel."]


def test_a_section_title_read_as_text_is_still_a_title():
    found = parse(
        heading("4 Wege"),
        heading("4.6 Leitern"),
        text("4.6.3 Schutz gegen Absturz"),
        text("(1) Ein Satz."),
    )
    assert found["4.6.3"].heading == "Regel, Wege, Leitern, Schutz gegen Absturz"
    assert found["4.6.3"].lines == ["(1) Ein Satz."]


def test_literature_is_left_out_but_a_following_annex_is_kept():
    found = parse(
        heading("2 Regel"),
        text("Satz."),
        heading("Literaturhinweise"),
        text("DGUV Information 000-000"),
        table([["Titel", "Jahr"], ["Buch", "2020"]]),
        heading("Anhang 2"),
        heading("Beispiele für Maßnahmen"),
        text("Eine Maßnahme."),
    )
    assert list(found) == ["2", "Anhang 2"]
    assert found["Anhang 2"].heading == "Regel, Beispiele für Maßnahmen"
    assert found["Anhang 2"].lines == ["Eine Maßnahme."]


def test_numbers_inside_an_annex_do_not_collide_with_chapters():
    found = parse(
        heading("2 Regel"),
        text("Satz."),
        heading("Anhang 1"),
        heading("2 Warnzeichen"),
        text("Ein Zeichen."),
    )
    assert found["2"].lines == ["Satz."]
    assert found["Anhang 1 Nr. 2"].lines == ["Ein Zeichen."]


def test_a_captioned_table_is_its_own_section():
    rows = [["Körperhaltung", "leicht"], ["Sitzen", "+20 °C"]]
    found = parse(
        heading("4.2 Temperatur"),
        text("(1) Siehe Tabelle 1."),
        table(rows, "Tabelle 1: Mindestwerte der Temperatur", page=5),
        text("(2) Weiter im Text."),
    )
    assert found["4.2"].lines == ["(1) Siehe Tabelle 1.", "(2) Weiter im Text."]
    section = found["4.2 Tabelle 1"]
    assert section.heading == "Regel, Mindestwerte der Temperatur"
    assert section.lines == ["Körperhaltung | leicht", "Sitzen | +20 °C"]
    assert section.page == 5


def test_a_short_form_caption_is_understood():
    found = parse(heading("5 Wege"), table([["a"], ["b"]], "Tab. 2: Breiten"))
    assert found["5 Tabelle 2"].heading == "Regel, Breiten"


def test_a_caption_with_a_dotted_number_is_understood():
    found = parse(heading("5 Wege"), table([["a"], ["b"]], "Tabelle 5.1: Plätze"))
    assert found["5 Tabelle 5.1"].heading == "Regel, Plätze"


def test_notes_and_a_second_part_stay_with_their_table():
    found = parse(
        heading("5 Wege"),
        text("(1) Siehe Tabelle 1."),
        table([["Nr.", "Breite"], ["7", "2,40"]], "Tab. 1: Breiten"),
        text("Zwischenwerte sind zulässig."),
        heading("*) Hinweis:"),
        text("Gilt für Neubauten."),
        table([["", "Abweichend"], ["8", "0,60"]]),
        heading("Hinweis:"),
        text("Die Werte gelten für die Flucht."),
        text("(2) Weiter."),
    )
    assert found["5 Tabelle 1"].lines == [
        "Nr. | Breite",
        "7 | 2,40",
        "Zwischenwerte sind zulässig.",
        "*) Hinweis:",
        "Gilt für Neubauten.",
        " | Abweichend",
        "8 | 0,60",
        "Hinweis:",
        "Die Werte gelten für die Flucht.",
    ]
    assert found["5"].lines == ["(1) Siehe Tabelle 1.", "(2) Weiter."]


def test_a_real_heading_ends_a_table():
    found = parse(
        heading("5 Wege"),
        table([["a"]], "Tab. 1: Breiten"),
        heading("6 Türen"),
        text("Satz."),
        table([["b"]]),
    )
    assert found["5 Tabelle 1"].lines == ["a"]
    assert found["6"].lines == ["Satz.", "b"]


def test_a_table_without_caption_stays_in_its_absatz():
    found = parse(
        heading("5 Wege"),
        text("(2) Die Länge beträgt:"),
        table([["1.", "für Räume", "bis zu 35 m"]]),
        text("(3) Weiter."),
    )
    assert found["5"].lines == [
        "(2) Die Länge beträgt:",
        "1. | für Räume | bis zu 35 m",
        "(3) Weiter.",
    ]


def test_a_footnote_follows_its_table():
    found = parse(
        heading("5 Wege"),
        table([["a"], ["b *)"]], "Tabelle 1: Titel"),
        text("*) Gilt nur für Neubauten."),
        text("(2) Weiter."),
    )
    assert found["5 Tabelle 1"].lines[-1] == "*) Gilt nur für Neubauten."
    assert found["5"].lines == ["(2) Weiter."]


def test_group_rows_split_a_table_and_it_continues_on_the_next_page():
    header = ["", "Bereich", "Lux"]
    found = parse(
        heading("Anhang 3"),
        heading("Beleuchtungsanforderungen"),
        table(
            [header, ["1 Wege"], ["1.1", "Flure", "50"], ["2 Lager"], ["2.1", "Versand", "300"]],
            page=19,
        ),
        table([header, ["2.2", "Lagerräume", "50"], ["3 Büros"], ["3.1", "Büro", "500"]], page=20),
    )
    assert found["Anhang 3 Nr. 1"].lines == [" | Bereich | Lux", "1.1 | Flure | 50"]
    assert found["Anhang 3 Nr. 1"].heading == "Regel, Beleuchtungsanforderungen, Wege"
    assert found["Anhang 3 Nr. 2"].lines == [
        " | Bereich | Lux",
        "2.1 | Versand | 300",
        "2.2 | Lagerräume | 50",
    ]
    assert found["Anhang 3 Nr. 3"].lines == [" | Bereich | Lux", "3.1 | Büro | 500"]
    assert found["Anhang 3 Nr. 3"].page == 20


def test_chunks_keep_the_page_of_a_pdf_section():
    section = Section("4.2", "Regel, Temperatur", ["Kurzer Satz."], URL, 4)
    chunk = chunk_section("ASR A3.5", section, "GMBl 2022, S. 198", count=lambda t: len(t.split()))[
        0
    ]
    assert chunk["page"] == 4
    assert chunk["id"] == "asr-a3.5-4.2"
