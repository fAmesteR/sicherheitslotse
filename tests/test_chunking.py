"""The XML snippets below are cut down from the real gesetze-im-internet.de files."""

from lxml import etree

from lotse.chunking import chunk_section, context_prefix
from lotse.config import MAX_CHUNK_TOKENS
from lotse.parsing.laws_xml import (
    Section,
    annex_sections,
    flatten,
    page_name,
    parse_law,
    table_lines,
    version_date,
)


def word_count(text: str) -> int:
    return len(text.split())


def content(xml: str) -> etree._Element:
    return etree.fromstring(f"<Content>{xml}</Content>")


def law(*norms: str) -> etree._Element:
    return etree.fromstring(f"<dokumente>{''.join(norms)}</dokumente>")


def norm(enbez: str, body: str, titel: str = "") -> str:
    return (
        f"<norm><metadaten><enbez>{enbez}</enbez><titel>{titel}</titel></metadaten>"
        f"<textdaten><text><Content>{body}</Content></text></textdaten></norm>"
    )


def annex(xml: str, name: str = "Anhang"):
    return annex_sections(name, "", flatten(content(xml)))


def test_flatten_gives_each_list_item_its_own_line_and_drops_notes():
    lines = flatten(
        content(
            "<P><noindex>(Fundstelle: BGBl. I)</noindex>(1) Muss<FnR ID='x'/> sein:"
            "<DL><DT>1.</DT><DD><LA>eins,</LA></DD><DT>2.</DT><DD><LA>zwei</LA></DD></DL>"
            "und weiter.<BR/>(2) Satz.</P>"
        )
    )
    assert [line.text for line in lines] == [
        "(1) Muss sein:",
        "1. eins,",
        "2. zwei",
        "und weiter.",
        "(2) Satz.",
    ]


def test_annex_entries_use_their_number_and_skip_the_annex_contents_table():
    sections = annex(
        "<P><table><tgroup cols='2'><tbody><row><entry>3.5</entry><entry>Raumtemperatur</entry>"
        "</row></tbody></tgroup></table></P>"
        "<P><DL><DT/><DD><LA><DL><DT><B>3</B></DT><DD><LA><B>Arbeitsbedingungen</B></LA></DD>"
        "<DT><B>3.5</B></DT><DD><LA><B>Raumtemperatur</B></LA></DD></DL></LA></DD></DL>"
        "(1) Arbeitsräume müssen warm sein.</P>"
    )
    assert [s.label for s in sections if s.lines] == ["Anhang Nr. 3.5"]
    assert sections[-1].heading == "Arbeitsbedingungen – Raumtemperatur"
    assert sections[-1].lines == ["(1) Arbeitsräume müssen warm sein."]


def test_plain_list_items_stay_inside_their_entry():
    sections = annex(
        "<P><B>1.1 Anwendungsbereich</B></P>"
        "<P>(1) Gilt für"
        "<DL><DT>1.</DT><DD><LA>Flucht-</LA></DD><DT>2.</DT><DD><LA>und Rettungswege.</LA></DD>"
        "</DL></P>"
    )
    assert [s.label for s in sections] == ["Anhang Nr. 1.1"]
    assert sections[0].lines == ["(1) Gilt für", "1. Flucht-", "2. und Rettungswege."]


def test_numbering_restarts_per_abschnitt_and_keeps_it_in_the_label():
    sections = annex(
        "<Title>Abschnitt 1</Title><Title>Krane</Title>"
        "<P><DL><DT><B>1.</B></DT><DD><LA><B>Ziel</B></LA></DD>"
        "<DT>1.1</DT><DD><LA>Dieser Abschnitt gilt.</LA></DD></DL></P>"
        "<Title>Abschnitt 2</Title><Title>Gasanlagen</Title>"
        "<P><DL><DT>1.1</DT><DD><LA>Der gilt auch.</LA></DD></DL></P>",
        "Anhang 3",
    )
    found = [s for s in sections if s.lines]
    assert [s.label for s in found] == [
        "Anhang 3 Abschnitt 1 Nr. 1.1",
        "Anhang 3 Abschnitt 2 Nr. 1.1",
    ]
    assert found[0].heading == "Krane – Ziel"
    assert found[1].heading == "Gasanlagen"


def test_short_numbered_line_is_a_title_for_the_entries_below_it():
    sections = annex(
        "<P><DL><DT>3.1</DT><DD><LA>Allgemeine Mindestanforderungen</LA></DD>"
        "<DT>3.1.1</DT><DD><LA>Gerüste sind zu prüfen.</LA></DD></DL></P>"
    )
    found = [s for s in sections if s.lines]
    assert [s.label for s in found] == ["Anhang Nr. 3.1.1"]
    assert found[0].heading == "Allgemeine Mindestanforderungen"


def test_nummer_title_and_abs_without_entries():
    sections = annex(
        "<Title>Nummer 4</Title><Title><B>Kühlschmierstoffe</B></Title><P>(1) Sind verboten.</P>"
    )
    assert [(s.label, s.heading) for s in sections] == [("Anhang Nr. 4", "Kühlschmierstoffe")]


def table_xml(caption: str, head: list[str], *rows: list[str]) -> str:
    def cells(row: list[str]) -> str:
        return "<row>" + "".join(f"<entry>{cell}</entry>" for cell in row) + "</row>"

    return (
        f"<table><tgroup cols='1'><tbody>{cells([caption])}</tbody></tgroup></table>"
        f"<table><tgroup cols='{len(head)}'><thead>{cells(head)}</thead>"
        f"<tbody>{''.join(cells(row) for row in rows)}</tbody></tgroup></table>"
    )


def test_captioned_table_is_cited_by_its_number_not_by_the_entry_before_it():
    sections = annex(
        "<P><DL><DT>3.4</DT><DD><LA>Krane sind zu prüfen.</LA></DD></DL></P>"
        + table_xml("Tabelle 1 Prüffristen", ["Kran", "Frist"], ["Turmdreh", "2 Jahre"])
    )
    assert [s.label for s in sections] == ["Anhang Nr. 3.4", "Anhang Tabelle 1"]
    assert sections[-1].heading == "Prüffristen"
    assert sections[-1].lines == ["Kran | Frist", "Turmdreh | 2 Jahre"]


def test_text_after_a_table_goes_back_to_the_entry_it_interrupted():
    sections = annex(
        "<P><B>5.3 Lagerung</B></P><P>(7) Davor.</P>"
        + table_xml("Tabelle 1", ["Gruppe", "Anteil"], ["A", "90 %"])
        + "<P>(8) Danach.</P>"
    )
    entry, table = sections
    assert (entry.label, entry.lines) == ("Anhang Nr. 5.3", ["(7) Davor.", "(8) Danach."])
    assert (table.label, table.lines) == ("Anhang Tabelle 1", ["Gruppe | Anteil", "A | 90 %"])


def test_numbered_table_rows_become_sections_that_keep_device_and_header():
    sections = annex(
        table_xml(
            "Tabelle 12 Prüfanforderungen",
            ["Nr.", "Frist"],
            ["7.2", "Wärmepumpen"],
            ["a)", "10 Jahre"],
            ["7.3", "Wärmeerzeuger"],
        )
    )
    assert [(s.label, s.heading, s.lines) for s in sections] == [
        (
            "Anhang Tabelle 12 Ziffer 7.2",
            "Prüfanforderungen – Wärmepumpen",
            ["Nr. | Frist", "7.2 | Wärmepumpen", "a) | 10 Jahre"],
        ),
        (
            "Anhang Tabelle 12 Ziffer 7.3",
            "Prüfanforderungen – Wärmeerzeuger",
            ["Nr. | Frist", "7.3 | Wärmeerzeuger"],
        ),
    ]


def test_title_wrapped_over_two_lines_is_joined_but_a_bold_absatz_is_not_a_title():
    wrapped = annex(
        "<Title>Nummer 4</Title><Title>Begasung mit</Title><Title>Biozid-Produkten</Title>"
        "<P>Regel.</P>"
    )
    assert [(s.heading, s.lines) for s in wrapped] == [
        ("Begasung mit Biozid-Produkten", ["Regel."])
    ]

    teil = annex("<P><B>Teil 1</B><BR/><B>Gefahrstoffe</B><BR/><B>(1) Pflichtvorsorge bei:</B></P>")
    assert [(s.label, s.heading, s.lines) for s in teil] == [
        ("Anhang Teil 1", "Gefahrstoffe", ["(1) Pflichtvorsorge bei:"])
    ]


def test_annex_without_numbering_stays_whole():
    sections = annex("<P>Merkmale:</P><P>(1) Gewicht</P>")
    assert [(s.label, s.lines) for s in sections] == [("Anhang", ["Merkmale:", "(1) Gewicht"])]


def test_merged_cells_keep_rows_self_contained():
    table = etree.fromstring(
        "<table><tgroup cols='3'>"
        "<colspec colname='c1'/><colspec colname='c2'/><colspec colname='c3'/>"
        "<thead><row><entry morerows='1'>Anlage</entry><entry namest='c2' nameend='c3'>Prüfung"
        "</entry></row><row><entry colname='c2'>innen</entry><entry colname='c3'>außen</entry>"
        "</row></thead>"
        "<tbody><row><entry morerows='1'>Kessel</entry><entry>5 Jahre</entry><entry>2 Jahre</entry>"
        "</row><row><entry>10 Jahre</entry><entry namest='c3' nameend='c3'>entfällt</entry></row>"
        "</tbody></tgroup></table>"
    )
    assert [line.text for line in table_lines(table)] == [
        "Anlage | Prüfung innen | außen",
        "Kessel | 5 Jahre | 2 Jahre",
        "Kessel | 10 Jahre | entfällt",
    ]


def test_parse_law_skips_everything_without_rule_text():
    root = law(
        "<norm><metadaten/><textdaten><text><Content><P>Präambel</P></Content></text></textdaten>"
        "</norm>",
        norm("Inhaltsübersicht", "<P>§ 1 Ziel</P>"),
        norm("Eingangsformel", "<P>Der Bundestag hat beschlossen:</P>"),
        norm("§ 1", "<P>(1) Regel.</P><P>(2) (weggefallen)</P>", "Ziel"),
        norm("§ 2", "", "(weggefallen)"),
        norm("§ 3", "<P>(weggefallen)</P>"),
        norm("§ 21", "<P>-</P>"),
        norm("Anhang EV", "<P><noindex>Auszug aus dem Einigungsvertrag</noindex></P>"),
    )
    sections = parse_law(root, "x")
    assert [s.label for s in sections] == ["§ 1"]
    assert sections[0].lines == ["(1) Regel."]


def test_annex_title_loses_its_reference_and_contents_note():
    root = law(
        norm("Anhang 1", "<P>Text.</P>", "(zu § 6 Absatz 1)<BR/>Besondere Vorschriften"),
        norm("Anhang", "<P>Text.</P>", "Anforderungen<BR/>Inhaltsübersicht"),
    )
    assert [s.heading for s in parse_law(root, "x")] == ["Besondere Vorschriften", "Anforderungen"]


def test_page_names_match_the_site():
    assert page_name("§ 24a") == "__24a.html"
    assert page_name("Anhang") == "anhang.html"
    assert page_name("Anhang II") == "anhang_ii.html"


def test_version_date_ignores_amendments_not_in_the_text():
    root = etree.fromstring(
        "<dokumente><norm><metadaten>"
        "<standangabe><standtyp>Stand</standtyp><standkommentar>Zuletzt geändert durch Art. 7 G "
        "v. 22.12.2025 I Nr. 369</standkommentar></standangabe>"
        "<standangabe><standtyp>Hinweis</standtyp><standkommentar>Änderung durch Art. 1 G "
        "v. 1.2.2026 noch nicht berücksichtigt</standkommentar></standangabe>"
        "</metadaten></norm></dokumente>"
    )
    assert version_date(root) == "2025-12-22"


def test_short_section_stays_one_chunk_and_matches_the_spec_format():
    section = Section("§ 5", "Beurteilung", ["(1) Kurz.", "(2) Auch."], "https://x/__5.html")
    chunk = chunk_section("ArbSchG", section, "2025-12-22", word_count)[0]
    assert chunk == {
        "id": "arbschg-p5",
        "doc": "ArbSchG",
        "section": "§ 5",
        "heading": "Beurteilung",
        "text": "(1) Kurz.\n(2) Auch.",
        "url": "https://x/__5.html",
        "page": None,
        "version_date": "2025-12-22",
    }


def test_long_section_splits_by_absatz():
    lines = [f"({i}) " + "wort " * 200 for i in (1, 2, "2a")]
    chunks = chunk_section("ArbZG", Section("§ 7", "Regeln", lines), "2024", word_count)
    assert [c["section"] for c in chunks] == ["§ 7 Abs. 1", "§ 7 Abs. 2", "§ 7 Abs. 2a"]
    assert [c["id"] for c in chunks] == ["arbzg-p7-a1", "arbzg-p7-a2", "arbzg-p7-a2a"]


def test_every_part_of_a_long_table_repeats_the_header_and_stays_in_budget():
    rows = ["Anlage | Frist"] + [f"Anlage{i} | " + "wort " * 45 for i in range(30)]
    chunks = chunk_section(
        "BetrSichV", Section("Anhang 2 Tabelle 12", "", rows), "2025", word_count
    )
    assert len(chunks) > 1
    assert len({c["id"] for c in chunks}) == len(chunks)
    assert all(c["text"].startswith("Anlage | Frist\n") for c in chunks)
    for chunk in chunks:
        prefix = context_prefix(chunk["doc"], chunk["section"], chunk["heading"])
        assert word_count(prefix + chunk["text"]) <= MAX_CHUNK_TOKENS
    assert sum(f"Anlage{i} " in c["text"] for c in chunks for i in range(30)) == 30


def test_a_single_overlong_sentence_is_still_cut_to_budget():
    chunks = chunk_section("X", Section("§ 1", "", ["wort " * 900]), "2025", word_count)
    assert len(chunks) > 1
    assert all(word_count(c["text"]) <= MAX_CHUNK_TOKENS for c in chunks)
    assert sum(word_count(c["text"]) for c in chunks) == 900
