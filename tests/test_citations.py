from lotse.generation.answer import Answer, Citation, check, numbers, sentences

LAW = {
    "doc": "ArbZG",
    "section": "§ 3",
    "heading": "Arbeitszeit der Arbeitnehmer",
    "text": "Die werktägliche Arbeitszeit der Arbeitnehmer darf acht Stunden nicht überschreiten. "
    "Sie kann auf bis zu zehn Stunden nur verlängert werden, wenn innerhalb von sechs "
    "Kalendermonaten oder innerhalb von 24 Wochen im Durchschnitt acht Stunden werktäglich nicht "
    "überschritten werden.",
}
# Invented, in the style of a technical rule.
RULE = {
    "doc": "Musterregel M1.2",
    "section": "4.1",
    "heading": "Größe der Räume",
    "text": "(1) Je Arbeitsplatz ist eine Fläche von mindestens 1,25 m² frei zu halten.\n"
    "(2) Die lichte Höhe beträgt in kleinen Hallen 2,50 m – in großen Hallen 15 m.",
}
CHUNKS = [LAW, RULE]
LAW_QUOTE = (1, "darf acht Stunden nicht überschreiten")
RULE_QUOTE = (2, "Fläche von mindestens 1,25 m²")


def answer(text: str, *citations: tuple[int, str]) -> Answer:
    quotes = [Citation(source=source, quote=quote) for source, quote in citations]
    return Answer(answerable=True, answer=text, citations=quotes)


def rejected(text: str, *citations: tuple[int, str], chunks: list[dict] = CHUNKS) -> str:
    return check(answer(text, *citations), chunks) or "passed"


def test_a_sourced_answer_passes():
    text = "Die Arbeitszeit darf acht Stunden am Tag nicht überschreiten [1]."
    assert check(answer(text, LAW_QUOTE), CHUNKS) is None


def test_an_answer_needs_citations_and_text():
    assert "fehlen Zitate" in rejected("Die Arbeitszeit ist begrenzt [1].")
    assert "leer" in rejected("", LAW_QUOTE)
    assert "leer" in rejected(" [1] ", LAW_QUOTE)


def test_every_sentence_needs_a_marker():
    text = "Die Arbeitszeit darf acht Stunden nicht überschreiten [1]. Das gilt auch am Samstag."
    assert "„Das gilt auch am Samstag.“ hat keine Quellenangabe" in rejected(text, LAW_QUOTE)


def test_a_marker_after_the_full_stop_belongs_to_the_sentence_before_it():
    text = (
        "Die Arbeitszeit darf acht Stunden nicht überschreiten. [1] Zehn Stunden sind möglich. [1]"
    )
    assert check(answer(text, LAW_QUOTE), CHUNKS) is None


def test_every_marker_needs_a_citation_for_that_source():
    text = "Die Arbeitszeit ist begrenzt [1]. Je Arbeitsplatz sind 1,25 m² frei zu halten [2]."
    assert "Zu [2] fehlt ein Zitat" in rejected(text, LAW_QUOTE)
    assert check(answer(text, LAW_QUOTE, RULE_QUOTE), CHUNKS) is None


def test_a_source_must_be_a_passage_that_was_sent():
    text = "Die Arbeitszeit ist begrenzt [1]."
    assert "Quelle [3] gibt es nicht" in rejected(text, (3, "darf acht Stunden nicht"))
    assert "Quelle [0] gibt es nicht" in rejected(text, (0, "darf acht Stunden nicht"))


def test_a_quote_needs_three_words():
    assert "weniger als drei Wörter" in rejected("Es gilt eine Grenze [1].", (1, "acht Stunden"))


def test_a_quote_must_be_word_for_word_from_its_own_passage():
    text = "Die Arbeitszeit ist begrenzt [1]."
    for quote in [
        "darf acht Stunden nicht übersteigen",  # one word changed
        "Die werktägliche Arbeitszeit ... nicht überschreiten",  # words left out
        "Fläche von mindestens 1,25 m²",  # from the other passage
    ]:
        assert "steht nicht wörtlich" in rejected(text, (1, quote))


def test_a_heading_is_no_quote():
    text = "Es gibt Regeln zur Raumgröße [2]."
    assert "steht nicht wörtlich" in rejected(text, (2, "Größe der Räume"))


def test_a_quote_survives_typography():
    chunk = dict(LAW, text="Der Arbeit\u00adgeber hat „geeignete Maßnahmen“\nzu\u00a0treffen.")
    quote = (1, '"geeignete Maßnahmen" zu treffen.')
    assert check(answer("Der Arbeitgeber muss handeln [1].", quote), [chunk]) is None
    dash = (2, "in kleinen Hallen 2,50 m - in großen Hallen 15 m")
    assert check(answer("Die Höhe hängt von der Halle ab [2].", dash), CHUNKS) is None


def test_a_quote_may_differ_in_case_and_spacing():
    chunk = dict(LAW, text="Insbesondere gilt die Verordnung (EG)Nr. 1907/2006 für alle Stoffe.")
    quote = (1, "insbesondere gilt die Verordnung (EG) Nr. 1907/2006")
    assert check(answer("Die Verordnung gilt für alle Stoffe [1].", quote), [chunk]) is None


LIST = dict(
    RULE,
    text="Zu dokumentieren sind:\n1. die Adressen der Unterkünfte,\n2. die Zahl der Betten\n"
    "a) je Raum und\nb) je Haus.\nDie Höhe beträgt 20. Danach gilt etwas anderes.",
)


def test_a_quote_may_leave_out_list_labels_but_no_words():
    text = "Adressen und Bettenzahl sind zu dokumentieren [1]."
    for quote in [
        "Zu dokumentieren sind: die Adressen der Unterkünfte, die Zahl der Betten je Raum",
        "sind: 1. die Adressen der Unterkünfte, 2. die Zahl der Betten",  # labels kept
        "sind: 1. die Adressen der Unterkünfte, die Zahl der Betten",  # one kept, one dropped
    ]:
        assert check(answer(text, (1, quote)), [LIST]) is None, quote
    dropped_word = "Zu dokumentieren sind: die Adressen, die Zahl der Betten"
    assert "steht nicht wörtlich" in rejected(text, (1, dropped_word), chunks=[LIST])
    assert "weniger als drei Wörter" in rejected(text, (1, "1. die Adressen"), chunks=[LIST])


def test_a_number_that_ends_a_sentence_is_no_list_label():
    text = "Es gilt eine Höhe [1]."
    assert check(answer(text, (1, "Die Höhe beträgt 20")), [LIST]) is None
    assert check(answer(text, (1, "Die Höhe beträgt 20. Danach gilt etwas")), [LIST]) is None
    assert "steht nicht wörtlich" in rejected(text, (1, "Die Höhe beträgt Danach"), chunks=[LIST])


def test_an_invented_number_fails():
    assert "Die Zahl 12" in rejected("Erlaubt sind höchstens 12 Stunden [1].", LAW_QUOTE)
    assert "Die Zahl 9" in rejected("Erlaubt sind höchstens neun Stunden [1].", LAW_QUOTE)


def test_digits_and_number_words_are_the_same_value():
    text = "Erlaubt sind 8 Stunden, mit Ausgleich in 24 Wochen bis zu 10 Stunden [1]."
    assert check(answer(text, LAW_QUOTE), CHUNKS) is None


def test_every_third_year_is_every_three_years():
    chunk = dict(LAW, text="Die Schutzmaßnahmen sind mindestens jedes dritte Jahr zu überprüfen.")
    quote = (1, "mindestens jedes dritte Jahr")
    assert check(answer("Geprüft wird mindestens alle drei Jahre [1].", quote), [chunk]) is None
    assert "Die Zahl 2" in rejected("Geprüft wird alle zwei Jahre [1].", quote, chunks=[chunk])
    assert numbers("zum Schutz Dritter ist auf Abstand zu achten") == set()


def test_a_number_must_come_from_a_source_its_sentence_names():
    quotes = (LAW_QUOTE, RULE_QUOTE)
    assert "Die Zahl 1,25" in rejected("Frei zu halten sind 1,25 m² [1].", *quotes)
    assert check(answer("Frei zu halten sind 1,25 m² [2].", *quotes), CHUNKS) is None
    assert check(answer("Es gelten 8 Stunden und 1,25 m² [1][2].", *quotes), CHUNKS) is None
    assert check(answer("Es gelten 8 Stunden und 1,25 m² [1, 2].", *quotes), CHUNKS) is None


def test_a_number_from_the_question_may_be_repeated():
    text = "Auch bei 12 Stunden gilt die Grenze von acht Stunden [1]."
    assert "Die Zahl 12" in rejected(text, LAW_QUOTE)
    question = "Darf ich zwölf Stunden am Stück arbeiten?"
    assert check(answer(text, LAW_QUOTE), CHUNKS, question) is None


def test_paragraph_and_rule_numbers_come_from_the_context_prefix():
    assert check(answer("Nach § 3 ArbZG gelten acht Stunden [1].", LAW_QUOTE), CHUNKS) is None
    assert "Die Zahl 4" in rejected("Nach § 4 ArbZG gelten acht Stunden [1].", LAW_QUOTE)
    text = "Nach M1.2 Nummer 4.1 sind 1,25 m² frei zu halten [2]."
    assert check(answer(text, RULE_QUOTE), CHUNKS) is None


def test_part_of_a_number_is_not_that_number():
    assert "Die Zahl 25" in rejected("Frei zu halten sind 25 m² [2].", RULE_QUOTE)
    assert "Die Zahl 5" in rejected("Die Höhe beträgt 5 m [2].", RULE_QUOTE)


def test_the_power_in_a_unit_is_no_number():
    assert numbers("Je Person sind 6 m² oder 15 m³ nötig [3].") == {"6", "15"}
    chunk = dict(RULE, section="5", text="Je Person ist eine Fläche von 6 m² nötig.")
    quote = (1, "eine Fläche von 6 m²")
    assert "Die Zahl 2" in rejected("Nötig sind 2 Räume [1].", quote, chunks=[chunk])


def test_abbreviations_and_dates_do_not_end_a_sentence():
    text = "Nach § 3 Abs. 1 Nr. 2 gilt das z. B. für Hallen bzw. Lager ab dem 1. Januar 2026 [1]."
    assert sentences(text) == [text]


def test_when_in_doubt_a_sentence_is_split():
    assert len(sentences("Das gilt immer [1]. Ausnahmen gibt es nicht.")) == 2
    assert len(sentences("Die Höhe beträgt 20. Danach gilt etwas anderes [1].")) == 2
    assert len(sentences('Die Regel lautet: "Das gilt." Danach folgt nichts [1].')) == 2
    assert sentences("Erster Punkt [1]\n\n- Zweiter Punkt [2]\n") == [
        "Erster Punkt [1]",
        "- Zweiter Punkt [2]",
    ]
