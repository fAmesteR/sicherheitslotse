from lotse.demo import (
    MAX_QUESTION_CHARS,
    TOO_LONG,
    TOO_MANY,
    Demo,
    RateLimit,
    footer,
    render,
    visitor,
)
from lotse.generation.answer import NO_ANSWER, UNVERIFIED, Answer, Citation, check, locate
from lotse.pipeline import Result

LAW = {
    "doc": "ArbZG",
    "section": "§ 4",
    "heading": "Ruhepausen",
    "text": "Die Arbeit ist durch Ruhepausen zu unterbrechen, wenn\na) sie <lang> dauert,\n"
    "b) der Arbeitnehmer es verlangt.",
    "url": "https://example.org/arbzg/__4.html",
    "version_date": "2025-12-22",
}
QUOTE = Citation(source=1, quote="durch Ruhepausen zu unterbrechen")
ANSWERED = Result("answered", "Pausen sind Pflicht [1].", [QUOTE], [LAW])


def counting(result: Result):
    """A stand-in for the pipeline that records the questions it was asked."""
    asked: list[str] = []

    def ask(question: str) -> Result:
        asked.append(question)
        return result

    return ask, asked


def test_a_quote_is_found_where_it_stands():
    start, end = locate(LAW["text"], "durch  ruhepausen zu Unterbrechen.")
    assert LAW["text"][start:end] == "durch Ruhepausen zu unterbrechen"
    assert locate(LAW["text"], "durch Pausen zu unterbrechen") is None


def test_a_quote_the_check_accepts_without_list_labels_is_found_too():
    quote = "zu unterbrechen, wenn sie <lang> dauert, der Arbeitnehmer"
    answer = Answer(
        answerable=True, answer="Pausen [1].", citations=[Citation(source=1, quote=quote)]
    )
    assert check(answer, [LAW]) is None
    start, end = locate(LAW["text"], quote)
    assert LAW["text"][start:end].startswith("zu unterbrechen, wenn\na) sie")
    assert LAW["text"][start:end].endswith("b) der Arbeitnehmer")


def test_the_limit_counts_per_visitor_and_forgets_after_the_window():
    now = [0.0]
    limit = RateLimit(limit=2, window=600, clock=lambda: now[0])
    assert [limit.allow("a"), limit.allow("a"), limit.allow("a")] == [True, True, False]
    assert limit.allow("b")
    now[0] = 601
    assert limit.allow("a")
    assert "b" not in limit.asked


def test_behind_a_proxy_the_visitor_is_the_last_public_forwarded_address():
    # The visitor wrote 1.2.3.4 into the header; the proxy appended the address it really saw.
    assert visitor("10.0.0.5", "1.2.3.4, 93.184.216.34, 10.0.0.9") == "93.184.216.34"
    assert visitor("10.0.0.5", "2a00:1450::1") == "2a00:1450::1"


def test_a_direct_or_unknown_connection_is_taken_as_it_is():
    assert visitor("93.184.216.34", "1.2.3.4") == "93.184.216.34"  # a header anyone can set
    assert visitor("127.0.0.1", "") == "127.0.0.1"
    assert visitor("10.0.0.5", "unknown, 192.168.1.2") == "10.0.0.5"


def test_the_source_is_shown_with_its_quote_marked_and_its_text_escaped():
    page = render(ANSWERED)
    assert "Pausen sind Pflicht [1]." in page
    assert "<b>[1] ArbZG § 4</b> · Ruhepausen · Stand 22.12.2025" in page
    assert 'href="https://example.org/arbzg/__4.html"' in page
    assert ">durch Ruhepausen zu unterbrechen</mark>" in page
    assert "&lt;lang&gt;" in page and "<lang>" not in page


def test_overlapping_quotes_are_marked_once():
    quotes = [QUOTE, Citation(source=1, quote="Ruhepausen zu unterbrechen, wenn")]
    page = render(Result("answered", "Pausen [1].", quotes, [LAW]))
    assert page.count("<mark") == 2 and page.count("Ruhepausen zu unterbrechen") == 1


def test_an_answer_is_reused_for_the_same_question_and_costs_no_quota():
    ask, asked = counting(ANSWERED)
    demo = Demo(ask, RateLimit(limit=1))
    first = demo.respond("Muss ich Pause machen?", "a")
    assert demo.respond("  muss ich  PAUSE machen? ", "a") == first
    assert asked == ["Muss ich Pause machen?"]
    assert demo.respond("Und am Sonntag?", "a") == f"<p>{TOO_MANY}</p>"


def test_a_rejected_answer_is_explained_and_asked_again_next_time():
    ask, asked = counting(Result("unverified", UNVERIFIED))
    demo = Demo(ask)
    assert UNVERIFIED in demo.respond("Frage?", "a") and "nicht angezeigt" in demo.respond(
        "Frage?", "a"
    )
    assert len(asked) == 2


def test_a_refusal_names_what_is_searched():
    ask, _ = counting(Result("no_answer", NO_ANSWER))
    assert "nur die unten genannten" in Demo(ask).respond("Wie wird das Wetter?", "a")


def test_a_long_or_empty_question_never_reaches_the_pipeline():
    ask, asked = counting(ANSWERED)
    demo = Demo(ask)
    assert demo.respond("x" * (MAX_QUESTION_CHARS + 1), "a") == f"<p>{TOO_LONG}</p>"
    assert demo.respond("   ", "a") == ""
    assert asked == []


def test_a_failure_shows_a_notice_instead_of_an_error():
    def ask(question: str) -> Result:
        raise ConnectionError("no network")

    demo = Demo(ask)
    assert "konnte gerade nicht" in demo.respond("Frage?", "a")
    assert demo.answers == {}


def test_the_footer_counts_the_documents_that_are_searched():
    text = footer()
    assert "11 Dokumente, Stand " in text and "ArbZG" in text and "ASR" not in text
