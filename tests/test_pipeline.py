import numpy as np

from lotse.chunking import passage
from lotse.generation.answer import NO_ANSWER, UNVERIFIED, Answer, Citation
from lotse.pipeline import answer
from lotse.retrieval import bm25, hybrid
from lotse.retrieval.index import Index, load_index, save_subset

LAW = {
    "doc": "ArbZG",
    "section": "§ 3",
    "heading": "Arbeitszeit der Arbeitnehmer",
    "text": "Die werktägliche Arbeitszeit der Arbeitnehmer darf acht Stunden nicht überschreiten.",
}
RANKED = [(2.5, LAW)]
QUOTE = Citation(source=1, quote="darf acht Stunden nicht überschreiten")
GOOD = Answer(answerable=True, answer="Erlaubt sind acht Stunden [1].", citations=[QUOTE])
BAD = Answer(answerable=True, answer="Erlaubt sind zwölf Stunden [1].", citations=[QUOTE])


def model(*replies: Answer | None):
    """A stand-in for the API that hands out the given replies and records what it was sent."""
    calls: list[list[dict]] = []
    queue = list(replies)

    def complete(messages: list[dict]) -> Answer | None:
        calls.append(messages)
        return queue.pop(0)

    return complete, calls


def roles(messages: list[dict]) -> list[str]:
    return [message["role"] for message in messages]


def test_a_checked_answer_is_shown_with_its_sources():
    complete, calls = model(GOOD)
    result = answer("Wie lange darf ich arbeiten?", RANKED, complete)
    assert (result.status, result.text) == ("answered", GOOD.answer)
    assert result.citations == [QUOTE] and result.passages == [LAW]
    assert len(calls) == 1
    assert "[1] ArbZG § 3 Arbeitszeit der Arbeitnehmer: Die werktägliche" in calls[0][1]["content"]
    assert calls[0][1]["content"].endswith("Frage: Wie lange darf ich arbeiten?")


def test_a_rejected_answer_gets_one_retry_that_names_the_reason():
    complete, calls = model(BAD, GOOD)
    result = answer("Wie lange darf ich arbeiten?", RANKED, complete)
    assert result.status == "answered" and len(result.failures) == 1
    assert roles(calls[1]) == ["system", "user", "assistant", "user"]
    assert "zwölf Stunden" in calls[1][2]["content"]
    assert "Die Zahl 12" in calls[1][3]["content"]


def test_a_second_failure_shows_no_answer():
    complete, calls = model(BAD, BAD)
    result = answer("Wie lange darf ich arbeiten?", RANKED, complete)
    assert (result.status, result.text, result.citations) == ("unverified", UNVERIFIED, [])
    assert len(calls) == 2 and len(result.failures) == 2


def test_the_model_can_say_that_the_sources_do_not_answer():
    complete, _ = model(Answer(answerable=False, answer="", citations=[]))
    result = answer("Wie viele Urlaubstage habe ich?", RANKED, complete)
    assert (result.status, result.text) == ("no_answer", NO_ANSWER)


def test_a_cut_off_reply_is_retried_in_a_single_user_turn():
    complete, calls = model(None, GOOD)
    result = answer("Wie lange darf ich arbeiten?", RANKED, complete)
    assert result.status == "answered"
    assert roles(calls[1]) == ["system", "user"]
    assert "kein vollständiges JSON" in calls[1][1]["content"]


def test_a_low_rerank_score_refuses_without_an_api_call():
    complete, calls = model(GOOD)
    result = answer("Wie backe ich Brot?", RANKED, complete, threshold=3.0)
    assert (result.status, result.text, calls) == ("no_answer", NO_ANSWER, [])
    assert (
        answer("Wie lange darf ich arbeiten?", RANKED, complete, threshold=2.0).status == "answered"
    )


def test_search_only_returns_documents_cleared_for_the_api():
    chunks = [
        {"doc": "Musterregel M1.2", "section": "5", "heading": "", "text": "Fluchtwege sind frei."},
        {"doc": "ArbZG", "section": "§ 1", "heading": "", "text": "Fluchtwege im Betrieb."},
        {"doc": "ArbZG", "section": "§ 4", "heading": "", "text": "Ruhepausen sind Pflicht."},
    ]
    embeddings = np.array([[1, 0, 0], [0.5, 0.5, 0], [0, 0, 1]], dtype=np.float32)
    index = Index(chunks, embeddings, bm25.build([passage(chunk) for chunk in chunks]))
    question = np.array([1, 0, 0], dtype=np.float32)
    assert hybrid.search(index, "Fluchtwege", question, frozenset({"ArbZG"})) == [1, 2]


def test_a_published_index_holds_only_the_documents_it_was_given(tmp_path):
    chunks = [
        {"doc": "Musterregel M1.2", "section": "5", "heading": "", "text": "Fluchtwege sind frei."},
        {"doc": "ArbZG", "section": "§ 4", "heading": "", "text": "Ruhepausen sind Pflicht."},
    ]
    embeddings = np.array([[1, 0], [0, 1]], dtype=np.float32)
    index = Index(chunks, embeddings, bm25.build([passage(chunk) for chunk in chunks]))
    save_subset(index, frozenset({"ArbZG"}), tmp_path)
    saved = load_index(tmp_path)
    assert saved.chunks == chunks[1:] and saved.embeddings.tolist() == [[0, 1]]
    assert bm25.search(saved.bm25, "Ruhepausen", k=1) == [0]
    assert "Musterregel" not in "".join(file.read_text("latin-1") for file in tmp_path.rglob("*.*"))
