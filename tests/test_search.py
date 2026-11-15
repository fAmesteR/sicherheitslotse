import numpy as np

from lotse.retrieval import bm25, dense

TEXTS = [
    "ArbSchG § 5 Beurteilung der Arbeitsbedingungen: Der Arbeitgeber hat zu ermitteln.",
    "ASR A2.3 Fluchtwege: Fluchtwege müssen frei sein.",
    "TRGS 510 Lagerung: Gefahrstoffe in ortsbeweglichen Behältern.",
    "ArbZG § 4 Ruhepausen: Die Arbeit ist durch Ruhepausen zu unterbrechen.",
]


def test_tokens_keep_rule_numbers_and_paragraph_numbers():
    tokens = bm25.tokenize(["ASR A2.3 und TRGS 510 nach § 5"])[0]
    assert {"a2.3", "510", "§5"} <= set(tokens)


def test_paragraph_number_is_not_confused_with_absatz_number():
    index = bm25.build(
        [
            "ArbSchG § 7 Abs. 5 Übertragung: Absatz fünf.",
            "ArbSchG § 22 Abs. 5 Befugnisse: Absatz fünf.",
            "ArbSchG § 5 Beurteilung der Arbeitsbedingungen: Der Arbeitgeber hat zu ermitteln.",
        ]
    )
    assert bm25.search(index, "§ 5 ArbSchG", k=3)[0] == 2
    assert bm25.search(index, "§5", k=3) == [2]


def test_double_paragraph_sign_and_letter_suffix():
    tokens = bm25.tokenize(["§§ 8 bis 11 und § 24a"])[0]
    assert {"§8", "11", "§24a"} <= set(tokens)


def test_german_stopwords_go_and_german_words_stem():
    tokens = bm25.tokenize(["Die Fluchtwege und der Fluchtweg"])[0]
    assert tokens == ["fluchtweg", "fluchtweg"]


def test_keyword_search_finds_the_exact_rule_number():
    index = bm25.build(TEXTS)
    assert bm25.search(index, "Was steht in TRGS 510?", k=2)[0] == 2
    assert bm25.search(index, "ASR A2.3", k=2)[0] == 1


def test_keyword_search_leaves_out_chunks_without_a_shared_word():
    index = bm25.build(TEXTS)
    assert bm25.search(index, "Ruhepausen", k=4) == [3]
    assert bm25.search(index, "Urlaubsgeld", k=4) == []
    assert bm25.search(index, "und oder", k=4) == []


def test_dense_search_ranks_by_dot_product():
    embeddings = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)
    assert dense.search(embeddings, np.array([0, 1], dtype=np.float32), k=2) == [1, 2]


def test_the_reranker_loads_once_whether_or_not_its_name_is_given(monkeypatch):
    from lotse.config import RERANKER_MODEL
    from lotse.retrieval import rerank

    class Model:
        def to(self, device):
            return self

        def eval(self):
            return self

    loads = []
    monkeypatch.setattr(rerank.AutoTokenizer, "from_pretrained", lambda name: loads.append(name))
    monkeypatch.setattr(
        rerank.AutoModelForSequenceClassification, "from_pretrained", lambda name: Model()
    )
    rerank.loaded.cache_clear()
    try:
        assert rerank.load() is rerank.load(RERANKER_MODEL)
        assert loads == [RERANKER_MODEL]
    finally:
        rerank.loaded.cache_clear()
