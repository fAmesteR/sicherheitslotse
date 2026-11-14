import pytest

from lotse.metrics import has_all_gold, in_section, over_runs, reciprocal_rank, wilson_interval


def chunk(doc: str, section: str) -> dict:
    return {"doc": doc, "section": section}


def test_label_matches_the_section_and_what_sits_under_it():
    assert in_section(chunk("ArbSchG", "§ 5"), "ArbSchG § 5")
    assert in_section(chunk("ArbZG", "§ 7 Abs. 2a"), "ArbZG § 7")
    assert in_section(chunk("ArbStättV", "Anhang Nr. 3.5"), "ArbStättV Anhang Nr. 3")


def test_label_does_not_match_a_neighbour():
    assert not in_section(chunk("ArbSchG", "§ 50"), "ArbSchG § 5")
    assert not in_section(chunk("ArbSchG", "§ 5a"), "ArbSchG § 5")
    assert not in_section(chunk("ArbZG", "§ 5"), "ArbSchG § 5")
    assert not in_section(chunk("ArbStättV", "Anhang Nr. 3"), "ArbStättV Anhang Nr. 3.5")


def test_a_two_source_question_needs_both_parts():
    gold = [["ArbSchG § 5"], ["ArbZG § 4"]]
    assert not has_all_gold([chunk("ArbSchG", "§ 5")], gold)
    assert has_all_gold([chunk("ArbZG", "§ 4"), chunk("ArbSchG", "§ 5 Abs. 1")], gold)


def test_either_alternative_can_supply_a_part():
    gold = [["ASR A3.5 4.2", "ArbStättV Anhang Nr. 3.5"]]
    assert has_all_gold([chunk("ArbStättV", "Anhang Nr. 3.5")], gold)


def test_reciprocal_rank_uses_the_first_gold_chunk():
    chunks = [chunk("ArbZG", "§ 1"), chunk("ArbZG", "§ 4"), chunk("ArbZG", "§ 4 Abs. 2")]
    assert reciprocal_rank(chunks, [["ArbZG § 4"]]) == 0.5
    assert reciprocal_rank(chunks, [["ArbZG § 9"]]) == 0.0


def test_wilson_interval_matches_a_worked_example():
    low, high = wilson_interval(12, 15)
    assert low == pytest.approx(0.5481, abs=1e-3)
    assert high == pytest.approx(0.9295, abs=1e-3)


def test_wilson_interval_stays_inside_zero_and_one():
    assert wilson_interval(0, 10)[0] == pytest.approx(0)
    assert wilson_interval(10, 10)[1] == pytest.approx(1)


def test_a_share_over_several_runs_keeps_each_run_and_the_mean():
    result = over_runs([20, 22, 21], [28, 28, 28])
    assert result["share"] == 0.75 and result["n"] == 28
    assert result["per_run"] == [0.714, 0.786, 0.75]
    low, high = result["interval_95"]
    assert low < 0.75 < high


def test_a_share_with_nothing_to_count_is_empty():
    assert over_runs([0, 0], [0, 0]) == {"share": None, "n": 0}
