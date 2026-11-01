from lotse.retrieval.hybrid import reciprocal_rank_fusion


def test_item_in_both_lists_beats_item_on_top_of_one():
    assert reciprocal_rank_fusion([[1, 2], [3, 2]]) == [2, 1, 3]


def test_scores_follow_the_formula():
    # Item 2 scores 1/62 + 1/61, item 1 scores 1/61 + 1/69, item 3 scores 1/62 alone.
    merged = reciprocal_rank_fusion([[1, 2], [2, 3, 4, 5, 6, 7, 8, 9, 1]], k=60)
    assert merged[:3] == [2, 1, 3]


def test_ties_keep_first_seen_order():
    assert reciprocal_rank_fusion([[1], [2]]) == [1, 2]


def test_limit_cuts_the_list():
    assert reciprocal_rank_fusion([[1, 2, 3], [4, 5, 6]], limit=2) == [1, 4]


def test_one_empty_list_changes_nothing():
    assert reciprocal_rank_fusion([[], [7, 8]]) == [7, 8]
