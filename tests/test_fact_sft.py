from datagen.tree import facts
from datagen.tree.fact_sft import preference_pairs, select


def _row(fen, sid, errors, supported):
    return dict(sample_id=sid, fen=fen, prompt="p", generation="ANALYSIS: x",
                fact_errors=errors, fact_supported=supported)


def test_fact_errors_sums_error_groups_only():
    by_kind = {"line_move": {"refuted": 2, "supported": 5},
               "evaluation": {"refuted": 1, "supported": 0},
               "outpost": {"refuted": 4, "supported": 0}}
    assert facts.fact_errors(by_kind) == 3
    assert facts.fact_supported(by_kind) == 5


def test_select_prefers_clean_and_supported_and_drops_unqualified():
    scored = [_row("A", "a0", 0, 1), _row("A", "a1", 0, 4), _row("A", "a2", 3, 9),
              _row("B", "b0", 1, 2), _row("B", "b1", 2, 2),
              _row("C", "c0", 0, 0)]
    fact, control, stats = select(scored, max_errors=0, seed=0)
    assert [r["sample_id"] for r in fact] == ["a1", "c0"]
    assert [r["fen"] for r in control] == ["A", "C"]  # same positions, B dropped
    assert stats["positions"] == 3 and stats["kept"] == 2
    assert stats["fact"]["clean_share"] == 1.0


def test_select_honours_exclude():
    scored = [_row("A", "a0", 0, 1), _row("B", "b0", 0, 1)]
    fact, control, _ = select(scored, exclude={"A"})
    assert [r["fen"] for r in fact] == ["B"] and [r["fen"] for r in control] == ["B"]


def test_select_best_of_k_keeps_every_position():
    scored = [_row("A", "a0", 2, 1), _row("A", "a1", 1, 0), _row("A", "a2", 1, 3),
              _row("B", "b0", 4, 2)]
    fact, control, stats = select(scored, seed=0)
    assert [r["sample_id"] for r in fact] == ["a2", "b0"]
    assert stats["kept"] == stats["positions"] == 2


def test_preference_pairs_need_an_error_gap():
    scored = [_row("A", "a0", 0, 2), _row("A", "a1", 3, 1), _row("A", "a2", 1, 1),
              _row("B", "b0", 1, 2), _row("B", "b1", 2, 2)]
    fact, _, _ = select(scored, seed=0)
    pairs = preference_pairs(fact, scored, min_gap=2)
    assert [(c["sample_id"], r["sample_id"]) for c, r in pairs] == [("a0", "a1")]


def test_select_prefer_uncut_breaks_ties_toward_fewer_line_errors():
    scored = [dict(_row("A", "a0", 1, 9), line_errors=1), dict(_row("A", "a1", 1, 2), line_errors=0),
              dict(_row("A", "a2", 2, 9), line_errors=0)]
    assert [r["sample_id"] for r in select(scored, seed=0)[0]] == ["a0"]
    assert [r["sample_id"] for r in select(scored, seed=0, prefer_uncut=True)[0]] == ["a1"]
