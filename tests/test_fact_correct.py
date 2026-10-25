import chess

from datagen.tree import facts
from datagen.tree.fact_correct import correct, material_statement

WHITE_UP_KNIGHT = "4k3/pp6/8/8/8/8/PP6/1N2K3 w - - 0 1"
START = chess.STARTING_FEN


def analysis(root, lines="Quiet.", best="", critical="", promising="", evaluation=""):
    return (f"ANALYSIS:\n{root}\n\n{lines}\n\nBEST_MOVE: {best}\nCRITICAL_LINE: {critical}\n"
            f"PROMISING_MOVES: {promising}\nEVALUATION: {evaluation}")


def refuted(fen, text, cp=None):
    return [f for f in facts.check_facts(fen, text, reference_cp=cp).findings
            if f.status == "refuted"]


def test_material_statement_is_true_and_checkable():
    sentence = material_statement(WHITE_UP_KNIGHT)
    assert sentence == "The player holds a material advantage, with an extra <PIECE_MN>."
    found = facts.check_material(WHITE_UP_KNIGHT, analysis(sentence))
    assert found and all(f.status == "supported" for f in found)
    assert material_statement(START) == "Material is level."


def test_false_material_sentence_is_replaced():
    text = analysis("Black holds a clear material advantage. The king is safe.")
    fixed, log = correct(WHITE_UP_KNIGHT, text)
    assert log["material"] == 1
    assert "The player holds a material advantage" in fixed and "The king is safe." in fixed
    assert not refuted(WHITE_UP_KNIGHT, fixed)


def test_illegal_line_paragraph_is_cut_at_its_sentence():
    e4, e5 = "<PIECE_MP><SQUARE_13><SQUARE_29>", "<PIECE_OP><SQUARE_53><SQUARE_37>"
    bad = "<PIECE_MN><SQUARE_2><SQUARE_29>"  # Nb1-e4: not a knight move
    lines = f"The best move is 1.{e4} 1...{e5}, taking space. Then 2.{bad} follows. More words."
    fixed, log = correct(START, analysis("Quiet.", lines=lines))
    assert log["line_paragraphs"] == 1
    assert "taking space." in fixed and "follows" not in fixed and "More words" not in fixed


def test_critical_line_and_promising_keep_legal_prefix():
    e4, e5 = "<PIECE_MP><SQUARE_13><SQUARE_29>", "<PIECE_OP><SQUARE_53><SQUARE_37>"
    bad = "<PIECE_MQ><SQUARE_4><SQUARE_60>"
    text = analysis("Quiet.", best=e4, critical=f"{e4} {e5} {bad}", promising=f"{e4} {bad}")
    fixed, log = correct(START, text)
    assert f"CRITICAL_LINE: {e4} {e5}\n" in fixed and f"PROMISING_MOVES: {e4}\n" in fixed
    assert log["critical_cut"] == 1 and log["promising_dropped"] == 1


def test_wrong_evaluation_is_replaced_with_engine_score():
    text = analysis("Quiet.", evaluation="Black is up approximately 3.00 pawns.")
    fixed, log = correct(START, text, reference_cp=30)
    assert log["evaluation"] == 1
    assert "EVALUATION: +0.30 pawns from the player's perspective." in fixed
    assert not refuted(START, fixed, cp=30)


def test_clean_text_is_unchanged_and_illegal_best_is_uncorrectable():
    e4 = "<PIECE_MP><SQUARE_13><SQUARE_29>"
    text = analysis("Material is level.", best=e4, critical=e4, promising=e4,
                    evaluation="+0.30 pawns from the player's perspective.")
    assert correct(START, text, reference_cp=30)[0] == text
    _, log = correct(START, analysis("Quiet.", best="<PIECE_MQ><SQUARE_4><SQUARE_60>"))
    assert log["uncorrectable"]


def test_short_fragment_of_a_cut_paragraph_is_dropped_whole():
    e4, e5 = "<PIECE_MP><SQUARE_13><SQUARE_29>", "<PIECE_OP><SQUARE_53><SQUARE_37>"
    bad = "<PIECE_MN><SQUARE_2><SQUARE_29>"
    lines = f"Next 1.{e4} 1...{e5} opens lines. Then 2.{bad} follows."
    text = analysis("Quiet.", lines=lines)
    kept, _ = correct(START, text)
    dropped, log = correct(START, text, min_kept_sentences=2)
    assert "opens lines." in kept and "opens lines." not in dropped
    assert log["line_paragraphs"] == 1 and "Quiet." in dropped
