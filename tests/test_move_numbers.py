from datagen.tree.move_numbers import renumber

FEN_W14 = "r3kb2/p1p1ppnr/n2q2p1/3p3P/1p1P4/2P1N2b/PP1KPP2/RNBQ1B1R w q - 0 14"
FEN_B18 = "rnb1kbnr/p1pp1pp1/2q4p/1p1P2N1/P4PP1/4P2P/1PPNK3/1RBQ1B1R b kq - 0 18"
MOVE = "<PIECE_MR><SQUARE_8><SQUARE_16>"


def test_shifts_every_number_by_one_offset():
    text = f"The best move is 9.{MOVE}, then 9...{MOVE} 10.{MOVE}.\n\nAlso 9.{MOVE}."
    out, offset = renumber(FEN_W14, text)
    assert offset == 10
    assert out == f"The best move is 14.{MOVE}, then 14...{MOVE} 15.{MOVE}.\n\nAlso 14.{MOVE}."


def test_black_to_move_keeps_dots_and_correct_text_is_unchanged():
    out, offset = renumber(FEN_B18, f"Play 3...{MOVE} 4.{MOVE}")
    assert (out, offset) == (f"Play 18...{MOVE} 19.{MOVE}", 30)
    assert renumber(FEN_B18, out) == (out, 0)


def test_root_move_written_for_the_wrong_side_gets_its_dots_fixed():
    out, _ = renumber(FEN_W14, f"Best 19...{MOVE} 20.{MOVE} 20...{MOVE}")
    assert out == f"Best 14.{MOVE} 14...{MOVE} 15.{MOVE}"


def test_prose_numbers_are_untouched():
    text = f"Material: 3. The plan has 2 ideas. 1.{MOVE}"
    out, _ = renumber(FEN_W14, text)
    assert out == f"Material: 3. The plan has 2 ideas. 14.{MOVE}"
    assert renumber(FEN_W14, "No moves here.") == ("No moves here.", 0)
