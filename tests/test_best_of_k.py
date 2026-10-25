import chess

from datagen.tree.best_of_k import choose, engine_requests, explain

WHITE_UP_KNIGHT = "4k3/pp6/8/8/8/8/PP6/1N2K3 w - - 0 1"
START = chess.STARTING_FEN
E4, E5 = "<PIECE_MP><SQUARE_13><SQUARE_29>", "<PIECE_OP><SQUARE_53><SQUARE_37>"
BAD = "<PIECE_MN><SQUARE_2><SQUARE_29>"  # Nb1-e4: not a knight move


def analysis(root, lines="Quiet.", best="", critical=""):
    return (f"ANALYSIS:\n{root}\n\n{lines}\n\nBEST_MOVE: {best}\nCRITICAL_LINE: {critical}\n"
            f"PROMISING_MOVES: \nEVALUATION: ")


def test_picks_fewest_errors_then_earliest():
    wrong = analysis("Black holds a clear material advantage.")
    right = analysis("The player holds a material advantage, with an extra <PIECE_MN>.")
    index, scores = choose(WHITE_UP_KNIGHT, [wrong, right, right])
    assert index == 1 and scores[0][0] == 1 and scores[1][0] == 0


def test_ties_prefer_fewer_illegal_line_moves_then_more_supported_facts():
    illegal = analysis("Quiet.", lines=f"Play 1.{BAD} now.")
    legal = analysis("Quiet.", lines=f"Play 1.{E4} 1...{E5} now.")
    bare = analysis("Quiet.")
    assert choose(START, [illegal, bare, legal])[0] == 2
    assert choose(START, [bare, legal])[0] == 1


def test_unusable_samples_are_skipped():
    good = analysis("Quiet.")
    assert choose(START, ["", "no analysis field", good])[0] == 2
    assert explain(START, ["", "cut off"]) is None


def test_explain_corrects_and_renumbers_the_pick():
    fen = "4k3/pp6/8/8/8/8/PP6/1N2K3 w - - 0 14"
    king = "<PIECE_MK><SQUARE_5><SQUARE_4>"  # Ke1-d1
    text = analysis("Black holds a clear material advantage.", lines=f"Then 1.{king} is calm.")
    result = explain(fen, [text])
    assert result["index"] == 0 and result["correction"]["material"] == 1
    assert result["renumber_offset"] == 26
    assert "The player holds a material advantage" in result["text"]
    assert f"14.{king}" in result["text"]


def test_engine_requests_are_deduplicated_across_samples():
    text = analysis("Quiet.", best=E4, critical=f"{E4} {E5}")
    requests = engine_requests(START, [text, text])
    assert requests == list(dict.fromkeys(requests)) and len(requests) == 2
