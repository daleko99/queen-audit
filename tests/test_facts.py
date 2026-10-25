"""board and engine fact checks: line replay, material, evaluation, engine audits."""

import chess
import pytest

from datagen.tree import facts

START = chess.STARTING_FEN
AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
WHITE_UP_KNIGHT = "4k3/pp6/8/8/8/8/PP6/1N2K3 w - - 0 1"


def tag(uci, piece, side="M", captured=None, pov=chess.WHITE):
    """a POV move tag the way the model writes it (squares mirrored when black is to move)."""
    def sq(name):
        index = chess.parse_square(name)
        return (index if pov == chess.WHITE else index ^ 56) + 1
    other = "O" if side == "M" else "M"
    capture = f"<PIECE_{other}{captured}>" if captured else ""
    return f"<PIECE_{side}{piece}><SQUARE_{sq(uci[:2])}>{capture}<SQUARE_{sq(uci[2:])}>"


def analysis(root="Quiet.", lines="", best="", critical="", promising="", evaluation=""):
    return (f"ANALYSIS:\n{root}\n\n{lines}\n\nBEST_MOVE: {best}\nCRITICAL_LINE: {critical}\n"
            f"PROMISING_MOVES: {promising}\nEVALUATION: {evaluation}")


E4, E5, NF3, NC6 = (tag("e2e4", "P"), tag("e7e5", "P", "O"), tag("g1f3", "N"), tag("b8c6", "N", "O"))


def statuses(findings, kind):
    return [f.status for f in findings if f.kind == kind]


def test_line_replay_follows_branches_and_continuations():
    lines = f"The best move is 1.{E4}, and 1...{E5} 2.{NF3} {NC6}. Instead 2.{tag('f1c4', 'B')} also works."
    findings = facts.check_line_moves(START, analysis(lines=lines))
    assert statuses(findings, "line_move") == ["supported"] * 5


def test_line_replay_refutes_illegal_and_skips_unreachable_plies():
    lines = f"The best move is 1.{tag('e2e5', 'P')}, then 1...{E5} 2.{NF3}."
    findings = facts.check_line_moves(START, analysis(lines=lines))
    assert statuses(findings, "line_move") == ["refuted", "uncheckable", "uncheckable"]
    assert findings[0].detail == "move is illegal"


def test_line_replay_flags_wrong_mover_and_ignores_mentions():
    lines = f"The best move is 1.{tag('d1d3', 'N')}. White is threatening {NF3}."
    findings = facts.check_line_moves(START, analysis(lines=lines))
    assert [f.detail for f in findings] == ["mover token does not match the board", "mention"]


@pytest.mark.parametrize("first, reply, second", [("5.", "5...", "6."), ("5...", "6.", "6...")])
def test_black_root_accepts_player_relative_and_standard_numbering(first, reply, second):
    m1, o1, m2 = (tag("e7e5", "P", "M", pov=chess.BLACK), tag("g1f3", "N", "O", pov=chess.BLACK),
                  tag("b8c6", "N", "M", pov=chess.BLACK))
    lines = f"The best move is {first}{m1}. White replies {reply}{o1} and {second}{m2} follows."
    findings = facts.check_line_moves(AFTER_E4, analysis(lines=lines))
    assert statuses(findings, "line_move") == ["supported"] * 3


def test_structured_fields_legality_and_consistency():
    report = facts.check_structured_moves(START, analysis(
        lines=f"The best move is 1.{E4}.", best=E4, critical=f"{E4} {E5} {tag('e4e5', 'P')}",
        promising=f"{E4} {tag('a2a5', 'P')}"))
    assert statuses(report, "critical_move") == ["supported", "supported", "refuted"]
    assert statuses(report, "promising_move") == ["supported", "refuted"]
    assert statuses(report, "best_consistency") == ["supported"]
    mismatch = facts.check_structured_moves(START, analysis(best=E4, critical=NF3))
    assert statuses(mismatch, "best_consistency") == ["refuted"]


def material(root):
    return facts.check_material(WHITE_UP_KNIGHT, analysis(root=root))


def test_material_side_and_extra():
    findings = material("White holds a material advantage with an extra white knight.")
    assert statuses(findings, "material_side") == ["supported"]
    assert statuses(findings, "material_extra") == ["supported"]
    findings = material("Black holds a clear material advantage with an extra black bishop and black pawn.")
    assert statuses(findings, "material_side") == ["refuted"]
    assert statuses(findings, "material_extra") == ["refuted", "refuted"]


def test_material_inventory_counts():
    findings = material("White has a knight against Black's two black rooks.")
    assert statuses(findings, "material_count") == ["supported", "refuted"]
    assert "black has 0 rook" in [f for f in findings if f.status == "refuted"][0].detail


def test_material_equal_and_conditionals():
    assert statuses(material("The material is equal."), "material_equal") == ["refuted"]
    assert statuses(material("White could win an extra pawn."), "material_extra") == ["uncheckable"]
    level = facts.check_material(START, analysis(root="Material is level."))
    assert statuses(level, "material_equal") == ["supported"]


def test_material_reads_only_the_root_paragraph():
    findings = facts.check_material(
        WHITE_UP_KNIGHT, analysis(root="Quiet.", lines="Black then holds a material advantage."))
    assert findings == []


@pytest.mark.parametrize("fen, text, cp", [
    (START, "the player is up approximately 1.50 pawns after the best line.", 150),
    (AFTER_E4, "the player is down approximately 2.00 pawns.", -200),
    (AFTER_E4, "White is up approximately 2 pawns.", -200),
    (AFTER_E4, "Decisively winning for Black, approximately +7.55 pawns.", 755),
    (START, "+0.40 pawns from Black's perspective.", -40),
])
def test_parse_evaluation_signs(fen, text, cp):
    assert facts.parse_evaluation(fen, analysis(evaluation=text))["root_cp"] == cp


def test_parse_evaluation_mate_equal_and_unparsed():
    mate = facts.parse_evaluation(AFTER_E4, analysis(evaluation="+M1 from White's perspective."))
    assert mate["kind"] == "mate" and mate["root_sign"] == -1
    assert facts.parse_evaluation(START, analysis(evaluation="The position is equal."))["root_cp"] == 0
    assert facts.parse_evaluation(START, analysis(evaluation="Unclear.")) is None


def test_check_evaluation_threshold():
    text = analysis(evaluation="the player is up approximately 1.50 pawns.")
    close, info = facts.check_evaluation(START, text, reference_cp=120)
    far, _ = facts.check_evaluation(START, text, reference_cp=-300)
    assert close.status == "supported" and info["gap"] < 0.1
    assert far.status == "refuted"


def test_engine_requests_and_audit_findings():
    text = analysis(best=E4, critical=f"{E4} {E5} {tag('a2a5', 'P')}")
    requests = facts.engine_requests(START, text)
    board = chess.Board()
    assert requests == [(board.fen(), "e2e4"),
                        (chess.Board(AFTER_E4).fen(), "e7e5")]
    audits = {requests[0]: dict(winrate_drop=0.02), requests[1]: dict(winrate_drop=0.25)}
    findings = facts.check_engine(START, text, audits)
    assert [(f.kind, f.status) for f in findings] == [
        ("best_quality", "supported"), ("critical_quality", "supported"),
        ("critical_quality", "refuted")]
