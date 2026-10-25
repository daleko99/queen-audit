"""translates the models side to move tokens into absolute ones."""
import re

import chess

PIECE_NAMES = {
    "P": "PAWN",
    "N": "KNIGHT",
    "B": "BISHOP",
    "R": "ROOK",
    "Q": "QUEEN",
    "K": "KING",
}

POV_SQUARE = re.compile(r"<SQUARE_(\d+)>")
POV_PIECE = re.compile(r"<PIECE_([MO])([PNBRQK])>")
ABS_SQUARE_PAIR = re.compile(
    r"<SQUARE_([A-H][1-8])>\s*[-–—]\s*<SQUARE_([A-H][1-8])>"
)


def _line_token(match: re.Match[str]) -> str:
    first, second = match.group(1), match.group(2)
    first_file, first_rank = first[0], int(first[1])
    second_file, second_rank = second[0], int(second[1])
    if first_file == second_file and {first_rank, second_rank} == {1, 8}:
        return f"<FILE_{first_file}>"
    if first_rank == second_rank and {first_file, second_file} == {"A", "H"}:
        return f"<RANK_{first_rank}>"
    first_square = chess.parse_square(first.lower())
    second_square = chess.parse_square(second.lower())
    file_distance = abs(chess.square_file(first_square) - chess.square_file(second_square))
    rank_distance = abs(chess.square_rank(first_square) - chess.square_rank(second_square))
    if file_distance == rank_distance:
        ends = []
        for delta in (-9, -7, 7, 9):
            square = first_square
            while True:
                following = square + delta
                if (
                    not 0 <= following < 64
                    or abs(chess.square_file(following) - chess.square_file(square)) != 1
                ):
                    break
                square = following
            ends.append(square)
        if second_square in ends:
            low, high = sorted((first, second))
            return f"<DIAGONAL_{low}_{high}>"
    return match.group(0)


def absolute_tokens(text: str, pov: chess.Color) -> str:

    def square(match: re.Match[str]) -> str:
        value = int(match.group(1)) - 1
        absolute = value if pov == chess.WHITE else value ^ 56
        return f"<SQUARE_{chess.square_name(absolute).upper()}>"

    def piece(match: re.Match[str]) -> str:
        colour = pov if match.group(1) == "M" else not pov
        side = "WHITE" if colour else "BLACK"
        return f"<{side}_{PIECE_NAMES[match.group(2)]}>"

    mover = "<WHITE>" if pov == chess.WHITE else "<BLACK>"
    opponent = "<BLACK>" if pov == chess.WHITE else "<WHITE>"
    translated = POV_SQUARE.sub(square, text)
    translated = POV_PIECE.sub(piece, translated)
    translated = translated.replace("<PLAYER>", mover).replace("<OPPONENT>", opponent)
    translated = re.sub(
        r"\b(?:the\s+)?player\b", mover, translated, flags=re.IGNORECASE
    )
    translated = re.sub(
        r"\b(?:the\s+)?opponent\b", opponent, translated, flags=re.IGNORECASE
    )
    return ABS_SQUARE_PAIR.sub(_line_token, translated)
