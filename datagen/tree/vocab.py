"""turns absolute board tokens ("<WHITE_KNIGHT> on <SQUARE_D5>") into plain words."""
import re

_ABS_SQUARE_TOKEN = re.compile(r"<SQUARE_([A-H][1-8])>")
_ABS_PIECE_TOKEN = re.compile(r"<(WHITE|BLACK)_(PAWN|KNIGHT|BISHOP|ROOK|QUEEN|KING)>")
_ABS_PLAYER_TOKEN = re.compile(r"<(WHITE|BLACK)>")
_FILE_TOKEN = re.compile(r"<FILE_([A-H])>")
_RANK_TOKEN = re.compile(r"<RANK_([1-8])>")
_DIAGONAL_TOKEN = re.compile(r"<DIAGONAL_([A-H][1-8])_([A-H][1-8])>")


def detokenize(text: str) -> str:
    """only absolute tokens get rewritten, everything else passes through unchanged."""
    text = _ABS_PIECE_TOKEN.sub(
        lambda match: f" {match.group(1).lower()} {match.group(2).lower()} ",
        text,
    )
    text = _ABS_PLAYER_TOKEN.sub(lambda match: f" {match.group(1).capitalize()} ", text)
    text = _ABS_SQUARE_TOKEN.sub(lambda match: f" {match.group(1).lower()} ", text)
    text = _FILE_TOKEN.sub(lambda match: f" {match.group(1).lower()}-file ", text)
    text = _RANK_TOKEN.sub(lambda match: f" rank {match.group(1)} ", text)
    text = _DIAGONAL_TOKEN.sub(
        lambda match: f" {match.group(1).lower()}-{match.group(2).lower()} diagonal ", text
    )
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?)'’])", r"\1", text)
    return text.strip()
