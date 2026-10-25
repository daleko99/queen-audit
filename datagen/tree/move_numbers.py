"""fixes move numbers in an analysis.

the model never sees the move number (the board encoding has no fullmove counter) so it
guesses one. its numbering is consistent inside an analysis though, so shifting every
marker by one offset (taken from the first marker) fixes the whole text.

    python -m datagen.tree.move_numbers --run outputs --output outputs_renumbered
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import chess

NUMBER = re.compile(r"(?<![\w>.])(?P<n>\d{1,3})(?P<dots>\.\.\.|\.)(?=\s*<PIECE_)")


def _ply(match):
    return 2 * int(match.group("n")) + (match.group("dots") == "...")


def renumber(fen, text):
    """(renumbered text, ply offset). the first marker is taken as the root move, this also
    fixes the dots when the model wrote the root move for the wrong side.
    """
    # note: breaks if the analysis opens with a side line instead of the root move
    first = NUMBER.search(text)
    if first is None:
        return text, 0
    board = chess.Board(fen)
    offset = 2 * board.fullmove_number + (board.turn == chess.BLACK) - _ply(first)
    if offset == 0:
        return text, 0

    def mark(match):
        ply = max(2, _ply(match) + offset)
        return f"{ply // 2}{'...' if ply % 2 else '.'}"

    return NUMBER.sub(mark, text), offset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run dir with sample.json + generations.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    run, out = Path(args.run), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    fens = {row["sample_id"]: row["fen"]
            for row in json.loads((run / "sample.json").read_text(encoding="utf-8-sig"))}
    rows = json.loads((run / "generations.json").read_text(encoding="utf-8-sig"))
    changed = 0
    for row in rows:
        row["generation"], offset = renumber(fens[row["sample_id"]], row["generation"])
        changed += offset != 0
    for name in ("sample.json", "annotations.json"):
        if (run / name).exists():
            shutil.copyfile(run / name, out / name)
    (out / "generations.json").write_text(json.dumps(rows), encoding="utf-8")
    print(f"{changed}/{len(rows)} renumbered -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
