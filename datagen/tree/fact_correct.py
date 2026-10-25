"""fixes false board facts in an analysis without using the model.

- material: a root paragraph sentence with a false material claim gets replaced by one
  true sentence built from the board
- lines: a paragraph gets cut before the sentence with its first illegal move.
  CRITICAL_LINE gets cut before its first illegal move and illegal PROMISING_MOVES get
  dropped
- evaluation: an EVALUATION thats too far from the engine score gets replaced by it

an illegal BEST_MOVE can not be fixed from the text, so those come back unchanged and
marked uncorrectable.

    python -m datagen.tree.fact_correct --input picked.jsonl \\
        --annotations annotations.json --output corrected.jsonl
"""
import argparse
import json
import re
from pathlib import Path

import chess

from datagen.self_distill.analysis import FIELDS, MOVE, pov_move
from datagen.self_distill.io import provenance
from datagen.tree import facts

_SENTENCE_START = re.compile(r"(?<=[.!?])\s+")
_COUNT = {1: "an", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
_LETTER = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B", chess.ROOK: "R",
           chess.QUEEN: "Q"}


def _surplus(board, side, other, tag):
    parts = []
    for piece in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN):
        n = len(board.pieces(piece, side)) - len(board.pieces(piece, other))
        if n > 0:
            token = f"<PIECE_{tag}{_LETTER[piece]}>"
            parts.append(f"{_COUNT.get(n, str(n))} extra {token}{'s' if n > 1 else ''}")
    return parts


def _join(parts):
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def material_statement(fen):
    board = chess.Board(fen)
    me, them = board.turn, not board.turn
    lead = facts.material(board, me) - facts.material(board, them)
    mine, theirs = _surplus(board, me, them, "M"), _surplus(board, them, me, "O")
    if abs(lead) < 2:
        return "Material is level." if not mine and not theirs else "Material is roughly level."
    if lead > 0:
        head, ahead, behind = "The player holds a material advantage", mine, theirs
    else:
        head, ahead, behind = "The opponent holds a material advantage", theirs, mine
    if not ahead:
        return head + "."
    text = f"{head}, with {_join(ahead)}"
    if behind:
        text += f" against {_join(behind)}"
    return text + "."


def _fix_material(fen, analysis):
    paragraphs = analysis.split("\n\n")
    if not paragraphs:
        return analysis, 0
    refuted = {f.text for f in facts.check_material(fen, f"ANALYSIS:\n{paragraphs[0]}")
               if f.status == "refuted"}
    if not refuted:
        return analysis, 0
    turn = chess.Board(fen).turn
    sentences = _SENTENCE_START.split(paragraphs[0].strip())
    out, replaced, fixed = [], False, 0
    for sentence in sentences:
        plain = facts.detokenize(facts.absolute_tokens(sentence, turn))
        if " ".join(plain.split()) in {" ".join(text.split()) for text in refuted}:
            fixed += 1
            if not replaced:  # only put the true sentence in once even if several were wrong
                out.append(material_statement(fen))
                replaced = True
            continue
        out.append(sentence)
    paragraphs[0] = " ".join(out)
    return "\n\n".join(paragraphs), fixed


def _cut_lines(fen, analysis, min_kept_sentences=1):
    """cuts each paragraph before its first illegal line move, and drops later paragraphs
    left with fewer than min_kept_sentences sentences.
    """
    paragraphs, cut = [], 0
    pieces = analysis.split("\n\n")
    # a line can run on into the next paragraph so check them all together
    remaining = iter(facts.check_line_moves(fen, "ANALYSIS:\n" + "\n\n".join(pieces)))
    for index, paragraph in enumerate(pieces):
        atoms = list(facts._line_atoms(paragraph))
        findings = [next(remaining) for _ in atoms]
        bad = next((atom for (_, atom), finding in zip(atoms, findings, strict=True)
                    if finding.status == "refuted"), None)
        if bad is None:
            paragraphs.append(paragraph)
            continue
        cut += 1
        starts = [0] + [m.end() for m in _SENTENCE_START.finditer(paragraph) if m.end() <= bad.start()]
        kept = paragraph[:starts[-1]].rstrip()
        sentences = len(_SENTENCE_START.split(kept)) if kept else 0
        if index == 0 or sentences >= min_kept_sentences:
            paragraphs.append(kept)
    return "\n\n".join(p for p in paragraphs if p.strip()), cut


def _legal_prefix(fen, body):
    # offset of the first illegal move, None if its all legal
    board = chess.Board(fen)
    pov = board.turn
    for atom in MOVE.finditer(body):
        move, _ = pov_move(atom, board, pov)
        if move is None:
            return atom.start()
        board.push(move)
        if board.is_game_over(claim_draw=False):
            return None
    return None


def evaluation(cp):
    """the EVALUATION field for an engine score (cp for the side to move, mates as
    +-100000).
    """
    if abs(cp) >= 99000:
        distance = 100000 - abs(int(cp))
        sign = "+" if cp > 0 else "-"
        return f"EVALUATION: {sign}M{distance} from the player's perspective."
    return f"EVALUATION: {cp / 100:+.2f} pawns from the player's perspective."


def correct(fen, generation, reference_cp=None, *, min_kept_sentences=1):
    """(corrected text, log dict). the text stays the same if nothing is refuted."""
    log = dict(material=0, line_paragraphs=0, critical_cut=0, promising_dropped=0,
               evaluation=0, uncorrectable=False)
    root = chess.Board(fen)
    pov = root.turn
    bodies = {m.group("name"): m for m in FIELDS.finditer(generation)}
    best = bodies.get("BEST_MOVE")
    best_atom = next(MOVE.finditer(best.group("body")), None) if best else None
    if best_atom is not None and pov_move(best_atom, root.copy(stack=False), pov)[0] is None:
        log["uncorrectable"] = True
        return generation, log

    replacements = {}
    if "ANALYSIS" in bodies:
        body = bodies["ANALYSIS"].group("body")
        trailing = body[len(body.rstrip()):]
        text, log["material"] = _fix_material(fen, body.rstrip())
        text, log["line_paragraphs"] = _cut_lines(fen, text, min_kept_sentences)
        if text != body.rstrip():
            replacements["ANALYSIS"] = text + trailing
    if "CRITICAL_LINE" in bodies:
        body = bodies["CRITICAL_LINE"].group("body")
        cut = _legal_prefix(fen, body)
        if cut is not None:
            log["critical_cut"] = 1
            kept = re.sub(r"[\s\d.]*$", "", body[:cut])
            if not kept and best_atom is not None:
                kept = best_atom.group(0)
            replacements["CRITICAL_LINE"] = kept + body[len(body.rstrip()):]
    if "PROMISING_MOVES" in bodies:
        body = bodies["PROMISING_MOVES"].group("body")
        legal = [a.group(0) for a in MOVE.finditer(body)
                 if pov_move(a, root.copy(stack=False), pov)[0] is not None]
        dropped = len(list(MOVE.finditer(body))) - len(legal)
        if dropped:
            log["promising_dropped"] = dropped
            replacements["PROMISING_MOVES"] = " ".join(legal) + body[len(body.rstrip()):]
    if "EVALUATION" in bodies and reference_cp is not None:
        finding, _ = facts.check_evaluation(fen, generation, reference_cp)
        if finding.status == "refuted":
            log["evaluation"] = 1
            body = bodies["EVALUATION"].group("body")
            replacements["EVALUATION"] = (evaluation(reference_cp).split(": ", 1)[1]
                                          + body[len(body.rstrip()):])

    text = generation
    for name, match in sorted(bodies.items(), key=lambda item: -item[1].start("body")):
        if name in replacements:
            text = text[:match.start("body")] + replacements[name] + text[match.end("body"):]
    return text, log


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="JSONL with fen, chosen, sample_id")
    parser.add_argument("--annotations", required=True, help="annotations.json with best_cp")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-kept-sentences", type=int, default=1,
                        help="drop a cut paragraph if fewer sentences than this remain")
    args = parser.parse_args(argv)
    cps = {row["sample_id"]: row["best_cp"] for row in
           json.loads(Path(args.annotations).read_text(encoding="utf-8-sig"))}
    rows = [json.loads(line) for line in
            Path(args.input).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    totals = dict(records=len(rows), changed=0, uncorrectable=0, material=0, line_paragraphs=0,
                  critical_cut=0, promising_dropped=0, evaluation=0)
    with open(args.output, "w", encoding="utf-8") as handle:
        for row in rows:
            text, log = correct(row["fen"], row["chosen"], cps.get(row["sample_id"]),
                                min_kept_sentences=args.min_kept_sentences)
            totals["changed"] += text != row["chosen"]
            totals["uncorrectable"] += log.pop("uncorrectable")
            for key, value in log.items():
                totals[key] += value
            handle.write(json.dumps(dict(row, chosen=text, original=row["chosen"],
                                         corrections=log)) + "\n")
    manifest = dict(input=args.input, annotations=args.annotations, totals=totals,
                    min_kept_sentences=args.min_kept_sentences,
                    provenance=provenance(["princeton-nlp/queen_pawn-8 (self-samples)"],
                                          labels="datagen.tree.fact_correct + Stockfish"))
    Path(args.output).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2),
                                                               encoding="utf-8")
    print(json.dumps(totals, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
