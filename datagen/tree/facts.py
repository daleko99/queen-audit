"""checks the facts in a models analysis against the board and stockfish.

precision over recall: only the structured fields, numbered move tags and a few fixed
phrasings get checked. anything hedged, conditional or unparseable is "uncheckable" and
never counts against the model. moves are checked on the models own move tags, so a
wrong piece or capture shows up.

kinds:

  line_move          numbered move in the ANALYSIS lines (branch aware replay)
  critical_move      CRITICAL_LINE move (replayed from the root)
  best_move          BEST_MOVE legal at the root
  promising_move     PROMISING_MOVES entry legal at the root
  best_consistency   BEST_MOVE == first CRITICAL_LINE move == "best move is" move
  material_side      "<side> holds a material advantage" / "<side>'s material edge"
  material_extra     "an extra black knight"
  material_count     inventories ("two white rooks ... against Black's queen")
  material_equal     "material is equal" (refuted only when off by 2+ points)
  evaluation         EVALUATION vs the stockfish root score (needs a reference)
  best_quality       BEST_MOVE win rate drop vs stockfish (needs move audits)
  critical_quality   CRITICAL_LINE win rate drops (needs move audits)

nothing here starts an engine. engine_requests lists the (fen, uci) pairs to audit and
the results come back in through audits.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import chess

from datagen.self_distill.analysis import FIELDS, MOVE, expected_winrate, field_map, pov_move
from datagen.tree.pov_tokens import absolute_tokens
from datagen.tree.vocab import detokenize

MISTAKE_DROP = 0.10  # win rate, the papers threshold for a mistake
VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
PIECE_WORDS = {"pawn": chess.PAWN, "knight": chess.KNIGHT, "bishop": chess.BISHOP,
               "rook": chess.ROOK, "queen": chess.QUEEN}
COUNT_WORDS = {"a": 1, "an": 1, "one": 1, "a single": 1, "single": 1, "two": 2, "three": 3,
               "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}


@dataclass(frozen=True)
class Finding:
    kind: str
    status: str  # "supported" | "refuted" | "uncheckable"
    detail: str = ""
    text: str = ""


@dataclass
class FactReport:
    findings: list[Finding] = field(default_factory=list)
    evaluation: dict | None = None  # parsed claim and gap, only when theres a reference

    def count(self, status: str, kinds=None) -> int:
        return sum(1 for f in self.findings
                   if f.status == status and (kinds is None or f.kind in kinds))

    def by_kind(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for f in self.findings:
            row = out.setdefault(f.kind, {"supported": 0, "refuted": 0, "uncheckable": 0})
            row[f.status] += 1
        return out


_NUMBERED = re.compile(r"(\d+)\s*(\.\.\.|\.)\s*(?=<PIECE_)")


def _field_span(generation: str, name: str) -> str:
    for match in FIELDS.finditer(generation):
        if match.group("name") == name:
            return match.group("body")
    return ""


def _line_atoms(paragraph: str):
    """(move number, None if it continues the last move, or "mention", atom) for each move
    tag.
    """
    numbered = {match.end(): int(match.group(1)) for match in _NUMBERED.finditer(paragraph)}
    previous_end = None
    for atom in MOVE.finditer(paragraph):
        if atom.start() in numbered:
            yield numbered[atom.start()], atom
        elif previous_end is not None and not paragraph[previous_end:atom.start()].strip():
            yield None, atom
        else:
            yield "mention", atom
        previous_end = atom.end()


def check_line_moves(fen: str, generation: str) -> list[Finding]:
    """replays the ANALYSIS lines as a tree from the root.

    plies come from the move numbers (relative to the first one) and the side comes from
    the move tag, not the dots, since the model gets the dots wrong. a move on a ply
    thats already been played starts a branch ("20.Qxf6+ fails to 20...Kxf6; instead
    20.e3"). moves that skip a ply or come after an illegal one are uncheckable.
    """
    root = chess.Board(fen)
    pov = root.turn
    findings = []
    paragraphs = [list(_line_atoms(p)) for p in _field_span(generation, "ANALYSIS").split("\n\n")]
    numbered = [(n, a) for atoms in paragraphs for n, a in atoms if isinstance(n, int)]
    base = numbered[0][0] if numbered else None
    # with black to move, normal numbering puts whites reply on the next number
    reply_shift = 1
    if pov == chess.BLACK and not any(
            n == base and a.group("mover_side") == "O" for n, a in numbered):
        reply_shift = -1
    path: list[chess.Move] = []
    for atoms in paragraphs:
        last_ply = None
        for number, atom in atoms:
            mover_is_root = atom.group("mover_side") == "M"
            if number == "mention" or (number is None and last_ply is None):
                findings.append(Finding("line_move", "uncheckable", "mention", atom.group(0)))
                continue
            if number is None:
                ply = last_ply + 1
            else:
                offset = 2 * (number - base)
                ply = offset if mover_is_root else offset + reply_shift
            last_ply = ply
            if ply < 0 or ply > len(path) or (ply % 2 == 0) != mover_is_root:
                findings.append(Finding("line_move", "uncheckable", "ply not on the replayed path",
                                        atom.group(0)))
                continue
            del path[ply:]
            board = root.copy(stack=False)
            for move in path:
                board.push(move)
            move, error = pov_move(atom, board, pov)
            if move is None:
                findings.append(Finding("line_move", "refuted", error, atom.group(0)))
                continue
            findings.append(Finding("line_move", "supported", "", atom.group(0)))
            path.append(move)
    return findings


def check_structured_moves(fen: str, generation: str) -> list[Finding]:
    root = chess.Board(fen)
    pov = root.turn
    fields = field_map(generation)
    findings = []

    board = root.copy(stack=False)
    for atom in MOVE.finditer(fields.get("CRITICAL_LINE", "")):
        move, error = pov_move(atom, board, pov)
        if move is None:
            findings.append(Finding("critical_move", "refuted", error, atom.group(0)))
            break
        findings.append(Finding("critical_move", "supported", "", atom.group(0)))
        board.push(move)
        if board.is_game_over(claim_draw=False):
            break

    best = next(MOVE.finditer(fields.get("BEST_MOVE", "")), None)
    if best is not None:
        move, error = pov_move(best, root.copy(stack=False), pov)
        findings.append(Finding("best_move", "supported" if move else "refuted", error or "",
                                best.group(0)))
    for atom in MOVE.finditer(fields.get("PROMISING_MOVES", "")):
        move, error = pov_move(atom, root.copy(stack=False), pov)
        findings.append(Finding("promising_move", "supported" if move else "refuted",
                                error or "", atom.group(0)))

    critical = next(MOVE.finditer(fields.get("CRITICAL_LINE", "")), None)
    prose = re.search(r"best move is\s*(?:\d+\s*\.(?:\.\.)?\s*)?(<PIECE_[^\s,.;]*)",
                      fields.get("ANALYSIS", ""), re.IGNORECASE)
    prose_atom = MOVE.match(prose.group(1)) if prose else None
    named = [m.group(0) for m in (best, critical, prose_atom) if m is not None]
    if best is not None and len(named) >= 2:
        same = len(set(named)) == 1
        findings.append(Finding("best_consistency", "supported" if same else "refuted",
                                "" if same else "best move differs across fields",
                                " | ".join(named)))
    return findings


_PIECE = r"(?:pawn|knight|bishop|rook|queen)"
_PLURAL = r"(?:\s?s\b)?"
_COUNT = r"(?:a single|a|an|one|single|two|three|four|five|six|seven|eight)"
_ITEM = rf"(?:{_COUNT}\s+)?(?:(?:white|black)\s+)?{_PIECE}{_PLURAL}"
_SEP = r"(?:\s*,\s*(?:and\s+)?|\s+and\s+)"
_LIST = rf"{_ITEM}(?:{_SEP}{_ITEM})*"
_ITEM_PARTS = re.compile(rf"(?:(?P<count>{_COUNT})\s+)?(?:(?P<colour>white|black)\s+)?"
                         rf"(?P<piece>{_PIECE}){_PLURAL}", re.IGNORECASE)
_INVENTORY = re.compile(rf"(?P<a>{_LIST})\s+(?:against|versus|vs\.?)\s+"
                        rf"(?P<owner>White|Black)['’]s\s+(?P<b>{_LIST})", re.IGNORECASE)
_EXTRA = re.compile(rf"\b(?P<count>an|a|one|two|three)\s+(?:extra|additional)\s+"
                    rf"(?P<items>{_LIST})", re.IGNORECASE)
_SIDE = (
    re.compile(r"\b(?P<side>White|Black)\s+(?:holds|has|possesses|enjoys|retains|maintains|"
               r"holding|having|possessing|enjoying)\s+(?:a|an|the)?\s*(?:[\w-]+\s+){0,2}?"
               r"material\s+(?:advantage|edge|superiority|lead|plus)\b"),
    re.compile(r"\b(?P<side>White|Black)['’]s\s+(?:[\w-]+\s+){0,2}?"
               r"material\s+(?:advantage|edge|superiority|lead|plus)\b"),
)
_EQUAL = re.compile(r"\bmaterial\s+(?:is\s+)?(?:(?:roughly|approximately|nearly)\s+)?"
                    r"(?:equal|balanced|level)\b|\b(?:equal|level)\s+material\b|"
                    r"\bmaterial\s+(?:parity|equality)\b", re.IGNORECASE)
_CONDITIONAL = re.compile(r"\b(?:would|could|might|will|if|once|after|regain\w*|win\s+back|"
                          r"to\s+win|wins?|winning|threaten\w*|no|not|without|lacks?|"
                          r"despite|compensat\w*)\b", re.IGNORECASE)


def material(board: chess.Board, colour: chess.Color) -> int:
    return sum(VALUES[piece] * len(board.pieces(piece, colour)) for piece in VALUES)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _root_paragraph(fen: str, generation: str) -> str:
    # todo: only the first paragraph, later ones are about lines so its unclear which position they mean
    paragraph = _field_span(generation, "ANALYSIS").strip().split("\n\n")[0]
    return detokenize(absolute_tokens(paragraph, chess.Board(fen).turn))


def _items(text: str, owner: chess.Color):
    for item in _ITEM_PARTS.finditer(text):
        count = COUNT_WORDS.get((item.group("count") or "a").lower(), 1)
        colour = item.group("colour")
        item_owner = owner if colour is None else (colour.lower() == "white")
        yield count, PIECE_WORDS[item.group("piece").lower()], item_owner, colour is not None


def check_material(fen: str, generation: str) -> list[Finding]:
    """material claims in the root paragraph, since thats the one describing the root
    position.
    """
    board = chess.Board(fen)
    balance = material(board, chess.WHITE) - material(board, chess.BLACK)
    findings = []
    for sentence in _sentences(_root_paragraph(fen, generation)):
        conditional = bool(_CONDITIONAL.search(sentence))

        for pattern in _SIDE:
            for match in pattern.finditer(sentence):
                side = match.group("side").lower() == "white"
                if conditional:
                    findings.append(Finding("material_side", "uncheckable", "conditional", sentence))
                    continue
                lead = balance if side else -balance
                findings.append(Finding(
                    "material_side", "supported" if lead > 0 else "refuted",
                    "" if lead > 0 else f"{match.group('side')} material lead is {lead:+d}",
                    sentence))

        if _EQUAL.search(sentence):
            if conditional:
                findings.append(Finding("material_equal", "uncheckable", "conditional", sentence))
            else:
                findings.append(Finding(
                    "material_equal", "supported" if abs(balance) < 2 else "refuted",
                    "" if abs(balance) < 2 else f"balance is {balance:+d}", sentence))

        for match in _EXTRA.finditer(sentence):
            n = COUNT_WORDS[match.group("count").lower()]
            for count, piece, owner, coloured in _items(match.group("items"), None):
                if conditional or owner is None or not coloured:
                    findings.append(Finding("material_extra", "uncheckable",
                                            "conditional" if conditional else "no owner", sentence))
                    continue
                surplus = len(board.pieces(piece, owner)) - len(board.pieces(piece, not owner))
                need = n if count == 1 else count
                findings.append(Finding(
                    "material_extra", "supported" if surplus >= need else "refuted",
                    "" if surplus >= need else
                    f"{'white' if owner else 'black'} {chess.piece_name(piece)} surplus is {surplus:+d}",
                    sentence))

        for match in _INVENTORY.finditer(sentence):
            owner_b = match.group("owner").lower() == "white"
            for part, owner in ((match.group("a"), not owner_b), (match.group("b"), owner_b)):
                for count, piece, item_owner, _ in _items(part, owner):
                    if item_owner != owner or conditional:
                        findings.append(Finding("material_count", "uncheckable",
                                                "colour disagrees with list owner"
                                                if item_owner != owner else "conditional", sentence))
                        continue
                    actual = len(board.pieces(piece, owner))
                    findings.append(Finding(
                        "material_count", "supported" if actual == count else "refuted",
                        "" if actual == count else
                        f"{'white' if owner else 'black'} has {actual} {chess.piece_name(piece)}(s), "
                        f"claimed {count}", sentence))
    return findings


_EVAL_PATTERNS = (
    # "White is up approximately 1.35 pawns", "Black is down by about 2 pawns"
    (re.compile(r"\b(?P<side>White|Black)\s+is\s+(?:\w+\s+)?(?P<verb>up|ahead|better|winning|"
                r"down|behind|worse|losing)\b[^.]*?(?P<num>\d+(?:\.\d+)?)\s*pawns?", re.I), "verb"),
    # "Decisively winning for Black, approximately +7.55 pawns"
    (re.compile(r"\b(?:winning|better|advantage|edge|favou?rs?|good)\s+for\s+(?P<side>White|Black)"
                r"\b[^.]*?[+-]?(?P<num>\d+(?:\.\d+)?)\s*pawns?", re.I), "favoured"),
    # "+1.20 pawns from White's perspective"
    (re.compile(r"(?P<sign>[+-])?(?P<num>\d+(?:\.\d+)?)\s*(?:pawns?\s+)?(?:from|for)\s+"
                r"(?P<side>White|Black)['’]s\s+perspective", re.I), "perspective"),
)
_EVAL_MATE = re.compile(r"(?P<sign>[+-])?\s*M(?:ate)?\s*(?P<n>\d+)\s+(?:from|for)\s+"
                        r"(?P<side>White|Black)['’]?s?(?:\s+perspective)?", re.I)
_EVAL_MATES = re.compile(r"\b(?P<side>White|Black)\s+(?:has|delivers|forces|wins\s+with)\s+"
                         r"(?:a\s+)?(?:forced\s+)?(?:check)?mate\b", re.I)
_EVAL_EQUAL = re.compile(r"\b(?:equal|balanced|level|drawn|drawish)\b", re.I)


def parse_evaluation(fen: str, generation: str) -> dict | None:
    """the EVALUATION field as cp for the side to move, or a mate sign. None if its
    unclear.
    """
    turn = chess.Board(fen).turn
    text = detokenize(absolute_tokens(field_map(generation).get("EVALUATION", ""), turn))
    if not text:
        return None

    def result(side_white: bool, sign: int, cp=None, mate=False):
        root_sign = sign if side_white == (turn == chess.WHITE) else -sign
        if mate:
            return dict(kind="mate", root_sign=root_sign, winrate=1.0 if root_sign > 0 else 0.0,
                        text=text)
        root_cp = root_sign * cp
        return dict(kind="pawn", root_cp=root_cp, winrate=expected_winrate(root_cp), text=text)

    match = _EVAL_MATE.search(text)
    if match:
        return result(match.group("side").lower() == "white",
                      -1 if match.group("sign") == "-" else 1, mate=True)
    match = _EVAL_MATES.search(text)
    if match:
        return result(match.group("side").lower() == "white", 1, mate=True)
    for pattern, form in _EVAL_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        side_white = match.group("side").lower() == "white"
        cp = round(float(match.group("num")) * 100)
        if form == "verb":
            sign = 1 if match.group("verb").lower() in ("up", "ahead", "better", "winning") else -1
        elif form == "perspective":
            sign = -1 if match.group("sign") == "-" else 1
        else:
            sign = 1
        return result(side_white, sign, cp)
    # todo: "slightly better for white" with no number ends up as None (uncheckable)
    if _EVAL_EQUAL.search(text) and not re.search(r"\d", text):
        return dict(kind="pawn", root_cp=0, winrate=0.5, text=text)
    return None


def check_evaluation(fen: str, generation: str, reference_cp: int,
                     threshold: float = MISTAKE_DROP) -> tuple[Finding, dict | None]:
    """EVALUATION vs a stockfish root score (cp for the side to move). refuted past
    threshold.
    """
    claim = parse_evaluation(fen, generation)
    if claim is None:
        return Finding("evaluation", "uncheckable", "unparsed evaluation"), None
    reference = expected_winrate(reference_cp)
    gap = abs(claim["winrate"] - reference)
    info = dict(claim, reference_cp=reference_cp, reference_winrate=reference, gap=gap)
    return Finding("evaluation", "refuted" if gap >= threshold else "supported",
                   f"win-rate gap {gap:.2f}", claim["text"]), info


def _engine_plan(fen: str, generation: str):
    # (best key, critical keys) as (fen, uci), stops at the first illegal move
    root = chess.Board(fen)
    pov = root.turn
    fields = field_map(generation)
    best_key = None
    best = next(MOVE.finditer(fields.get("BEST_MOVE", "")), None)
    if best is not None:
        move, _ = pov_move(best, root.copy(stack=False), pov)
        if move is not None:
            best_key = (root.fen(), move.uci())
    critical = []
    board = root.copy(stack=False)
    for atom in MOVE.finditer(fields.get("CRITICAL_LINE", "")):
        if board.is_game_over(claim_draw=False):
            break
        move, _ = pov_move(atom, board, pov)
        if move is None:
            break
        critical.append((board.fen(), move.uci()))
        board.push(move)
    return best_key, critical


def engine_requests(fen: str, generation: str) -> list[tuple[str, str]]:
    """(fen, uci) pairs to audit: BEST_MOVE plus every legal CRITICAL_LINE move."""
    best_key, critical = _engine_plan(fen, generation)
    return list(dict.fromkeys(([best_key] if best_key else []) + critical))


def check_engine(fen: str, generation: str, audits: dict,
                 threshold: float = MISTAKE_DROP) -> list[Finding]:
    """best move and critical line moves against the oracles move audits."""
    best_key, critical = _engine_plan(fen, generation)
    findings = []
    keyed = ([("best_quality", best_key)] if best_key else []) + \
        [("critical_quality", key) for key in critical]
    for kind, key in keyed:
        drop = audits[key]["winrate_drop"]
        findings.append(Finding(kind, "refuted" if drop >= threshold else "supported",
                                f"win-rate drop {drop:.2f}", key[1]))
    return findings


def check_facts(fen: str, generation: str, *, reference_cp: int | None = None,
                audits: dict | None = None) -> FactReport:
    report = FactReport()
    report.findings += check_line_moves(fen, generation)
    report.findings += check_structured_moves(fen, generation)
    report.findings += check_material(fen, generation)
    if reference_cp is not None:
        finding, info = check_evaluation(fen, generation, reference_cp)
        report.findings.append(finding)
        report.evaluation = info
    if audits is not None:
        report.findings += check_engine(fen, generation, audits)
    return report


ERROR_GROUPS = {
    "line_illegal": ("line_move",),
    "critical_illegal": ("critical_move", "best_move", "promising_move"),
    "material_wrong": ("material_side", "material_extra", "material_count", "material_equal"),
    "eval_wrong": ("evaluation",),
    "best_mistake": ("best_quality",),
    "critical_mistakes": ("critical_quality",),
}


def fact_errors(by_kind: dict[str, dict[str, int]]) -> int:
    """refuted findings over ERROR_GROUPS."""
    return sum(by_kind.get(kind, {}).get("refuted", 0)
               for kinds in ERROR_GROUPS.values() for kind in kinds)


def fact_supported(by_kind: dict[str, dict[str, int]]) -> int:
    return sum(by_kind.get(kind, {}).get("supported", 0)
               for kinds in ERROR_GROUPS.values() for kind in kinds)


def summarize(reports: list[FactReport]) -> dict:
    kinds: Counter = Counter()
    for report in reports:
        for f in report.findings:
            kinds[(f.kind, f.status)] += 1
    names = sorted({kind for kind, _ in kinds})
    return {name: {status: kinds[(name, status)] for status in ("supported", "refuted", "uncheckable")}
            for name in names}
