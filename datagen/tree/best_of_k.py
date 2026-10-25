"""picks the best of K sampled analyses, then corrects and renumbers it.

the sample with the fewest fact errors wins (ties go to fewer illegal line moves, then
more supported facts, then the earlier sample). engine work is left to the caller: pass
the root score as reference_cp and the move audits from eval.facts_eval audit as audits,
or leave them out to skip those checks.

    python -m datagen.tree.best_of_k --run samples --k 4 --audits audits.json \\
        --like positions --output picked
"""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

from datagen.tree import facts
from datagen.tree.fact_correct import correct
from datagen.tree.move_numbers import renumber

MIN_KEPT_SENTENCES = 2


def engine_requests(fen, samples):
    return list(dict.fromkeys(key for text in samples
                              for key in facts.engine_requests(fen, text)))


def score(fen, text, *, reference_cp=None, audits=None):
    by_kind = facts.check_facts(fen, text, reference_cp=reference_cp, audits=audits).by_kind()
    line_errors = sum(by_kind.get(kind, {}).get("refuted", 0)
                      for kind in facts.ERROR_GROUPS["line_illegal"])
    return facts.fact_errors(by_kind), line_errors, facts.fact_supported(by_kind)


def choose(fen, samples, *, reference_cp=None, audits=None):
    """(index, scores) of the best sample. samples with no ANALYSIS field score None."""
    scores = [score(fen, text, reference_cp=reference_cp, audits=audits)
              if "ANALYSIS" in text else None for text in samples]
    usable = [i for i, s in enumerate(scores) if s is not None]
    if not usable:
        return None, scores
    return min(usable, key=lambda i: (scores[i][0], scores[i][1], -scores[i][2], i)), scores


def explain(fen, samples, *, reference_cp=None, audits=None,
            min_kept_sentences=MIN_KEPT_SENTENCES):
    """dict(text, index, fact_errors, correction, renumber_offset), or None if no sample is
    usable.
    """
    index, scores = choose(fen, samples, reference_cp=reference_cp, audits=audits)
    if index is None:
        return None
    text, log = correct(fen, samples[index], reference_cp, min_kept_sentences=min_kept_sentences)
    text, offset = renumber(fen, text)
    return dict(text=text, index=index, fact_errors=scores[index][0], correction=log,
                renumber_offset=offset)


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main(argv=None):
    from eval.facts_eval import load_audits

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="dir with sample.json, generations.json, annotations.json")
    parser.add_argument("--k", type=int, required=True, help="use the first k samples (ids ending -r00, -r01, ...)")
    parser.add_argument("--audits", help="move audits from eval.facts_eval (optional)")
    parser.add_argument("--like", required=True,
                        help="dir whose positions and sample ids the output uses")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-kept-sentences", type=int, default=MIN_KEPT_SENTENCES)
    args = parser.parse_args(argv)

    run = Path(args.run)
    fens = {row["sample_id"]: row["fen"] for row in _read(run / "sample.json")}
    cps = {row["sample_id"]: row.get("best_cp") for row in _read(run / "annotations.json")}
    by_fen = defaultdict(dict)
    for row in _read(run / "generations.json"):
        repeat = int(row["sample_id"].rsplit("-r", 1)[1])
        if repeat < args.k:
            text = "" if row.get("hit_token_cap") else row["generation"]
            by_fen[fens[row["sample_id"]]][repeat] = (text, cps.get(row["sample_id"]))
    audits = load_audits(args.audits) if args.audits else None

    like = Path(args.like)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    generations, chosen_errors, missing = [], [], 0
    for row in _read(like / "sample.json"):
        repeats = by_fen.get(row["fen"], {})
        samples = [repeats[i][0] for i in sorted(repeats)]
        reference_cp = next((cp for _, cp in repeats.values() if cp is not None), None)
        result = explain(row["fen"], samples, reference_cp=reference_cp, audits=audits,
                         min_kept_sentences=args.min_kept_sentences) if samples else None
        if result is None:
            missing += 1
            continue
        generations.append(dict(sample_id=row["sample_id"], generation=result["text"]))
        chosen_errors.append(result["fact_errors"])
    shutil.copy(like / "sample.json", output / "sample.json")
    if (like / "annotations.json").exists():
        shutil.copy(like / "annotations.json", output / "annotations.json")
    (output / "generations.json").write_text(json.dumps(generations), encoding="utf-8")
    n = max(len(chosen_errors), 1)
    print(json.dumps(dict(k=args.k, positions=len(generations), missing=missing,
                          chosen_mean_fact_errors=round(sum(chosen_errors) / n, 3),
                          chosen_clean_share=round(sum(e == 0 for e in chosen_errors) / n, 3))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
