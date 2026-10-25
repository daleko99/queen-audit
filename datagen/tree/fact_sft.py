"""builds fine tuning sets from K sampled analyses per position.

needs a run dir with sample.json, generations.json and facts.jsonl (from eval.facts_eval
check). writes these over the same positions:

- fact.jsonl: per position, the sample with the fewest fact errors (or any with at most
  --max-errors), preferring more supported facts so vague analyses dont win
- control.jsonl: a random sample of the same position
- dpo.jsonl: the fact pick vs the positions worst sample, when they differ by at least
  --min-gap errors

    python -m datagen.tree.fact_sft --run samples --exclude eval_positions.jsonl \\
        --output sets
"""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from datagen.self_distill.io import provenance
from datagen.tree import facts


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _jsonl(path):
    with open(path, encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_scored(run_dir):
    run_dir = Path(run_dir)
    sample = {row["sample_id"]: row for row in _read(run_dir / "sample.json")}
    fact_rows = {row["sample_id"]: row for row in _jsonl(run_dir / "facts.jsonl")}
    scored = []
    for row in _read(run_dir / "generations.json"):
        sid = row["sample_id"]
        # skip samples that hit the token cap, theyre cut off mid analysis
        if row.get("hit_token_cap") or sid not in fact_rows or "ANALYSIS" not in row["generation"]:
            continue
        by_kind = fact_rows[sid]["by_kind"]
        scored.append(dict(sample_id=sid, fen=sample[sid]["fen"], prompt=sample[sid]["prompt"],
                           generation=row["generation"],
                           fact_errors=facts.fact_errors(by_kind),
                           fact_supported=facts.fact_supported(by_kind),
                           line_errors=sum(by_kind.get(kind, {}).get("refuted", 0)
                                           for kind in facts.ERROR_GROUPS["line_illegal"])))
    return scored


def select(scored, *, max_errors=None, exclude=(), seed=0, prefer_uncut=False):
    """(fact_rows, control_rows, stats). prefer_uncut favours samples with fewer illegal
    line moves, since correction cuts a paragraph at each one.
    """
    rng = random.Random(seed)
    by_fen = defaultdict(list)
    for row in scored:
        if row["fen"] not in exclude:
            by_fen[row["fen"]].append(row)
    fact_rows, control_rows = [], []
    for fen in sorted(by_fen):
        group = by_fen[fen]
        limit = min(row["fact_errors"] for row in group) if max_errors is None else max_errors
        clean = [row for row in group if row["fact_errors"] <= limit]
        if not clean:
            continue
        if prefer_uncut:
            fewest = min(row.get("line_errors", 0) for row in clean)
            clean = [row for row in clean if row.get("line_errors", 0) == fewest]
        best = max(row["fact_supported"] for row in clean)
        fact_rows.append(rng.choice([row for row in clean if row["fact_supported"] == best]))
        control_rows.append(rng.choice(group))
    stats = dict(positions=len(by_fen), kept=len(fact_rows),
                 samples=sum(len(group) for group in by_fen.values()))
    for name, rows in (("fact", fact_rows), ("control", control_rows)):
        n = max(len(rows), 1)
        stats[name] = dict(mean_fact_errors=round(sum(r["fact_errors"] for r in rows) / n, 3),
                           mean_fact_supported=round(sum(r["fact_supported"] for r in rows) / n, 3),
                           clean_share=round(sum(r["fact_errors"] == 0 for r in rows) / n, 3))
    return fact_rows, control_rows, stats


def preference_pairs(fact_rows, scored, *, min_gap=2, seed=0):
    rng = random.Random(seed)
    by_fen = defaultdict(list)
    for row in scored:
        by_fen[row["fen"]].append(row)
    pairs = []
    for chosen in fact_rows:
        worst = max(row["fact_errors"] for row in by_fen[chosen["fen"]])
        if worst - chosen["fact_errors"] < min_gap:
            continue
        rejected = rng.choice([row for row in by_fen[chosen["fen"]] if row["fact_errors"] == worst])
        pairs.append((chosen, rejected))
    return pairs


def _record(row):
    return dict(id=hashlib.sha1(row["sample_id"].encode()).hexdigest()[:16], fen=row["fen"],
                prompt=row["prompt"], chosen=row["generation"], sample_id=row["sample_id"],
                fact_errors=row["fact_errors"], fact_supported=row["fact_supported"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run dir with sample/generations/facts")
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude", action="append", default=[],
                        help="positions.jsonl whose FENs are held out (repeatable)")
    parser.add_argument("--max-errors", type=int, default=None,
                        help="keep samples with at most N errors (default: best per position)")
    parser.add_argument("--min-gap", type=int, default=2,
                        help="minimum fact-error gap for a DPO pair")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefer-uncut", action="store_true",
                        help="break ties toward samples with fewer illegal line moves")
    args = parser.parse_args(argv)
    exclude = {row["fen"] for path in args.exclude for row in _jsonl(path)}
    scored = load_scored(args.run)
    fact_rows, control_rows, stats = select(scored, max_errors=args.max_errors,
                                            exclude=exclude, seed=args.seed,
                                            prefer_uncut=args.prefer_uncut)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("fact", fact_rows), ("control", control_rows)):
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(_record(row)) + "\n")
    manifest = dict(source_run=args.run, excluded_positions=len(exclude), exclude=args.exclude,
                    max_errors=args.max_errors, min_gap=args.min_gap, seed=args.seed,
                    prefer_uncut=args.prefer_uncut,
                    provenance=provenance(["princeton-nlp/queen_pawn-8 (self-samples)"],
                                          labels="datagen.tree.facts + Stockfish (external)"),
                    stats=stats)
    pairs = preference_pairs(fact_rows, scored, min_gap=args.min_gap, seed=args.seed)
    with (output / "dpo.jsonl").open("w", encoding="utf-8") as handle:
        for chosen, rejected in pairs:
            handle.write(json.dumps(dict(_record(chosen), rejected=rejected["generation"],
                                         rejected_sample_id=rejected["sample_id"],
                                         rejected_fact_errors=rejected["fact_errors"])) + "\n")
    stats["dpo"] = dict(pairs=len(pairs), mean_gap=round(
        sum(r["fact_errors"] - c["fact_errors"] for c, r in pairs) / max(len(pairs), 1), 3))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
