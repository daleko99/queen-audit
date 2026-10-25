"""runs the fact checker over one or more run dirs.

a run dir has sample.json and generations.json, plus annotations.json with stockfish
root scores for the evaluation check.

    # engine audits of BEST_MOVE and CRITICAL_LINE moves, cached by (fen, uci)
    python -m eval.facts_eval audit --run base=outputs --audits audits.json

    # findings per analysis -> <run>/facts.jsonl, plus a summary table
    python -m eval.facts_eval check --run base=outputs --audits audits.json
"""
import argparse
import json
from pathlib import Path

from datagen.tree import facts


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_run(run_dir):
    run_dir = Path(run_dir)
    sample = _read(run_dir / "sample.json")
    generations = {row["sample_id"]: row["generation"] for row in _read(run_dir / "generations.json")}
    annotations = {}
    if (run_dir / "annotations.json").exists():
        annotations = {row["sample_id"]: row for row in _read(run_dir / "annotations.json")}
    return [dict(sample_id=row["sample_id"], fen=row["fen"],
                 generation=generations[row["sample_id"]],
                 reference_cp=(annotations.get(row["sample_id"]) or {}).get("best_cp"))
            for row in sample if row["sample_id"] in generations]


def _key(fen, uci):
    return f"{fen}|{uci}"


def load_audits(path):
    path = Path(path)
    if not path.exists():
        return {}
    return {tuple(key.split("|", 1)): value for key, value in _read(path).items()}


def audit(runs, audits_path, stockfish, workers, nodes, chunk=2000):
    from datagen.self_distill.oracle import ParallelOracle

    audits = load_audits(audits_path)
    requests = []
    for records in runs.values():
        for record in records:
            requests += facts.engine_requests(record["fen"], record["generation"])
    missing = [key for key in dict.fromkeys(requests) if key not in audits]
    print(f"[audit] {len(set(requests))} unique moves, {len(missing)} to search", flush=True)
    oracle = ParallelOracle(Path(stockfish), workers, nodes)
    try:
        for start in range(0, len(missing), chunk):
            audits.update(oracle.move_audits(missing[start:start + chunk]))
            # save after every chunk so a crash doesnt lose hours of searches
            Path(audits_path).write_text(json.dumps(
                {_key(*key): value for key, value in audits.items()}), encoding="utf-8")
            print(f"[audit] {min(start + chunk, len(missing))}/{len(missing)}", flush=True)
    finally:
        oracle.close()


def record_row(record, audits):
    report = facts.check_facts(record["fen"], record["generation"],
                               reference_cp=record["reference_cp"],
                               audits=audits if audits else None)
    return dict(sample_id=record["sample_id"], by_kind=report.by_kind(),
                evaluation=report.evaluation,
                refuted=[dict(kind=f.kind, detail=f.detail, text=f.text)
                         for f in report.findings if f.status == "refuted"])


def check(runs, run_dirs, audits_path):
    audits = load_audits(audits_path) if audits_path else {}
    table = {}
    for label, records in runs.items():
        rows = [record_row(record, audits) for record in records]
        with (Path(run_dirs[label]) / "facts.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        totals = {}
        for row in rows:
            for kind, counts in row["by_kind"].items():
                slot = totals.setdefault(kind, {"supported": 0, "refuted": 0, "records": 0})
                slot["supported"] += counts["supported"]
                slot["refuted"] += counts["refuted"]
                slot["records"] += bool(counts["refuted"])
        table[label] = totals
    kinds = sorted({kind for totals in table.values() for kind in totals})
    print("refuted / checkable (records with >=1 refuted, of n)")
    print(f"{'kind':18}" + "".join(f"{label:>24}" for label in table))
    for kind in kinds:
        cells = []
        for label, totals in table.items():
            slot = totals.get(kind, {"supported": 0, "refuted": 0, "records": 0})
            checkable = slot["supported"] + slot["refuted"]
            rate = f"{slot['refuted'] / checkable:.1%}" if checkable else "n/a"
            cells.append(f"{rate} ({slot['records']}/{len(runs[label])})")
        print(f"{kind:18}" + "".join(f"{cell:>24}" for cell in cells))
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=("audit", "check"))
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--audits", default=None, help="engine audit cache (json)")
    parser.add_argument("--stockfish", default="stockfish", help="stockfish binary (default: whatever is on the PATH)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--nodes", type=int, default=100_000)
    args = parser.parse_args(argv)
    run_dirs = dict(spec.split("=", 1) for spec in args.run)
    runs = {label: load_run(path) for label, path in run_dirs.items()}
    if args.phase == "audit":
        if not args.audits:
            raise SystemExit("the audit phase needs --audits")
        audit(runs, args.audits, args.stockfish, args.workers, args.nodes)
    else:
        check(runs, run_dirs, args.audits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
