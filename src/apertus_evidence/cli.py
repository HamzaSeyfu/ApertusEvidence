from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from enum import Enum
from typing import Any

from .ingest import ingest_path
from .pipeline import EvidenceCourt
from .retrieval import InMemoryEvidenceIndex


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apertus-evidence",
        description="Evidence-first fact checking over official source material.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser(
        "ingest",
        help="Normalize PDF/TXT/MD source documents into a provenance-rich JSONL corpus.",
    )
    ingest.add_argument("--input", required=True, help="Source file or directory.")
    ingest.add_argument("--output", required=True, help="Output JSONL corpus.")
    ingest.add_argument(
        "--chunk-chars",
        type=int,
        default=1400,
        help="Approximate maximum characters per chunk (default: 1400).",
    )

    evaluate = sub.add_parser(
        "evaluate",
        help="Evaluate verdict quality against labeled JSONL claims.",
    )
    evaluate.add_argument("--index", required=True, help="Normalized evidence JSONL.")
    evaluate.add_argument("--cases", required=True, help="Labeled claim JSONL.")
    evaluate.add_argument("--json", action="store_true", help="Emit structured JSON.")

    check = sub.add_parser("check", help="Check one claim against official evidence.")
    check.add_argument("--claim", required=True, help="Factual claim to verify.")
    source_group = check.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--corpus",
        help="Directory containing local .txt/.md sources (legacy quick path).",
    )
    source_group.add_argument(
        "--index",
        help="Normalized JSONL corpus produced by the ingest command.",
    )
    check.add_argument("--json", action="store_true", help="Emit structured JSON.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "ingest":
        count = ingest_path(args.input, args.output, chunk_chars=args.chunk_chars)
        print(f"Wrote {count} evidence chunk(s) to {args.output}")
        return 0

    if args.command == "evaluate":
        from .evaluation import evaluate_files

        summary = evaluate_files(index_path=args.index, cases_path=args.cases)
        if args.json:
            print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(f"Cases:     {summary.total}")
            print(f"Correct:   {summary.correct}")
            print(f"Accuracy:  {summary.accuracy:.3f}")
            print(f"Macro F1:  {summary.macro_f1:.3f}")
            if summary.failures:
                print("\nFailures:")
                for failure in summary.failures:
                    print(
                        f"  - {failure['case_id']}: expected={failure['expected']} "
                        f"predicted={failure['predicted']}"
                    )
        return 0

    if args.command == "check":
        index = (
            InMemoryEvidenceIndex.from_jsonl(args.index)
            if args.index
            else InMemoryEvidenceIndex.from_directory(args.corpus)
        )
        result = EvidenceCourt(index).check(args.claim)

        if args.json:
            print(json.dumps(_jsonable(asdict(result)), indent=2, ensure_ascii=False))
            return 0

        print(f"Claim:      {result.claim.text}")
        print(f"Verdict:    {result.verdict.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print()

        for title, items in (
            ("Supporting evidence", result.supporting),
            ("Contradicting evidence", result.contradicting),
        ):
            print(f"{title}:")
            if not items:
                print("  (none)")
            for item in items:
                print(
                    f"  - [{item.evidence.locator}] score={item.score:.2f} "
                    f"retrieval={item.evidence.retrieval_score:.2f}"
                )
                print(f"    {item.evidence.text}")
            print()

        print("Audit:")
        for note in result.audit_notes:
            print(f"  - {note}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
