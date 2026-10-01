from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from enum import Enum
from typing import Any

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
        description="Evidence-first fact checking over a local official-document corpus.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Check one claim against a local corpus.")
    check.add_argument("--claim", required=True, help="Factual claim to verify.")
    check.add_argument("--corpus", required=True, help="Directory containing .txt/.md sources.")
    check.add_argument("--json", action="store_true", help="Emit structured JSON.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "check":
        index = InMemoryEvidenceIndex.from_directory(args.corpus)
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
