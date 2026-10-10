from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from enum import Enum
from typing import Any

from .agents import AdversarialEvidenceCourt, CitationAuditor
from .analysis import ApertusEvidenceAnalyst
from .backend import OpenAICompatibleJsonBackend\nfrom .decomposition import ApertusClaimDecomposer
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


def _add_reasoning_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reasoner",
        choices=("heuristic", "apertus"),
        default="heuristic",
        help="Semantic evidence reasoner (default: heuristic).",
    )
    parser.add_argument(
        "--apertus-base-url",
        help="OpenAI-compatible Apertus endpoint. Can also use APERTUS_BASE_URL.",
    )
    parser.add_argument(
        "--apertus-model",
        help="Apertus model identifier. Can also use APERTUS_MODEL.",
    )
    parser.add_argument(
        "--api-key-env",
        default="APERTUS_API_KEY",
        help="Environment variable containing an optional bearer token.",
    )
    parser.add_argument(
        "--court",
        choices=("single", "adversarial"),
        default="single",
        help="Use single-pass analysis or the two-sided Evidence Court.",
    )


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
    _add_reasoning_options(evaluate)

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
    _add_reasoning_options(check)
    return parser


def _build_court(args: argparse.Namespace, index: InMemoryEvidenceIndex) -> EvidenceCourt:
    if args.reasoner == "heuristic":
        return EvidenceCourt(index)

    backend = OpenAICompatibleJsonBackend.from_env(
        base_url=args.apertus_base_url,
        model=args.apertus_model,
        api_key_env=args.api_key_env,
    )
    if args.court == "adversarial":
        return AdversarialEvidenceCourt(
            index,
            backend,
            decomposer=ApertusClaimDecomposer(backend),
            citation_auditor=CitationAuditor(backend),
        )
    return EvidenceCourt(index, analyst=ApertusEvidenceAnalyst(backend))


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "ingest":
        count = ingest_path(args.input, args.output, chunk_chars=args.chunk_chars)
        print(f"Wrote {count} evidence chunk(s) to {args.output}")
        return 0

    if args.command == "evaluate":
        from .evaluation import evaluate, load_cases

        index = InMemoryEvidenceIndex.from_jsonl(args.index)
        court = _build_court(args, index)
        summary = evaluate(court, load_cases(args.cases))
        if args.json:
            print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(f"Cases:          {summary.total}")
            print(f"Correct:        {summary.correct}")
            print(f"Accuracy:       {summary.accuracy:.3f}")
            print(f"Macro F1:       {summary.macro_f1:.3f}")
            print(f"Decisive rate:  {summary.decisive_rate:.3f}")
            print(f"Abstention:     {summary.abstention_rate:.3f}")
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
        court = _build_court(args, index)
        result = court.check(args.claim)

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
