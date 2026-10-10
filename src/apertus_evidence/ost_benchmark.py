from __future__ import annotations

import json
from pathlib import Path

from .analysis import EvidenceAnalyst
from .evaluation import EvaluationCase, EvaluationSummary, summarize_predictions
from .models import AssessmentLabel, Claim, EvidenceSpan, Verdict


def evaluate_ost_pairs(
    analyst: EvidenceAnalyst,
    *,
    evidence_path: str | Path,
    cases_path: str | Path,
) -> EvaluationSummary:
    """Evaluate NLI reasoning on each OST claim and its paired reference.

    This intentionally bypasses retrieval. It measures whether the reasoning
    layer can classify the official reference provided by the benchmark.
    End-to-end retrieval quality is evaluated separately with the generic
    evaluate command.
    """

    evidence_by_locator = _load_evidence(evidence_path)
    cases: list[EvaluationCase] = []
    predictions: list[Verdict] = []

    source = Path(cases_path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            metadata = payload.get("metadata") or {}
            try:
                claim_text = str(payload["claim"]).strip()
                case_id = str(payload.get("case_id") or f"case-{line_number}")
                expected = Verdict(str(payload["expected_verdict"]).upper())
                locators = metadata["paired_locators"]
                source_id = str(metadata["paired_source_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid OST evaluation case at {source}:{line_number}"
                ) from exc

            if not claim_text:
                raise ValueError(
                    f"Empty claim at OST evaluation case {source}:{line_number}"
                )
            if not isinstance(locators, list) or not locators:
                raise ValueError(
                    f"Missing paired_locators at OST evaluation case "
                    f"{source}:{line_number}"
                )

            passages: list[str] = []
            for locator in locators:
                record = evidence_by_locator.get(str(locator))
                if record is None:
                    raise ValueError(
                        f"Unknown paired evidence locator {locator!r} "
                        f"at OST case {case_id}"
                    )
                passages.append(str(record["text"]))

            evidence = EvidenceSpan(
                text="\n\n".join(passages),
                source_id=source_id,
                locator=";".join(str(locator) for locator in locators),
                retrieval_score=1.0,
                metadata={
                    "benchmark_mode": "paired_reference",
                    **dict(metadata),
                },
            )
            assessment = analyst.assess(Claim(claim_text), evidence)
            predicted = _assessment_to_verdict(assessment.label)

            cases.append(
                EvaluationCase(
                    case_id=case_id,
                    claim=claim_text,
                    expected_verdict=expected,
                )
            )
            predictions.append(predicted)

    return summarize_predictions(cases, predictions)


def _load_evidence(path: str | Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                locator = str(payload["locator"])
                text = str(payload["text"])
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f"Invalid evidence record at {source}:{line_number}"
                ) from exc
            records[locator] = {"text": text, **payload}
    if not records:
        raise ValueError(f"No evidence records found in {source}")
    return records


def _assessment_to_verdict(label: AssessmentLabel) -> Verdict:
    if label == AssessmentLabel.SUPPORT:
        return Verdict.SUPPORTED
    if label == AssessmentLabel.CONTRADICTION:
        return Verdict.CONTRADICTED
    return Verdict.INSUFFICIENT_EVIDENCE
