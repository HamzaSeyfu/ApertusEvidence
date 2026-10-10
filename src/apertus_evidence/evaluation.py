from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence

from .models import FactCheckResult, Verdict


class FactChecker(Protocol):
    def check(self, claim_text: str) -> FactCheckResult:
        ...


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    claim: str
    expected_verdict: Verdict


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    total: int
    correct: int
    accuracy: float
    macro_f1: float
    decisive_rate: float
    abstention_rate: float
    per_label: dict[str, dict[str, float | int]]
    failures: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "decisive_rate": self.decisive_rate,
            "abstention_rate": self.abstention_rate,
            "per_label": self.per_label,
            "failures": list(self.failures),
        }


def load_cases(path: str | Path) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                verdict = Verdict(str(payload["expected_verdict"]).upper())
                claim = str(payload["claim"]).strip()
                if not claim:
                    raise ValueError("claim cannot be empty")
                cases.append(
                    EvaluationCase(
                        case_id=str(payload.get("case_id") or f"case-{line_number}"),
                        claim=claim,
                        expected_verdict=verdict,
                    )
                )
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(
                    f"Invalid evaluation case at {source}:{line_number}: {exc}"
                ) from exc

    if not cases:
        raise ValueError(f"No evaluation cases found in {source}")
    return cases


def evaluate(
    court: FactChecker,
    cases: Iterable[EvaluationCase],
) -> EvaluationSummary:
    materialized = list(cases)
    predictions = [court.check(case.claim).verdict for case in materialized]
    return summarize_predictions(materialized, predictions)


def summarize_predictions(
    cases: Sequence[EvaluationCase],
    predictions: Sequence[Verdict],
) -> EvaluationSummary:
    if len(cases) != len(predictions):
        raise ValueError("cases and predictions must have the same length")
    if not cases:
        raise ValueError("at least one evaluation case is required")

    labels = list(Verdict)
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}
    failures: list[dict[str, str]] = []
    correct = 0
    abstentions = 0

    for case, predicted in zip(cases, predictions, strict=True):
        expected = case.expected_verdict

        if predicted == Verdict.INSUFFICIENT_EVIDENCE:
            abstentions += 1

        if predicted == expected:
            correct += 1
        else:
            failures.append(
                {
                    "case_id": case.case_id,
                    "claim": case.claim,
                    "expected": expected.value,
                    "predicted": predicted.value,
                }
            )

        for label in labels:
            if predicted == label and expected == label:
                counts[label]["tp"] += 1
            elif predicted == label and expected != label:
                counts[label]["fp"] += 1
            elif predicted != label and expected == label:
                counts[label]["fn"] += 1

    per_label: dict[str, dict[str, float | int]] = {}
    f1_scores: list[float] = []
    for label in labels:
        tp = counts[label]["tp"]
        fp = counts[label]["fp"]
        fn = counts[label]["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_label[label.value] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        f1_scores.append(f1)

    total = len(cases)
    abstention_rate = abstentions / total
    return EvaluationSummary(
        total=total,
        correct=correct,
        accuracy=round(correct / total, 4),
        macro_f1=round(sum(f1_scores) / len(f1_scores), 4),
        decisive_rate=round(1.0 - abstention_rate, 4),
        abstention_rate=round(abstention_rate, 4),
        per_label=per_label,
        failures=tuple(failures),
    )
