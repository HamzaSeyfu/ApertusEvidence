from __future__ import annotations

from dataclasses import dataclass

from .analysis import ConservativeHeuristicAnalyst
from .models import (
    AssessmentLabel,
    Claim,
    EvidenceAssessment,
    FactCheckResult,
    Verdict,
)
from .retrieval import InMemoryEvidenceIndex


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    top_k: int = 6
    min_retrieval_score: float = 0.20
    decision_threshold: float = 0.78


class EvidenceCourt:
    """Evidence-first orchestration layer.

    Retrieval and semantic analysis are separate on purpose. The heuristic
    analyst can later be replaced by an Apertus-backed structured analyst.
    """

    def __init__(
        self,
        index: InMemoryEvidenceIndex,
        *,
        analyst: ConservativeHeuristicAnalyst | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        self.index = index
        self.analyst = analyst or ConservativeHeuristicAnalyst()
        self.config = config or PipelineConfig()

    def check(self, claim_text: str) -> FactCheckResult:
        claim = Claim(claim_text)
        retrieved = [
            item
            for item in self.index.search(claim.text, top_k=self.config.top_k)
            if item.retrieval_score >= self.config.min_retrieval_score
        ]

        if not retrieved:
            return FactCheckResult(
                claim=claim,
                verdict=Verdict.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                audit_notes=("No retrieved passage met the minimum evidence score.",),
            )

        assessments = tuple(self.analyst.assess(claim, evidence) for evidence in retrieved)
        supporting = tuple(
            item for item in assessments if item.label == AssessmentLabel.SUPPORT
        )
        contradicting = tuple(
            item for item in assessments if item.label == AssessmentLabel.CONTRADICTION
        )
        unclear = tuple(
            item for item in assessments if item.label == AssessmentLabel.UNCLEAR
        )

        support_strength = max((item.score for item in supporting), default=0.0)
        contradiction_strength = max((item.score for item in contradicting), default=0.0)
        threshold = self.config.decision_threshold

        if support_strength >= threshold and contradiction_strength >= threshold:
            verdict = Verdict.MIXED
            confidence = min(support_strength, contradiction_strength)
            note = "Strong evidence exists on both sides; returning MIXED."
        elif support_strength >= threshold:
            verdict = Verdict.SUPPORTED
            confidence = support_strength
            note = "At least one supporting passage cleared the decision threshold."
        elif contradiction_strength >= threshold:
            verdict = Verdict.CONTRADICTED
            confidence = contradiction_strength
            note = "At least one contradicting passage cleared the decision threshold."
        else:
            verdict = Verdict.INSUFFICIENT_EVIDENCE
            confidence = max(support_strength, contradiction_strength)
            note = "Retrieved evidence did not justify a decisive verdict."

        citation_notes = self._audit_citations(supporting + contradicting)
        return FactCheckResult(
            claim=claim,
            verdict=verdict,
            confidence=round(confidence, 4),
            supporting=supporting,
            contradicting=contradicting,
            unclear=unclear,
            audit_notes=(note, *citation_notes),
        )

    @staticmethod
    def _audit_citations(
        assessments: tuple[EvidenceAssessment, ...],
    ) -> tuple[str, ...]:
        invalid = [
            item
            for item in assessments
            if not item.evidence.source_id.strip()
            or not item.evidence.locator.strip()
            or not item.evidence.text.strip()
        ]
        if invalid:
            return (f"Citation audit failed for {len(invalid)} evidence item(s).",)
        if assessments:
            return ("Citation audit passed: all decisive evidence has provenance.",)
        return ("Citation audit: no decisive evidence to validate.",)
