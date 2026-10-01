from __future__ import annotations

import re
from typing import Mapping, Protocol

from .models import AssessmentLabel, Claim, EvidenceAssessment, EvidenceSpan
from .prompts import EVIDENCE_ANALYST_SYSTEM, build_evidence_analysis_prompt
from .retrieval import tokenize

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were",
    "with", "de", "des", "du", "et", "la", "le", "les", "un", "une", "est",
    "sont", "dans", "pour", "sur", "avec", "que", "qui",
}
_NEGATIONS = {"not", "no", "never", "without", "ne", "pas", "aucun", "aucune", "jamais"}
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")


class EvidenceAnalyst(Protocol):
    def assess(self, claim: Claim, evidence: EvidenceSpan) -> EvidenceAssessment:
        ...


class JsonGenerationBackend(Protocol):
    """Provider-neutral contract for Apertus structured generation."""

    def generate_json(self, *, system: str, user: str) -> Mapping[str, object]:
        ...


class ApertusEvidenceAnalyst:
    """Structured semantic analyst backed by an Apertus inference provider.

    The provider itself is intentionally not hard-coded here. Hackathon compute
    can therefore be swapped between CSCS, Hugging Face or a local endpoint.
    """

    def __init__(self, backend: JsonGenerationBackend) -> None:
        self.backend = backend

    def assess(self, claim: Claim, evidence: EvidenceSpan) -> EvidenceAssessment:
        try:
            payload = self.backend.generate_json(
                system=EVIDENCE_ANALYST_SYSTEM,
                user=build_evidence_analysis_prompt(claim, evidence),
            )
            raw_label = str(payload.get("label", "UNCLEAR")).upper()
            label = AssessmentLabel(raw_label)
            score = float(payload.get("score", 0.0))
            rationale = str(payload.get("rationale", "")).strip()

            if not 0.0 <= score <= 1.0:
                raise ValueError("score must be in [0, 1]")
            if label == AssessmentLabel.UNCLEAR:
                score = 0.0
            if not rationale:
                rationale = "Apertus returned no rationale; assessment downgraded."
                label = AssessmentLabel.UNCLEAR
                score = 0.0

            return EvidenceAssessment(
                label=label,
                score=round(score, 4),
                rationale=rationale,
                evidence=evidence,
            )
        except (TypeError, ValueError, KeyError) as exc:
            return EvidenceAssessment(
                label=AssessmentLabel.UNCLEAR,
                score=0.0,
                rationale=f"Structured model output rejected: {exc}",
                evidence=evidence,
            )


class ConservativeHeuristicAnalyst:
    """Offline baseline.

    It intentionally abstains often. The Apertus analyst replaces semantic
    judgment while keeping the same structured contract.
    """

    def assess(self, claim: Claim, evidence: EvidenceSpan) -> EvidenceAssessment:
        claim_terms = _content_terms(claim.text)
        evidence_terms = _content_terms(evidence.text)

        if not claim_terms:
            return self._unclear(evidence, "Claim has no analyzable content terms.")

        overlap = len(claim_terms & evidence_terms) / len(claim_terms)
        if overlap < 0.55:
            return self._unclear(
                evidence,
                f"Low lexical coverage ({overlap:.2f}); refusing semantic inference.",
            )

        claim_numbers = set(_NUMBER_RE.findall(claim.text))
        evidence_numbers = set(_NUMBER_RE.findall(evidence.text))
        if claim_numbers and evidence_numbers and not claim_numbers.issubset(evidence_numbers):
            score = min(0.98, 0.65 + 0.30 * overlap)
            return EvidenceAssessment(
                label=AssessmentLabel.CONTRADICTION,
                score=round(score, 4),
                rationale="High lexical overlap but conflicting explicit numeric values.",
                evidence=evidence,
            )

        claim_negated = bool(set(tokenize(claim.text)) & _NEGATIONS)
        evidence_negated = bool(set(tokenize(evidence.text)) & _NEGATIONS)
        if claim_negated != evidence_negated and overlap >= 0.65:
            score = min(0.95, 0.62 + 0.28 * overlap)
            return EvidenceAssessment(
                label=AssessmentLabel.CONTRADICTION,
                score=round(score, 4),
                rationale="High lexical overlap with a negation mismatch.",
                evidence=evidence,
            )

        if overlap >= 0.75:
            score = min(0.95, 0.58 + 0.34 * overlap)
            return EvidenceAssessment(
                label=AssessmentLabel.SUPPORT,
                score=round(score, 4),
                rationale="Strong lexical coverage with compatible explicit values.",
                evidence=evidence,
            )

        return self._unclear(
            evidence,
            f"Partial lexical coverage ({overlap:.2f}); baseline abstains.",
        )

    @staticmethod
    def _unclear(evidence: EvidenceSpan, rationale: str) -> EvidenceAssessment:
        return EvidenceAssessment(
            label=AssessmentLabel.UNCLEAR,
            score=0.0,
            rationale=rationale,
            evidence=evidence,
        )


def _content_terms(text: str) -> set[str]:
    return {
        token
        for token in tokenize(text)
        if token not in _STOPWORDS and len(token) > 1
    }
