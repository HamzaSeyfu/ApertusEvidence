from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from hashlib import sha256
from typing import Any


class Verdict(StrEnum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    MIXED = "MIXED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class AssessmentLabel(StrEnum):
    SUPPORT = "SUPPORT"
    CONTRADICTION = "CONTRADICTION"
    UNCLEAR = "UNCLEAR"


@dataclass(frozen=True, slots=True)
class Claim:
    text: str
    claim_id: str = ""

    def __post_init__(self) -> None:
        cleaned = " ".join(self.text.split())
        if not cleaned:
            raise ValueError("Claim text cannot be empty.")
        object.__setattr__(self, "text", cleaned)
        if not self.claim_id:
            digest = sha256(cleaned.encode("utf-8")).hexdigest()[:12]
            object.__setattr__(self, "claim_id", f"claim_{digest}")


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    text: str
    source_id: str
    locator: str
    retrieval_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    label: AssessmentLabel
    score: float
    rationale: str
    evidence: EvidenceSpan


@dataclass(frozen=True, slots=True)
class FactCheckResult:
    claim: Claim
    verdict: Verdict
    confidence: float
    supporting: tuple[EvidenceAssessment, ...] = ()
    contradicting: tuple[EvidenceAssessment, ...] = ()
    unclear: tuple[EvidenceAssessment, ...] = ()
    audit_notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
