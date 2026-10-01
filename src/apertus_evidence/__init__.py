"""ApertusEvidence: evidence-first fact checking over official documents."""

from .models import Claim, EvidenceSpan, FactCheckResult, Verdict
from .pipeline import EvidenceCourt, PipelineConfig

__all__ = [
    "Claim",
    "EvidenceSpan",
    "FactCheckResult",
    "Verdict",
    "EvidenceCourt",
    "PipelineConfig",
]

__version__ = "0.1.0"
