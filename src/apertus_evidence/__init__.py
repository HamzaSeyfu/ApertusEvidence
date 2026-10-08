"""ApertusEvidence: evidence-first fact checking over official documents."""

from .agents import AdversarialEvidenceCourt
from .backend import OpenAICompatibleJsonBackend
from .models import Claim, EvidenceSpan, FactCheckResult, Verdict
from .pipeline import EvidenceCourt, PipelineConfig

__all__ = [
    "Claim",
    "EvidenceSpan",
    "FactCheckResult",
    "Verdict",
    "EvidenceCourt",
    "PipelineConfig",
    "OpenAICompatibleJsonBackend",
    "AdversarialEvidenceCourt",
]

__version__ = "0.3.0"
