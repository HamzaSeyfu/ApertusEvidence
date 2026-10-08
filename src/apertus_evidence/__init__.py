"""ApertusEvidence: evidence-first fact checking over official documents."""

from .models import Claim, EvidenceSpan, FactCheckResult, Verdict
from .agents import AdversarialEvidenceCourt\nfrom .backend import OpenAICompatibleJsonBackend\nfrom .pipeline import EvidenceCourt, PipelineConfig

__all__ = [
    "Claim",
    "EvidenceSpan",
    "FactCheckResult",
    "Verdict",
    "EvidenceCourt",
    "PipelineConfig",\n    "OpenAICompatibleJsonBackend",\n    "AdversarialEvidenceCourt",
]

__version__ = "0.1.0"
