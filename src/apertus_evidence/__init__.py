"""ApertusEvidence: evidence-first fact checking over official documents."""

from .agents import AdversarialEvidenceCourt, CitationAuditor
from .backend import OpenAICompatibleJsonBackend
from .decomposition import ApertusClaimDecomposer
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
    "CitationAuditor",
    "ApertusClaimDecomposer",
]

__version__ = "0.4.0"
