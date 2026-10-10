from apertus_evidence.agents import (
    AdversarialEvidenceCourt,
    CitationAuditor,
)
from apertus_evidence.decomposition import ApertusClaimDecomposer
from apertus_evidence.models import Verdict
from apertus_evidence.retrieval import CorpusChunk, InMemoryEvidenceIndex


class QueueBackend:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def generate_json(self, *, system: str, user: str):
        self.calls.append((system, user))
        return self.payloads.pop(0)


def compound_index() -> InMemoryEvidenceIndex:
    return InMemoryEvidenceIndex(
        [
            CorpusChunk(
                text="The reform allocates 10 million francs to the programme.",
                source_id="booklet.pdf",
                locator="booklet.pdf#page=4&chunk=1",
            ),
            CorpusChunk(
                text="Implementation begins in 2027.",
                source_id="booklet.pdf",
                locator="booklet.pdf#page=5&chunk=1",
            ),
        ]
    )


def test_enhanced_court_uses_decomposition_and_independent_audit() -> None:
    backend = QueueBackend(
        [
            {
                "claims": [
                    "The reform allocates 10 million francs.",
                    "Implementation begins in 2027.",
                ]
            },
            {"relevant": True, "score": 0.94, "rationale": "Direct amount support."},
            {"relevant": False, "score": 0.1, "rationale": "No amount support."},
            {"relevant": False, "score": 0.1, "rationale": "No conflict."},
            {"relevant": False, "score": 0.1, "rationale": "No conflict."},
            {
                "verdict": "SUPPORTED",
                "confidence": 0.91,
                "rationale": "The cited official passage directly supports the claim.",
                "citations": ["S1"],
            },
            {
                "valid": True,
                "score": 0.88,
                "rationale": "S1 directly supports the amount stated.",
            },
        ]
    )

    court = AdversarialEvidenceCourt(
        compound_index(),
        backend,
        decomposer=ApertusClaimDecomposer(backend),
        citation_auditor=CitationAuditor(backend),
    )
    result = court.check(
        "The reform allocates 10 million francs and implementation begins in 2027."
    )

    assert result.verdict == Verdict.SUPPORTED
    assert result.confidence == 0.88
    assert "3 query formulation" in result.audit_notes[0]
    assert "Citation audit passed" in result.audit_notes[-1]


def test_citation_auditor_can_veto_judge() -> None:
    backend = QueueBackend(
        [
            {"claims": ["The reform allocates 10 million francs."]},
            {"relevant": True, "score": 0.94, "rationale": "Direct support."},
            {"relevant": False, "score": 0.1, "rationale": "No conflict."},
            {
                "verdict": "SUPPORTED",
                "confidence": 0.95,
                "rationale": "Supported by S1.",
                "citations": ["S1"],
            },
            {
                "valid": False,
                "score": 0.2,
                "rationale": "The citation does not justify the full claim.",
            },
        ]
    )

    court = AdversarialEvidenceCourt(
        InMemoryEvidenceIndex(
            [
                CorpusChunk(
                    text="The reform allocates 10 million francs.",
                    source_id="booklet.pdf",
                    locator="booklet.pdf#page=4&chunk=1",
                )
            ]
        ),
        backend,
        decomposer=ApertusClaimDecomposer(backend),
        citation_auditor=CitationAuditor(backend),
    )
    result = court.check("The reform allocates 10 million francs.")

    assert result.verdict == Verdict.INSUFFICIENT_EVIDENCE
    assert result.confidence == 0.0
    assert "Citation audit rejected verdict" in result.audit_notes[-1]
