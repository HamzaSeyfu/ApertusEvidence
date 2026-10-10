from apertus_evidence.agents import AdversarialEvidenceCourt
from apertus_evidence.models import Verdict
from apertus_evidence.retrieval import CorpusChunk, InMemoryEvidenceIndex


class QueueBackend:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def generate_json(self, *, system: str, user: str):
        self.calls.append((system, user))
        return self.payloads.pop(0)


def index() -> InMemoryEvidenceIndex:
    return InMemoryEvidenceIndex(
        [
            CorpusChunk(
                text="The measure allocates 10 million francs to the programme.",
                source_id="booklet.pdf",
                locator="booklet.pdf#page=4&chunk=1",
            )
        ]
    )


def test_adversarial_court_support_flow() -> None:
    backend = QueueBackend(
        [
            {"relevant": True, "score": 0.92, "rationale": "Direct support."},
            {"relevant": False, "score": 0.1, "rationale": "No conflict."},
            {
                "verdict": "SUPPORTED",
                "confidence": 0.9,
                "rationale": "The official passage directly states the amount.",
                "citations": ["S1"],
            },
        ]
    )
    result = AdversarialEvidenceCourt(index(), backend).check(
        "The measure allocates 10 million francs."
    )

    assert result.verdict == Verdict.SUPPORTED
    assert result.confidence == 0.9
    assert len(result.supporting) == 1
    assert result.contradicting == ()
    assert any("S1" in note for note in result.audit_notes)


def test_adversarial_court_rejects_hallucinated_citation() -> None:
    backend = QueueBackend(
        [
            {"relevant": True, "score": 0.92, "rationale": "Direct support."},
            {"relevant": False, "score": 0.1, "rationale": "No conflict."},
            {
                "verdict": "SUPPORTED",
                "confidence": 0.99,
                "rationale": "Supported.",
                "citations": ["S99"],
            },
        ]
    )
    result = AdversarialEvidenceCourt(index(), backend).check(
        "The measure allocates 10 million francs."
    )

    assert result.verdict == Verdict.INSUFFICIENT_EVIDENCE
    assert result.confidence == 0.0
    assert any("failed closed" in note.lower() for note in result.audit_notes)


def test_adversarial_court_abstains_without_agent_findings() -> None:
    backend = QueueBackend(
        [
            {"relevant": False, "score": 0.1, "rationale": "Not direct."},
            {"relevant": False, "score": 0.1, "rationale": "Not direct."},
        ]
    )
    result = AdversarialEvidenceCourt(index(), backend).check(
        "The measure allocates 10 million francs."
    )

    assert result.verdict == Verdict.INSUFFICIENT_EVIDENCE
    assert len(backend.calls) == 2
