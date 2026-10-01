from apertus_evidence.analysis import ApertusEvidenceAnalyst
from apertus_evidence.models import AssessmentLabel, Claim, EvidenceSpan


class FakeBackend:
    def __init__(self, payload):
        self.payload = payload
        self.last_system = ""
        self.last_user = ""

    def generate_json(self, *, system: str, user: str):
        self.last_system = system
        self.last_user = user
        return self.payload


def evidence() -> EvidenceSpan:
    return EvidenceSpan(
        text="The measure allocates 10 million francs.",
        source_id="booklet.pdf",
        locator="booklet.pdf#page-4",
        retrieval_score=0.9,
    )


def test_apertus_adapter_accepts_structured_support() -> None:
    backend = FakeBackend(
        {"label": "SUPPORT", "score": 0.91, "rationale": "Direct entailment."}
    )
    result = ApertusEvidenceAnalyst(backend).assess(
        Claim("The measure allocates 10 million francs."),
        evidence(),
    )

    assert result.label == AssessmentLabel.SUPPORT
    assert result.score == 0.91
    assert "official" in backend.last_system.lower()
    assert "booklet.pdf#page-4" in backend.last_user


def test_apertus_adapter_fails_closed_on_invalid_output() -> None:
    backend = FakeBackend(
        {"label": "SUPPORTED_MAYBE", "score": 5, "rationale": "Guess"}
    )
    result = ApertusEvidenceAnalyst(backend).assess(
        Claim("The measure allocates 10 million francs."),
        evidence(),
    )

    assert result.label == AssessmentLabel.UNCLEAR
    assert result.score == 0.0
    assert "rejected" in result.rationale.lower()
