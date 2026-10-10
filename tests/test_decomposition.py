from apertus_evidence.decomposition import ApertusClaimDecomposer
from apertus_evidence.models import Claim


class FakeBackend:
    def __init__(self, payload):
        self.payload = payload

    def generate_json(self, *, system: str, user: str):
        return self.payload


def test_decomposer_returns_atomic_claims() -> None:
    decomposer = ApertusClaimDecomposer(
        FakeBackend(
            {
                "claims": [
                    "The reform costs 10 million francs.",
                    "Implementation begins in 2027.",
                ]
            }
        )
    )

    result = decomposer.decompose(
        Claim(
            "The reform costs 10 million francs and implementation begins in 2027."
        )
    )

    assert result == (
        "The reform costs 10 million francs.",
        "Implementation begins in 2027.",
    )


def test_decomposer_falls_back_to_original_claim() -> None:
    claim = Claim("The reform costs 10 million francs.")
    decomposer = ApertusClaimDecomposer(FakeBackend({"claims": "not-a-list"}))

    assert decomposer.decompose(claim) == (claim.text,)
