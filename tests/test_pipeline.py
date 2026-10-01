from apertus_evidence.models import Verdict
from apertus_evidence.pipeline import EvidenceCourt
from apertus_evidence.retrieval import InMemoryEvidenceIndex, _Chunk


def court_for(text: str) -> EvidenceCourt:
    index = InMemoryEvidenceIndex(
        [_Chunk(text=text, source_id="official.txt", locator="official.txt#chunk-1")]
    )
    return EvidenceCourt(index)


def test_support_requires_grounded_evidence() -> None:
    result = court_for(
        "The measure allocates 10 million francs to the programme."
    ).check("The measure allocates 10 million francs.")

    assert result.verdict == Verdict.SUPPORTED
    assert result.supporting
    assert result.supporting[0].evidence.source_id == "official.txt"


def test_numeric_conflict_is_contradiction() -> None:
    result = court_for(
        "The measure allocates 12 million francs to the programme."
    ).check("The measure allocates 10 million francs.")

    assert result.verdict == Verdict.CONTRADICTED
    assert result.contradicting


def test_unrelated_material_abstains() -> None:
    result = court_for(
        "The official document describes railway timetable maintenance."
    ).check("The measure allocates 10 million francs.")

    assert result.verdict == Verdict.INSUFFICIENT_EVIDENCE
