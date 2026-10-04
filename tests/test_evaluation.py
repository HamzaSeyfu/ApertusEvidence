from apertus_evidence.evaluation import EvaluationCase, evaluate
from apertus_evidence.models import Verdict
from apertus_evidence.pipeline import EvidenceCourt
from apertus_evidence.retrieval import CorpusChunk, InMemoryEvidenceIndex


def build_court() -> EvidenceCourt:
    return EvidenceCourt(
        InMemoryEvidenceIndex(
            [
                CorpusChunk(
                    text="The measure allocates 10 million francs to the programme.",
                    source_id="official.txt",
                    locator="official.txt#chunk=1",
                ),
                CorpusChunk(
                    text="The implementation period begins in 2027.",
                    source_id="official.txt",
                    locator="official.txt#chunk=2",
                ),
            ]
        )
    )


def test_evaluation_reports_accuracy_and_failure_details() -> None:
    cases = [
        EvaluationCase(
            case_id="supported",
            claim="The measure allocates 10 million francs.",
            expected_verdict=Verdict.SUPPORTED,
        ),
        EvaluationCase(
            case_id="wrong-number",
            claim="The measure allocates 12 million francs.",
            expected_verdict=Verdict.CONTRADICTED,
        ),
    ]

    summary = evaluate(build_court(), cases)

    assert summary.total == 2
    assert summary.correct == 2
    assert summary.accuracy == 1.0
    assert summary.failures == ()
    assert summary.per_label["SUPPORTED"]["tp"] == 1
    assert summary.per_label["CONTRADICTED"]["tp"] == 1
