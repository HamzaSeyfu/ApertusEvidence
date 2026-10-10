from pathlib import Path

from apertus_evidence.analysis import ConservativeHeuristicAnalyst
from apertus_evidence.ost_benchmark import evaluate_ost_pairs
from apertus_evidence.ost_dataset import convert_ost_rows


def rows():
    return [
        {
            "claim": "The reform allocates 10 million francs.",
            "claim_language": "en",
            "reference_string": (
                "The reform allocates 10 million francs to the programme."
            ),
            "entailment_label": 0,
            "baseline_score": 0.92,
            "reference_language": "en",
            "booklet_url": "https://example.test/a.pdf",
            "vote": "A",
        },
        {
            "claim": "The reform allocates 12 million francs.",
            "claim_language": "en",
            "reference_string": (
                "The reform allocates 10 million francs to the programme."
            ),
            "entailment_label": 2,
            "baseline_score": 0.88,
            "reference_language": "en",
            "booklet_url": "https://example.test/a.pdf",
            "vote": "A",
        },
        {
            "claim": "The reform abolishes railway timetables.",
            "claim_language": "en",
            "reference_string": "The booklet describes programme financing.",
            "entailment_label": 1,
            "baseline_score": 0.75,
            "reference_language": "en",
            "booklet_url": "https://example.test/a.pdf",
            "vote": "A",
        },
    ]


def test_pairwise_ost_benchmark_isolated_from_retrieval(tmp_path: Path) -> None:
    convert_ost_rows(rows(), tmp_path)

    summary = evaluate_ost_pairs(
        ConservativeHeuristicAnalyst(),
        evidence_path=tmp_path / "ost_evidence.jsonl",
        cases_path=tmp_path / "ost_cases.jsonl",
    )

    assert summary.total == 3
    assert summary.correct == 3
    assert summary.accuracy == 1.0
    assert summary.abstention_rate == 1 / 3
