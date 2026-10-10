import json
from pathlib import Path

from apertus_evidence.ost_dataset import convert_ost_rows


def sample_rows():
    return [
        {
            "claim": "The reform allocates 10 million francs.",
            "claim_language": "en",
            "reference_string": (
                "The official voting booklet states that the reform allocates "
                "10 million francs to the programme."
            ),
            "entailment_label": 0,
            "baseline_score": 0.91,
            "reference_language": "en",
            "booklet_publish_date": "2026-01-01",
            "booklet_download_date": "2026-10-02",
            "booklet_url": "https://example.test/booklet.pdf",
            "vote": "Example vote",
        },
        {
            "claim": "The reform allocates 12 million francs.",
            "claim_language": "en",
            "reference_string": (
                "The official voting booklet states that the reform allocates "
                "10 million francs to the programme."
            ),
            "entailment_label": 2,
            "baseline_score": 0.87,
            "reference_language": "en",
            "booklet_publish_date": "2026-01-01",
            "booklet_download_date": "2026-10-02",
            "booklet_url": "https://example.test/booklet.pdf",
            "vote": "Example vote",
        },
        {
            "claim": "The reform changes railway timetables.",
            "claim_language": "en",
            "reference_string": "The booklet discusses programme financing.",
            "entailment_label": 1,
            "baseline_score": 0.78,
            "reference_language": "en",
            "booklet_publish_date": "2026-01-01",
            "booklet_download_date": "2026-10-02",
            "booklet_url": "https://example.test/booklet.pdf",
            "vote": "Example vote",
        },
    ]


def test_convert_ost_rows_maps_labels_and_deduplicates_evidence(
    tmp_path: Path,
) -> None:
    summary = convert_ost_rows(sample_rows(), tmp_path)

    assert summary.rows == 3
    assert summary.labels == {
        "CONTRADICTED": 1,
        "INSUFFICIENT_EVIDENCE": 1,
        "SUPPORTED": 1,
    }

    evidence_lines = (
        tmp_path / "ost_evidence.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    case_lines = (
        tmp_path / "ost_cases.jsonl"
    ).read_text(encoding="utf-8").splitlines()

    assert len(evidence_lines) == 2
    assert len(case_lines) == 3

    cases = [json.loads(line) for line in case_lines]
    assert cases[0]["expected_verdict"] == "SUPPORTED"
    assert cases[1]["expected_verdict"] == "CONTRADICTED"
    assert cases[2]["expected_verdict"] == "INSUFFICIENT_EVIDENCE"
    assert cases[0]["metadata"]["paired_source_id"].startswith("ost:")


def test_convert_ost_rows_writes_manifest(tmp_path: Path) -> None:
    summary = convert_ost_rows(sample_rows(), tmp_path)
    manifest = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["rows"] == 3
    assert manifest["dataset"] == "OSTswiss/MNLIoverSwissVotingBooklets"
    assert manifest["label_mapping"]["0"] == "SUPPORTED"
    assert manifest["label_mapping"]["1"] == "INSUFFICIENT_EVIDENCE"
    assert manifest["label_mapping"]["2"] == "CONTRADICTED"
    assert summary.manifest_path.endswith("manifest.json")
