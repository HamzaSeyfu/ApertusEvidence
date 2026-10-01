from pathlib import Path

from apertus_evidence.retrieval import InMemoryEvidenceIndex


def test_retrieval_ranks_relevant_passage_first(tmp_path: Path) -> None:
    (tmp_path / "booklet.txt").write_text(
        "The measure allocates 10 million francs to the programme.\n\n"
        "The proposal also changes reporting requirements.",
        encoding="utf-8",
    )
    (tmp_path / "other.txt").write_text(
        "This document concerns railway timetables.",
        encoding="utf-8",
    )

    index = InMemoryEvidenceIndex.from_directory(tmp_path)
    results = index.search("The measure allocates 10 million francs", top_k=2)

    assert results
    assert results[0].source_id == "booklet.txt"
    assert "10 million francs" in results[0].text
