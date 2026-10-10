from pathlib import Path

from apertus_evidence.retrieval import CorpusChunk, InMemoryEvidenceIndex


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
    assert "retrieval_bm25" in results[0].metadata


def test_numeric_conflict_remains_retrievable() -> None:
    index = InMemoryEvidenceIndex(
        [
            CorpusChunk(
                text="The measure allocates 12 million francs to the programme.",
                source_id="official.txt",
                locator="official.txt#1",
            ),
            CorpusChunk(
                text="The measure introduces annual reporting obligations.",
                source_id="official.txt",
                locator="official.txt#2",
            ),
        ]
    )

    results = index.search("The measure allocates 10 million francs", top_k=2)

    assert results[0].locator == "official.txt#1"
    assert results[0].metadata["retrieval_signal"] >= 0.5


def test_search_many_deduplicates_and_tracks_queries() -> None:
    index = InMemoryEvidenceIndex(
        [
            CorpusChunk(
                text=(
                    "The reform costs 10 million francs and implementation "
                    "begins in 2027."
                ),
                source_id="booklet.txt",
                locator="booklet.txt#1",
            )
        ]
    )

    results = index.search_many(
        [
            "The reform costs 10 million francs.",
            "Implementation begins in 2027.",
        ],
        top_k=5,
    )

    assert len(results) == 1
    assert len(results[0].metadata["matched_queries"]) == 2
