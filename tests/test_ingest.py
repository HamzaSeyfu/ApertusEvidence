import json
from pathlib import Path

from apertus_evidence.ingest import ingest_path
from apertus_evidence.retrieval import InMemoryEvidenceIndex


def test_ingestion_writes_provenance_rich_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "official"
    source.mkdir()
    (source / "booklet.txt").write_text(
        "The measure allocates 10 million francs.\n\n"
        "Implementation begins in 2027.",
        encoding="utf-8",
    )
    output = tmp_path / "index.jsonl"

    count = ingest_path(source, output, chunk_chars=80)

    assert count == 1
    record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert record["source_id"] == "booklet.txt"
    assert record["locator"] == "booklet.txt#chunk=1"
    assert len(record["metadata"]["sha256"]) == 64


def test_jsonl_index_preserves_metadata(tmp_path: Path) -> None:
    index_file = tmp_path / "index.jsonl"
    index_file.write_text(
        json.dumps(
            {
                "text": "The measure allocates 10 million francs.",
                "source_id": "vote-booklet.pdf",
                "locator": "vote-booklet.pdf#page=7&chunk=1",
                "metadata": {"page": 7, "format": "pdf"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    index = InMemoryEvidenceIndex.from_jsonl(index_file)
    result = index.search("allocates 10 million francs", top_k=1)[0]

    assert result.locator == "vote-booklet.pdf#page=7&chunk=1"
    assert result.metadata["page"] == 7
