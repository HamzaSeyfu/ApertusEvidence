from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .retrieval import CorpusChunk, _chunk_text

_SUPPORTED = {".pdf", ".txt", ".md"}


def ingest_path(
    input_path: str | Path,
    output_path: str | Path,
    *,
    chunk_chars: int = 1400,
) -> int:
    """Normalize official source documents into provenance-rich JSONL.

    PDF chunks never cross page boundaries, preserving page-level citation
    provenance for later entailment and citation auditing.
    """

    source = Path(input_path)
    output = Path(output_path)

    files = _discover(source)
    chunks: list[CorpusChunk] = []
    root = source if source.is_dir() else source.parent

    for path in files:
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() == ".pdf":
            chunks.extend(_ingest_pdf(path, relative, chunk_chars=chunk_chars))
        else:
            chunks.extend(_ingest_text(path, relative, chunk_chars=chunk_chars))

    if not chunks:
        raise ValueError(f"No extractable evidence found under {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(
                json.dumps(
                    {
                        "text": chunk.text,
                        "source_id": chunk.source_id,
                        "locator": chunk.locator,
                        "metadata": chunk.metadata,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    return len(chunks)


def _discover(source: Path) -> list[Path]:
    if not source.exists():
        raise FileNotFoundError(f"Input path does not exist: {source}")
    if source.is_file():
        if source.suffix.lower() not in _SUPPORTED:
            raise ValueError(f"Unsupported evidence format: {source.suffix}")
        return [source]
    return sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in _SUPPORTED
    )


def _ingest_text(path: Path, source_id: str, *, chunk_chars: int) -> Iterable[CorpusChunk]:
    digest = _sha256(path)
    text = path.read_text(encoding="utf-8")
    for index, chunk in enumerate(_chunk_text(text, chunk_chars=chunk_chars), start=1):
        yield CorpusChunk(
            text=chunk,
            source_id=source_id,
            locator=f"{source_id}#chunk={index}",
            metadata={
                "format": path.suffix.lower().lstrip("."),
                "sha256": digest,
                "chunk": index,
            },
        )


def _ingest_pdf(path: Path, source_id: str, *, chunk_chars: int) -> Iterable[CorpusChunk]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            'PDF ingestion requires the optional dependency: pip install -e ".[pdf]"'
        ) from exc

    digest = _sha256(path)
    reader = PdfReader(str(path))
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for chunk_number, chunk in enumerate(
            _chunk_text(text, chunk_chars=chunk_chars),
            start=1,
        ):
            yield CorpusChunk(
                text=chunk,
                source_id=source_id,
                locator=f"{source_id}#page={page_number}&chunk={chunk_number}",
                metadata={
                    "format": "pdf",
                    "sha256": digest,
                    "page": page_number,
                    "chunk": chunk_number,
                },
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
