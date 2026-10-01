from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .models import EvidenceSpan

_TOKEN_RE = re.compile(r"[\wÀ-ÖØ-öø-ÿ]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text)]


@dataclass(frozen=True, slots=True)
class _Chunk:
    text: str
    source_id: str
    locator: str


class InMemoryEvidenceIndex:
    """Small deterministic lexical index for local/offline development.

    This is intentionally dependency-free. It gives the project a reproducible
    retrieval baseline before embeddings or hybrid retrieval are introduced.
    """

    def __init__(self, chunks: list[_Chunk]) -> None:
        if not chunks:
            raise ValueError("Evidence index requires at least one chunk.")
        self._chunks = chunks
        self._term_sets = [set(tokenize(chunk.text)) for chunk in chunks]
        self._idf = self._build_idf(self._term_sets)

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        chunk_chars: int = 1400,
    ) -> "InMemoryEvidenceIndex":
        root = Path(directory)
        if not root.exists():
            raise FileNotFoundError(f"Corpus directory does not exist: {root}")

        chunks: list[_Chunk] = []
        allowed = {".txt", ".md"}

        for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in allowed):
            text = path.read_text(encoding="utf-8")
            relative = path.relative_to(root).as_posix()
            for idx, chunk_text in enumerate(_chunk_text(text, chunk_chars=chunk_chars), start=1):
                chunks.append(
                    _Chunk(
                        text=chunk_text,
                        source_id=relative,
                        locator=f"{relative}#chunk-{idx}",
                    )
                )

        if not chunks:
            raise ValueError(f"No .txt or .md evidence documents found in {root}")
        return cls(chunks)

    @staticmethod
    def _build_idf(term_sets: list[set[str]]) -> dict[str, float]:
        document_frequency: Counter[str] = Counter()
        for terms in term_sets:
            document_frequency.update(terms)

        total = len(term_sets)
        return {
            term: math.log((1 + total) / (1 + freq)) + 1.0
            for term, freq in document_frequency.items()
        }

    def search(self, query: str, *, top_k: int = 6) -> list[EvidenceSpan]:
        query_terms = set(tokenize(query))
        if not query_terms:
            return []

        scored: list[EvidenceSpan] = []
        query_weight = sum(self._idf.get(term, 1.0) for term in query_terms) or 1.0

        for chunk, terms in zip(self._chunks, self._term_sets, strict=True):
            overlap = query_terms & terms
            if not overlap:
                continue
            overlap_weight = sum(self._idf.get(term, 1.0) for term in overlap)
            coverage = overlap_weight / query_weight
            specificity = overlap_weight / (
                sum(self._idf.get(term, 1.0) for term in terms) ** 0.5 or 1.0
            )
            score = min(1.0, 0.82 * coverage + 0.18 * min(1.0, specificity))
            scored.append(
                EvidenceSpan(
                    text=chunk.text,
                    source_id=chunk.source_id,
                    locator=chunk.locator,
                    retrieval_score=round(score, 4),
                )
            )

        scored.sort(key=lambda item: item.retrieval_score, reverse=True)
        return scored[:top_k]


def _chunk_text(text: str, *, chunk_chars: int) -> list[str]:
    paragraphs = [" ".join(part.split()) for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for paragraph in paragraphs:
        projected = current_size + len(paragraph) + (2 if current else 0)
        if current and projected > chunk_chars:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_size = len(paragraph)
        else:
            current.append(paragraph)
            current_size = projected

    if current:
        chunks.append("\n\n".join(current))
    return chunks
