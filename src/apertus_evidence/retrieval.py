from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .models import EvidenceSpan

_TOKEN_RE = re.compile(r"[\wÀ-ÖØ-öø-ÿ]+", re.UNICODE)
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text)]


@dataclass(frozen=True, slots=True)
class CorpusChunk:
    text: str
    source_id: str
    locator: str
    metadata: dict[str, Any] = field(default_factory=dict)


# Backwards-compatible alias for the initial tests/API.
_Chunk = CorpusChunk


class InMemoryEvidenceIndex:
    """Deterministic multi-signal evidence index for offline/reproducible use.

    Ranking combines a BM25-style lexical score, IDF-weighted query coverage,
    phrase overlap and numeric signals. Numeric conflicts remain retrievable:
    a passage mentioning a different amount is potentially contradiction
    evidence and must not be filtered out merely because the number differs.
    """

    def __init__(self, chunks: Iterable[CorpusChunk]) -> None:
        materialized = list(chunks)
        if not materialized:
            raise ValueError("Evidence index requires at least one chunk.")

        self._chunks = materialized
        self._tokens = [tokenize(chunk.text) for chunk in materialized]
        self._term_counts = [Counter(tokens) for tokens in self._tokens]
        self._term_sets = [set(tokens) for tokens in self._tokens]
        self._doc_lengths = [len(tokens) for tokens in self._tokens]
        self._avg_doc_length = (
            sum(self._doc_lengths) / len(self._doc_lengths)
            if self._doc_lengths
            else 1.0
        )
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

        chunks: list[CorpusChunk] = []
        allowed = {".txt", ".md"}

        for path in sorted(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in allowed
        ):
            text = path.read_text(encoding="utf-8")
            relative = path.relative_to(root).as_posix()
            for idx, chunk_text in enumerate(
                _chunk_text(text, chunk_chars=chunk_chars),
                start=1,
            ):
                chunks.append(
                    CorpusChunk(
                        text=chunk_text,
                        source_id=relative,
                        locator=f"{relative}#chunk-{idx}",
                        metadata={"format": path.suffix.lower().lstrip(".")},
                    )
                )

        if not chunks:
            raise ValueError(f"No .txt or .md evidence documents found in {root}")
        return cls(chunks)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "InMemoryEvidenceIndex":
        index_path = Path(path)
        chunks: list[CorpusChunk] = []
        with index_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                try:
                    chunks.append(
                        CorpusChunk(
                            text=str(payload["text"]),
                            source_id=str(payload["source_id"]),
                            locator=str(payload["locator"]),
                            metadata=dict(payload.get("metadata") or {}),
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid corpus record at {index_path}:{line_number}: {exc}"
                    ) from exc
        return cls(chunks)

    @staticmethod
    def _build_idf(term_sets: list[set[str]]) -> dict[str, float]:
        document_frequency: Counter[str] = Counter()
        for terms in term_sets:
            document_frequency.update(terms)

        total = len(term_sets)
        return {
            term: math.log(1.0 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def _bm25(self, query_terms: Sequence[str], index: int) -> float:
        counts = self._term_counts[index]
        doc_length = self._doc_lengths[index]
        k1 = 1.5
        b = 0.75
        score = 0.0

        for term in set(query_terms):
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            denominator = frequency + k1 * (
                1.0 - b + b * doc_length / (self._avg_doc_length or 1.0)
            )
            score += self._idf.get(term, 0.0) * (
                frequency * (k1 + 1.0) / denominator
            )
        return score

    def search(self, query: str, *, top_k: int = 6) -> list[EvidenceSpan]:
        query_terms = tokenize(query)
        query_set = set(query_terms)
        if not query_terms:
            return []

        raw_bm25 = [
            self._bm25(query_terms, index)
            for index in range(len(self._chunks))
        ]
        max_bm25 = max(raw_bm25, default=0.0) or 1.0

        query_weight = sum(
            self._idf.get(term, 0.25) for term in query_set
        ) or 1.0
        query_numbers = set(_NUMBER_RE.findall(query.casefold()))
        normalized_query = " ".join(query.casefold().split())

        scored: list[EvidenceSpan] = []
        for idx, (chunk, terms) in enumerate(
            zip(self._chunks, self._term_sets, strict=True)
        ):
            overlap = query_set & terms
            if not overlap:
                continue

            overlap_weight = sum(
                self._idf.get(term, 0.25) for term in overlap
            )
            coverage = min(1.0, overlap_weight / query_weight)
            bm25 = raw_bm25[idx] / max_bm25

            normalized_text = " ".join(chunk.text.casefold().split())
            phrase_bonus = 1.0 if (
                normalized_query and normalized_query in normalized_text
            ) else 0.0

            evidence_numbers = set(_NUMBER_RE.findall(chunk.text.casefold()))
            numeric_signal = 0.0
            if query_numbers and evidence_numbers:
                # Different explicit numbers are still highly relevant because
                # they may be direct contradiction evidence.
                numeric_signal = 0.5
                if query_numbers & evidence_numbers:
                    numeric_signal = 1.0

            signal = max(phrase_bonus, numeric_signal)
            score = min(
                1.0,
                0.55 * bm25 + 0.35 * coverage + 0.10 * signal,
            )

            metadata = dict(chunk.metadata)
            metadata.update(
                {
                    "retrieval_bm25": round(bm25, 4),
                    "retrieval_coverage": round(coverage, 4),
                    "retrieval_signal": round(signal, 4),
                }
            )
            scored.append(
                EvidenceSpan(
                    text=chunk.text,
                    source_id=chunk.source_id,
                    locator=chunk.locator,
                    retrieval_score=round(score, 4),
                    metadata=metadata,
                )
            )

        scored.sort(key=lambda item: item.retrieval_score, reverse=True)
        return scored[:top_k]

    def search_many(
        self,
        queries: Iterable[str],
        *,
        top_k: int = 6,
        per_query_k: int | None = None,
    ) -> list[EvidenceSpan]:
        """Search several atomic formulations and fuse evidence by max score."""

        fused: dict[tuple[str, str], EvidenceSpan] = {}
        matched_queries: dict[tuple[str, str], list[str]] = {}
        local_k = per_query_k or top_k

        for query in queries:
            cleaned = " ".join(query.split())
            if not cleaned:
                continue
            for item in self.search(cleaned, top_k=local_k):
                key = (item.source_id, item.locator)
                matched_queries.setdefault(key, []).append(cleaned)
                previous = fused.get(key)
                if previous is None or item.retrieval_score > previous.retrieval_score:
                    fused[key] = item

        results: list[EvidenceSpan] = []
        for key, item in fused.items():
            metadata = dict(item.metadata)
            metadata["matched_queries"] = tuple(dict.fromkeys(matched_queries[key]))
            results.append(
                EvidenceSpan(
                    text=item.text,
                    source_id=item.source_id,
                    locator=item.locator,
                    retrieval_score=item.retrieval_score,
                    metadata=metadata,
                )
            )

        results.sort(key=lambda item: item.retrieval_score, reverse=True)
        return results[:top_k]


def _chunk_text(text: str, *, chunk_chars: int) -> list[str]:
    paragraphs = [
        " ".join(part.split())
        for part in re.split(r"\n\s*\n", text)
        if part.strip()
    ]
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
