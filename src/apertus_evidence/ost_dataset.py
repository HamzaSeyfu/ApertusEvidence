from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .models import Verdict
from .retrieval import _chunk_text

OST_DATASET_NAME = "OSTswiss/MNLIoverSwissVotingBooklets"

_LABEL_TO_VERDICT = {
    0: Verdict.SUPPORTED,
    1: Verdict.INSUFFICIENT_EVIDENCE,
    2: Verdict.CONTRADICTED,
}


@dataclass(frozen=True, slots=True)
class OstPreparationSummary:
    rows: int
    evidence_chunks: int
    labels: dict[str, int]
    claim_languages: dict[str, int]
    reference_languages: dict[str, int]
    evidence_path: str
    cases_path: str
    manifest_path: str

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "evidence_chunks": self.evidence_chunks,
            "labels": self.labels,
            "claim_languages": self.claim_languages,
            "reference_languages": self.reference_languages,
            "evidence_path": self.evidence_path,
            "cases_path": self.cases_path,
            "manifest_path": self.manifest_path,
        }


def prepare_ost_dataset(
    output_dir: str | Path,
    *,
    dataset_name: str = OST_DATASET_NAME,
    split: str = "train",
    chunk_chars: int = 1800,
    limit: int | None = None,
) -> OstPreparationSummary:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            'OST dataset preparation requires: pip install -e ".[ost]"'
        ) from exc

    dataset = load_dataset(dataset_name, split=split)
    rows: Iterable[Mapping[str, Any]]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        rows = (dataset[index] for index in range(min(limit, len(dataset))))
    else:
        rows = dataset

    return convert_ost_rows(
        rows,
        output_dir,
        dataset_name=dataset_name,
        split=split,
        chunk_chars=chunk_chars,
    )


def convert_ost_rows(
    rows: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    dataset_name: str = OST_DATASET_NAME,
    split: str = "train",
    chunk_chars: int = 1800,
) -> OstPreparationSummary:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    evidence_path = root / "ost_evidence.jsonl"
    cases_path = root / "ost_cases.jsonl"
    manifest_path = root / "manifest.json"

    evidence_records: dict[tuple[str, int], dict[str, object]] = {}
    case_records: list[dict[str, object]] = []
    label_counts: Counter[str] = Counter()
    claim_languages: Counter[str] = Counter()
    reference_languages: Counter[str] = Counter()

    for row_index, row in enumerate(rows, start=1):
        claim = _required_text(row, "claim", row_index)
        reference = _required_text(row, "reference_string", row_index)

        try:
            raw_label = int(row["entailment_label"])
            verdict = _LABEL_TO_VERDICT[raw_label]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid entailment_label at OST row {row_index}"
            ) from exc

        claim_language = str(row.get("claim_language") or "unknown")
        reference_language = str(row.get("reference_language") or "unknown")
        booklet_url = str(row.get("booklet_url") or "")
        vote = str(row.get("vote") or "")
        publish_date = str(row.get("booklet_publish_date") or "")
        baseline_score = _optional_float(row.get("baseline_score"))

        source_digest = hashlib.sha256(
            (
                booklet_url
                + "\n"
                + vote
                + "\n"
                + reference_language
                + "\n"
                + reference
            ).encode("utf-8")
        ).hexdigest()[:16]
        source_id = f"ost:{source_digest}"

        chunks = _chunk_text(reference, chunk_chars=chunk_chars)
        if not chunks:
            raise ValueError(f"Empty reference_string at OST row {row_index}")

        locators: list[str] = []
        for chunk_number, chunk in enumerate(chunks, start=1):
            locator = f"{source_id}#chunk={chunk_number}"
            locators.append(locator)
            key = (source_id, chunk_number)
            evidence_records.setdefault(
                key,
                {
                    "text": chunk,
                    "source_id": source_id,
                    "locator": locator,
                    "metadata": {
                        "dataset": dataset_name,
                        "split": split,
                        "reference_language": reference_language,
                        "booklet_url": booklet_url,
                        "booklet_publish_date": publish_date,
                        "vote": vote,
                    },
                },
            )

        case_records.append(
            {
                "case_id": f"ost-{row_index:04d}",
                "claim": claim,
                "expected_verdict": verdict.value,
                "metadata": {
                    "dataset": dataset_name,
                    "split": split,
                    "entailment_label": raw_label,
                    "baseline_score": baseline_score,
                    "claim_language": claim_language,
                    "reference_language": reference_language,
                    "booklet_url": booklet_url,
                    "booklet_publish_date": publish_date,
                    "vote": vote,
                    "paired_source_id": source_id,
                    "paired_locators": locators,
                },
            }
        )

        label_counts[verdict.value] += 1
        claim_languages[claim_language] += 1
        reference_languages[reference_language] += 1

    if not case_records:
        raise ValueError("No OST rows were provided.")

    with evidence_path.open("w", encoding="utf-8") as handle:
        for record in evidence_records.values():
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    with cases_path.open("w", encoding="utf-8") as handle:
        for record in case_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    summary = OstPreparationSummary(
        rows=len(case_records),
        evidence_chunks=len(evidence_records),
        labels=dict(sorted(label_counts.items())),
        claim_languages=dict(sorted(claim_languages.items())),
        reference_languages=dict(sorted(reference_languages.items())),
        evidence_path=str(evidence_path),
        cases_path=str(cases_path),
        manifest_path=str(manifest_path),
    )

    manifest = {
        "dataset": dataset_name,
        "split": split,
        "label_mapping": {
            "0": Verdict.SUPPORTED.value,
            "1": Verdict.INSUFFICIENT_EVIDENCE.value,
            "2": Verdict.CONTRADICTED.value,
        },
        **summary.to_dict(),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _required_text(row: Mapping[str, Any], key: str, row_index: int) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise ValueError(f"Missing {key} at OST row {row_index}")
    return value


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
