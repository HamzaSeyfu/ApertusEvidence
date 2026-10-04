# ApertusEvidence

**Evidence-first fact checking for official voting materials, built for Hack Apertus 2026.**

ApertusEvidence is a neutral, auditable fact-checking pipeline designed to verify claims against **primary official documents** rather than asking a language model for an unsupported yes/no answer.

The core idea is an **Evidence Court**:

```text
Claim
  ↓
Claim decomposition
  ↓
Evidence retrieval from official material
  ↓
┌───────────────────────┬────────────────────────┐
│ Support analyst       │ Contradiction analyst  │
│ finds confirming text │ finds conflicting text │
└───────────────────────┴────────────────────────┘
              ↓
        Evidence judge
              ↓
        Citation auditor
              ↓
SUPPORTED / CONTRADICTED / MIXED / INSUFFICIENT_EVIDENCE
              +
confidence, cited passages, provenance, audit trail
```

## Why this architecture?

A fact-checking demo is easy to make and easy to hallucinate.

ApertusEvidence is intentionally conservative:

- **official-source-first**: decisions must be tied to retrieved source passages;
- **abstention is a feature**: missing evidence returns `INSUFFICIENT_EVIDENCE`;
- **adversarial analysis**: support and contradiction are searched independently;
- **citation provenance**: every evidence span carries source, locator and score;
- **auditable output**: the result exposes the path from claim to verdict;
- **model-agnostic core**: Apertus can be plugged into the reasoning layer without coupling the data pipeline to a paid API.

## Current status

### v0.1 — local evidence engine

The first milestone provides:

- typed claim/evidence/result models;
- dependency-free lexical retrieval over local official documents;
- conservative local support/contradiction baseline;
- evidence aggregation and verdict logic;
- citation audit hooks;
- command-line fact checking;
- unit tests.

The local baseline is deliberately modest. Its purpose is to make the end-to-end system testable **before** Apertus inference is connected.


## Provenance-first ingestion

Official vote material is typically distributed as PDFs. ApertusEvidence keeps
the extraction boundary aligned with source pages so a downstream verdict can
always point back to the exact page that produced the evidence.

Install PDF support:

```bash
python -m pip install -e ".[pdf,dev]"
```

Normalize an official-material directory:

```bash
apertus-evidence ingest \
  --input data/raw \
  --output data/processed/official.jsonl
```

Each JSONL record contains:

- extracted text;
- stable `source_id`;
- page/chunk locator such as `booklet.pdf#page=7&chunk=2`;
- SHA-256 of the source file;
- page/chunk metadata.

Run the court against the normalized corpus:

```bash
apertus-evidence check \
  --claim "The measure allocates 10 million francs." \
  --index data/processed/official.jsonl \
  --json
```

## Planned Hack Apertus path

1. **Official material ingestion**
   - voting booklet / PDF extraction;
   - stable document and page/section identifiers;
   - multilingual metadata where available.

2. **Apertus reasoning layer**
   - atomic claim decomposition;
   - structured support and contradiction assessments;
   - evidence-grounded judge;
   - citation entailment audit.

3. **Evaluation**
   - retrieval recall;
   - verdict accuracy / macro F1;
   - citation precision;
   - abstention quality;
   - calibration and failure analysis.

4. **Demo**
   - enter a claim;
   - inspect supporting and contradicting passages;
   - view final verdict and confidence;
   - expand the complete audit trail.

## Design principle

> No evidence, no verdict.

ApertusEvidence does **not** recommend political choices, rank political actors or attempt persuasion. It verifies factual claims against the supplied official source material and exposes the evidence used.

## Quick start

Requires Python 3.11+.

```bash
python -m pip install -e ".[dev]"
pytest
```

Create a small corpus:

```text
data/official/example.txt
```

Then run:

```bash
apertus-evidence check \
  --claim "The measure allocates 10 million francs." \
  --corpus data/official
```

JSON output:

```bash
apertus-evidence check \
  --claim "The measure allocates 10 million francs." \
  --corpus data/official \
  --json
```

## Repository layout

```text
src/apertus_evidence/
  models.py       # domain model and verdicts
  retrieval.py    # local evidence index
  analysis.py     # conservative evidence assessments
  pipeline.py     # Evidence Court orchestration
  cli.py          # command-line interface
tests/
  test_retrieval.py
  test_pipeline.py
```

## Cost / infrastructure

The baseline runs locally with **no cloud account, no API key and no bank card**.

The Apertus adapter will remain optional so the repository stays reproducible even when external inference is unavailable.

## Licensing

Hack Apertus requires submitted outputs to remain open under category-specific licenses:

- **source code and model weights:** Apache License 2.0;
- **documentation, designs and text:** Creative Commons Attribution 4.0 (CC BY 4.0);
- **submitted datasets:** Community Data License Agreement – Permissive 2.0.

See the repository license files and per-directory notices before reuse.
