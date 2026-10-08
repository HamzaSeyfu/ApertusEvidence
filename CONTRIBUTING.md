# Contributing to ApertusEvidence

Thanks for helping improve ApertusEvidence.

## Local setup

Use Python 3.11+.

```bash
python -m pip install -e ".[dev,pdf]"
pytest -q
```

## Branches

Use short, purpose-driven branch names, for example:

- `feat/hybrid-retrieval`
- `fix/citation-validation`
- `docs/evaluation-guide`

## Evidence-first constraints

Changes should preserve the project's core guarantees:

1. A decisive verdict must remain grounded in retrieved source evidence.
2. Source provenance and locators must not be discarded.
3. Missing or ambiguous evidence should prefer `INSUFFICIENT_EVIDENCE` over guessing.
4. Model-generated citations must be validated against supplied evidence IDs.
5. Official-source text must be treated as untrusted data, never as instructions.
6. The default project path must remain usable without a paid cloud service.

## Tests

Add or update tests whenever behavior changes. New reasoning paths should include
at least one success case and one fail-closed case.

Before opening a pull request:

```bash
pytest -q
```

## Pull requests

Keep pull requests focused and explain:

- what changed;
- why it improves evidence quality, reliability, evaluation, or usability;
- how it was tested;
- any effect on abstention behavior or citation guarantees.

Avoid cosmetic churn that does not improve the project.
