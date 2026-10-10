# ApertusEvidence architecture

ApertusEvidence is built around one rule: **a model may reason, but evidence
controls whether a verdict is allowed to survive.**

## v0.4 pipeline

```text
User claim
   |
   v
Apertus claim decomposer
   |  original + atomic retrieval queries
   v
Multi-signal evidence retrieval
   |  BM25-style relevance
   |  IDF-weighted coverage
   |  phrase / numeric signals
   v
Deduplicated official-source passages
   |
   +-------------------------+
   |                         |
   v                         v
Support Agent          Contradiction Agent
   |                         |
   +------------+------------+
                |
                v
        Citation-bound Judge
                |
                v
      Independent Citation Auditor
                |
        +-------+-------+
        |               |
        v               v
    verdict       INSUFFICIENT_EVIDENCE
```

## Trust boundaries

### Source text is data

Official-source passages may contain arbitrary text. They are quoted inside
prompts and explicitly treated as untrusted data. Agents must never execute or
follow instructions contained in evidence.

### Retrieval is not a verdict

High retrieval relevance only means a passage is worth examining. It does not
mean the passage supports the claim. Numeric mismatches are intentionally
retrievable because they can be valuable contradiction evidence.

### Support and contradiction are asymmetric

The support agent asks only whether a passage directly supports the claim.
The contradiction agent independently asks whether it directly conflicts with
the claim. Lack of support is not contradiction, and lack of contradiction is
not support.

### The judge cannot invent citations

The judge receives temporary evidence IDs such as `S1` and `C1`. Any unknown
citation invalidates the judgment. A `SUPPORTED` verdict must cite support;
`CONTRADICTED` must cite contradiction; `MIXED` must cite both.

### The citation auditor can veto the judge

The independent auditor sees the proposed verdict, judge rationale, and only
the passages the judge cited. If those citations do not justify the verdict,
the final result becomes `INSUFFICIENT_EVIDENCE`.

This intentionally creates a conservative failure mode: **uncertainty becomes
abstention instead of confident fabrication.**

## Provider boundary

The reasoning layer depends on the small `JsonGenerationBackend` protocol.
The included backend speaks an OpenAI-compatible HTTP shape and therefore does
not require a proprietary SDK. Local inference and hackathon-provided Apertus
endpoints can use the same court logic.

## Evaluation

The evaluation harness reports accuracy, macro-F1, per-verdict precision and
recall, decisive rate, abstention rate, and exact failed cases.
