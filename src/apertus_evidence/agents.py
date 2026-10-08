from __future__ import annotations

from dataclasses import dataclass

from .analysis import JsonGenerationBackend
from .models import (
    AssessmentLabel,
    Claim,
    EvidenceAssessment,
    EvidenceSpan,
    FactCheckResult,
    Verdict,
)
from .pipeline import PipelineConfig
from .prompts import (
    CONTRADICTION_AGENT_SYSTEM,
    JUDGE_SYSTEM,
    SUPPORT_AGENT_SYSTEM,
    build_judge_prompt,
    build_role_prompt,
)
from .retrieval import InMemoryEvidenceIndex


@dataclass(frozen=True, slots=True)
class JudgeDecision:
    verdict: Verdict
    confidence: float
    rationale: str
    citations: tuple[str, ...]


class RoleEvidenceAgent:
    def __init__(
        self,
        backend: JsonGenerationBackend,
        *,
        target: AssessmentLabel,
    ) -> None:
        if target not in (AssessmentLabel.SUPPORT, AssessmentLabel.CONTRADICTION):
            raise ValueError("Role agent target must be SUPPORT or CONTRADICTION.")
        self.backend = backend
        self.target = target

    @property
    def system_prompt(self) -> str:
        return (
            SUPPORT_AGENT_SYSTEM
            if self.target == AssessmentLabel.SUPPORT
            else CONTRADICTION_AGENT_SYSTEM
        )

    def assess(self, claim: Claim, evidence: EvidenceSpan) -> EvidenceAssessment:
        try:
            payload = self.backend.generate_json(
                system=self.system_prompt,
                user=build_role_prompt(claim, evidence),
            )
            relevant = payload.get("relevant")
            if not isinstance(relevant, bool):
                raise ValueError("relevant must be boolean")
            score = float(payload.get("score", 0.0))
            if not 0.0 <= score <= 1.0:
                raise ValueError("score must be in [0, 1]")
            rationale = str(payload.get("rationale", "")).strip()
            if not rationale:
                raise ValueError("rationale cannot be empty")

            if not relevant:
                return EvidenceAssessment(
                    label=AssessmentLabel.UNCLEAR,
                    score=0.0,
                    rationale=rationale,
                    evidence=evidence,
                )

            return EvidenceAssessment(
                label=self.target,
                score=round(score, 4),
                rationale=rationale,
                evidence=evidence,
            )
        except (TypeError, ValueError, KeyError) as exc:
            return EvidenceAssessment(
                label=AssessmentLabel.UNCLEAR,
                score=0.0,
                rationale=f"{self.target.value} agent output rejected: {exc}",
                evidence=evidence,
            )


class EvidenceJudge:
    def __init__(self, backend: JsonGenerationBackend) -> None:
        self.backend = backend

    def decide(
        self,
        claim: Claim,
        supporting: tuple[EvidenceAssessment, ...],
        contradicting: tuple[EvidenceAssessment, ...],
    ) -> JudgeDecision:
        support_payload = [
            (
                f"S{idx}",
                item.evidence.locator,
                item.evidence.text,
                item.score,
                item.rationale,
            )
            for idx, item in enumerate(supporting, start=1)
        ]
        contradiction_payload = [
            (
                f"C{idx}",
                item.evidence.locator,
                item.evidence.text,
                item.score,
                item.rationale,
            )
            for idx, item in enumerate(contradicting, start=1)
        ]
        allowed = {item[0] for item in support_payload + contradiction_payload}

        payload = self.backend.generate_json(
            system=JUDGE_SYSTEM,
            user=build_judge_prompt(claim, support_payload, contradiction_payload),
        )
        verdict = Verdict(str(payload.get("verdict", "")).upper())
        confidence = float(payload.get("confidence", 0.0))
        rationale = str(payload.get("rationale", "")).strip()
        citations_raw = payload.get("citations", [])

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Judge confidence must be in [0, 1].")
        if not rationale:
            raise ValueError("Judge rationale cannot be empty.")
        if not isinstance(citations_raw, list) or not all(
            isinstance(item, str) for item in citations_raw
        ):
            raise ValueError("Judge citations must be a list of evidence IDs.")

        citations = tuple(citations_raw)
        unknown = set(citations) - allowed
        if unknown:
            raise ValueError(f"Judge cited unknown evidence IDs: {sorted(unknown)}")

        support_ids = {item[0] for item in support_payload}
        contradiction_ids = {item[0] for item in contradiction_payload}

        if verdict == Verdict.SUPPORTED and not (set(citations) & support_ids):
            raise ValueError("SUPPORTED verdict requires at least one support citation.")
        if verdict == Verdict.CONTRADICTED and not (set(citations) & contradiction_ids):
            raise ValueError("CONTRADICTED verdict requires at least one contradiction citation.")
        if verdict == Verdict.MIXED and not (
            set(citations) & support_ids and set(citations) & contradiction_ids
        ):
            raise ValueError("MIXED verdict requires support and contradiction citations.")

        return JudgeDecision(
            verdict=verdict,
            confidence=round(confidence, 4),
            rationale=rationale,
            citations=citations,
        )


class AdversarialEvidenceCourt:
    """Two-sided evidence search followed by a citation-constrained judge."""

    def __init__(
        self,
        index: InMemoryEvidenceIndex,
        backend: JsonGenerationBackend,
        *,
        config: PipelineConfig | None = None,
        agent_threshold: float = 0.60,
    ) -> None:
        self.index = index
        self.config = config or PipelineConfig()
        self.agent_threshold = agent_threshold
        self.support_agent = RoleEvidenceAgent(
            backend, target=AssessmentLabel.SUPPORT
        )
        self.contradiction_agent = RoleEvidenceAgent(
            backend, target=AssessmentLabel.CONTRADICTION
        )
        self.judge = EvidenceJudge(backend)

    def check(self, claim_text: str) -> FactCheckResult:
        claim = Claim(claim_text)
        retrieved = [
            item
            for item in self.index.search(claim.text, top_k=self.config.top_k)
            if item.retrieval_score >= self.config.min_retrieval_score
        ]

        if not retrieved:
            return FactCheckResult(
                claim=claim,
                verdict=Verdict.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                audit_notes=("No retrieved passage met the minimum evidence score.",),
            )

        support_all = tuple(
            self.support_agent.assess(claim, evidence) for evidence in retrieved
        )
        contradiction_all = tuple(
            self.contradiction_agent.assess(claim, evidence) for evidence in retrieved
        )

        supporting = tuple(
            item
            for item in support_all
            if item.label == AssessmentLabel.SUPPORT
            and item.score >= self.agent_threshold
        )
        contradicting = tuple(
            item
            for item in contradiction_all
            if item.label == AssessmentLabel.CONTRADICTION
            and item.score >= self.agent_threshold
        )
        unclear = tuple(
            item
            for item in support_all + contradiction_all
            if item.label == AssessmentLabel.UNCLEAR
        )

        if not supporting and not contradicting:
            return FactCheckResult(
                claim=claim,
                verdict=Verdict.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                supporting=(),
                contradicting=(),
                unclear=unclear,
                audit_notes=(
                    "Both adversarial agents abstained below the evidence threshold.",
                    "Citation audit: no decisive evidence to validate.",
                ),
            )

        try:
            decision = self.judge.decide(claim, supporting, contradicting)
        except (TypeError, ValueError, KeyError) as exc:
            return FactCheckResult(
                claim=claim,
                verdict=Verdict.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                supporting=supporting,
                contradicting=contradicting,
                unclear=unclear,
                audit_notes=(
                    f"Judge output rejected; failed closed: {exc}",
                    "Citation audit failed at judge validation.",
                ),
            )

        return FactCheckResult(
            claim=claim,
            verdict=decision.verdict,
            confidence=decision.confidence,
            supporting=supporting,
            contradicting=contradicting,
            unclear=unclear,
            audit_notes=(
                f"Judge: {decision.rationale}",
                "Judge citations validated: " + ", ".join(decision.citations),
            ),
        )
