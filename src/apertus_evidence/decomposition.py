from __future__ import annotations

from .analysis import JsonGenerationBackend
from .models import Claim
from .prompts import CLAIM_DECOMPOSER_SYSTEM, build_claim_decomposition_prompt


class ApertusClaimDecomposer:
    """Fail-safe claim decomposition for retrieval expansion."""

    def __init__(self, backend: JsonGenerationBackend, *, max_claims: int = 4) -> None:
        if max_claims < 1:
            raise ValueError("max_claims must be at least 1")
        self.backend = backend
        self.max_claims = max_claims

    def decompose(self, claim: Claim) -> tuple[str, ...]:
        try:
            payload = self.backend.generate_json(
                system=CLAIM_DECOMPOSER_SYSTEM,
                user=build_claim_decomposition_prompt(claim),
            )
            raw = payload.get("claims")
            if not isinstance(raw, list):
                raise ValueError("claims must be a list")

            cleaned: list[str] = []
            seen: set[str] = set()
            for item in raw[: self.max_claims]:
                if not isinstance(item, str):
                    raise ValueError("every decomposed claim must be text")
                normalized = " ".join(item.split())
                if len(normalized) < 4:
                    continue
                key = normalized.casefold()
                if key not in seen:
                    seen.add(key)
                    cleaned.append(normalized)

            if not cleaned:
                raise ValueError("no usable decomposed claims")
            return tuple(cleaned)
        except (TypeError, ValueError, KeyError):
            # Retrieval expansion is optional. A bad model response must never
            # prevent the original factual claim from being checked.
            return (claim.text,)
