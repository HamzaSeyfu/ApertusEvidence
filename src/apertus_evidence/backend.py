from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Mapping


class BackendError(RuntimeError):
    """Raised when an inference backend cannot return valid structured output."""


@dataclass(frozen=True, slots=True)
class OpenAICompatibleJsonBackend:
    """Minimal dependency-free client for Apertus-compatible chat endpoints.

    The backend intentionally targets the common OpenAI-compatible HTTP shape so
    the project can use local servers, hackathon-provided inference, or hosted
    endpoints without coupling the core pipeline to a proprietary SDK.
    """

    base_url: str
    model: str
    api_key: str | None = None
    timeout_seconds: float = 60.0

    @classmethod
    def from_env(
        cls,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key_env: str = "APERTUS_API_KEY",
        timeout_seconds: float = 60.0,
    ) -> "OpenAICompatibleJsonBackend":
        resolved_url = base_url or os.getenv("APERTUS_BASE_URL")
        resolved_model = model or os.getenv("APERTUS_MODEL")
        if not resolved_url:
            raise ValueError(
                "Missing Apertus endpoint. Pass --apertus-base-url or set APERTUS_BASE_URL."
            )
        if not resolved_model:
            raise ValueError(
                "Missing Apertus model. Pass --apertus-model or set APERTUS_MODEL."
            )
        return cls(
            base_url=resolved_url,
            model=resolved_model,
            api_key=os.getenv(api_key_env) or None,
            timeout_seconds=timeout_seconds,
        )

    def generate_json(self, *, system: str, user: str) -> Mapping[str, object]:
        endpoint = _chat_completions_url(self.base_url)
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise BackendError(
                f"Inference endpoint returned HTTP {exc.code}: {detail[:500]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise BackendError(f"Could not reach inference endpoint: {exc.reason}") from exc

        try:
            payload = json.loads(raw)
            content = payload["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise BackendError("Inference endpoint returned an unexpected response shape.") from exc

        if not isinstance(content, str):
            raise BackendError("Model response content is not text.")

        cleaned = _strip_json_fence(content)
        try:
            decoded = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise BackendError("Model response was not valid JSON.") from exc

        if not isinstance(decoded, dict):
            raise BackendError("Model JSON response must be an object.")
        return decoded


def _chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _strip_json_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned
