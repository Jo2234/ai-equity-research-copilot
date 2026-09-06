from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import Settings
from .schemas import Confidence


@dataclass(frozen=True)
class GroundedDraft:
    answer: str
    key_points: list[str]
    citation_indices: list[int]
    confidence: Confidence
    limitations: list[str]
    model: str
    provider: str


class InvalidGroundedDraft(RuntimeError):
    """The model responded, but its answer cannot be attributed to supplied context."""


class _DraftPayload(BaseModel):
    model_config = ConfigDict(strict=True)
    answer: str = Field(min_length=1)
    key_points: list[str] = Field(min_length=1)
    citation_indices: list[int] = Field(min_length=1)
    confidence: str
    limitations: list[str]


CITATION_REFERENCE = re.compile(r"\[(\d+)\]")


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def answer(self, question: str, contexts: list[dict[str, Any]]) -> GroundedDraft:
        context_text = "\n\n".join(
            f"[{idx}] {item['label']}\n{item['excerpt']}"
            for idx, item in enumerate(contexts, start=1)
        )
        prompt = f"""You are an equity research assistant. Answer using only the filing excerpts below.

Rules:
- Do not use news, market rumors, prior knowledge, price targets, or investment recommendations.
- Every factual claim in answer and key_points must include inline references such as [1] or [2].
- Use separate brackets for each reference, never [1, 2]. citation_indices must list exactly the referenced indices.
- If the excerpts are insufficient, say what is missing.
- Return only valid JSON with this schema:
{{"answer": "...", "key_points": ["..."], "citation_indices": [1, 2], "confidence": "high|medium|low", "limitations": ["..."]}}

Question:
{question}

Filing excerpts:
{context_text}
"""
        payload = {
            "model": self.settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.15,
                "top_p": 0.9,
                "num_ctx": 8192,
            },
        }
        raw = self._post("/api/generate", payload)
        try:
            parsed = _DraftPayload.model_validate_json(raw.get("response", ""))
        except (ValidationError, TypeError, AttributeError) as exc:
            raise InvalidGroundedDraft("Ollama returned an invalid answer schema") from exc
        answer = parsed.answer.strip()
        key_points = [point.strip() for point in parsed.key_points]
        indices = list(dict.fromkeys(parsed.citation_indices))
        texts = [answer, *key_points]
        references = {int(index) for text in texts for index in CITATION_REFERENCE.findall(text)}
        if (
            not answer or any(not point for point in key_points)
            or any(not CITATION_REFERENCE.search(text) for text in texts)
            or set(indices) != references
            or any(index < 1 or index > len(contexts) for index in indices)
            or parsed.confidence not in {"high", "medium", "low"}
        ):
            raise InvalidGroundedDraft("Ollama returned missing or inconsistent citation references")
        return GroundedDraft(
            answer=answer,
            key_points=key_points,
            citation_indices=indices,
            confidence=Confidence(parsed.confidence),
            limitations=parsed.limitations,
            model=self.settings.ollama_model,
            provider="ollama",
        )

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = Request(
            f"{self.settings.ollama_base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.settings.ollama_timeout_seconds) as response:
                parsed = json.loads(response.read().decode("utf-8", errors="replace"))
                if not isinstance(parsed, dict):
                    raise RuntimeError("Ollama returned a non-object response")
                return parsed
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned an invalid JSON response") from exc
        except HTTPError as exc:
            raise RuntimeError(f"Ollama request failed with HTTP {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"Ollama is unavailable: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError("Ollama request timed out") from exc
