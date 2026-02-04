from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def is_available(self) -> bool:
        try:
            self._post("/api/tags", {})
            return True
        except RuntimeError:
            return False

    def answer(self, question: str, contexts: list[dict[str, Any]]) -> GroundedDraft:
        context_text = "\n\n".join(
            f"[{idx}] {item['label']}\n{item['excerpt']}"
            for idx, item in enumerate(contexts, start=1)
        )
        prompt = f"""You are an equity research assistant. Answer using only the filing excerpts below.

Rules:
- Do not use news, market rumors, prior knowledge, price targets, or investment recommendations.
- Every factual claim must be supported by one or more citation indices.
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
        response_text = str(raw.get("response") or "").strip()
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned invalid JSON") from exc
        citation_indices = [
            idx for idx in parsed.get("citation_indices", [])
            if isinstance(idx, int) and 1 <= idx <= len(contexts)
        ]
        answer = str(parsed.get("answer") or "").strip()
        key_points = [str(item).strip() for item in parsed.get("key_points", []) if str(item).strip()]
        limitations = [str(item).strip() for item in parsed.get("limitations", []) if str(item).strip()]
        confidence_value = str(parsed.get("confidence") or "medium").lower()
        confidence = confidence_value if confidence_value in {"high", "medium", "low"} else "medium"
        if not answer:
            raise RuntimeError("Ollama returned an empty answer")
        if not citation_indices:
            confidence = "low"
            limitations.append("The local model did not attach citation indices; answer should be reviewed.")
        return GroundedDraft(
            answer=answer,
            key_points=key_points or [answer],
            citation_indices=citation_indices,
            confidence=Confidence(confidence),
            limitations=limitations,
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
                return json.loads(response.read().decode("utf-8", errors="replace"))
        except HTTPError as exc:
            raise RuntimeError(f"Ollama request failed with HTTP {exc.code}") from exc
        except URLError as exc:
            raise RuntimeError(f"Ollama is unavailable: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError("Ollama request timed out") from exc
