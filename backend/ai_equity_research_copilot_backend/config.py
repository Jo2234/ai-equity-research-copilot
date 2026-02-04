from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    raw_dir: Path
    state_dir: Path
    state_file: Path
    embedding_dimensions: int = 128
    chunk_target_tokens: int = 800
    chunk_overlap_tokens: int = 80
    retrieval_min_score: float = 0.04
    local_model_name: str = "local-deterministic-grounded-v1"
    llm_provider: str = "auto"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "gemma3:4b"
    ollama_timeout_seconds: float = 180.0
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    sec_user_agent: str = "AI Equity Research Copilot local demo contact@example.com"
    sec_timeout_seconds: float = 20.0
    sec_max_filing_chars: int = 600_000


def get_settings(data_dir: str | Path | None = None) -> Settings:
    root = Path(data_dir or os.getenv("AIERC_DATA_DIR", DEFAULT_DATA_DIR)).resolve()
    raw_dir = root / "storage" / "raw"
    state_dir = root / "storage" / "state"
    return Settings(
        data_dir=root,
        raw_dir=raw_dir,
        state_dir=state_dir,
        state_file=state_dir / "store.json",
        embedding_dimensions=int(os.getenv("AIERC_EMBEDDING_DIMENSIONS", "128")),
        chunk_target_tokens=int(os.getenv("AIERC_CHUNK_TARGET_TOKENS", "800")),
        chunk_overlap_tokens=int(os.getenv("AIERC_CHUNK_OVERLAP_TOKENS", "80")),
        retrieval_min_score=float(os.getenv("AIERC_RETRIEVAL_MIN_SCORE", "0.04")),
        local_model_name=os.getenv("AIERC_LOCAL_MODEL_NAME", "local-deterministic-grounded-v1"),
        llm_provider=os.getenv("AIERC_LLM_PROVIDER", "auto").lower(),
        ollama_base_url=os.getenv("AIERC_OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
        ollama_model=os.getenv("AIERC_OLLAMA_MODEL", "gemma3:4b"),
        ollama_timeout_seconds=float(os.getenv("AIERC_OLLAMA_TIMEOUT_SECONDS", "180")),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        sec_user_agent=os.getenv(
            "AIERC_SEC_USER_AGENT",
            "AI Equity Research Copilot local demo contact@example.com",
        ),
        sec_timeout_seconds=float(os.getenv("AIERC_SEC_TIMEOUT_SECONDS", "20")),
        sec_max_filing_chars=int(os.getenv("AIERC_SEC_MAX_FILING_CHARS", "600000")),
    )
