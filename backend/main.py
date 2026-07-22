"""Compatibility entry point; all behavior lives in the packaged backend."""

from .ai_equity_research_copilot_backend.main import app, create_app

__all__ = ["app", "create_app"]
