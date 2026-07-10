from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

os.environ.setdefault("AIERC_DATA_DIR", "/tmp/aierc-public-demo-v1")
os.environ.setdefault("AIERC_SEED_DIR", str(PROJECT_ROOT / "data" / "sample_documents"))
os.environ.setdefault("AIERC_LLM_PROVIDER", "local")
os.environ.setdefault("AIERC_DEMO_MODE", "true")

from ai_equity_research_copilot_backend.main import create_app  # noqa: E402


app = FastAPI(title="AI Equity Research Copilot")
app.mount("/api", create_app(seed=True))
