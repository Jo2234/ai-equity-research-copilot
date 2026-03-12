from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient  # noqa: E402

from ai_equity_research_copilot_backend.main import create_app  # noqa: E402

ARTIFACT_DIR = ROOT / "demo_artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "demo_smoke_output.json"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="aierc-demo-") as data_dir:
        demo_data_dir = Path(data_dir)
        shutil.copytree(ROOT / "data" / "sample_documents", demo_data_dir / "sample_documents")
        app = create_app(demo_data_dir, seed=True)
        client = TestClient(app)

        health = client.get("/health")
        health.raise_for_status()
        companies = client.get("/companies")
        companies.raise_for_status()
        nvda = next(company for company in companies.json() if company["ticker"] == "NVDA")

        retrieve = client.post(
            "/research/retrieve",
            json={
                "company_ids": [nvda["id"]],
                "query": "What drove data center revenue growth?",
                "top_k": 3,
            },
        )
        retrieve.raise_for_status()
        chat = client.post(
            "/research/chat",
            json={
                "company_ids": [nvda["id"]],
                "question": "What drove NVIDIA data center revenue growth?",
                "top_k": 3,
            },
        )
        chat.raise_for_status()
        memo = client.post("/research/memo", json={"company_id": nvda["id"]})
        memo.raise_for_status()

        chat_payload = chat.json()
        artifact = {
            "status": "ok",
            "health": health.json(),
            "selected_company": {"id": nvda["id"], "ticker": nvda["ticker"], "name": nvda["name"]},
            "retrieval": {
                "total": retrieve.json()["total"],
                "top_citations": [
                    {
                        "document_title": row["document_title"],
                        "page_start": row["page_start"],
                        "score": row["score"],
                        "excerpt": row["excerpt"][:240],
                    }
                    for row in retrieve.json()["results"][:3]
                ],
            },
            "chat": {
                "answer": chat_payload["answer"],
                "citation_count": len(chat_payload["citations"]),
                "provider": chat_payload["usage"]["provider"],
                "model": chat_payload["usage"]["model"],
            },
            "memo_sections": {
                "business_summary": memo.json()["business_summary"],
                "growth_drivers": memo.json()["growth_drivers"][:3],
                "risk_factors": memo.json()["risk_factors"][:3],
            },
        }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"wrote {ARTIFACT_PATH.relative_to(ROOT)}")
    print(json.dumps({"status": "ok", "artifact": str(ARTIFACT_PATH.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
