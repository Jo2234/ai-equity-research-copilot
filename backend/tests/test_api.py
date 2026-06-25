from __future__ import annotations

from pathlib import Path
from datetime import date

from fastapi.testclient import TestClient

from ai_equity_research_copilot_backend.main import create_app
from ai_equity_research_copilot_backend.schemas import CompanyCreate, DocumentCreate, DocumentType
from ai_equity_research_copilot_backend.sec import SecCompanyMatch, SecFiling


def test_company_upload_chat_memo_and_delete_flow(tmp_path: Path) -> None:
    app = create_app(tmp_path, seed=False)
    client = TestClient(app)

    company_response = client.post("/companies", json={"ticker": "nvda", "name": "NVIDIA Corporation"})
    assert company_response.status_code == 201
    company_id = company_response.json()["id"]

    upload_response = client.post(
        f"/companies/{company_id}/documents",
        data={
            "title": "NVIDIA Local Note",
            "document_type": "manual_note",
            "fiscal_year": "2026",
        },
        files={
            "file": (
                "nvda-note.txt",
                b"MANAGEMENT COMMENTARY\n\nGross margin expanded because mix shifted toward accelerated computing platforms. Demand for AI infrastructure remained broad across cloud and enterprise customers.",
                "text/plain",
            )
        },
    )
    assert upload_response.status_code == 201
    document_payload = upload_response.json()
    assert document_payload["status"] == "ready"
    assert document_payload["chunk_count"] >= 1

    chat_response = client.post(
        "/research/chat",
        json={
            "company_ids": [company_id],
            "question": "What drove gross margin expansion?",
            "top_k": 4,
        },
    )
    assert chat_response.status_code == 200
    chat_payload = chat_response.json()
    assert chat_payload["citations"]
    assert chat_payload["usage"]["provider"] == "local"
    assert chat_payload["conversation_id"]
    assert "gross margin" in chat_payload["answer"].lower()

    retrieve_response = client.post(
        "/research/retrieve",
        json={
            "company_ids": [company_id],
            "query": "gross margin accelerated computing",
            "top_k": 3,
        },
    )
    assert retrieve_response.status_code == 200
    retrieve_payload = retrieve_response.json()
    assert retrieve_payload["total"] >= 1
    assert retrieve_payload["results"][0]["ticker"] == "NVDA"
    assert retrieve_payload["results"][0]["chunk_id"]
    assert "score" in retrieve_payload["results"][0]

    conversation_response = client.get(f"/conversations/{chat_payload['conversation_id']}")
    assert conversation_response.status_code == 200
    conversation_payload = conversation_response.json()
    assert len(conversation_payload["messages"]) == 2
    assistant_messages = [
        message for message in conversation_payload["messages"] if message["role"] == "assistant"
    ]
    assert assistant_messages
    assert assistant_messages[0]["citations"]
    assert assistant_messages[0]["structured_payload"]["retrieval_debug"]

    conversations_response = client.get("/conversations")
    assert conversations_response.status_code == 200
    assert any(row["id"] == chat_payload["conversation_id"] for row in conversations_response.json())

    missing_context_response = client.post(
        "/research/chat",
        json={
            "company_ids": [],
            "question": "What drove gross margin expansion?",
        },
    )
    assert missing_context_response.status_code == 422

    memo_response = client.post("/research/memo", json={"company_id": company_id})
    assert memo_response.status_code == 200
    memo_payload = memo_response.json()
    assert memo_payload["company"]["ticker"] == "NVDA"
    assert isinstance(memo_payload["risk_factors"], list)
    assert memo_payload["usage"]["estimated_cost_usd"] == 0.0

    chunks_response = client.get(f"/documents/{document_payload['id']}/chunks")
    assert chunks_response.status_code == 200
    assert chunks_response.json()["total"] >= 1

    delete_response = client.delete(f"/documents/{document_payload['id']}")
    assert delete_response.status_code == 204
    assert client.get(f"/documents/{document_payload['id']}").status_code == 404


def test_duplicate_ticker_rejected(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path, seed=False))

    assert client.post("/companies", json={"ticker": "MSFT", "name": "Microsoft"}).status_code == 201
    duplicate = client.post("/companies", json={"ticker": "msft", "name": "Microsoft Duplicate"})

    assert duplicate.status_code == 409


def test_sec_company_search_and_discover_imports_latest_filing(tmp_path: Path, monkeypatch) -> None:
    app = create_app(tmp_path, seed=False)
    match = SecCompanyMatch(ticker="AAPL", name="Apple Inc.", cik=320193)
    filing = SecFiling(
        accession_number="0000320193-25-000079",
        filing_date=date(2025, 10, 31),
        report_date=date(2025, 9, 27),
        form="10-K",
        primary_document="aapl-20250927.htm",
        source_url="https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm",
    )

    monkeypatch.setattr(app.state.sec_client, "search_companies", lambda query, limit=8: [match])
    monkeypatch.setattr(app.state.sec_client, "company_corpus_filings", lambda cik, **kwargs: [filing])
    monkeypatch.setattr(app.state.sec_client, "latest_filing", lambda cik, forms: filing)
    monkeypatch.setattr(
        app.state.sec_client,
        "download_filing_text",
        lambda imported_filing: (
            "BUSINESS\n\nApple designs products and services. Services revenue and installed-base engagement "
            "were important growth drivers. RISK FACTORS\n\nSupply constraints and regulatory matters could affect results."
        ),
    )
    monkeypatch.setattr(
        app.state.sec_client,
        "company_create_payload",
        lambda imported_match: CompanyCreate(ticker=imported_match.ticker, name=imported_match.name, industry="SEC CIK 320193"),
    )
    monkeypatch.setattr(
        app.state.sec_client,
        "document_create_payload",
        lambda imported_match, imported_filing: DocumentCreate(
            title="AAPL 10-K filed 2025-10-31",
            document_type=DocumentType.ten_k,
            source_url=imported_filing.source_url,
            filing_date=imported_filing.filing_date,
            period_end_date=imported_filing.report_date,
            fiscal_year=2025,
        ),
    )

    client = TestClient(app)
    search_response = client.get("/companies/search", params={"q": "apple"})
    assert search_response.status_code == 200
    assert search_response.json()[0]["ticker"] == "AAPL"
    assert search_response.json()[0]["already_in_workspace"] is False

    discover_response = client.post("/companies/discover", json={"query": "AAPL", "form_type": "10-k"})
    assert discover_response.status_code == 201
    payload = discover_response.json()
    assert payload["company"]["ticker"] == "AAPL"
    assert payload["company"]["ready_document_count"] == 1
    assert payload["imported_document"]["status"] == "ready"
    assert payload["imported_document"]["chunk_count"] >= 1
    assert len(payload["imported_documents"]) == 1

    chat_response = client.post(
        "/research/chat",
        json={"company_ids": [payload["company"]["id"]], "question": "What were Apple's growth drivers?"},
    )
    assert chat_response.status_code == 200
    assert chat_response.json()["citations"]


def test_chat_uses_ollama_provider_when_configured_and_grounded(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AIERC_LLM_PROVIDER", "ollama")
    app = create_app(tmp_path, seed=False)
    client = TestClient(app)
    company_response = client.post("/companies", json={"ticker": "MU", "name": "Micron Technology"})
    company_id = company_response.json()["id"]
    upload_response = client.post(
        f"/companies/{company_id}/documents",
        json={
            "title": "Micron Filing Excerpt",
            "document_type": "10-k",
            "fiscal_year": 2025,
            "text": "Micron sells memory and storage products. Data center demand supported revenue growth, while cyclical pricing and capital intensity were risk factors.",
            "filename": "mu.txt",
        },
    )
    assert upload_response.status_code == 201

    def fake_answer(question, contexts):
        from ai_equity_research_copilot_backend.llm import GroundedDraft
        from ai_equity_research_copilot_backend.schemas import Confidence

        return GroundedDraft(
            answer="Micron's cited filing excerpt points to data center demand as a revenue driver, with cyclical pricing and capital intensity as risks.",
            key_points=["Data center demand supported revenue growth.", "Pricing cyclicality and capital intensity are risk factors."],
            citation_indices=[1],
            confidence=Confidence.high,
            limitations=[],
            model="gemma3:4b",
            provider="ollama",
        )

    monkeypatch.setattr(app.state.research.ollama, "answer", fake_answer)
    response = client.post(
        "/research/chat",
        json={"company_ids": [company_id], "question": "What drives Micron revenue and what are the risks?"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["usage"]["provider"] == "ollama"
    assert payload["usage"]["model"] == "gemma3:4b"
    assert payload["citations"]
    assert "data center demand" in payload["answer"].lower()
