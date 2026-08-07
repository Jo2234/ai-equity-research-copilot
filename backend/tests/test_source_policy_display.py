from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from ai_equity_research_copilot_backend.llm import GroundedDraft
from ai_equity_research_copilot_backend.main import create_app
from ai_equity_research_copilot_backend.schemas import (
    CompanyCreate,
    Confidence,
    DocumentCreate,
    DocumentType,
)


SOURCE = "Subscription revenue was $ 7.42 billion, representing growth of 16 %."


@pytest.fixture
def research_workspace(tmp_path):
    app = create_app(tmp_path, seed=False)
    company = app.state.repo.create_company(
        CompanyCreate(ticker="MAPL", name="Maple Corporation")
    )
    documents = []
    for title, year, quarter, kind, text in (
        ("Annual results", 2031, None, DocumentType.ten_k, SOURCE),
        ("Quarter results", 2032, 1, DocumentType.ten_q,
         "Subscription revenue was $ 2.10 billion, representing growth of 22 %."),
    ):
        path = tmp_path / f"{title}.txt"
        path.write_text(text)
        document = app.state.repo.create_document(
            company.id,
            DocumentCreate(title=title, fiscal_year=year, fiscal_quarter=quarter,
                           document_type=kind),
            str(path),
        )
        app.state.ingestion.ingest_document(document.id)
        documents.append(document)
    return app, company, documents[0]


@pytest.mark.parametrize("provider", ["auto", "ollama"])
def test_successful_model_answer_displays_source_policy(research_workspace, monkeypatch, provider):
    app, company, annual = research_workspace
    app.state.research.settings = replace(app.state.research.settings, llm_provider=provider)
    calls = []

    def answer(question, contexts):
        calls.append((question, contexts))
        assert len(contexts) == 1
        assert contexts[0]["excerpt"] == SOURCE
        return GroundedDraft(
            answer="Subscription revenue grew 16% to $7.42 billion. [1]",
            key_points=["Subscription revenue grew 16%. [1]"],
            citation_indices=[1], confidence=Confidence.medium,
            limitations=["Only the supplied evidence was considered."],
            model="test-local-model", provider="ollama",
        )

    monkeypatch.setattr(app.state.research.ollama, "answer", answer)
    response = TestClient(app).post("/research/chat", json={
        "company_ids": [str(company.id)], "question": "What was subscription revenue growth?",
    })
    assert response.status_code == 200
    payload = response.json()
    assert len(calls) == 1
    assert payload["answer"] == "Subscription revenue grew 16% to $7.42 billion. [1]"
    assert {citation["document_id"] for citation in payload["citations"]} == {str(annual.id)}
    assert "Only the supplied evidence was considered." in payload["limitations"]
    assert any("latest available annual" in limitation for limitation in payload["limitations"])


def test_rendered_answer_preserves_original_evidence(research_workspace, monkeypatch):
    app, company, annual = research_workspace
    app.state.research.settings = replace(app.state.research.settings, llm_provider="local")

    def unexpected_model_call(*args, **kwargs):
        raise AssertionError("Deterministic rendering must not invoke a model")

    monkeypatch.setattr(app.state.research.ollama, "answer", unexpected_model_call)
    response = TestClient(app).post("/research/chat", json={
        "company_ids": [str(company.id)], "question": "What was subscription revenue growth?",
    })
    assert response.status_code == 200
    payload = response.json()
    assert "MAPL 10-k FY2031:" in payload["answer"]
    assert "$7.42 billion" in payload["answer"] and "16%" in payload["answer"]
    assert "$ 7.42" not in payload["answer"] and "16 %" not in payload["answer"]
    assert payload["key_points"] == [f"{SOURCE} [1]"]
    assert len(payload["citations"]) == 1
    assert payload["citations"][0]["excerpt"] == SOURCE
    assert payload["citations"][0]["document_id"] == str(annual.id)
    assert any("latest available annual" in limitation for limitation in payload["limitations"])
    chunks = app.state.repo.list_chunks(document_id=annual.id)
    assert len(chunks) == 1 and chunks[0].text == SOURCE


def test_memo_discloses_annual_baseline_once(research_workspace):
    app, company, annual = research_workspace
    response = TestClient(app).post("/research/memo", json={"company_id": str(company.id)})
    assert response.status_code == 200
    payload = response.json()
    assert {citation["document_id"] for citation in payload["source_citations"]} == {str(annual.id)}
    assert sum("latest available annual" in limitation for limitation in payload["limitations"]) == 1
    assert any("live prices" in limitation for limitation in payload["limitations"])


def test_compare_discloses_source_policy_for_each_company(research_workspace, tmp_path):
    app, company, annual = research_workspace
    other = app.state.repo.create_company(CompanyCreate(ticker="PINE", name="Pine Corporation"))
    path = tmp_path / "other.txt"
    path.write_text(SOURCE)
    document = app.state.repo.create_document(other.id,
        DocumentCreate(title="Other annual report", fiscal_year=2030, document_type=DocumentType.ten_k), str(path))
    app.state.ingestion.ingest_document(document.id)
    response = TestClient(app).post("/research/compare", json={
        "company_ids": [str(company.id), str(other.id)], "question": "Compare subscription revenue growth.",
    })
    assert response.status_code == 200
    payload = response.json()
    for ticker in ("MAPL", "PINE"):
        assert any(limitation.startswith(f"{ticker}: ") and "latest available annual" in limitation
                   for limitation in payload["limitations"])
    assert any("market data" in limitation for limitation in payload["limitations"])
    assert {citation["document_id"] for item in payload["comparisons"] for citation in item["citations"]} == {
        str(annual.id), str(document.id),
    }
