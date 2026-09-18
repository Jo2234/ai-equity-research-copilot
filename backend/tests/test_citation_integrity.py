from __future__ import annotations

import json
import re

import pytest
from fastapi.testclient import TestClient

from ai_equity_research_copilot_backend.main import create_app
from ai_equity_research_copilot_backend.schemas import (
    CompanyCreate, CompareRequest, DocumentChunk, DocumentCreate, DocumentStatus,
    DocumentType, MemoRequest, RetrievalDebugResult,
)


def corpus_app(tmp_path, monkeypatch, count=6):
    app = create_app(tmp_path, seed=False)
    repo = app.state.repo
    company = repo.create_company(CompanyCreate(ticker="PROOF", name="Proof Company"))
    document = repo.create_document(company.id, DocumentCreate(
        title="Source report", document_type=DocumentType.ten_k, fiscal_year=2025,
    ), None)
    document.status = DocumentStatus.ready
    repo.update_document(document)
    results = []
    for index in range(count):
        text = ("General corporate background and employee policies are described here." if index < count - 1
                else "Revenue growth was driven exclusively by the new zephyr product launch.")
        chunk = DocumentChunk(document_id=document.id, company_id=company.id,
            chunk_index=index, text=text, embedding=app.state.retrieval.embedder.embed(text), token_count=15)
        results.append(RetrievalDebugResult(query="revenue growth", chunk=chunk, document=document,
            company=company, score=.9 - index * .05, keyword_score=0, vector_score=.5))
    repo.replace_chunks(document.id, [result.chunk for result in results])
    monkeypatch.setattr(app.state.retrieval, "search", lambda *args, **kwargs: results)
    return app, company, results


def assert_point_sources(payload):
    for point in payload["key_points"]:
        match = re.search(r" \[(\d+)\]$", point)
        assert match, point
        assert point[:match.start()] in payload["citations"][int(match[1]) - 1]["excerpt"]


@pytest.mark.parametrize("question", [
    "What was robotaxi revenue?", "What was revenue from lunar tourism?",
])
def test_related_financial_words_do_not_answer_an_unmentioned_business_line(tmp_path, monkeypatch, question):
    app, company, results = corpus_app(tmp_path, monkeypatch, count=1)
    results[0].chunk.text = "Revenue grew 15% to $4.2 billion because pricing improved and demand increased."
    payload = TestClient(app).post("/research/chat", json={
        "company_ids": [str(company.id)], "question": question,
    }).json()
    assert not payload["citations"]
    assert not payload["key_points"]
    assert "$4.2" not in payload["answer"]


def test_lower_ranked_support_is_cited_and_audit_matches_response(tmp_path, monkeypatch):
    app, company, results = corpus_app(tmp_path, monkeypatch)
    client = TestClient(app)
    response = client.post("/research/chat", json={"company_ids": [str(company.id)],
        "question": "What drove revenue growth from the zephyr product launch?"})
    assert response.status_code == 200
    payload = response.json()
    assert "zephyr" in payload["answer"]
    assert str(results[5].chunk.id) in {citation["chunk_id"] for citation in payload["citations"]}
    assert_point_sources(payload)
    cited = {citation["chunk_id"] for citation in payload["citations"]}
    stored = client.get(f"/conversations/{payload['conversation_id']}").json()["messages"][-1]
    assert {citation["document_chunk_id"] for citation in stored["citations"]} == cited
    assert {chunk["id"] for chunk in payload["retrieval_debug"]["chunks"] if chunk["cited"]} == cited
    assert payload["usage"]["retrieval"]["cited_chunks"] == len(cited)


def test_memo_and_comparison_keep_sentence_provenance(tmp_path, monkeypatch):
    app, company, results = corpus_app(tmp_path, monkeypatch)
    memo = app.state.research.memo(MemoRequest(company_id=company.id))
    assert results[5].chunk.id in {citation.chunk_id for citation in memo.source_citations}
    for field in ["business_summary", "recent_performance", "growth_drivers", "margin_analysis",
                  "capital_allocation", "risk_factors", "management_commentary"]:
        value = getattr(memo, field)
        for point in [value] if isinstance(value, str) else value:
            match = re.search(r" \[(\d+)\]$", point)
            if match:
                assert point[:match.start()] in memo.source_citations[int(match[1]) - 1].excerpt
    other = app.state.repo.create_company(CompanyCreate(ticker="OTHER", name="Other Company"))
    # Provide only this company's real retrieval results; no cross-company source sharing.
    monkeypatch.setattr(app.state.retrieval, "search", lambda *a, **kw: results if company.id in kw["company_ids"] else [])
    compared = app.state.research.compare(CompareRequest(company_ids=[company.id, other.id], question="Revenue growth zephyr product launch"))
    assert_point_sources(compared.comparisons[0].model_dump(mode="json"))
    assert not compared.comparisons[1].citations


@pytest.mark.parametrize("provider", ["auto", "ollama"])
@pytest.mark.parametrize("model_payload", [
    [], {"answer": "No citations"},
    {"answer": "Unsupported claim.", "key_points": ["Unsupported claim."], "citation_indices": [], "confidence": "high", "limitations": []},
    {"answer": "Claim [99].", "key_points": ["Claim [99]."], "citation_indices": [99], "confidence": "high", "limitations": []},
    {"answer": "Claim [1].", "key_points": ["Claim [1]."], "citation_indices": [True], "confidence": "high", "limitations": []},
    {"answer": "Claim [1].", "key_points": ["Claim without citation."], "citation_indices": [1], "confidence": "high", "limitations": []},
    {"answer": "Claim [1].", "key_points": ["Claim [1]."], "citation_indices": [2], "confidence": "high", "limitations": []},
])
def test_invalid_model_output_falls_back_without_attaching_arbitrary_sources(tmp_path, monkeypatch, provider, model_payload):
    monkeypatch.setenv("AIERC_LLM_PROVIDER", provider)
    app, company, _ = corpus_app(tmp_path, monkeypatch)
    monkeypatch.setattr(app.state.research.ollama, "_post", lambda *a: {"response": json.dumps(model_payload)})
    payload = TestClient(app).post("/research/chat", json={"company_ids": [str(company.id)],
        "question": "What drove zephyr revenue growth?"}).json()
    assert payload["usage"]["provider"] == "local"
    assert "zephyr" in payload["answer"]
    assert any("failed citation validation" in limitation for limitation in payload["limitations"])
    assert_point_sources(payload)


def test_model_references_are_renumbered_and_full_context_is_available(tmp_path, monkeypatch):
    monkeypatch.setenv("AIERC_LLM_PROVIDER", "ollama")
    app, company, results = corpus_app(tmp_path, monkeypatch)
    results[2].chunk.text = "Background material. " * 30 + "Distinct supporting evidence at the end."
    def model(path, payload):
        assert "Distinct supporting evidence at the end." in payload["prompt"]
        return {"response": json.dumps({"answer": "Third source claim [3]. First source claim [1].",
            "key_points": ["Third source claim [3].", "First source claim [1]."],
            "citation_indices": [3, 1], "confidence": "medium", "limitations": []})}
    monkeypatch.setattr(app.state.research.ollama, "_post", model)
    client = TestClient(app)
    payload = client.post("/research/chat", json={"company_ids": [str(company.id)], "question": "Revenue growth"}).json()
    assert payload["usage"]["provider"] == "ollama"
    assert payload["answer"] == "Third source claim [1]. First source claim [2]."
    assert [citation["chunk_id"] for citation in payload["citations"]] == [str(results[2].chunk.id), str(results[0].chunk.id)]
    assert "Distinct supporting evidence at the end." in payload["citations"][0]["excerpt"]
    stored = client.get(f"/conversations/{payload['conversation_id']}").json()["messages"][-1]
    assert [citation["document_chunk_id"] for citation in stored["citations"]] == [citation["chunk_id"] for citation in payload["citations"]]
    assert payload["usage"]["retrieval"]["cited_chunks"] == 2


@pytest.mark.parametrize("error, expected", [
    ("Ollama unavailable", "unavailable"),
    ("Ollama request timed out", "timed out"),
])
def test_required_unavailable_model_does_not_invent_citations(tmp_path, monkeypatch, error, expected):
    monkeypatch.setenv("AIERC_LLM_PROVIDER", "ollama")
    app, company, _ = corpus_app(tmp_path, monkeypatch)
    def unavailable(*args):
        raise RuntimeError(error)
    monkeypatch.setattr(app.state.research.ollama, "_post", unavailable)
    payload = TestClient(app).post("/research/chat", json={"company_ids": [str(company.id)], "question": "Revenue growth"}).json()
    assert payload["usage"]["provider"] == "ollama"
    assert expected in payload["answer"]
    assert payload["citations"] == []
    assert payload["key_points"] == []
    assert payload["limitations"] == [error]
    if expected == "timed out":
        assert "Start Ollama" not in payload["answer"]
        assert "pull" not in payload["answer"]


@pytest.mark.parametrize("body", [b"<html>bad gateway</html>", b"[]"])
def test_malformed_provider_transport_response_falls_back(tmp_path, monkeypatch, body):
    from io import BytesIO
    from ai_equity_research_copilot_backend import llm

    monkeypatch.setenv("AIERC_LLM_PROVIDER", "auto")
    app, company, _ = corpus_app(tmp_path, monkeypatch)
    monkeypatch.setattr(llm, "urlopen", lambda *args, **kwargs: BytesIO(body))
    payload = TestClient(app).post("/research/chat", json={
        "company_ids": [str(company.id)], "question": "Revenue growth zephyr product launch",
    }).json()
    assert payload["usage"]["provider"] == "local"
    assert_point_sources(payload)


def test_comparison_keeps_each_source_even_when_scores_differ(tmp_path, monkeypatch):
    app, company, results = corpus_app(tmp_path, monkeypatch, count=1)
    annual = results[0]
    annual.score = 100
    annual.chunk.text = (
        "Subscription revenue growth reached 28 percent because enterprise demand increased.\n\n"
        "Advertising revenue growth reached 19 percent because customer spending increased."
    )
    quarterly = annual.model_copy(deep=True)
    quarterly.document = app.state.repo.create_document(company.id, DocumentCreate(
        title="Quarterly report", document_type=DocumentType.ten_q, fiscal_year=2026, fiscal_quarter=1,
    ), None)
    quarterly.chunk = DocumentChunk(document_id=quarterly.document.id, company_id=company.id,
        chunk_index=0, text="Subscription revenue growth reached 12 percent because customer demand improved.",
        token_count=15, embedding=[0.0] * 128)
    quarterly.score = 0.1
    points = app.state.research._points_from_results(
        "Compare revenue growth in the annual and quarterly filings.", [annual, quarterly], max_points=2,
    )
    assert {point.result.document.id for point in points} == {annual.document.id, quarterly.document.id}
