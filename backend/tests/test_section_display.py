"""Section context must survive real ingestion without rewriting evidence."""

from dataclasses import replace
from uuid import UUID

from fastapi.testclient import TestClient

from ai_equity_research_copilot_backend.main import create_app


def ingest_annual(tmp_path, source):
    app = create_app(tmp_path, seed=False)
    app.state.research.settings = replace(app.state.research.settings, llm_provider="local")
    client = TestClient(app)
    company = client.post("/companies", json={"ticker": "ELMR", "name": "Elm River Group"})
    assert company.status_code == 201
    company_id = company.json()["id"]
    upload = client.post(
        f"/companies/{company_id}/documents",
        data={"title": "Annual financial report", "document_type": "10-k", "fiscal_year": "2031"},
        files={"file": ("annual.txt", source.encode(), "text/plain")},
    )
    assert upload.status_code == 201
    assert upload.json()["status"] == "ready"
    chunks = app.state.repo.list_chunks(document_id=UUID(upload.json()["id"]))
    assert len(chunks) == 1
    # Normalizing line endings and paragraph boundaries must not alter facts,
    # punctuation, headings, or spaces within the original source paragraphs.
    expected = source.replace("\r\n", "\n").replace("\r", "\n").strip()
    assert chunks[0].text == expected
    return app, client, company_id, chunks[0]


def ask(client, company_id, question):
    response = client.post("/research/chat", json={"company_ids": [company_id], "question": question})
    assert response.status_code == 200
    return response.json()


def test_mixed_business_segments_keep_distinct_credit_provision_scopes(tmp_path):
    consumer = "The provision for credit losses was $ 3.7 billion, driven by consumer loan growth."
    commercial = "The provision for credit losses was $ 1.4 billion, driven by commercial borrower defaults."
    source = f"Consumer Banking\r\n\r\n{consumer}\r\n\r\nCommercial Banking\r\n\r\n{commercial}\r\n"
    app, client, company_id, chunk = ingest_annual(tmp_path, source)

    payload = ask(client, company_id, "What drove the annual provision for credit losses?")

    assert f"Consumer Banking: {consumer.replace('$ 3.7', '$3.7')}" in payload["answer"]
    assert f"Commercial Banking: {commercial.replace('$ 1.4', '$1.4')}" in payload["answer"]
    # Citations are deduplicated by chunk; both scoped passages must survive
    # within that shared citation, even when their amounts differ.
    assert len(payload["citations"]) == 1
    citation = payload["citations"][0]
    assert citation["section_title"] is None
    consumer_excerpt = f"Consumer Banking\n...\n{consumer}"
    commercial_excerpt = f"Commercial Banking\n...\n{commercial}"
    assert citation["excerpt"] in {
        consumer_excerpt + "\n" + commercial_excerpt,
        commercial_excerpt + "\n" + consumer_excerpt,
    }
    assert citation["chunk_id"] == str(chunk.id)
    for header, body in (("Consumer Banking", consumer), ("Commercial Banking", commercial)):
        assert f"{header}\n\n{body}" in chunk.text
    assert app.state.repo.list_chunks(document_id=chunk.document_id)[0].text == chunk.text


def test_prior_year_section_is_excluded_from_current_annual_answer(tmp_path):
    current = "The provision for credit losses was $ 4.6 billion, driven by loan growth."
    prior = "The provision for credit losses was $ 8.9 billion, driven by borrower defaults."
    source = f"2031 Financial Results\n\n{current}\n\n2030 Financial Results\n\n{prior}"
    _, client, company_id, chunk = ingest_annual(tmp_path, source)

    payload = ask(client, company_id, "What drove the provision for credit losses in fiscal 2031?")

    assert "2031 Financial Results:" in payload["answer"]
    assert "$4.6 billion" in payload["answer"]
    assert "2030" not in payload["answer"] and "$8.9" not in payload["answer"]
    assert payload["citations"]
    assert {citation["excerpt"] for citation in payload["citations"]} == {
        f"2031 Financial Results\n...\n{current}"
    }
    assert prior in chunk.text


def test_unrecognized_heading_does_not_invent_a_business_segment(tmp_path):
    source = "Notes from the desk\n\nThe provision for credit losses was $ 2.6 billion, driven by loan growth."
    _, client, company_id, chunk = ingest_annual(tmp_path, source)

    payload = ask(client, company_id, "What drove the provision for credit losses?")

    assert "$2.6 billion" in payload["answer"]
    assert "Notes from the desk:" not in payload["answer"]
    assert "Consumer Banking" not in payload["answer"]
    assert "Commercial Banking" not in payload["answer"]
    assert chunk.section_title is None
    assert all(span["section"] is None for span in chunk.metadata["section_spans"])
    assert {citation["excerpt"] for citation in payload["citations"]} == {source.split("\n\n")[1]}


def test_loaded_model_timeout_reports_timeout_without_install_guidance(tmp_path, monkeypatch):
    source = "The provision for credit losses was $ 2.6 billion, driven by loan growth."
    app, client, company_id, _ = ingest_annual(tmp_path, source)
    app.state.research.settings = replace(
        app.state.research.settings, llm_provider="ollama", ollama_model="existing-local-model",
    )
    calls = []

    def timed_out(question, contexts):
        calls.append((question, contexts))
        assert contexts and contexts[0]["excerpt"] == source
        raise RuntimeError("Ollama request timed out")

    monkeypatch.setattr(app.state.research.ollama, "answer", timed_out)
    payload = ask(client, company_id, "What drove the provision for credit losses?")

    assert len(calls) == 1
    assert "timed out" in payload["answer"].lower()
    assert "timeout" in payload["answer"].lower()
    assert "pull" not in payload["answer"].lower()
    assert "start ollama" not in payload["answer"].lower()
    assert payload["citations"] == [] and payload["key_points"] == []
    assert payload["confidence"] == "low"
    assert payload["limitations"] == ["Ollama request timed out"]
    assert payload["usage"]["provider"] == "ollama"
    assert payload["usage"]["model"] == "existing-local-model"


def test_consumer_excerpt_does_not_display_neighboring_commercial_section(tmp_path):
    source = ("Consumer Banking\n\nConsumer loan growth drove provisions of $3.7 billion.\n\n"
              "Commercial Banking\n\nCommercial borrower defaults drove provisions of $1.4 billion.")
    _, client, company_id, _ = ingest_annual(tmp_path, source)
    payload = ask(client, company_id, "What was consumer loan growth?")
    assert payload["citations"]
    assert "Consumer Banking:" in payload["answer"]
    assert all(citation["section_title"] is None for citation in payload["citations"])


def test_uniform_section_is_still_shown_in_citations(tmp_path):
    source = "Consumer Banking\n\nConsumer loan growth drove provisions of $3.7 billion."
    _, client, company_id, _ = ingest_annual(tmp_path, source)
    payload = ask(client, company_id, "What was consumer loan growth?")
    assert payload["citations"]
    assert all(citation["section_title"] == "Consumer Banking" for citation in payload["citations"])


def test_unmentioned_product_revenue_is_not_inferred_from_generic_table(tmp_path):
    source = "Results (USD millions)\n\n2031 2030\n\nRevenue from product sales 4200 3650"
    _, client, company_id, _ = ingest_annual(tmp_path, source)
    for question in ("What was robotaxi revenue?", "What was revenue from lunar tourism?"):
        payload = ask(client, company_id, question)
        assert not payload["citations"]
        assert not payload["key_points"]
    assert ask(client, company_id, "What was product sales revenue?")["citations"]
