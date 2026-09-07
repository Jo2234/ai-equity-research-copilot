from __future__ import annotations

from ai_equity_research_copilot_backend.main import create_app
from ai_equity_research_copilot_backend.schemas import CompanyCreate, DocumentCreate, DocumentType, MemoRequest, CompareRequest


def add_document(app, company, tmp_path, title, year, document_type, text):
    source = tmp_path / (title + ".txt")
    source.write_text(text)
    doc = app.state.repo.create_document(company.id, DocumentCreate(title=title,
        fiscal_year=year, document_type=document_type), str(source))
    app.state.ingestion.ingest_document(doc.id)
    return doc


def test_one_snapshot_for_each_memo_and_comparison(tmp_path, monkeypatch):
    app = create_app(tmp_path, seed=True)
    companies = app.state.repo.list_companies()
    original = app.state.repo._read_state
    reads = []
    def counted():
        reads.append(1)
        return original()
    monkeypatch.setattr(app.state.repo, "_read_state", counted)
    memo = app.state.research.memo(MemoRequest(company_id=companies[0].id))
    assert memo.source_citations
    assert len(reads) == 1
    reads.clear()
    compared = app.state.research.compare(CompareRequest(company_ids=[company.id for company in companies]))
    assert len(compared.comparisons) == 2
    assert len(reads) == 1


def test_snapshot_preserves_filters_and_next_request_observes_ingestion(tmp_path):
    app = create_app(tmp_path, seed=False)
    company = app.state.repo.create_company(CompanyCreate(ticker="ONE", name="One"))
    other = app.state.repo.create_company(CompanyCreate(ticker="TWO", name="Two"))
    text = "Revenue growth reflected sustained customer demand for enterprise software."
    wanted = add_document(app, company, tmp_path, "wanted", 2025, DocumentType.ten_k, text)
    add_document(app, company, tmp_path, "older", 2024, DocumentType.ten_k, text)
    add_document(app, company, tmp_path, "note", 2025, DocumentType.manual_note, text)
    add_document(app, other, tmp_path, "foreign", 2025, DocumentType.ten_k, text)
    prepared = app.state.retrieval.prepare([company.id])
    assert all(chunk.company_id == company.id for chunk in prepared.snapshot.chunks)
    params = dict(company_ids=[company.id], document_types=[DocumentType.ten_k], fiscal_years=[2025], min_score=0)
    first = app.state.retrieval.search("Revenue growth", **params, corpus=prepared)
    assert {result.document.id for result in first} == {wanted.id}
    added = add_document(app, company, tmp_path, "new", 2025, DocumentType.ten_k, text)
    during_request = app.state.retrieval.search("Revenue growth", **params, corpus=prepared)
    assert [(row.chunk.id, row.score) for row in during_request] == [(row.chunk.id, row.score) for row in first]
    next_request = app.state.retrieval.search("Revenue growth", **params)
    assert {row.document.id for row in next_request} == {wanted.id, added.id}
