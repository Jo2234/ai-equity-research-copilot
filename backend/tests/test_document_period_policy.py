from __future__ import annotations

from datetime import date

import pytest

from ai_equity_research_copilot_backend.embeddings import HashingEmbedder
from ai_equity_research_copilot_backend.query import content_terms, explicit_periods, passage_period, question_scope
from ai_equity_research_copilot_backend.retrieval import RetrievalService
from ai_equity_research_copilot_backend.schemas import CompanyCreate, DocumentChunk, DocumentCreate, DocumentStatus, DocumentType
from ai_equity_research_copilot_backend.storage import JsonRepository


@pytest.fixture
def corpus(tmp_path):
    repo = JsonRepository(tmp_path / "corpus.json")
    company = repo.create_company(CompanyCreate(ticker="ARC", name="Arc Corporation"))
    embedder = HashingEmbedder()

    def add(year, quarter=None, kind=DocumentType.ten_k, text="Revenue grew 14 percent because demand increased.", owner=None, end=None):
        owner = owner or company
        document = repo.create_document(owner.id, DocumentCreate(title=f"{kind.value} {year} {quarter}",
            fiscal_year=year, fiscal_quarter=quarter, document_type=kind, period_end_date=end), None)
        chunk = DocumentChunk(document_id=document.id, company_id=owner.id, chunk_index=0,
            text=text, embedding=embedder.embed(text), token_count=len(text.split()))
        repo.replace_chunks(document.id, [chunk])
        document.status = DocumentStatus.ready
        repo.update_document(document)
        return document

    return repo, company, add, RetrievalService(repo, embedder)


def test_unspecified_period_uses_latest_annual_and_exposes_assumption(corpus):
    repo, company, add, service = corpus
    add(2023)
    latest_annual = add(2024)
    add(2025, 1, DocumentType.ten_q)
    results = service.search("What drove revenue growth?", [company.id])
    assert {r.document.id for r in results} == {latest_annual.id}
    assert all("latest available annual" in r.chunk.metadata["retrieval_period_policy"] for r in results)
    assert all("retrieval_period_policy" not in c.metadata for c in service.prepare([company.id]).snapshot.chunks)
    assert all("retrieval_period_policy" not in c.metadata for c in repo.list_chunks())


def test_explicit_source_and_year_filters_override_default(corpus):
    _, company, add, service = corpus
    annual = add(2024)
    quarter = add(2025, 1, DocumentType.ten_q)
    assert {r.document.id for r in service.search("Revenue growth", [company.id], document_types=[DocumentType.ten_q])} == {quarter.id}
    assert {r.document.id for r in service.search("Revenue growth", [company.id], fiscal_years=[2024])} == {annual.id}
    assert not service.search("Revenue growth in fiscal 2022", [company.id])


def test_each_company_gets_its_own_latest_annual(corpus):
    repo, company, add, service = corpus
    one = add(2024)
    other = repo.create_company(CompanyCreate(ticker="BIR", name="Birch Corporation"))
    add(2023, owner=other)
    two = add(2025, owner=other)
    results = service.search("Compare revenue growth", [company.id, other.id])
    assert {r.document.id for r in results} == {one.id, two.id}


def test_quarter_results_and_call_comparison_keeps_both(corpus):
    _, company, add, service = corpus
    add(2024)
    quarter = add(2025, 1, DocumentType.ten_q)
    call = add(2025, 1, DocumentType.earnings_transcript)
    results = service.search("Compare quarterly revenue results with earnings call commentary.", [company.id])
    assert {r.document.id for r in results} == {quarter.id, call.id}


def test_quarter_disclosure_and_release_comparison_keeps_both(corpus):
    _, company, add, service = corpus
    quarter = add(2025, 1, DocumentType.ten_q)
    release = add(2025, 1, DocumentType.eight_k)
    results = service.search("Compare Q1 FY2025 revenue disclosure with comments in the earnings release.", [company.id])
    assert {r.document.id for r in results} == {quarter.id, release.id}


def test_mixed_release_quarter_requires_explicit_matching_period(corpus):
    repo, company, add, service = corpus
    mixed = add(2025, kind=DocumentType.eight_k, text="Fiscal 2025 Q4 Results\n\nRevenue grew 14 percent because demand increased.")
    add(2025, kind=DocumentType.eight_k, text="Revenue grew 99 percent because demand increased.")
    results = service.search("What was revenue growth in Q4 FY2025?", [company.id])
    assert {r.document.id for r in results} == {mixed.id}
    assert results[0].chunk.metadata["retrieval_require_passage_period"] == [[2025, 4]]
    assert not service.search("What was revenue growth in Q3 FY2025?", [company.id])
    assert all("retrieval_require_passage_period" not in c.metadata for c in repo.list_chunks())


def test_short_fiscal_year_requires_matching_document_year():
    assert explicit_periods("Q4 FY25 results", 2025) == {(2025, 4)}
    assert not explicit_periods("Q4 FY25 results", 2026)
    assert not explicit_periods("Q4 FY25 results")
    assert passage_period("Revenue grew during fiscal year 2025.", (2025, 4)) == (2025, None)


def test_latest_request_does_not_default_to_older_annual(corpus):
    _, company, add, service = corpus
    add(2024)
    latest = add(2025, 1, DocumentType.ten_q)
    results = service.search("What was the latest revenue growth?", [company.id])
    assert {r.document.id for r in results} == {latest.id}
    assert "latest available fiscal period" in results[0].chunk.metadata["retrieval_period_policy"]


def test_management_commentary_comparison_permits_transcript_alongside_filing(corpus):
    _, company, add, service = corpus
    quarter = add(2025, 1, DocumentType.ten_q, text="Average deposits increased because new accounts grew.")
    call = add(2025, 1, DocumentType.earnings_transcript, text="Deposit behavior reflects growth in new accounts and lower outflows.")
    add(2024, kind=DocumentType.ten_k, text="Deposit balances declined during the fiscal year.")
    query = "Compare quarterly deposit disclosures with management commentary on deposit behavior."
    results = service.search(query, [company.id])
    assert {r.document.id for r in results} == {quarter.id, call.id}
    filtered = service.search(query, [company.id], document_types=[DocumentType.ten_q])
    assert {r.document.id for r in filtered} == {quarter.id}


def test_explicit_management_commentary_source_is_not_replaced_by_transcript():
    from ai_equity_research_copilot_backend.query import question_scope
    scope = question_scope("Compare quarterly deposit disclosures with management commentary in the annual filing.")
    assert DocumentType.earnings_transcript not in scope.types
    assert DocumentType.ten_k in scope.types
    ordinary = question_scope("Summarize management commentary on deposit behavior.")
    assert not ordinary.types


def test_latest_full_annual_period_does_not_sort_before_same_year_q1(corpus):
    _, company, add, service = corpus
    annual = add(2031)
    add(2031, 1, DocumentType.ten_q)
    results = service.search("What was the latest revenue?", [company.id])
    assert {result.document.id for result in results} == {annual.id}


def test_latest_prefers_actual_period_end_dates_over_fiscal_label_order(corpus):
    _, company, add, service = corpus
    # Fiscal labels alone would put the annual after Q1. Explicit end dates
    # identify the actual chronology, even when the labels do not agree.
    add(2031, end=date(2031, 1, 31))
    quarter = add(2031, 1, DocumentType.ten_q, end=date(2031, 4, 30))
    results = service.search("What was the latest revenue?", [company.id])
    assert {result.document.id for result in results} == {quarter.id}


def test_unknown_release_period_is_retained_with_explicit_ambiguity(corpus):
    _, company, add, service = corpus
    quarter = add(2031, 1, DocumentType.ten_q)
    release = add(2031, kind=DocumentType.eight_k)
    results = service.search("What was the latest revenue?", [company.id])
    assert {result.document.id for result in results} == {quarter.id, release.id}
    assert all("cannot be ordered" in result.chunk.metadata["retrieval_period_policy"] for result in results)


def test_unknown_fiscal_year_is_not_silently_assumed_older(corpus):
    _, company, add, service = corpus
    annual = add(2031)
    release = add(None, kind=DocumentType.eight_k)
    results = service.search("What was the latest revenue?", [company.id])
    assert {result.document.id for result in results} == {annual.id, release.id}
    assert all("cannot be ordered" in result.chunk.metadata["retrieval_period_policy"] for result in results)


@pytest.mark.parametrize("ordinal, quarter", [("first", 1), ("second", 2), ("third", 3), ("fourth", 4)])
def test_ordinal_quarter_query_does_not_default_to_annual(corpus, ordinal, quarter):
    _, company, add, service = corpus
    add(2031)
    selected = add(2031, quarter, DocumentType.ten_q)
    query = f"What was revenue in the {ordinal} quarter of fiscal 2031?"
    scope = question_scope(query)
    assert scope.periods == ((2031, quarter),)
    assert scope.quarters == {quarter}
    assert content_terms(query) == content_terms(f"What was revenue in Q{quarter} FY2031?")
    assert {result.document.id for result in service.search(query, [company.id])} == {selected.id}


@pytest.mark.parametrize("text", ["Revenue grew 15% in Q4 FY30.", "Revenue grew 15% in FY30."])
def test_conflicting_short_year_never_inherits_current_year(text):
    assert passage_period(text, (2031, 4), 2031) is None


def test_annual_and_quarter_window_has_no_exclusive_period():
    text = ("Subscription revenue was $2.9 billion in Q4 FY2031. "
            "Full-year FY2031 subscription revenue was $10.2 billion.")
    assert passage_period(text, (2031, 4), 2031) is None
