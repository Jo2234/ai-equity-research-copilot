import pytest

from ai_equity_research_copilot_backend.main import create_app
from ai_equity_research_copilot_backend.schemas import (
    Company, Document, DocumentChunk, DocumentType, RetrievalDebugResult,
)


def source(company, text, year, quarter, kind):
    document = Document(company_id=company.id, title="Financial results",
                        fiscal_year=year, fiscal_quarter=quarter, document_type=kind)
    chunk = DocumentChunk(document_id=document.id, company_id=company.id,
                          chunk_index=0, text=text, embedding=[], token_count=60)
    return RetrievalDebugResult(query="", company=company, document=document,
                                chunk=chunk, score=0.8, keyword_score=0.8, vector_score=0.8)


def test_undated_annual_component_keeps_own_period_when_quarter_is_named(tmp_path):
    company = Company(ticker="PINE", name="Pine Corporation")
    results = [
        source(company, "Subscription revenue grew 19% in fiscal 2030 because demand increased.",
               2030, None, DocumentType.ten_k),
        source(company, "Subscription revenue grew 24% in Q1 FY2031 because adoption increased.",
               2031, 1, DocumentType.ten_q),
    ]
    service = create_app(tmp_path, seed=False).state.research
    points = service._points_from_results(
        "Compare annual subscription revenue growth with Q1 FY2031 revenue growth.", results, 4)
    assert {point.result.document.id for point in points} == {item.document.id for item in results}


def test_named_annual_year_still_rejects_other_passage_year(tmp_path):
    company = Company(ticker="PINE", name="Pine Corporation")
    results = [source(company,
        "Subscription revenue grew 19% in fiscal 2030 because demand increased.\n\n"
        "Subscription revenue grew 11% in fiscal 2029 because demand increased.",
        2030, None, DocumentType.ten_k)]
    service = create_app(tmp_path, seed=False).state.research
    points = service._points_from_results(
        "Compare annual FY2030 subscription revenue with Q1 FY2031 subscription revenue.", results, 4)
    assert points and all("19%" in point.text and "11%" not in point.text for point in points)


def test_future_guidance_does_not_answer_historical_quarter(tmp_path):
    company = Company(ticker="PINE", name="Pine Corporation")
    results = [source(company,
        "Q1 FY2031 Financial Highlights\n\n"
        "Subscription revenue was $9.2 billion, up 24%.\n\n"
        "Guidance for Q2 FY2031\n\n"
        "Subscription revenue is expected to increase 27% to $10.1 billion.",
        2031, 1, DocumentType.eight_k)]
    service = create_app(tmp_path, seed=False).state.research
    points = service._points_from_results("What was subscription revenue in Q1 FY2031?", results, 4)
    assert points and all("$9.2" in point.text and "$10.1" not in point.text for point in points)


def test_bare_year_results_heading_preserves_prior_year_boundary(tmp_path):
    company = Company(ticker="PINE", name="Pine Corporation")
    results = [source(company,
        "2031 Results\n\nSubscription revenue was $9.2 billion, up 24%.\n\n"
        "2030 Results\n\nSubscription revenue was $7.4 billion, up 19%.",
        2031, None, DocumentType.ten_k)]
    service = create_app(tmp_path, seed=False).state.research
    points = service._points_from_results("What was subscription revenue in FY2031?", results, 4)
    assert points and all("$9.2" in point.text and "$7.4" not in point.text for point in points)


def test_mixed_release_requires_matching_local_period_for_every_point(tmp_path):
    company = Company(ticker="PINE", name="Pine Corporation")
    result = source(company,
        "Subscription revenue was $2.5 billion, up 13%.\n\n"
        "Q4 FY2031 Results\n\nSubscription revenue was $2.9 billion, up 16%.\n\n"
        "Full-year FY2031 Results\n\nSubscription revenue was $10.2 billion, up 15%.",
        2031, None, DocumentType.eight_k)
    result.chunk.metadata["retrieval_require_passage_period"] = [[2031, 4]]
    service = create_app(tmp_path, seed=False).state.research
    points = service._points_from_results("What was subscription revenue in Q4 FY2031?", [result], 4)
    assert points and all("$2.9" in point.text and "$2.5" not in point.text
                          and "$10.2" not in point.text for point in points)


@pytest.mark.parametrize("require_passage_period", [False, True])
def test_conflicting_short_year_is_not_current_quarter_evidence(tmp_path, require_passage_period):
    company = Company(ticker="PINE", name="Pine Corporation")
    result = source(company,
        "Q4 FY2031 Results\n\nSubscription revenue grew 15% in Q4 FY30 because demand increased.",
        2031, 4, DocumentType.eight_k)
    if require_passage_period:
        result.chunk.metadata["retrieval_require_passage_period"] = [[2031, 4]]
    service = create_app(tmp_path, seed=False).state.research
    assert service._points_from_results("What was subscription revenue in Q4 FY2031?", [result], 4) == []


@pytest.mark.parametrize("require_passage_period", [False, True])
def test_mixed_period_window_is_excluded_but_correct_single_sentence_remains(tmp_path, require_passage_period):
    company = Company(ticker="PINE", name="Pine Corporation")
    quarter = "Subscription revenue was $2.9 billion in Q4 FY2031 because demand increased."
    annual = "Full-year FY2031 subscription revenue was $10.2 billion because adoption increased."
    result = source(company, quarter + " " + annual, 2031, 4, DocumentType.eight_k)
    if require_passage_period:
        result.chunk.metadata["retrieval_require_passage_period"] = [[2031, 4]]
    service = create_app(tmp_path, seed=False).state.research
    points = service._points_from_results("What was subscription revenue in Q4 FY2031?", [result], 4)
    assert [point.text for point in points] == [quarter]


def test_ordinal_quarter_query_returns_matching_passage(tmp_path):
    company = Company(ticker="PINE", name="Pine Corporation")
    result = source(company, "Revenue grew 15% because demand increased.", 2031, 1, DocumentType.ten_q)
    service = create_app(tmp_path, seed=False).state.research
    points = service._points_from_results("What was revenue in the first quarter of fiscal 2031?", [result], 4)
    assert points and points[0].text == result.chunk.text
