from __future__ import annotations

from ai_equity_research_copilot_backend.main import create_app
from ai_equity_research_copilot_backend.query import question_scope
from ai_equity_research_copilot_backend.schemas import Company, Document, DocumentChunk, DocumentType, RetrievalDebugResult


def result(text, *, company=None, document=None, score=0.5):
    company = company or Company(ticker="TRLS", name="Trellis Corporation")
    document = document or Document(company_id=company.id, title="Results", document_type=DocumentType.eight_k, fiscal_year=2024)
    chunk = DocumentChunk(document_id=document.id, company_id=company.id, chunk_index=0,
                          text=text, embedding=[], token_count=len(text.split()))
    return RetrievalDebugResult(query="", company=company, document=document, chunk=chunk,
                                score=score, keyword_score=score, vector_score=score)


def select(tmp_path, question, text, count=2):
    app = create_app(tmp_path, seed=False)
    return app.state.research._points_from_results(question, [result(text)], max_points=count)


def test_adjacent_windows_leave_space_for_other_requested_metric(tmp_path):
    points = select(tmp_path, "What drove revenue and operating cash flows?", (
        "Revenue grew 30 percent as demand increased. Revenue growth was driven by software adoption. "
        "Revenue reached $90 million as subscriptions expanded.\n\n"
        "Operating cash flows grew to $24 million because collections improved."
    ))
    assert any("$90 million" in point.text or "30 percent" in point.text for point in points)
    assert any("$24 million" in point.text for point in points)


def test_generic_key_word_does_not_select_employee_service(tmp_path):
    points = select(tmp_path, "Summarize Services segment performance and key disclosures.", (
        "The continued service of key employees is essential to corporate performance.\n\n"
        "Services segment revenue grew 19 percent because customer demand increased."
    ), count=1)
    assert all("employees" not in point.text for point in points)
    assert any("19 percent" in point.text for point in points)


def test_explicit_full_year_heading_beats_quarter_in_same_release(tmp_path):
    points = select(tmp_path, "What was Digital Media revenue in fiscal 2024?", (
        "Fourth Quarter Fiscal Year 2024 Financial Highlights\n\n"
        "Digital Media revenue was $3.40 billion, representing 12 percent growth.\n\n"
        "Fiscal Year 2024 Financial Highlights\n\n"
        "Digital Media revenue was $12.70 billion, representing 14 percent growth."
    ), count=1)
    assert "$12.70 billion" in points[0].text


def test_explicit_quarterly_filing_and_earnings_call_include_both_source_types():
    scope = question_scope("Compare the quarterly filing with earnings call commentary.")
    assert scope.types == {DocumentType.ten_q, DocumentType.earnings_transcript}
    release = question_scope("Summarize the quarterly earnings release.")
    assert release.types == {DocumentType.eight_k}


def test_quarterly_metric_does_not_exclude_earnings_releases():
    scope = question_scope("What was quarterly revenue in Q1 FY2025?")
    assert not scope.types
    assert scope.matches(Document(company_id=Company(ticker="T", name="Trellis").id,
                                  title="Quarter results", document_type=DocumentType.eight_k,
                                  fiscal_year=2025, fiscal_quarter=1))


def test_mismatched_quarter_section_alone_refuses_full_year_answer(tmp_path):
    app = create_app(tmp_path, seed=False)
    source = result("Fourth Quarter Fiscal Year 2024 Financial Highlights\n\n"
                    "Digital Media revenue was $3.40 billion, representing 12 percent growth.")
    answer, _, _ = app.state.research._answer_from_context(
        "What was full-year Digital Media revenue in fiscal 2024?", [source])
    assert answer.confidence == "low"
    assert not answer.citations
    assert not answer.key_points


def test_explicit_quarter_heading_matches_quarter_separated_from_year(tmp_path):
    text = ("First Quarter Fiscal Year 2024 Financial Highlights\n\n"
            "Digital Media revenue was $3.40 billion, representing 12 percent growth.\n\n"
            "Fiscal Year 2024 Financial Highlights\n\n"
            "Digital Media revenue was $12.70 billion, representing 14 percent growth.")
    for question in ("What was Digital Media revenue in Q1 FY2024?",
                     "What was Q1 Digital Media revenue in fiscal 2024?"):
        points = select(tmp_path, question, text)
        assert points
        assert all("$3.40 billion" in point.text for point in points)
        assert all("$12.70 billion" not in point.text for point in points)


def test_explicit_annual_prose_overrides_inherited_quarter_heading(tmp_path):
    points = select(tmp_path, "What was operating cash flow in fiscal 2024?", (
        "Fourth Quarter Fiscal Year 2024 Financial Highlights\n\n"
        "Operating cash flow was $3.20 billion during fiscal year 2024, up 11 percent."
    ))
    assert len(points) == 1
    assert "$3.20 billion" in points[0].text


def test_source_instructions_do_not_hide_single_topic_evidence(tmp_path):
    points = select(tmp_path, "Compare the quarterly filing supply disclosures with management's earnings call comments.",
                    "Supply remains constrained because manufacturing capacity is limited.")
    assert points
    assert "manufacturing capacity" in points[0].text


def test_overlapping_context_preserves_new_offsetting_cause(tmp_path):
    points = select(tmp_path, "What factors changed net interest income?", (
        "The bank reported net income of $70 billion, down 3 percent. "
        "Net interest income was $120 billion, up 4 percent, driven by higher loan balances. "
        "These factors were offset by lower deposit margins.\n\n"
        "Derivatives are classified as net interest income and compensation expense.\n\n"
        "Markets revenue includes fees and net interest income."
    ))
    assert any("lower deposit margins" in point.text for point in points)
