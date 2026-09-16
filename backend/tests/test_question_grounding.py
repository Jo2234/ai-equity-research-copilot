from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from ai_equity_research_copilot_backend.main import create_app
from ai_equity_research_copilot_backend.schemas import (
    CompanyCreate,
    DocumentCreate,
    DocumentType,
)


@pytest.fixture
def workspace(tmp_path):
    app = create_app(tmp_path, seed=False)
    company = app.state.repo.create_company(
        CompanyCreate(ticker="ATLS", name="Atlas Corporation")
    )

    def add(title, text, year=2024, quarter=None, kind=DocumentType.ten_k):
        path = tmp_path / (title + ".txt")
        path.write_text(text)
        document = app.state.repo.create_document(
            company.id,
            DocumentCreate(
                title=title,
                fiscal_year=year,
                fiscal_quarter=quarter,
                document_type=kind,
            ),
            str(path),
        )
        app.state.ingestion.ingest_document(document.id)
        return document

    def ask(question, **kwargs):
        response = TestClient(app).post(
            "/research/chat",
            json={
                "company_ids": [str(company.id)],
                "question": question,
                **kwargs,
            },
        )
        assert response.status_code == 200
        return response.json()

    return app, company, add, ask


def test_explicit_period_and_form_do_not_fall_back_to_other_sources(workspace):
    app, company, add, ask = workspace
    desired = add(
        "Annual 2024",
        "Subscription revenue grew 14 percent as enterprise demand increased.",
    )
    add(
        "Annual 2023",
        "Subscription revenue grew 90 percent as enterprise demand increased.",
        year=2023,
    )
    add(
        "Quarter 2024",
        "Subscription revenue grew 50 percent as enterprise demand increased.",
        quarter=1,
        kind=DocumentType.ten_q,
    )
    answer = ask("What drove subscription revenue in the fiscal 2024 annual filing?")
    assert {c["document_id"] for c in answer["citations"]} == {str(desired.id)}
    assert "14 percent" in answer["answer"]
    missing = ask("What drove subscription revenue in the fiscal 2022 annual filing?")
    assert not missing["citations"]
    assert missing["confidence"] == "low"
    conflict = ask(
        "What drove subscription revenue in fiscal 2024?", fiscal_years=[2023]
    )
    assert not conflict["citations"]


@pytest.mark.parametrize("period", ["Q2 FY2025", "fiscal 2025 Q2", "Q2 2025"])
def test_question_quarter_metadata_is_respected(workspace, period):
    _, _, add, ask = workspace
    add(
        "Quarter one",
        "Subscription revenue rose 2 percent because demand increased.",
        year=2025,
        quarter=1,
        kind=DocumentType.ten_q,
    )
    wanted = add(
        "Quarter two",
        "Subscription revenue rose 7 percent because demand increased.",
        year=2025,
        quarter=2,
        kind=DocumentType.ten_q,
    )
    add(
        "Unknown quarter",
        "Subscription revenue rose 99 percent because demand increased.",
        year=2025,
        kind=DocumentType.ten_q,
    )
    answer = ask(f"What drove subscription revenue in {period}?")
    assert {c["document_id"] for c in answer["citations"]} == {str(wanted.id)}


def test_annual_quarter_comparison_retains_both_documents(workspace):
    _, _, add, ask = workspace
    annual = add(
        "Annual 2024",
        "Subscription revenue grew 14 percent because enterprise demand increased.",
    )
    quarter = add(
        "Quarter 2025",
        "Subscription revenue grew 7 percent because enterprise demand increased.",
        year=2025,
        quarter=2,
        kind=DocumentType.ten_q,
    )
    add(
        "Distractor Q1 2024",
        "Subscription revenue grew 90 percent because enterprise demand increased.",
        year=2024,
        quarter=1,
        kind=DocumentType.ten_q,
    )
    answer = ask(
        "Compare annual subscription revenue growth with fiscal 2025 Q2 subscription revenue growth."
    )
    assert {c["document_id"] for c in answer["citations"]} == {
        str(annual.id),
        str(quarter.id),
    }
    explicit = ask(
        "Compare FY2024 annual subscription revenue with Q2 FY2025 subscription revenue."
    )
    assert {c["document_id"] for c in explicit["citations"]} == {
        str(annual.id),
        str(quarter.id),
    }
    unspecified = ask(
        "Compare annual subscription revenue with Q2 subscription revenue."
    )
    assert {c["document_id"] for c in unspecified["citations"]} == {
        str(annual.id),
        str(quarter.id),
    }


def test_hash_similarity_alone_cannot_supply_evidence(workspace, monkeypatch):
    app, _, add, ask = workspace
    add(
        "Policy",
        "The company maintains office lease agreements and employee vacation policies.",
    )
    # Simulate the strongest possible vector collision on unrelated material.
    monkeypatch.setattr(
        "ai_equity_research_copilot_backend.retrieval.cosine_similarity",
        lambda *args: 1.0,
    )
    answer = ask("What caused satellite launch delays?")
    assert not answer["citations"]
    assert answer["key_points"] == []


@pytest.mark.parametrize(
    "question",
    [
        "What is the exact price target for Atlas?",
        "Which private enterprise customer generated the most subscription revenue?",
        "What did the board decide in non-public minutes about dividends?",
        "Give the internal gross margin target to two decimals.",
        "What exact commodity price is forecast for every month next year?",
        "What revenue was reported after the provided documents end?",
        "Should I buy the stock before earnings?",
    ],
)
def test_missing_requested_facts_are_not_replaced_with_related_text(
    workspace, question
):
    _, _, add, ask = workspace
    add(
        "Results",
        "Subscription revenue and gross margin increased as enterprise customer demand grew. "
        "The board declared a dividend while commodity prices and customer concentration remain risks.",
    )
    answer = ask(question)
    assert not answer["citations"]
    assert not answer["key_points"]
    assert answer["confidence"] == "low"


def test_publicly_supplied_price_target_can_be_quoted(workspace):
    _, _, add, ask = workspace
    source = add(
        "Analyst note",
        "The published analyst price target for Atlas was 42 dollars, based on the stated valuation assumptions.",
        kind=DocumentType.manual_note,
    )
    answer = ask("What price target was published for Atlas?")
    assert "42 dollars" in answer["answer"]
    assert {c["document_id"] for c in answer["citations"]} == {str(source.id)}


def test_ordinary_risk_question_is_not_blocked_by_advice_policy(workspace):
    _, _, add, ask = workspace
    add(
        "Risks",
        "Credit risk increased because customers missed payments during the economic downturn.",
    )
    assert ask("What credit risks were disclosed?")["citations"]


def test_refusal_gate_applies_before_model_generation(workspace, monkeypatch):
    app, _, add, ask = workspace
    add(
        "Results",
        "Revenue increased with enterprise customer demand and subscription renewals.",
    )
    app.state.research.settings = replace(
        app.state.research.settings, llm_provider="ollama"
    )

    def unexpected(*args):
        pytest.fail("Unsupported requests must not be sent to a model")

    monkeypatch.setattr(app.state.research.ollama, "answer", unexpected)
    assert not ask("What is the exact price target?")["citations"]


def test_generic_request_words_do_not_outweigh_the_financial_topic(workspace):
    _, _, add, ask = workspace
    source = add(
        "Results",
        "Subscription sales increased 18 percent as renewal demand improved. "
        "Liquidity and capital resources remained sufficient to meet operating requirements.",
    )
    add(
        "Accounting policies",
        "Our disclosure policies describe the position of each document and its commentary trend.",
    )
    for question in (
        "Summarize subscription growth disclosure and its trend.",
        "Summarize the liquidity position.",
    ):
        answer = ask(question)
        assert {c["document_id"] for c in answer["citations"]} == {str(source.id)}


def test_compact_table_keeps_column_headers_units_and_negative_values(workspace):
    _, _, add, ask = workspace
    source = add("Annual table", "Results (USD millions)\n2024 2023\nOperating income (18) 25\nNet income (22) 19")
    answer = ask("What operating income was reported in the annual filing?")
    assert answer["citations"]
    for value in ("2024", "2023", "USD millions", "(18)", "25"):
        assert value in answer["answer"]
    assert {c["document_id"] for c in answer["citations"]} == {str(source.id)}


def test_headerless_numeric_row_is_not_interpreted(workspace):
    _, _, add, ask = workspace
    add("Ambiguous table", "Subscription revenue 481 329 207")
    assert not ask("What subscription revenue was reported?")["citations"]


def test_identical_passages_in_distinct_periods_keep_both_citations(workspace):
    _, _, add, ask = workspace
    sentence = "Operating margin increased 14 percent because freight costs declined."
    annual = add("Annual", sentence)
    quarter = add("Quarter", sentence, year=2025, quarter=2, kind=DocumentType.ten_q)
    answer = ask("Compare annual operating margin with Q2 FY2025 operating margin.")
    assert {c["document_id"] for c in answer["citations"]} == {str(annual.id), str(quarter.id)}
    assert "FY2024" in answer["answer"] and "FY2025 Q2" in answer["answer"]


def test_direct_margin_evidence_outranks_scattered_topic_terms(workspace):
    _, _, add, ask = workspace
    # The query words occur in unrelated sentences, so chunk overlap is not enough.
    distractor = "Automotive production capacity increased during the year. " + "We maintain facilities in many countries and manage employee benefits. " * 4 + "Interest rates include a variable margin over the benchmark."
    add("Distractor", distractor)
    annual = add("Annual margins", "Gross margin for automotive decreased from 20.8% to 18.6% in 2024 due to a decrease in credit revenue.")
    quarter = add("Quarter margins", "Gross margin for automotive increased from 17.5% to 20.2% in the first quarter of 2025 due to lower costs.", year=2025, quarter=1, kind=DocumentType.ten_q)
    answer = ask("Compare annual automotive margin with fiscal 2025 Q1 margin.", top_k=2)
    assert {c["document_id"] for c in answer["citations"]} == {str(annual.id), str(quarter.id)}
    assert "18.6%" in answer["answer"] and "20.2%" in answer["answer"]
