import pytest

from ai_equity_research_copilot_backend.schemas import DocumentType
from ai_equity_research_copilot_backend.source_query import source_clause_query


@pytest.mark.parametrize("document_type, expected", [
    (DocumentType.ten_q, "Maple's quarterly filing geographic sales"),
    ("8-k", "the earnings release regional demand commentary"),
])
def test_explicit_filing_release_topics(document_type, expected):
    question = "Compare Maple's quarterly filing geographic sales with the earnings release regional demand commentary"
    assert source_clause_query(question, document_type) == expected


@pytest.mark.parametrize("document_type, expected", [
    ("10-q", "Elm's quarterly deposit disclosures"),
    (DocumentType.earnings_transcript, "management commentary on deposit behavior"),
])
def test_deposit_comparison_keeps_clause_specific_qualifiers(document_type, expected):
    question = "Compare Elm's quarterly deposit disclosures with management commentary on deposit behavior"
    assert source_clause_query(question, document_type) == expected


@pytest.mark.parametrize("document_type", ["10-k", "annual_report", DocumentType.ten_k])
def test_reversed_comparison_and_annual_report_family(document_type):
    question = "Earnings call customer demand versus annual filing geographic sales"
    assert source_clause_query(question, document_type) == "annual filing geographic sales"
    assert source_clause_query(question, "earnings_transcript") == "Earnings call customer demand"


@pytest.mark.parametrize("question", [
    "Compare annual filing revenue with earnings call revenue",
    "Compare the annual filing with the earnings call on revenue",
    "Compare Maple's quarterly filing with management commentary on deposits",
    "Compare revenue in the annual filing and earnings call",
    "Explain annual filing revenue with earnings call demand",
    "Compare quarterly revenue with management commentary on demand",
    "Compare annual filing revenue with management commentary in the annual filing on demand",
    "Compare annual filing revenue with earnings call and earnings release demand",
    "Compare annual filing revenue with earnings call demand with regional commentary",
    "Compare earnings call revenue with earnings release demand",
    "Compare annual and quarterly filing revenue with earnings call demand",
])
def test_shared_or_ambiguous_question_is_unchanged(question):
    for document_type in ("10-k", "10-q", "8-k", "earnings_transcript"):
        assert source_clause_query(question, document_type) == question


@pytest.mark.parametrize("document_type", ["manual_note", "investor_presentation", "other", "unknown", "10-k"])
def test_unmatched_source_type_keeps_full_query(document_type):
    question = "Compare quarterly filing sales with earnings call demand"
    assert source_clause_query(question, document_type) == question


def test_returned_clause_keeps_verbatim_period_and_topic_words():
    question = "Compare Elm's 10-Q Q2 FY2031 deposit balances versus management's earnings call remarks on deposit pricing"
    assert source_clause_query(question, "10-q") == "Elm's 10-Q Q2 FY2031 deposit balances"
    assert source_clause_query(question, "earnings_transcript") == "management's earnings call remarks on deposit pricing"
    assert question == "Compare Elm's 10-Q Q2 FY2031 deposit balances versus management's earnings call remarks on deposit pricing"
