import pytest

from ai_equity_research_copilot_backend.evidence import evidence_passages, usable_prose


@pytest.mark.parametrize("headline", [
    "• Q3 Cloud Revenue (Hosted plus Licensed) $13.8 billion, up 24%",
    "Q2 Subscription Sales € 845 million, down 6.5 percent",
    "Net income £3.24 billion, up 8% year-over-year",
    "Revenue $1.2 billion, up 7%",
    "GAAP earnings per share $2.16, down 4%",
])
def test_explicit_metric_amount_and_change_headlines_are_evidence(headline):
    assert usable_prose(headline)
    assert headline in evidence_passages(headline)


@pytest.mark.parametrize("row", [
    "Subscription revenue and related services $7,420 $6,396 16%",
    "Q3 Cloud Revenue (Hosted plus Licensed) 13800 11129 24%",
    "Customer Alpha €845 million, down 6.5 percent",
    "Net income $3.24 billion $2.91 billion up 8%",
    "Revenue $1.2 billion, up 7% 2025 2024",
])
def test_headline_exception_does_not_admit_ambiguous_numeric_rows(row):
    assert not usable_prose(row)


def test_multiline_highlights_preserve_individual_metric_labels():
    lines = [
        "• Q3 Subscription Revenue $13.8 billion, up 24%",
        "• Q3 Operating Income $3.24 billion, down 8%",
        "• Q3 Net Income $2.16 billion, up 4%",
    ]
    passages = evidence_passages("\n".join(lines))
    assert passages == lines


@pytest.mark.parametrize("fragment", [
    "Cloud services and license support revenues were up 14% in",
    "Subscription revenue is expected to grow between 16% and 20% and",
    "Subscription revenue increased by 12 percent,",
    "The increase in revenue was driven by stronger year-",
])
def test_physical_line_fragments_are_not_standalone_evidence(fragment):
    assert not usable_prose(fragment)
    assert fragment not in evidence_passages(fragment)


def test_wrapped_sentence_keeps_complete_context_without_truncated_prefix():
    first = "Cloud services revenue was up 14% in"
    second = "constant currency to $4.12 billion."
    full = first + " " + second
    passages = evidence_passages(first + "\n" + second)
    assert full in passages
    assert first not in passages


def test_fragment_guard_preserves_complete_source_sentence():
    sentence = "Subscription revenue was $4.12 billion and operating income was $1.05 billion."
    assert sentence in evidence_passages(sentence)
