from ai_equity_research_copilot_backend.sec import html_to_text


def test_financial_disclosure_with_three_inline_bullets_is_preserved():
    disclosure = (
        "During the year: • operating cash flow was $12 billion; "
        "▪ share repurchases were $3 billion; • dividends paid were $1 billion."
    )
    raw = f"<p>{disclosure}</p>"

    assert html_to_text(raw) == disclosure
