from ai_equity_research_copilot_backend.financial_text import format_financial_text


def test_financial_typography_preserves_values_and_signs():
    assert format_financial_text("Cash of $ 284 billion, down 3.5 percent; expenses $ (6.2) million.") == (
        "Cash of $284 billion, down 3.5 percent; expenses $(6.2) million."
    )
    assert format_financial_text("Growth 12 %; base 0.2 percentage points.") == (
        "Growth 12%; base 0.2 percentage points."
    )
