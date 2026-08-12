"""Presentation-only financial typography; original evidence stays in citations."""
import re


def format_financial_text(text: str) -> str:
    text = re.sub(r"([$€£])\s+(?=[(\-\d])", r"\1", text)
    text = re.sub(r"(?<=\d)\s+%", "%", text)
    return text
