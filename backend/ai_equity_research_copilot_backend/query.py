"""Question-derived constraints only; never consume evaluation labels or answer keys."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schemas import Document, DocumentType

STOPWORDS = set(
    """a an and are as at be been by can did do does for from had has have how
in into is it its of on or that the their these this those to was were what which who
why with would you your according cite cites cited describe described disclose disclosed
disclosure disclosures discussion commentary position trend explain identify main primary report reported say says summarize
compare comparison drove driven driver drivers factor factors change changes recent
annual quarterly fiscal year filing filings document documents fy q1 q2 q3 q4
about related key said comments comment update provide information based only
call transcript transcripts official""".split()
)


def content_terms(text: str) -> set[str]:
    terms = set(re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.lower())) - STOPWORDS
    terms = {
        "growth"
        if term
        in {
            "grow",
            "grew",
            "growing",
            "increase",
            "increased",
            "increases",
            "increasing",
            "rose",
        }
        else term
        for term in terms
    }
    return {
        term[:-1] if term.endswith("s") and not term.endswith(("ss", "is")) else term
        for term in terms
        if len(term) > 1 and not re.fullmatch(r"\d+(?:\.\d+)?", term)
    }


@dataclass(frozen=True)
class QuestionScope:
    types: frozenset[DocumentType]
    periods: tuple[tuple[int, int | None], ...]
    quarters: frozenset[int]
    diversify: bool

    def matches(self, document: Document) -> bool:
        if self.types and document.document_type not in self.types:
            return False
        annual = document.document_type in {
            DocumentType.ten_k,
            DocumentType.annual_report,
        }
        periods = self.periods
        annual_requested = DocumentType.ten_k in self.types
        if annual_requested and self.quarters:
            # Keep annual years separate from explicitly paired quarterly years.
            if annual:
                periods = tuple((year, q) for year, q in periods if q is None)
                if not periods:
                    return True
            elif any(q is not None for _, q in periods):
                periods = tuple((year, q) for year, q in periods if q is not None)
        if periods and not any(
            document.fiscal_year == year
            and (quarter is None or document.fiscal_quarter == quarter)
            for year, quarter in periods
        ):
            return False
        if (
            self.quarters
            and not annual
            and not any(q is not None for _, q in periods)
            and document.fiscal_quarter not in self.quarters
        ):
            return False
        return True


def question_scope(query: str) -> QuestionScope:
    text = query.lower()
    types = set()
    annual = bool(re.search(r"\b(?:annual|10[- ]?k)\b", text))
    quarterly = bool(re.search(r"\b(?:quarterly|10[- ]?q|q[1-4])\b", text))
    if annual:
        types.update((DocumentType.ten_k, DocumentType.annual_report))
    if quarterly and (annual or re.search(r"quarterly\s+(?:filing|report)|10[- ]?q", text)):
        types.add(DocumentType.ten_q)
    if re.search(r"earnings\s+call|transcript", text):
        types.add(DocumentType.earnings_transcript)
    if re.search(r"earnings\s+release", text):
        types.add(DocumentType.eight_k)
    periods = []
    spans = []
    for pattern, year_group, quarter_group in (
        (r"\b(?:fy|fiscal(?:\s+year)?)\s*(20\d{2})\s*[,/-]?\s*q([1-4])\b", 1, 2),
        (r"\bq([1-4])\s*(?:(?:fy|fiscal(?:\s+year)?)\s*)?(20\d{2})\b", 2, 1),
    ):
        for match in re.finditer(pattern, text):
            periods.append((int(match[year_group]), int(match[quarter_group])))
            spans.append(match.span())
    for match in re.finditer(r"\b20\d{2}\b", text):
        if not any(start <= match.start() < end for start, end in spans):
            periods.append((int(match[0]), None))
    quarters = frozenset(int(value) for value in re.findall(r"\bq([1-4])\b", text))
    return QuestionScope(
        frozenset(types),
        tuple(dict.fromkeys(periods)),
        quarters,
        bool(re.search(r"\bcompare|\bacross|\bevolve|\bversus", text)),
    )


def refusal_reason(question: str) -> str | None:
    text = question.lower()
    if re.search(
        r"\bshould\s+(?:i|we)\s+(?:buy|sell|hold)\b|\b(?:recommend|tell me to)\s+(?:buy|sell|hold)",
        text,
    ):
        return "I cannot provide a buy, sell, or hold recommendation from filings alone. Ask about the disclosed business results or supply explicit valuation assumptions for analysis."
    if re.search(
        r"\b(?:after|beyond)\b.{0,60}\b(?:documents?|filings?|corpus)\b.{0,20}\b(?:end|coverage|cutoff)\b",
        text,
    ):
        return "The requested period is beyond the supplied documents. I do not have enough cited context; upload the filing for that period."
    return None


def required_evidence_patterns(question: str) -> list[str]:
    """Conservative anchors for requests easily confused with nearby public facts.

    These checks are necessary evidence, not semantic proof of an answer.
    """
    patterns = []
    for request, evidence in (
        (
            r"\bprice\s+target|\btarget\s+price",
            r"\b(?:price\s+target|target\s+price)\b",
        ),
        (r"\bforecast\b", r"\b(?:forecast\w*|project\w*|expect\w*|guidance)\b"),
        (r"\bprivate\b.{0,40}\bcustomer", r"\bprivate\b.{0,60}\bcustomer"),
        (r"\b(?:non-public|confidential)\b", r"\b(?:non-public|confidential)\b"),
        (r"\bminutes\b", r"\bminutes\b"),
        (r"\binternal\b.{0,50}\btarget", r"\binternal\b.{0,70}\btarget"),
        (r"\bevery\s+month\b", r"\b(?:monthly|each\s+month|every\s+month)\b"),
    ):
        if re.search(request, question, re.I):
            patterns.append(evidence)
    return patterns
