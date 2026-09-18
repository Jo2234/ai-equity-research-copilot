"""Question-derived constraints only; never consume evaluation labels or answer keys."""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID

from .schemas import Document, DocumentType

QUARTER_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4}


def _normalize_quarters(text: str) -> str:
    return re.sub(r"\b(first|second|third|fourth)\s+quarter\b",
                  lambda match: f"Q{QUARTER_WORDS[match[1].lower()]}", text, flags=re.I)


STOPWORDS = set(
    """a an and are as at be been by can did do does for from had has have how
in into is it its of on or that the their these this those to was were what which who
why with would you your according cite cites cited describe described disclose disclosed
disclosure disclosures discussion commentary position trend explain identify main primary report reported say says summarize
compare comparison drive drives drove driven driver drivers factor factors change changes recent
annual quarterly fiscal year filing filings document documents fy q1 q2 q3 q4
about related key said comments comment update provide information based only
call transcript transcripts official""".split()
)


def content_terms(text: str) -> set[str]:
    text = _normalize_quarters(text)
    # Source-type instructions do not describe the requested financial topic.
    text = re.sub(r"\b(?:management(?:'s)?\s+)?earnings\s+(?:call|release)\b", " ", text, flags=re.I)
    terms = set(re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.lower())) - STOPWORDS
    terms = {re.sub(r"^(repurchas|purchas)(?:ed|ing)$", r"\1e", term) for term in terms}
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
    } - STOPWORDS


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
    text = _normalize_quarters(query).lower()
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
    # A named call/release in one side of a comparison is supplementary to
    # the financial disclosure requested on the other side.
    clauses = re.split(r"\bwith\b|\bversus\b", text, maxsplit=1)
    supplement = r"earnings\s+(?:call|release)|transcript"
    if (len(clauses) == 2 and re.search(r"\bcompare|\bacross|\bversus", text)
            and not re.search(supplement, clauses[0]) and re.search(supplement, clauses[1])
            and re.search(r"\b(?:quarterly|q[1-4])\b", clauses[0])):
        types.add(DocumentType.ten_q)
    # A comparison can ask for management's separate commentary without
    # naming its recording format. Permit earnings-call transcripts alongside
    # the filing, but do not override a source explicitly named in that clause.
    management = r"\bmanagement(?:['’]s)?\s+(?:commentary|comments|remarks)\b"
    if (len(clauses) == 2 and re.search(r"\bcompare|\bacross|\bversus", text)
            and re.search(management, clauses[1]) and not re.search(management, clauses[0])
            and re.search(r"\b(?:quarterly|annual|filing|q[1-4]|10[- ]?[qk])\b", clauses[0])
            and not re.search(r"\b(?:filing|report|release|10[- ]?[qk])\b", clauses[1])):
        types.add(DocumentType.earnings_transcript)
        if re.search(r"\b(?:quarterly|q[1-4]|10[- ]?q)\b", clauses[0]):
            types.add(DocumentType.ten_q)
    periods = []
    spans = []
    for pattern, year_group, quarter_group in (
        (r"\b(?:fy|fiscal(?:\s+year)?)\s*(20\d{2})\s*[,/-]?\s*q([1-4])\b", 1, 2),
        (r"\bq([1-4])\s*(?:of\s+)?(?:(?:fy|fiscal(?:\s+year)?)\s*)?(20\d{2})\b", 2, 1),
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
        return "I cannot determine whether you should buy, sell, or hold from filings alone. Ask about the disclosed business results or supply explicit valuation assumptions for analysis."
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


ANNUAL_TYPES = frozenset({DocumentType.ten_k, DocumentType.annual_report})


def explicit_periods(text: str, document_year: int | None = None) -> set[tuple[int, int | None]]:
    """Read stated fiscal periods; a short FY year must agree with metadata."""
    normalized = _normalize_quarters(text)
    normalized = re.sub(r"\bFY\s*(\d{2})(?!\d)",
                        lambda m: f"FY{document_year}" if document_year is not None
                        and document_year % 100 == int(m[1]) else m[0], normalized, flags=re.I)
    periods = set()
    paired_spans = []
    for pattern, quarter_group, year_group in (
        (r"\bQ([1-4])\s*(?:of\s+)?(?:(?:FY|fiscal(?:\s+year)?)\s*)?(20\d{2})\b", 1, 2),
        (r"\b(?:FY|fiscal(?:\s+year)?)\s*(20\d{2})\s*[,/-]?\s*Q([1-4])\b", 2, 1),
    ):
        for match in re.finditer(pattern, normalized, re.I):
            periods.add((int(match[year_group]), int(match[quarter_group])))
            paired_spans.append(match.span())
    for match in re.finditer(r"\b(?:FY|fiscal(?:\s+year)?|full[- ]year)\s*(20\d{2})\b", normalized, re.I):
        if not any(start <= match.start() < end for start, end in paired_spans):
            periods.add((int(match[1]), None))
    return periods


def passage_period(text: str, inherited: tuple[int, int | None] | None = None,
                   document_year: int | None = None) -> tuple[int, int | None] | None:
    """Prefer an explicit passage period to its surrounding section label."""
    # An explicit short year cannot become a bare quarter if it fails to
    # resolve against document metadata. Do not guess its century or inherit.
    short_years = re.findall(r"\bFY\s*(\d{2})(?!\d)", text, re.I)
    if any(document_year is None or document_year % 100 != int(year) for year in short_years):
        return None
    periods = explicit_periods(text, document_year)
    if len(periods) > 1:
        return None  # A multi-period passage has no single exclusive period.
    if periods and next(iter(periods))[1] is not None:
        return next(iter(periods))
    if len(periods) == 1 and not re.search(r"\b(?:quarter|Q[1-4]|three months)\b", text, re.I):
        return next(iter(periods))
    # A standalone Q4 may inherit its year only from verified surrounding text.
    local_quarters = {int(q) for q in re.findall(r"\bQ([1-4])\b", _normalize_quarters(text), re.I)}
    if periods and local_quarters:
        return None  # An unpaired annual year and quarter do not establish scope.
    if inherited and len(local_quarters) == 1:
        return inherited[0], next(iter(local_quarters))
    return inherited


def document_period_policy(query: str, documents: list[Document], *,
                           filtered_types: bool = False, filtered_years: bool = False) -> dict[UUID, str]:
    """Choose reproducible source periods, with assumptions returned for display.

    Inputs already satisfy explicit user filters. No answer relevance or expected
    facts enter the policy. A missing date never becomes an invented fiscal year.
    """
    scope = question_scope(query)
    by_company: dict[UUID, list[Document]] = {}
    for document in documents:
        by_company.setdefault(document.company_id, []).append(document)
    selected: dict[UUID, str] = {}
    quarterly = bool(scope.quarters or re.search(r"\bquarterly\b", query, re.I))
    latest_requested = bool(re.search(r"\b(?:latest|most recent|current)\b", query, re.I))

    def family(document: Document) -> str:
        return "annual" if document.document_type in ANNUAL_TYPES else document.document_type.value

    def latest(rows: list[Document]) -> tuple[list[Document], bool]:
        if all(document.period_end_date is not None for document in rows):
            newest = max(document.period_end_date for document in rows)
            return [document for document in rows if document.period_end_date == newest], False
        # Fiscal ordering is the fallback when end dates are incomplete. A full
        # annual period ends in Q4; an unlabelled release is not an annual period.
        years = [document.fiscal_year for document in rows if document.fiscal_year is not None]
        if not years:
            return rows, True
        newest_year = max(years)
        current = [document for document in rows if document.fiscal_year == newest_year]
        unknown = [document for document in rows if document.fiscal_year is None]
        quarters = {document.id: 4 if document.document_type in ANNUAL_TYPES else document.fiscal_quarter
                    for document in current}
        known_quarters = [quarter for quarter in quarters.values() if quarter is not None]
        newest_quarter = max(known_quarters, default=None)
        selected = [document for document in current if quarters[document.id] in {None, newest_quarter}]
        ambiguous = bool(unknown or any(quarter is None for quarter in quarters.values()))
        return selected + unknown, ambiguous

    for company_documents in by_company.values():
        candidates = company_documents
        assumption = ""
        if not scope.types and not filtered_types:
            preferred = []
            if quarterly:
                preferred = [d for d in candidates if d.document_type == DocumentType.ten_q]
                if preferred:
                    assumption = "Quarterly disclosure questions use the quarterly filing when one is available."
            elif not latest_requested:
                preferred = [d for d in candidates if d.document_type in ANNUAL_TYPES]
                if preferred:
                    assumption = ("No source type was specified; using annual filings for the requested fiscal period."
                                  if scope.periods or filtered_years else
                                  "No source type was specified; using the latest available annual filing as the research baseline.")
            if preferred:
                candidates = preferred
            if latest_requested and not scope.periods and not filtered_years:
                candidates, ambiguous = latest(candidates)
                assumption = ("Some source periods cannot be ordered from the available metadata; retaining those sources alongside the latest identifiable fiscal period."
                              if ambiguous else "Using the latest available fiscal period requested, across available source types.")
        # Explicit periods are not replaced by a more recent period. An annual
        # component left undated in an annual/quarter comparison is independent.
        groups: dict[str, list[Document]] = {}
        for document in candidates:
            groups.setdefault(family(document), []).append(document)
        for group, rows in groups.items():
            explicitly_dated = bool(scope.periods or filtered_years)
            if group == "annual" and scope.quarters and scope.types & ANNUAL_TYPES:
                explicitly_dated = filtered_years or any(q is None for _, q in scope.periods)
            if not explicitly_dated:
                rows, ambiguous = latest(rows)
                if ambiguous:
                    assumption = "Some source periods cannot be ordered from the available metadata; retaining those sources alongside the latest identifiable fiscal period."
                elif not assumption:
                    assumption = "No period was specified for this source type; using its latest available fiscal period."
            for document in rows:
                selected[document.id] = assumption
    return selected
