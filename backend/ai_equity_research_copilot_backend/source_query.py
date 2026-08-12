"""Query-only relevance hints for explicitly compared source families.

This module does not select documents or establish financial facts. Callers must
retain the original question for source, fiscal-period, company and API filters.
"""
from __future__ import annotations

import re

from .query import content_terms
from .schemas import DocumentType


_SPLIT = re.compile(r"\b(?:with|versus|vs\.?)\s+", re.I)
_CALL = re.compile(r"\b(?:earnings\s+call|(?:earnings\s+)?transcripts?)\b", re.I)
_MANAGEMENT = re.compile(r"\bmanagement(?:['’]s)?\s+(?:commentary|comments|remarks)\b", re.I)
_RELEASE = re.compile(r"\b(?:earnings\s+release|8[- ]?k)\b", re.I)
_FILING = re.compile(r"\b(?:filings?|10[- ]?[qk]|annual\s+report|quarterly\s+report)\b", re.I)
_PERIOD_DISCLOSURE = re.compile(r"\b(?:annual|quarterly)\b[^.!?;]{0,100}\b(?:disclosures?|results)\b", re.I)
_FILING_TYPES = frozenset({DocumentType.ten_k, DocumentType.ten_q, DocumentType.annual_report})


def _family(clause: str) -> frozenset[DocumentType]:
    release = bool(_RELEASE.search(clause))
    call = bool(_CALL.search(clause))
    filing = bool(_FILING.search(clause) or _PERIOD_DISCLOSURE.search(clause))
    # Management can speak inside an explicitly named release or filing.
    # Only unqualified management commentary suggests the transcript family.
    call |= bool(_MANAGEMENT.search(clause)) and not (release or filing)
    if sum((release, call, filing)) != 1:
        return frozenset()
    if release:
        return frozenset({DocumentType.eight_k})
    if call:
        return frozenset({DocumentType.earnings_transcript})
    annual = bool(re.search(r"\b(?:annual|10[- ]?k)\b", clause, re.I))
    quarterly = bool(re.search(r"\b(?:quarterly|10[- ]?q|q[1-4])\b", clause, re.I))
    if annual and quarterly:
        return frozenset()
    if annual:
        return frozenset({DocumentType.ten_k, DocumentType.annual_report})
    if quarterly:
        return frozenset({DocumentType.ten_q})
    return _FILING_TYPES


def _topic(clause: str) -> set[str]:
    text = re.sub(r"^\s*compare\s+", "", clause, flags=re.I)
    # Ignore a possessive owner only for ambiguity detection, never in output.
    text = re.sub(r"^[A-Z][\w&.-]*(?:\s+[A-Z][\w&.-]*)*['’]s\s+", "", text)
    for marker in (_CALL, _MANAGEMENT, _RELEASE, _FILING):
        text = marker.sub(" ", text)
    text = re.sub(r"\b(?:fy\s*)?20\d{2}\b", " ", text, flags=re.I)
    return content_terms(text) - {"financial", "result", "versus", "vs"}


def source_clause_query(query: str, document_type: DocumentType | str) -> str:
    """Return the appropriate source clause for relevance ranking, or ``query``.

    Only explicit two-sided comparisons separated by ``with``/``versus``/``vs``
    qualify. One side must name a filing/disclosure and the other a distinct
    earnings call, transcript, management commentary or earnings release. Each
    side needs its own topic; shared or ambiguous topics retain the full query.

    Returned text is an original clause with only surrounding whitespace and a
    leading "Compare" removed. Unknown/unmatched document types also return the
    original query. This hint must never replace the original question in period
    or source policy, or broaden explicit API document filters.
    """
    if not re.search(r"\b(?:compare|versus|vs\.?)\b", query, re.I):
        return query
    clauses = _SPLIT.split(query)
    if len(clauses) != 2:
        return query
    families = [_family(clause) for clause in clauses]
    if not all(families) or families[0] & families[1]:
        return query
    if sum(bool(family & _FILING_TYPES) for family in families) != 1:
        return query
    topics = [_topic(clause) for clause in clauses]
    if not all(topics) or topics[0] == topics[1]:
        return query
    for clause, family in zip(clauses, families):
        if document_type in family:
            return re.sub(r"^\s*compare\s+", "", clause, flags=re.I).strip()
    return query
