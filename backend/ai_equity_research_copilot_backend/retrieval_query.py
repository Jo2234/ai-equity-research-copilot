"""Conservative, query-only discovery hints; none establish answer facts."""
from __future__ import annotations

import re
from collections.abc import Iterable
from uuid import UUID

from .chunking import detect_heading
from .query import content_terms
from .schemas import Company, DocumentChunk


def company_clause_queries(query: str, companies: Iterable[Company]) -> dict[UUID, str]:
    """Split only explicit comparisons with a separate named topic on each side.

    The caller must retain the original question for source and period policy.
    Shared-topic and ambiguous comparisons deliberately retain the full query.
    """
    companies = list(companies)
    fallback = {company.id: query for company in companies}
    if len(companies) != 2 or not re.search(r"\b(?:compare|versus|vs\.?)\b", query, re.IGNORECASE):
        return fallback
    parts = re.split(r"\b(?:with|versus|vs\.?)\s+", query, flags=re.IGNORECASE)
    if len(parts) != 2:
        return fallback
    identifiers: dict[UUID, re.Pattern[str]] = {}
    for company in companies:
        names = {company.name, company.ticker}
        # A unique first name token supports ordinary shortened issuer names.
        # Never use a shared prefix (e.g. two companies beginning 'United').
        words = company.name.split()
        for length in range(1, len(words)):
            prefix = " ".join(words[:length])
            if (len(prefix) >= 4 and sum(c.name.casefold().startswith(prefix.casefold())
                                         for c in companies) == 1):
                names.add(prefix)
        identifiers[company.id] = re.compile(
            r"(?<!\w)(?:" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
            + r")(?!\w)(?:['’]s)?", re.IGNORECASE)
    owners = [[identity for identity, pattern in identifiers.items() if pattern.search(part)]
              for part in parts]
    if any(len(owner) != 1 for owner in owners) or owners[0] == owners[1]:
        return fallback
    clauses = {}
    for part, owner in zip(parts, owners):
        topic = identifiers[owner[0]].sub(" ", part)
        topic = re.sub(r"\bcompare\b", " ", topic, flags=re.IGNORECASE)
        if not content_terms(topic) - {"sensitivity", "condition", "versus", "vs"}:
            return fallback
        clauses[owner[0]] = topic.strip()
    topics = [content_terms(clause) - {"sensitivity", "condition", "versus", "vs"}
              for clause in clauses.values()]
    if topics[0] == topics[1]:
        return fallback
    return clauses


def contextual_passages(chunk: DocumentChunk) -> list[tuple[str, str]]:
    """Return exact span text plus its verified heading, without cross-section joins."""
    spans = chunk.metadata.get("section_spans")
    if isinstance(spans, list):
        passages = []
        for span in spans:
            if not isinstance(span, dict):
                continue
            start, end, heading = span.get("start"), span.get("end"), span.get("section")
            if (type(start) is int and type(end) is int and 0 <= start < end <= len(chunk.text)
                    and isinstance(heading, str) and heading.strip()):
                body = chunk.text[start:end]
                if body.strip() != heading.strip():
                    passages.append((body, heading))
        return passages
    # Older corpora can still use headings physically present in this chunk.
    # Never apply the chunk's final section_title to preceding source sections.
    passages = []
    heading = None
    for block in re.split(r"\n\s*\n|\n", chunk.text):
        detected = detect_heading(block)
        if detected:
            heading = detected
        elif heading and block.strip():
            passages.append((block, heading))
    return passages


def investment_discovery_query(query: str) -> bool:
    return bool(re.search(r"\b(?:AI|cloud|computing)\b", query, re.IGNORECASE)
                and re.search(r"\b(?:infrastructure|investment|capex|capital\s+expenditures?)\b", query, re.IGNORECASE))


def capital_investment_passage(text: str) -> bool:
    """Recognize cash-capital vocabulary, not a claim that all PPE funds AI."""
    return bool(re.search(r"\bcapital\s+expenditures?\b|\badditions?\s+to\s+(?:property|plant|equipment)\b", text, re.IGNORECASE))
