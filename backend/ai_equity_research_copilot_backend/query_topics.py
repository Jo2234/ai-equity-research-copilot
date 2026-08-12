"""Small, auditable financial topic vocabulary, independent of issuers and facts.

Related words broaden discovery; they never establish a factual answer or its
period. Exact source excerpts and citation checks remain the evidence boundary.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .query import content_terms


# Trigger, related financial vocabulary. Deliberately omit brands, products,
# geographic names, dates and numerical values.
TOPICS = (
    (r"\bliquidity\b|\bcash\s+flows?\b|\bcapital\s+(?:allocation|resources)\b",
     "cash operating operations financing dividends repurchases buybacks debt borrowing capital expenditures property plant equipment"),
    (r"\bcredit\b|\bprovisions?\b|\bloan\s+loss",
     "provision allowance reserve charge-offs losses loans borrower delinquency default consumer wholesale"),
    (r"\bsupply\b|\bsuppliers?\b|\binventory\b|\bpurchase\s+obligations?\b",
     "supply supplier foundries subcontractors manufacturing fabrication assembly testing packaging capacity inventory purchase obligations shipment yield commitment demand forecast estimate deposit shortage"),
    (r"\bmargins?\b",
     "cost costs pricing prices mix materials overhead tariffs volume absorption warranty currency"),
    (r"\bsegments?\b.{0,35}\b(?:performance|results|sales|revenue)\b|\bgeographic\b",
     "revenue sales operating income growth margin segment regional geographic"),
    (r"\bregulatory\s+capital\b|\bcapital\s+(?:ratio|requirement|adequacy)\b",
     "CET1 tier equity standardized ratio requirement buffer leverage capital"),
    (r"\brevenue\b|\bsales\b",
     "revenue sales growth grew demand volume prices pricing"),
    (r"\b(?:cloud|AI)\b.{0,35}\b(?:infrastructure|investment|capex)\b|\bcapital\s+expenditures?\b",
     "datacenter data center computing compute infrastructure investment capital expenditures additions property plant equipment capacity"),
    (r"\bcustomer\s+concentration\b",
     "customer direct indirect revenue represented percentage accounted"),
    (r"\bdeposits?\b",
     "deposit balance average accounts inflow outflow yield growth"),
    (r"\binterest[- ]rates?\b",
     "interest income deposit balance rates securities lending loan margin"),
    (r"\bcommodit(?:y|ies)\b|\bupstream\b",
     "commodity realization price production volume earnings costs depreciation divestment"),
)


@dataclass(frozen=True)
class QueryTopics:
    terms: frozenset[str]
    related_terms: frozenset[str]
    groups: tuple[frozenset[str], ...]
    aliases: tuple[tuple[str, tuple[frozenset[str], ...]], ...] = ()

    def matched_terms(self, passage_terms: Iterable[str]) -> set[str]:
        """Credit direct financial concept aliases, separately from related words."""
        terms = set(passage_terms)
        matched = set(self.terms & terms)
        for original, alternatives in self.aliases:
            if any(alternative <= terms for alternative in alternatives):
                matched.add(original)
        return matched

    def related_score(self, passage_terms: Iterable[str]) -> float:
        """Bound expansion support; isolated generic related words score weakly."""
        terms = set(passage_terms)
        return max((min(1.0, len(terms & group) / 4) for group in self.groups), default=0.0)


def query_topics(query: str, excluded_terms: Iterable[str] = ()) -> QueryTopics:
    # Preserve source/period parsing in question_scope(query). These framing
    # instructions are not financial topics and must not displace the metric.
    text = re.sub(r"\b(?:your\s+)?(?:latest\s+)?market\s+knowledge\b", " ", query, flags=re.I)
    text = re.sub(r"\bignore\s+(?:the\s+)?(?:documents|filings)\s+and\s+(?:say|claim)\b", " ", text, flags=re.I)
    text = re.sub(r"\bcapital\s+resources\b", "liquidity", text, flags=re.I)
    text = re.sub(r"\bsupply\s+chain\b", "supply", text, flags=re.I)
    text = re.sub(r"\bmanagement(?:['’]s)?\s+(?:commentary|comments|remarks)\b", " ", text, flags=re.I)
    excluded = set(excluded_terms)
    terms = content_terms(text) - excluded - {"use", "latest", "knowledge", "update", "ignore", "prove", "no", "sensitivity", "condition", "behavior", "evolve", "metric"}
    terms = (terms - {"risk", "result", "performance"}) or terms
    groups = tuple(frozenset(content_terms(vocabulary) - terms - excluded)
                   for trigger, vocabulary in TOPICS if re.search(trigger, text, re.I))
    groups = tuple(group for group in groups if group)
    aliases = []
    for original, trigger, alternatives in (
        ("liquidity", r"\bliquidity\b", ("cash flow", "cash equivalents", "cash operating activities", "cash operations", "capital resources")),
        ("regulatory", r"\bregulatory\s+capital\b", ("CET1 capital", "tier capital", "common equity ratio")),
        ("supply", r"\bsupply\b", ("suppliers", "foundries", "subcontractors")),
        ("obligation", r"\bpurchase\s+obligations?\b", ("purchase commitments", "capacity commitments")),
        ("infrastructure", r"\b(?:AI|cloud|computing)\b.{0,35}\binfrastructure\b|\binfrastructure\b.{0,35}\b(?:AI|cloud|computing)\b",
         ("data center", "datacenter", "computing platform", "compute platform")),
        ("commodity", r"\bcommodit(?:y|ies)\b",
         ("crude prices", "oil prices", "natural gas prices", "price realizations")),
        ("repurchase", r"\brepurchases?\b", ("buybacks",)),
        ("buyback", r"\bbuybacks?\b", ("repurchases",)),
    ):
        if original in terms and re.search(trigger, text, re.I):
            aliases.append((original, tuple(frozenset(content_terms(phrase)) for phrase in alternatives)))
    return QueryTopics(frozenset(terms), frozenset().union(*groups), groups, tuple(aliases))
