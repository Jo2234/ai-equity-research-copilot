from __future__ import annotations

import math
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, field, replace
from threading import RLock
from uuid import UUID

from .embeddings import HashingEmbedder, cosine_similarity
from .evidence import QUANTITY_RE, evidence_passages
from .query import (
    content_terms,
    document_period_policy,
    explicit_periods,
    question_scope,
)
from .query_topics import query_topics
from .retrieval_query import (
    capital_investment_passage,
    company_clause_queries,
    contextual_passages,
    investment_discovery_query,
)
from .schemas import DocumentType, RetrievalDebugResult
from .source_query import source_clause_query
from .storage import CorpusSnapshot, JsonRepository


@dataclass(frozen=True)
class PreparedCorpus:
    snapshot: CorpusSnapshot
    terms: dict[UUID, frozenset[str]]
    chunk_frequency: dict[str, int]
    passage_terms: dict[UUID, tuple[frozenset[str], ...]]
    passage_lock: RLock = field(default_factory=RLock, repr=False, compare=False)
    numerical_terms: dict[UUID, tuple[frozenset[str], ...]] = field(default_factory=dict)
    capital_investment: dict[UUID, bool] = field(default_factory=dict)
    contextual_changes: dict[UUID, tuple[tuple[frozenset[str], frozenset[str]], ...]] = field(default_factory=dict)


class RetrievalService:
    def __init__(self, repo: JsonRepository, embedder: HashingEmbedder, min_score: float = 0.04) -> None:
        self.repo = repo
        self.embedder = embedder
        self.min_score = min_score
        self._cache_lock = RLock()
        self._cache_revision = -1
        self._prepared: OrderedDict[frozenset[UUID], PreparedCorpus] = OrderedDict()

    def prepare(self, company_ids: list[UUID]) -> PreparedCorpus:
        # Bound memory across scope combinations. Chat history does not invalidate
        # document-derived data; ingestion/deletion and external changes do.
        scope = frozenset(company_ids)
        with self._cache_lock:
            revision = self.repo.corpus_revision()
            if revision != self._cache_revision:
                self._prepared.clear()
                self._cache_revision = revision
            if scope in self._prepared:
                self._prepared.move_to_end(scope)
                return self._prepared[scope]
            snapshot = self.repo.corpus_snapshot(company_ids)
            terms = {chunk.id: frozenset(content_terms(chunk.text) | set().union(
                *(content_terms(heading) for _, heading in contextual_passages(chunk))))
                     for chunk in snapshot.chunks}
            prepared = PreparedCorpus(
                snapshot=snapshot,
                terms=terms,
                chunk_frequency=dict(Counter(term for row in terms.values() for term in row)),
                passage_terms={},
            )
            self._prepared[scope] = prepared
            if len(self._prepared) > 8:
                self._prepared.popitem(last=False)
            return prepared

    def search(
        self,
        query: str,
        company_ids: list[UUID],
        top_k: int = 8,
        document_types: list[DocumentType] | None = None,
        fiscal_years: list[int] | None = None,
        min_score: float | None = None,
        corpus: PreparedCorpus | None = None,
    ) -> list[RetrievalDebugResult]:
        scope_query = question_scope(query)
        prepared = corpus if corpus is not None else self.prepare(company_ids)
        docs = prepared.snapshot.documents
        companies = prepared.snapshot.companies
        scope = set(company_ids)
        company_terms = set().union(*(content_terms(c.name) | content_terms(c.ticker)
                                     for c in companies.values()))
        # Company-specific discovery must never change the original source/period policy.
        clauses = company_clause_queries(query, (c for c in companies.values() if c.id in scope))
        features = {}
        for identity, company_clause in clauses.items():
            for document_type in {d.document_type for d in docs.values() if d.company_id == identity}:
                # A separately named company's topic takes priority over a source
                # family hint. Both remain relevance-only; policy uses query below.
                clause = (source_clause_query(query, document_type)
                          if company_clause == query else company_clause)
                topics = query_topics(clause, company_terms)
                weights = {term: math.log(1 + len(prepared.terms) / (1 + prepared.chunk_frequency.get(term, 0)))
                           for term in topics.terms}
                features[(identity, document_type)] = (topics, weights, max(sum(weights.values()), 1e-9),
                                                      self.embedder.embed(clause), investment_discovery_query(clause))
        # Resolve document periods before relevance scoring, so stronger lexical
        # matches cannot silently replace a requested or assumed fiscal period.
        explicit_docs = {document.id: document for document in docs.values()
                         if document.company_id in scope and document.status == "ready"
                         and (not document_types or document.document_type in document_types)
                         and (not fiscal_years or document.fiscal_year in fiscal_years)}
        eligible = {identity: document for identity, document in explicit_docs.items()
                    if scope_query.matches(document)}
        scoped_chunks: dict[UUID, tuple[tuple[int, int], ...]] = {}
        without_quarters = replace(scope_query,
                                   periods=tuple((year, None) for year, _ in scope_query.periods),
                                   quarters=frozenset())
        if scope_query.quarters:
            for chunk in prepared.snapshot.chunks:
                document = explicit_docs.get(chunk.document_id)
                if (document is None or document.id in eligible or document.fiscal_quarter is not None
                        or document.fiscal_year is None or not without_quarters.matches(document)):
                    continue
                verified = tuple(sorted(period for period in explicit_periods(chunk.text, document.fiscal_year)
                                        if period[0] == document.fiscal_year and period[1] in scope_query.quarters
                                        and (not scope_query.periods or period in scope_query.periods
                                             or (period[0], None) in scope_query.periods)))
                if verified:
                    scoped_chunks[chunk.id] = verified
        for chunk in prepared.snapshot.chunks:
            if chunk.id in scoped_chunks:
                eligible[chunk.document_id] = explicit_docs[chunk.document_id]
        policies = document_period_policy(query, list(eligible.values()),
                                          filtered_types=bool(document_types), filtered_years=bool(fiscal_years))
        results: list[RetrievalDebugResult] = []
        threshold = self.min_score if min_score is None else min_score

        for chunk in prepared.snapshot.chunks:
            if chunk.company_id not in scope:
                continue
            document = docs.get(chunk.document_id)
            company = companies.get(chunk.company_id)
            if not document or not company or document.status != "ready":
                continue
            if document_types and document.document_type not in document_types:
                continue
            if fiscal_years and document.fiscal_year not in fiscal_years:
                continue
            if document.id not in policies:
                continue
            if not scope_query.matches(document) and chunk.id not in scoped_chunks:
                continue
            topics, weights, query_weight, query_embedding, investment_query = features[(chunk.company_id, document.document_type)]
            chunk_terms = prepared.terms[chunk.id]
            overlap = topics.matched_terms(chunk_terms)
            # Hash collisions and generic company names cannot establish relevance.
            if not overlap and len(topics.related_terms & chunk_terms) < 2:
                continue
            vector_score = max(0.0, cosine_similarity(query_embedding, chunk.embedding))
            keyword_score = sum(weights[term] for term in overlap) / query_weight
            with prepared.passage_lock:
                if chunk.id not in prepared.passage_terms:
                    passages_to_index = evidence_passages(chunk.text)
                    # Table labels are useful for discovery even when their
                    # column context is not yet sufficient to quote as evidence.
                    # Answer construction must still verify headers and units.
                    passages_to_index.extend(line for line in chunk.text.splitlines()
                                             if len(line) <= 240 and re.search(r"[A-Za-z]{3}", line)
                                             and len(re.findall(r"\d+(?:[.,]\d+)*", line)) >= 2)
                    indexed = [(p, frozenset(content_terms(p))) for p in passages_to_index]
                    contextual_changes = []
                    for body, heading in contextual_passages(chunk):
                        heading_terms = content_terms(heading)
                        for passage in evidence_passages(body):
                            terms = frozenset(content_terms(passage))
                            indexed.append((passage, terms | heading_terms))
                            if (QUANTITY_RE.search(passage) and re.search(
                                    r"\b(?:increased?|decreased?|grew|declined?|rose|driven|due)\b", passage, re.IGNORECASE)):
                                contextual_changes.append((frozenset(heading_terms), terms))
                    prepared.contextual_changes[chunk.id] = tuple(contextual_changes)
                    prepared.passage_terms[chunk.id] = tuple(terms for _, terms in indexed)
                    prepared.numerical_terms[chunk.id] = tuple(
                        terms for p, terms in indexed if QUANTITY_RE.search(p)
                    )
                    prepared.capital_investment[chunk.id] = any(
                        QUANTITY_RE.search(p) and capital_investment_passage(p) for p, _ in indexed
                    )
                passages = prepared.passage_terms[chunk.id]
                numerical = prepared.numerical_terms.get(chunk.id, ())
            passage_score = max((sum(weights[term] for term in topics.matched_terms(passage)) / query_weight
                                 for passage in passages), default=0.0)
            related_score = max((topics.related_score(passage) for passage in passages), default=0.0)
            # Keep the original topic/IDF score dominant. A bounded expansion
            # bonus promotes financial measures over repeated topic definitions.
            score = ((0.20 * vector_score) + (0.15 * keyword_score)
                     + (0.65 * passage_score) + (0.25 * related_score))
            # The requested topic and amount must occur together, regardless
            # of the wording used to ask for them.
            numeric_score = max((sum(weights[term] for term in topics.matched_terms(passage)) / query_weight
                                 for passage in numerical), default=0.0)
            score += 0.2 * numeric_score
            # A quantified change under a matching financial heading is stronger
            # sensitivity evidence than a generic discussion of economic exposure.
            if re.search(r"\bsensitivity\b", query, re.IGNORECASE):
                score += 0.2 * max((topics.related_score(body)
                                   for heading, body in prepared.contextual_changes.get(chunk.id, ())
                                   if heading & (topics.terms | topics.related_terms)
                                   and topics.matched_terms(body)), default=0.0)
            if investment_query and prepared.capital_investment.get(chunk.id, False):
                # Cash-capital measures are relevant discovery context for AI/cloud
                # investment, but the excerpt must retain their broader PPE meaning.
                score += 0.4
            if score >= threshold:
                results.append(
                    RetrievalDebugResult(
                        query=query,
                        chunk=chunk.model_copy(update={"metadata": {
                            **chunk.metadata,
                            **({"retrieval_period_policy": policies[document.id]} if policies[document.id] else {}),
                            **({"retrieval_require_passage_period": [list(period) for period in scoped_chunks[chunk.id]]}
                               if chunk.id in scoped_chunks else {}),
                        }}),
                        document=document,
                        company=company,
                        score=round(score, 6),
                        keyword_score=round(keyword_score, 6),
                        vector_score=round(vector_score, 6),
                    )
                )

        results.sort(key=lambda item: (-item.score, item.company.ticker, item.document.title, item.chunk.chunk_index))
        if scope_query.diversify or len(scope) > 1:
            selected: list[RetrievalDebugResult] = []
            # Cover companies first, then documents, before adding redundant chunks.
            for attribute in ("company_id", "document_id"):
                seen = {getattr(item.chunk, attribute) for item in selected}
                for item in results:
                    value = getattr(item.chunk, attribute)
                    if value not in seen:
                        selected.append(item)
                        seen.add(value)
                        if len(selected) == top_k:
                            return selected
            chosen = {item.chunk.id for item in selected}
            selected.extend(item for item in results if item.chunk.id not in chosen)
            return selected[:top_k]
        return results[:top_k]
