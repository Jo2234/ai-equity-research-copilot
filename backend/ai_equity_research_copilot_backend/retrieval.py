from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from uuid import UUID

from .embeddings import HashingEmbedder, cosine_similarity
from .evidence import evidence_passages
from .query import content_terms, question_scope
from .schemas import DocumentType, RetrievalDebugResult
from .storage import CorpusSnapshot, JsonRepository


@dataclass(frozen=True)
class PreparedCorpus:
    snapshot: CorpusSnapshot
    terms: dict[UUID, frozenset[str]]
    chunk_frequency: dict[str, int]
    passage_terms: dict[UUID, tuple[frozenset[str], ...]]


class RetrievalService:
    def __init__(self, repo: JsonRepository, embedder: HashingEmbedder, min_score: float = 0.04) -> None:
        self.repo = repo
        self.embedder = embedder
        self.min_score = min_score

    def prepare(self, company_ids: list[UUID]) -> PreparedCorpus:
        snapshot = self.repo.corpus_snapshot(company_ids)
        terms = {chunk.id: frozenset(content_terms(chunk.text)) for chunk in snapshot.chunks}
        return PreparedCorpus(
            snapshot=snapshot,
            terms=terms,
            chunk_frequency=dict(Counter(term for row in terms.values() for term in row)),
            passage_terms={},
        )

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
        query_embedding = self.embedder.embed(query)
        scope_query = question_scope(query)
        query_terms = content_terms(query)
        prepared = corpus if corpus is not None else self.prepare(company_ids)
        docs = prepared.snapshot.documents
        companies = prepared.snapshot.companies
        scope = set(company_ids)
        company_terms = set().union(*(content_terms(c.name) | content_terms(c.ticker)
                                     for c in companies.values()))
        query_terms -= company_terms
        weights = {term: math.log(1 + len(prepared.terms) / (1 + prepared.chunk_frequency.get(term, 0))) for term in query_terms}
        query_weight = max(sum(weights.values()), 1e-9)
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
            if not scope_query.matches(document):
                continue
            chunk_terms = prepared.terms[chunk.id]
            overlap = query_terms & chunk_terms
            # Hash collisions and generic company names cannot establish relevance.
            if not overlap:
                continue
            vector_score = max(0.0, cosine_similarity(query_embedding, chunk.embedding))
            keyword_score = sum(weights[term] for term in overlap) / query_weight
            if chunk.id not in prepared.passage_terms:
                prepared.passage_terms[chunk.id] = tuple(
                    frozenset(content_terms(p)) for p in evidence_passages(chunk.text)
                )
            passage_score = max((sum(weights[term] for term in query_terms & passage) / query_weight
                                 for passage in prepared.passage_terms[chunk.id]), default=0.0)
            score = (0.20 * vector_score) + (0.15 * keyword_score) + (0.65 * passage_score)
            if score >= threshold:
                results.append(
                    RetrievalDebugResult(
                        query=query,
                        chunk=chunk,
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
