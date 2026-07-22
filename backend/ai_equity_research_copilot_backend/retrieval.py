from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from .embeddings import HashingEmbedder, cosine_similarity, tokenize
from .schemas import DocumentType, RetrievalDebugResult
from .storage import CorpusSnapshot, JsonRepository


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "which",
    "with",
}


@dataclass(frozen=True)
class PreparedCorpus:
    snapshot: CorpusSnapshot
    terms: dict[UUID, frozenset[str]]


class RetrievalService:
    def __init__(self, repo: JsonRepository, embedder: HashingEmbedder, min_score: float = 0.04) -> None:
        self.repo = repo
        self.embedder = embedder
        self.min_score = min_score

    def prepare(self, company_ids: list[UUID]) -> PreparedCorpus:
        snapshot = self.repo.corpus_snapshot(company_ids)
        return PreparedCorpus(
            snapshot=snapshot,
            terms={chunk.id: frozenset(tokenize(chunk.text)) for chunk in snapshot.chunks},
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
        query_terms = set(token for token in tokenize(query) if token not in STOPWORDS)
        prepared = corpus if corpus is not None else self.prepare(company_ids)
        docs = prepared.snapshot.documents
        companies = prepared.snapshot.companies
        scope = set(company_ids)
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
            vector_score = max(0.0, cosine_similarity(query_embedding, chunk.embedding))
            chunk_terms = prepared.terms[chunk.id]
            keyword_score = len(query_terms & chunk_terms) / max(len(query_terms), 1)
            score = (0.75 * vector_score) + (0.25 * keyword_score)
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

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]
