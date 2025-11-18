from __future__ import annotations

from uuid import UUID

from .embeddings import HashingEmbedder, cosine_similarity, tokenize
from .schemas import DocumentType, RetrievalDebugResult
from .storage import JsonRepository


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


class RetrievalService:
    def __init__(self, repo: JsonRepository, embedder: HashingEmbedder, min_score: float = 0.04) -> None:
        self.repo = repo
        self.embedder = embedder
        self.min_score = min_score

    def search(
        self,
        query: str,
        company_ids: list[UUID],
        top_k: int = 8,
        document_types: list[DocumentType] | None = None,
        fiscal_years: list[int] | None = None,
        min_score: float | None = None,
    ) -> list[RetrievalDebugResult]:
        query_embedding = self.embedder.embed(query)
        query_terms = set(token for token in tokenize(query) if token not in STOPWORDS)
        docs = {document.id: document for document in self.repo.list_documents()}
        companies = {company.id: company for company in self.repo.list_companies()}
        results: list[RetrievalDebugResult] = []
        threshold = self.min_score if min_score is None else min_score

        for chunk in self.repo.list_chunks(company_ids=company_ids):
            document = docs.get(chunk.document_id)
            company = companies.get(chunk.company_id)
            if not document or not company or document.status != "ready":
                continue
            if document_types and document.document_type not in document_types:
                continue
            if fiscal_years and document.fiscal_year not in fiscal_years:
                continue
            vector_score = max(0.0, cosine_similarity(query_embedding, chunk.embedding))
            chunk_terms = set(tokenize(chunk.text))
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
