from __future__ import annotations

import re
import time
from pathlib import Path
from uuid import UUID

from .chunking import chunk_pages
from .config import Settings
from .embeddings import HashingEmbedder, estimate_tokens, tokenize
from .parsing import parse_document
from .retrieval import STOPWORDS, RetrievalService
from .schemas import (
    ChatAnswerPayload,
    ChatRequest,
    ChatResponse,
    Citation,
    CompanyComparison,
    CompareRequest,
    CompareResponse,
    Confidence,
    DocumentChunk,
    DocumentStatus,
    Message,
    MessageRole,
    MemoCompany,
    MemoRequest,
    ResearchMemo,
    StoredCitation,
    UsageMetadata,
)
from .storage import JsonRepository


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class IngestionService:
    def __init__(self, repo: JsonRepository, embedder: HashingEmbedder, settings: Settings) -> None:
        self.repo = repo
        self.embedder = embedder
        self.settings = settings

    def ingest_document(self, document_id: UUID) -> None:
        document = self.repo.get_document(document_id)
        if not document:
            raise KeyError(f"Document '{document_id}' not found")
        document.status = DocumentStatus.processing
        document.parse_error = None
        self.repo.update_document(document)
        try:
            if not document.file_path:
                raise ValueError("Document has no stored file path")
            pages = parse_document(Path(document.file_path))
            drafts = chunk_pages(
                pages,
                target_tokens=self.settings.chunk_target_tokens,
                overlap_tokens=self.settings.chunk_overlap_tokens,
            )
            chunks = [
                DocumentChunk(
                    document_id=document.id,
                    company_id=document.company_id,
                    chunk_index=draft.chunk_index,
                    text=draft.text,
                    embedding=self.embedder.embed(draft.text),
                    page_start=draft.page_start,
                    page_end=draft.page_end,
                    section_title=draft.section_title,
                    token_count=draft.token_count,
                    metadata={
                        **draft.metadata,
                        "source_file": document.file_path,
                        "document_title": document.title,
                        "document_type": document.document_type,
                    },
                )
                for draft in drafts
                if draft.text.strip()
            ]
            if not chunks:
                raise ValueError("No text chunks were extracted")
            self.repo.replace_chunks(document.id, chunks)
            document.status = DocumentStatus.ready
        except Exception as exc:
            document.status = DocumentStatus.failed
            document.parse_error = str(exc)
            self.repo.replace_chunks(document.id, [])
        self.repo.update_document(document)


class ResearchService:
    def __init__(
        self,
        repo: JsonRepository,
        retrieval: RetrievalService,
        settings: Settings,
    ) -> None:
        self.repo = repo
        self.retrieval = retrieval
        self.settings = settings

    def chat(self, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        for company_id in request.company_ids:
            if not self.repo.get_company(company_id):
                raise KeyError(f"Company '{company_id}' not found")

        results = self.retrieval.search(
            request.question,
            company_ids=request.company_ids,
            top_k=request.top_k,
            document_types=request.document_types,
            fiscal_years=request.fiscal_years,
        )
        citations = [self._citation(result) for result in results[: min(5, len(results))]]
        answer_payload = self._answer_from_context(request.question, results, citations)
        conversation = (
            self.repo.get_conversation(request.conversation_id)
            if request.conversation_id
            else self.repo.create_conversation(request.question)
        )
        if not conversation:
            raise KeyError(f"Conversation '{request.conversation_id}' not found")

        user_message = Message(conversation_id=conversation.id, role=MessageRole.user, content=request.question)
        self.repo.create_message(user_message)
        usage = self._usage(
            started,
            input_text=request.question + "\n" + "\n".join(result.chunk.text for result in results),
            output_text=answer_payload.answer,
            retrieval_count=len(results),
            cited_count=len(citations),
        )
        retrieval_debug = [
            {
                "chunk_id": str(result.chunk.id),
                "document_id": str(result.document.id),
                "company_id": str(result.company.id),
                "ticker": result.company.ticker,
                "document_title": result.document.title,
                "score": result.score,
                "keyword_score": result.keyword_score,
                "vector_score": result.vector_score,
                "cited": any(citation.chunk_id == result.chunk.id for citation in citations),
            }
            for result in results
        ]
        assistant_message = Message(
            conversation_id=conversation.id,
            role=MessageRole.assistant,
            content=answer_payload.answer,
            structured_payload={
                **answer_payload.model_dump(mode="json"),
                "usage": usage.model_dump(mode="json"),
                "retrieval_debug": retrieval_debug,
            },
            model=usage.model,
            latency_ms=usage.latency_ms,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            estimated_cost_usd=usage.estimated_cost_usd,
        )
        self.repo.create_message(assistant_message)
        self.repo.create_citations(
            [
                StoredCitation(
                    message_id=assistant_message.id,
                    document_chunk_id=citation.chunk_id,
                    citation_label=citation.label,
                    excerpt=citation.excerpt,
                    relevance_score=citation.score,
                )
                for citation in citations
            ]
        )
        return ChatResponse(
            message_id=assistant_message.id,
            conversation_id=conversation.id,
            usage=usage,
            **answer_payload.model_dump(),
        )

    def memo(self, request: MemoRequest) -> ResearchMemo:
        started = time.perf_counter()
        company = self.repo.get_company(request.company_id)
        if not company:
            raise KeyError(f"Company '{request.company_id}' not found")

        section_queries = {
            "business_summary": "business model revenue products customers segments",
            "recent_performance": "recent performance revenue growth margin operating income quarter year",
            "growth_drivers": "growth drivers demand pricing adoption expansion guidance",
            "margin_analysis": "gross margin operating margin cost efficiency mix",
            "capital_allocation": "capital allocation cash flow repurchases dividends investment debt",
            "risk_factors": "risk factors competition regulation supply macro customer concentration",
            "management_commentary": "management commentary outlook guidance expects priorities",
        }
        all_citations: list[Citation] = []
        sections: dict[str, list[str] | str] = {}
        for name, query in section_queries.items():
            results = self.retrieval.search(
                query,
                company_ids=[request.company_id],
                top_k=max(3, request.top_k // 2),
                document_types=request.document_types,
                fiscal_years=request.fiscal_years,
            )
            citations = [self._citation(result) for result in results[:2]]
            all_citations.extend(citations)
            points = self._points_from_results(query, results, max_points=3)
            if name == "business_summary":
                sections[name] = points[0] if points else "Insufficient cited context for a business summary."
            else:
                sections[name] = points or ["Insufficient cited context for this section."]

        unique_citations = _dedupe_citations(all_citations)
        usage = self._usage(
            started,
            input_text="\n".join(section_queries.values()),
            output_text="\n".join(str(value) for value in sections.values()),
            retrieval_count=len(unique_citations),
            cited_count=len(unique_citations),
        )
        return ResearchMemo(
            company=MemoCompany(ticker=company.ticker, name=company.name),
            business_summary=str(sections["business_summary"]),
            recent_performance=list(sections["recent_performance"]),
            growth_drivers=list(sections["growth_drivers"]),
            margin_analysis=list(sections["margin_analysis"]),
            capital_allocation=list(sections["capital_allocation"]),
            risk_factors=list(sections["risk_factors"]),
            management_commentary=list(sections["management_commentary"]),
            bull_case=[
                "Bull case: cited context indicates execution upside if growth drivers persist and margin factors remain favorable."
            ],
            bear_case=[
                "Bear case: cited context highlights risk if demand, competition, costs, or execution pressures worsen."
            ],
            open_questions=[
                "What current valuation assumptions should be used?",
                "Which period should anchor the forecast baseline?",
                "Are there newer filings or transcripts that should be ingested?",
            ],
            source_citations=unique_citations,
            limitations=["This memo is document-grounded and does not include live prices or investment advice."],
            usage=usage,
        )

    def compare(self, request: CompareRequest) -> CompareResponse:
        started = time.perf_counter()
        comparisons: list[CompanyComparison] = []
        limitations: list[str] = []
        all_output: list[str] = []
        total_retrievals = 0
        total_citations = 0

        for company_id in request.company_ids:
            company = self.repo.get_company(company_id)
            if not company:
                raise KeyError(f"Company '{company_id}' not found")
            results = self.retrieval.search(
                request.question,
                company_ids=[company_id],
                top_k=request.top_k_per_company,
                document_types=request.document_types,
                fiscal_years=request.fiscal_years,
            )
            citations = [self._citation(result) for result in results[:3]]
            points = self._points_from_results(request.question, results, max_points=4)
            if not points:
                limitations.append(f"{company.ticker}: no sufficiently relevant cited context was found.")
                points = ["Insufficient cited context for this company."]
            summary = f"{company.ticker}: " + " ".join(points[:2])
            comparisons.append(
                CompanyComparison(
                    company=MemoCompany(ticker=company.ticker, name=company.name),
                    summary=summary,
                    key_points=points,
                    citations=citations,
                )
            )
            all_output.extend(points)
            total_retrievals += len(results)
            total_citations += len(citations)

        usage = self._usage(
            started,
            input_text=request.question,
            output_text="\n".join(all_output),
            retrieval_count=total_retrievals,
            cited_count=total_citations,
        )
        if not limitations:
            limitations.append("Comparison is limited to ingested documents and does not include market data.")
        return CompareResponse(
            question=request.question,
            comparisons=comparisons,
            limitations=limitations,
            usage=usage,
        )

    def _answer_from_context(self, question: str, results: list, citations: list[Citation]) -> ChatAnswerPayload:
        if not results or not citations:
            return ChatAnswerPayload(
                answer="I do not have enough cited context to answer that. Upload or select relevant company documents first.",
                key_points=[],
                citations=[],
                confidence=Confidence.low,
                limitations=["No retrieved chunks cleared the local relevance threshold."],
            )
        points = self._points_from_results(question, results, max_points=4)
        if not points:
            return ChatAnswerPayload(
                answer="I do not have enough cited context to answer that. The retrieved documents do not directly address the question.",
                key_points=[],
                citations=citations,
                confidence=Confidence.low,
                limitations=["Retrieved chunks were available but did not contain direct support."],
            )
        citation_refs = " ".join(f"[{idx + 1}]" for idx in range(min(len(citations), 3)))
        answer = " ".join(points)
        answer = f"{answer} {citation_refs}".strip()
        confidence = Confidence.high if len(citations) >= 3 else Confidence.medium
        return ChatAnswerPayload(
            answer=answer,
            key_points=points,
            citations=citations,
            confidence=confidence,
            limitations=["Answer is based only on ingested local documents; no live market data was used."],
        )

    def _points_from_results(self, query: str, results: list, max_points: int) -> list[str]:
        query_terms = set(token for token in tokenize(query) if token not in STOPWORDS)
        scored: list[tuple[float, str]] = []
        for result in results:
            for sentence in SENTENCE_RE.split(result.chunk.text.replace("\n", " ")):
                cleaned = " ".join(sentence.split())
                if len(cleaned) < 35:
                    continue
                sentence_terms = set(tokenize(cleaned))
                overlap = len(query_terms & sentence_terms) / max(len(query_terms), 1)
                score = result.score + overlap
                if overlap > 0 or result.score > 0.16:
                    scored.append((score, cleaned))
        scored.sort(key=lambda item: item[0], reverse=True)
        points: list[str] = []
        seen: set[str] = set()
        for _, sentence in scored:
            normalized = sentence.lower()[:120]
            if normalized in seen:
                continue
            points.append(sentence)
            seen.add(normalized)
            if len(points) >= max_points:
                break
        return points

    def _citation(self, result) -> Citation:
        document = result.document
        page = ""
        if result.chunk.page_start:
            page = f", p. {result.chunk.page_start}"
            if result.chunk.page_end and result.chunk.page_end != result.chunk.page_start:
                page = f", pp. {result.chunk.page_start}-{result.chunk.page_end}"
        label = f"{result.company.ticker} {document.title}{page}"
        excerpt = result.chunk.text[:420].strip()
        if len(result.chunk.text) > 420:
            excerpt += "..."
        return Citation(
            label=label,
            document_id=document.id,
            chunk_id=result.chunk.id,
            excerpt=excerpt,
            score=result.score,
            company_id=result.company.id,
            title=document.title,
        )

    def _usage(
        self,
        started: float,
        input_text: str,
        output_text: str,
        retrieval_count: int,
        cited_count: int,
    ) -> UsageMetadata:
        return UsageMetadata(
            model=self.settings.local_model_name,
            latency_ms=max(1, int((time.perf_counter() - started) * 1000)),
            input_tokens=estimate_tokens(input_text),
            output_tokens=estimate_tokens(output_text),
            estimated_cost_usd=0.0,
            provider="local",
            retrieval={"retrieved_chunks": retrieval_count, "cited_chunks": cited_count},
        )


def _dedupe_citations(citations: list[Citation]) -> list[Citation]:
    seen: set[UUID] = set()
    unique: list[Citation] = []
    for citation in citations:
        if citation.chunk_id in seen:
            continue
        unique.append(citation)
        seen.add(citation.chunk_id)
    return unique
