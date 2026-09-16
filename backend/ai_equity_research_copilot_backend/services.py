from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from .chunking import chunk_pages
from .config import Settings
from .embeddings import HashingEmbedder, estimate_tokens
from .llm import CITATION_REFERENCE, InvalidGroundedDraft, OllamaClient
from .parsing import parse_document
from .retrieval import RetrievalService
from .query import content_terms, question_scope, refusal_reason, required_evidence_patterns
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
    RetrievalDebugResult,
    StoredCitation,
    UsageMetadata,
)
from .storage import JsonRepository


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class EvidencePoint:
    text: str
    result: RetrievalDebugResult


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
        self.ollama = OllamaClient(settings)

    def chat(self, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        corpus = self.retrieval.prepare(request.company_ids)
        for company_id in request.company_ids:
            if company_id not in corpus.snapshot.companies:
                raise KeyError(f"Company '{company_id}' not found")

        results = self.retrieval.search(
            request.question,
            company_ids=request.company_ids,
            top_k=request.top_k,
            document_types=request.document_types,
            fiscal_years=request.fiscal_years,
            corpus=corpus,
        )
        answer_payload, synthesis_provider, synthesis_model = self._answer_from_context(request.question, results)
        citations = answer_payload.citations
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
            model=synthesis_model,
            provider=synthesis_provider,
        )
        retrieval_debug = [
            {
                "id": str(result.chunk.id),
                "document_id": str(result.document.id),
                "company_id": str(result.company.id),
                "company_ticker": result.company.ticker,
                "document_title": result.document.title,
                "score": result.score,
                "excerpt": result.chunk.text[:500] + ("..." if len(result.chunk.text) > 500 else ""),
                "page_start": result.chunk.page_start,
                "section_title": result.chunk.section_title,
                "keyword_score": result.keyword_score,
                "vector_score": result.vector_score,
                "cited": any(citation.chunk_id == result.chunk.id for citation in citations),
            }
            for result in results
        ]
        retrieval_debug = {"query": request.question, "top_k": request.top_k,
                           "threshold": self.retrieval.min_score, "chunks": retrieval_debug}
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
            retrieval_debug=retrieval_debug,
            **answer_payload.model_dump(),
        )

    def memo(self, request: MemoRequest) -> ResearchMemo:
        started = time.perf_counter()
        corpus = self.retrieval.prepare([request.company_id])
        company = corpus.snapshot.companies.get(request.company_id)
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
        retrieved_ids: set[UUID] = set()
        sections: dict[str, list[str] | str] = {}
        for name, query in section_queries.items():
            results = self.retrieval.search(
                query,
                company_ids=[request.company_id],
                top_k=max(3, request.top_k // 2),
                document_types=request.document_types,
                fiscal_years=request.fiscal_years,
                corpus=corpus,
            )
            retrieved_ids.update(result.chunk.id for result in results)
            evidence = self._points_from_results(query, results, max_points=1 if name == "business_summary" else 3)
            points = self._cited_points(evidence, all_citations)
            if name == "business_summary":
                sections[name] = points[0] if points else "Insufficient cited context for a business summary."
            else:
                sections[name] = points or ["Insufficient cited context for this section."]

        usage = self._usage(
            started,
            input_text="\n".join(section_queries.values()),
            output_text="\n".join(str(value) for value in sections.values()),
            retrieval_count=len(retrieved_ids),
            cited_count=len(all_citations),
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
                "Analyst scenario to evaluate: could the cited growth drivers persist and support margins? This is not a filing conclusion."
            ],
            bear_case=[
                "Analyst scenario to evaluate: could demand, competition, costs, or execution worsen? This is not a filing conclusion."
            ],
            open_questions=[
                "What current valuation assumptions should be used?",
                "Which period should anchor the forecast baseline?",
                "Are there newer filings or transcripts that should be ingested?",
            ],
            source_citations=all_citations,
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

        corpus = self.retrieval.prepare(request.company_ids)
        for company_id in request.company_ids:
            company = corpus.snapshot.companies.get(company_id)
            if not company:
                raise KeyError(f"Company '{company_id}' not found")
            results = self.retrieval.search(
                request.question,
                company_ids=[company_id],
                top_k=request.top_k_per_company,
                document_types=request.document_types,
                fiscal_years=request.fiscal_years,
                corpus=corpus,
            )
            citations: list[Citation] = []
            points = self._cited_points(self._points_from_results(request.question, results, max_points=4), citations)
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

    def _answer_from_context(self, question: str, results: list[RetrievalDebugResult]) -> tuple[ChatAnswerPayload, str, str]:
        evidence = self._points_from_results(question, results, max_points=4)
        reason = refusal_reason(question)
        if reason or not evidence:
            return (
                ChatAnswerPayload(
                    answer=reason or "I do not have enough cited context to answer that. No retrieved passage directly supports the requested fact and period; upload or select relevant documents.",
                    key_points=[], citations=[], confidence=Confidence.low,
                    limitations=["A relevance check is not semantic entailment. Missing support is not proof that the information does not exist."],
                ), "local", self.settings.local_model_name,
            )
        fallback_note = ""
        if self.settings.llm_provider in {"auto", "ollama"}:
            # UI previews are not model context. Supply bounded source text and retain
            # exactly those excerpts for the model's selected citation references.
            contexts = []
            candidates = []
            remaining_chars = 24_000
            for result in results[:5]:
                excerpt = result.chunk.text[:min(6_000, remaining_chars)]
                if not excerpt:
                    break
                remaining_chars -= len(excerpt)
                citation = self._citation(result, excerpt)
                candidates.append(citation)
                contexts.append({"label": citation.label, "excerpt": excerpt})
            try:
                draft = self.ollama.answer(question, contexts)
                if not draft.citation_indices or any(
                    index < 1 or index > len(candidates) for index in draft.citation_indices
                ):
                    raise InvalidGroundedDraft("The local model did not select valid supporting citations")
                indices = list(dict.fromkeys(draft.citation_indices))
                number_map = {old: new for new, old in enumerate(indices, 1)}
                def renumber(text: str) -> str:
                    references = [int(value) for value in CITATION_REFERENCE.findall(text)]
                    if not references or any(index not in number_map for index in references):
                        raise InvalidGroundedDraft("The local model omitted valid inline citations")
                    return CITATION_REFERENCE.sub(lambda match: f"[{number_map[int(match.group(1))]}]", text)
                answer = renumber(draft.answer)
                points = [renumber(point) for point in draft.key_points]
                limitations = [*draft.limitations,
                    "Local model synthesis uses the supplied filing excerpts; citation membership does not independently verify every claim."]
                if any(len(context["excerpt"]) < len(result.chunk.text) for context, result in zip(contexts, results)):
                    limitations.append("Long retrieved chunks were shortened to fit the local model context.")
                return (
                    ChatAnswerPayload(answer=answer, key_points=points,
                        citations=[candidates[index - 1] for index in indices],
                        confidence=draft.confidence, limitations=limitations),
                    draft.provider, draft.model,
                )
            except InvalidGroundedDraft as exc:
                fallback_note = f"Local model output failed citation validation; deterministic cited synthesis used instead ({exc})."
            except RuntimeError as exc:
                if self.settings.llm_provider == "ollama":
                    return (
                        ChatAnswerPayload(
                            answer="The local LLM is configured but unavailable. Start Ollama and pull the configured model, then retry.",
                            key_points=[], citations=[], confidence=Confidence.low, limitations=[str(exc)]),
                        "ollama", self.settings.ollama_model,
                    )
                fallback_note = f"Local LLM unavailable; deterministic cited synthesis used instead ({exc})."
        citations: list[Citation] = []
        points = self._cited_points(evidence, citations)
        limitations = ["Answer is based only on ingested local documents; no live market data was used."]
        if not question_scope(question).periods and len({r.document.fiscal_year for r in results}) > 1:
            limitations.append("No fiscal year was specified; retrieved sources cover different fiscal years. Specify a year or document filter for a period-specific answer.")
        limitations.append("Extracted passages are supporting context, not a verified complete answer or a calculation across periods.")
        if fallback_note:
            limitations.append(fallback_note)
        return (
            ChatAnswerPayload(
                answer=" ".join(points),
                key_points=points, citations=citations,
                confidence=Confidence.medium,
                limitations=limitations,
            ),
            "local", self.settings.local_model_name,
        )

    def _cited_points(self, points: list[EvidencePoint], citations: list[Citation]) -> list[str]:
        rendered = []
        for point in points:
            index = next((i for i, citation in enumerate(citations) if citation.chunk_id == point.result.chunk.id), None)
            if index is None:
                index = len(citations)
                citations.append(self._citation(point.result, point.text))
            elif point.text not in citations[index].excerpt:
                citations[index].excerpt += "\n" + point.text
            rendered.append(f"{point.text} [{index + 1}]")
        return rendered

    def _points_from_results(self, query: str, results: list[RetrievalDebugResult], max_points: int) -> list[EvidencePoint]:
        company_terms = set().union(*(content_terms(r.company.name) | content_terms(r.company.ticker) for r in results))
        query_terms = content_terms(query) - company_terms
        anchors = required_evidence_patterns(query)
        scored: list[tuple[float, str, RetrievalDebugResult]] = []
        for result in results:
            for sentence in SENTENCE_RE.split(result.chunk.text.replace("\n", " ")):
                cleaned = " ".join(sentence.split())
                if not _usable_sentence(cleaned):
                    continue
                sentence_terms = content_terms(cleaned)
                matched = query_terms & sentence_terms
                overlap = len(matched) / max(len(query_terms), 1)
                score = result.score + overlap
                if (query_terms and len(matched) >= min(2, len(query_terms))
                        and all(re.search(pattern, cleaned, re.I) for pattern in anchors)):
                    scored.append((score, cleaned, result))
        scored.sort(key=lambda item: item[0], reverse=True)
        if question_scope(query).diversify or len({r.company.id for r in results}) > 1:
            preferred = []
            for attribute in ("company_id", "document_id"):
                seen_ids = {getattr(row[2].chunk, attribute) for row in preferred}
                for row in scored:
                    identity = getattr(row[2].chunk, attribute)
                    if identity not in seen_ids:
                        preferred.append(row)
                        seen_ids.add(identity)
            scored = preferred + [row for row in scored if row not in preferred]
        points: list[EvidencePoint] = []
        seen: set[str] = set()
        for _, sentence, result in scored:
            normalized = sentence.lower()[:120]
            if normalized in seen:
                continue
            points.append(EvidencePoint(sentence, result))
            seen.add(normalized)
            if len(points) >= max_points:
                break
        return points

    def _citation(self, result: RetrievalDebugResult, excerpt: str) -> Citation:
        document = result.document
        page = ""
        if result.chunk.page_start:
            page = f", p. {result.chunk.page_start}"
            if result.chunk.page_end and result.chunk.page_end != result.chunk.page_start:
                page = f", pp. {result.chunk.page_start}-{result.chunk.page_end}"
        label = f"{result.company.ticker} {document.title}{page}"
        return Citation(
            label=label,
            document_id=document.id,
            chunk_id=result.chunk.id,
            excerpt=excerpt,
            score=result.score,
            company_id=result.company.id,
            title=document.title,
            company_ticker=result.company.ticker,
            page_start=result.chunk.page_start,
            page_end=result.chunk.page_end,
            section_title=result.chunk.section_title,
        )

    def _usage(
        self,
        started: float,
        input_text: str,
        output_text: str,
        retrieval_count: int,
        cited_count: int,
        model: str | None = None,
        provider: str = "local",
    ) -> UsageMetadata:
        return UsageMetadata(
            model=model or self.settings.local_model_name,
            latency_ms=max(1, int((time.perf_counter() - started) * 1000)),
            input_tokens=estimate_tokens(input_text),
            output_tokens=estimate_tokens(output_text),
            estimated_cost_usd=0.0,
            provider=provider,
            retrieval={"retrieved_chunks": retrieval_count, "cited_chunks": cited_count},
        )


def _usable_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    if len(sentence) < 35 or len(sentence) > 650:
        return False
    if "table of contents" in lowered:
        return False
    if lowered.count("▪") + lowered.count("•") > 2:
        return False
    if sentence.count(";") > 5:
        return False
    alpha = sum(1 for char in sentence if char.isalpha())
    return alpha / max(len(sentence), 1) > 0.55
