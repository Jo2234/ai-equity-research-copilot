from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from .chunking import chunk_pages
from .config import Settings
from .embeddings import HashingEmbedder, estimate_tokens
from .evidence import CAUSE_RE, QUANTITY_RE, evidence_passages
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
                    message = "The local LLM is configured but unavailable. Start Ollama and pull the configured model, then retry."
                    if "timed out" in str(exc).lower():
                        message = "The local model request timed out before an answer was available. Try fewer documents or increase the configured timeout."
                    return (
                        ChatAnswerPayload(
                            answer=message,
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
        multiple_documents = len({item.result.document.id for item in points}) > 1
        for point in points:
            index = next((i for i, citation in enumerate(citations) if citation.chunk_id == point.result.chunk.id), None)
            if index is None:
                index = len(citations)
                citations.append(self._citation(point.result, point.text))
            elif point.text not in citations[index].excerpt:
                citations[index].excerpt += "\n" + point.text
            period = ""
            if multiple_documents:
                doc = point.result.document
                quarter = f" Q{doc.fiscal_quarter}" if doc.fiscal_quarter else ""
                period = f"{point.result.company.ticker} {doc.document_type.value} FY{doc.fiscal_year or 'unknown'}{quarter}: "
            rendered.append(f"{period}{point.text} [{index + 1}]")
        return rendered

    def _points_from_results(self, query: str, results: list[RetrievalDebugResult], max_points: int) -> list[EvidencePoint]:
        company_terms = set().union(*(content_terms(r.company.name) | content_terms(r.company.ticker) for r in results))
        query_terms = content_terms(query) - company_terms
        anchors = required_evidence_patterns(query)
        explanatory = bool(re.search(r"\b(?:why|drove|driver|drivers|factor|factors|discussion|commentary)\b", query, re.I))
        # Preserve query word order for phrase matches: a "gross margin"
        # passage is stronger evidence than unrelated mentions of either word.
        def sequence(text: str) -> list[str]:
            return [next(iter(terms)) if terms else ""
                    for token in re.findall(r"[A-Za-z]+", text)
                    for terms in [content_terms(token)]]

        query_sequence = sequence(query)
        phrases = {(left, right) for left, right in zip(query_sequence, query_sequence[1:])
                   if left in query_terms and right in query_terms}
        scope = question_scope(query)
        requested_periods = set(scope.periods)
        if scope.quarters and not any(quarter is not None for _, quarter in scope.periods):
            # A quarter can be separated from its year by the requested metric,
            # e.g. "Q1 revenue in fiscal 2024". Preserve an annual component only
            # when the question actually requests both annual and quarter data.
            requested_periods = {(year, quarter) for year, _ in scope.periods for quarter in scope.quarters}
            if re.search(r"\b(?:annual|full[- ]year|10[- ]?k)\b", query, re.I):
                requested_periods.update(scope.periods)
        facets = [content_terms(part) - company_terms for part in re.split(r"\band\b|\bor\b", query, flags=re.I)]
        scored: list[tuple[float, str, RetrievalDebugResult, set[str], set[tuple[str, ...]]]] = []
        for result in results:
            # A fiscal-year release may contain both Q4 and full-year sections.
            # Use an explicit local heading when present; metadata alone cannot
            # distinguish those figures. Unknown section periods remain unknown.
            blocks = []
            local_period = None
            for block in re.split(r"\n\s*\n", result.chunk.text):
                text = " ".join(block.split())
                if len(text) < 120 and re.search(r"(?:highlights|results|performance)$", text, re.I):
                    year = re.search(r"\b20\d{2}\b", text)
                    if year:
                        quarter = re.search(r"\b(first|second|third|fourth)\s+quarter|\bq([1-4])\b", text, re.I)
                        quarter_number = None
                        if quarter:
                            quarter_number = (int(quarter[2]) if quarter[2] else
                                              {"first": 1, "second": 2, "third": 3, "fourth": 4}[quarter[1].lower()])
                        local_period = (int(year[0]), quarter_number)
                blocks.append((text, local_period))
            for cleaned in evidence_passages(result.chunk.text):
                sentence_terms = content_terms(cleaned)
                matched = query_terms & sentence_terms
                if not (query_terms and (len(matched) >= min(2, len(query_terms))
                                        or any(facet and facet <= matched for facet in facets))
                        and all(re.search(pattern, cleaned, re.I) for pattern in anchors)):
                    continue
                overlap = len(matched) / len(query_terms)
                words = sequence(cleaned)
                phrase_overlap = len(phrases & set(zip(words, words[1:]))) / max(len(phrases), 1)
                score = result.score + overlap + 0.3 * phrase_overlap
                score -= 0.2 if explanatory and "\n...\n" in cleaned else 0.0
                # Actual amounts and causal explanations answer financial
                # questions more directly than repeated accounting definitions.
                score += 0.05 * min(3, len(set(QUANTITY_RE.findall(cleaned))))
                if explanatory and CAUSE_RE.search(cleaned):
                    score += 0.2
                # Full-year releases can also contain their final quarter. Do
                # not treat an explicitly quarterly figure as a full-year total.
                normalized = " ".join(cleaned.split())
                local_period = next((period for text, period in blocks if normalized in text), None)
                # A release can switch to full-year results in prose without
                # introducing a new heading. The passage's explicit period is
                # stronger evidence than an inherited section label.
                if not re.search(r"\b(?:quarter|q[1-4]|three months)\b", cleaned, re.I):
                    explicit_years = set(re.findall(
                        r"\b(?:in|for|during)\s+(?:the\s+)?fiscal(?:\s+year)?\s+(20\d{2})\b", cleaned, re.I))
                    if len(explicit_years) == 1:
                        local_period = (int(next(iter(explicit_years))), None)
                if requested_periods and local_period:
                    if local_period not in requested_periods:
                        continue
                    score += 0.2
                tokens = re.findall(r"\w+", cleaned.lower())
                shingles = {tuple(tokens[i:i + 5]) for i in range(max(1, len(tokens) - 4))}
                scored.append((score, cleaned, result, matched, shingles))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = []
        diversify = scope.diversify or len({r.company.id for r in results}) > 1
        while scored and len(selected) < max_points:
            def priority(row):
                score, _, result, terms, shingles = row
                comparable = [item for item in selected if item[2].document.id == result.document.id]
                redundancy = max((len(shingles & item[4]) / max(1, min(len(shingles), len(item[4])))
                                  for item in comparable), default=0.0)
                covered = set().union(*(item[3] for item in comparable))
                novelty = len(terms - covered) / max(1, len(query_terms))
                # Reserve room for requested companies and documents without
                # letting four overlapping sentence windows fill the answer.
                coverage = 0.0
                if diversify:
                    if result.company.id not in {item[2].company.id for item in selected}:
                        coverage += 1.0
                    if result.document.id not in {item[2].document.id for item in selected}:
                        coverage += 0.5
                # Redundancy matters when it crowds out another requested
                # topic. For one topic, overlapping context may supply its
                # essential explanation or offsetting factors.
                penalty = 0.8 * redundancy if len(facets) > 1 else 0.0
                return score + 0.2 * novelty + coverage - penalty

            row = max(scored, key=priority)
            scored.remove(row)
            normalized = " ".join(row[1].lower().split())
            if any(item[2].document.id == row[2].document.id
                   and (normalized in " ".join(item[1].lower().split())
                        or " ".join(item[1].lower().split()) in normalized)
                   for item in selected):
                continue
            selected.append(row)
        return [EvidencePoint(row[1], row[2]) for row in selected]

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
