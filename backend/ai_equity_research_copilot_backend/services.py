from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import UUID

from .chunking import chunk_pages
from .config import Settings
from .embeddings import HashingEmbedder, estimate_tokens
from .evidence import CAUSE_RE, QUANTITY_RE, evidence_passages
from .financial_text import format_financial_text
from .financial_tables import financial_table_facts
from .financial_table_passages import table_passages
from .table_context import attach_table_context
from .llm import CITATION_REFERENCE, InvalidGroundedDraft, OllamaClient
from .parsing import parse_document
from .retrieval import RetrievalService
from .retrieval_query import company_clause_queries
from .source_query import source_clause_query
from .query import content_terms, explicit_periods, question_scope, refusal_reason, required_evidence_patterns, passage_period
from .query_topics import query_topics
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
    source_text: str | None = None


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
            drafts = attach_table_context(pages, drafts)
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
        period_policies: list[str] = []
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
            period_policies.extend(result.chunk.metadata["retrieval_period_policy"] for result in results
                                   if result.chunk.metadata.get("retrieval_period_policy"))
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
            limitations=["This memo is document-grounded and does not include live prices or investment advice.",
                         *dict.fromkeys(period_policies)],
            usage=usage,
        )

    def compare(self, request: CompareRequest) -> CompareResponse:
        started = time.perf_counter()
        comparisons: list[CompanyComparison] = []
        limitations: list[str] = []
        period_policies: list[str] = []
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
            period_policies.extend(f"{company.ticker}: {result.chunk.metadata['retrieval_period_policy']}"
                                   for result in results if result.chunk.metadata.get("retrieval_period_policy"))
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
        limitations.extend(dict.fromkeys(period_policies))
        return CompareResponse(
            question=request.question,
            comparisons=comparisons,
            limitations=limitations,
            usage=usage,
        )

    def _answer_from_context(self, question: str, results: list[RetrievalDebugResult]) -> tuple[ChatAnswerPayload, str, str]:
        evidence = self._points_from_results(question, results, max_points=8)
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
                limitations.extend(dict.fromkeys(
                    r.chunk.metadata["retrieval_period_policy"] for r in results
                    if r.chunk.metadata.get("retrieval_period_policy")
                ))
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
        limitations.extend(dict.fromkeys(
            r.chunk.metadata["retrieval_period_policy"] for r in results
            if r.chunk.metadata.get("retrieval_period_policy")
        ))
        if not question_scope(question).periods and len({r.document.fiscal_year for r in results}) > 1:
            limitations.append("No fiscal year was specified; retrieved sources cover different fiscal years. Specify a year or document filter for a period-specific answer.")
        limitations.append("Extracted passages are supporting context, not a verified complete answer or a calculation across periods.")
        if fallback_note:
            limitations.append(fallback_note)
        return (
            ChatAnswerPayload(
                answer=" ".join(self._cited_points(evidence, citations, render=True)),
                key_points=points, citations=citations,
                confidence=Confidence.medium,
                limitations=limitations,
            ),
            "local", self.settings.local_model_name,
        )

    def _cited_points(self, points: list[EvidencePoint], citations: list[Citation], *, render: bool = False) -> list[str]:
        rendered = []
        for point in points:
            source_text = point.source_text or point.text
            index = next((i for i, citation in enumerate(citations) if citation.chunk_id == point.result.chunk.id), None)
            if index is None:
                index = len(citations)
                citations.append(self._citation(point.result, source_text))
            elif source_text not in citations[index].excerpt:
                citations[index].excerpt += "\n" + source_text
            if render:
                doc = point.result.document
                quarter = f" Q{doc.fiscal_quarter}" if doc.fiscal_quarter else ""
                period = f"{point.result.company.ticker} {doc.document_type.value} FY{doc.fiscal_year or 'unknown'}{quarter}: "
                rendered.append(f"{period}{format_financial_text(point.text)} [{index + 1}]")
            else:
                # Evidence points remain verbatim, independently of readable
                # answer formatting and derived table normalization.
                rendered.append(f"{source_text} [{index + 1}]")
        return rendered

    def _points_from_results(self, query: str, results: list[RetrievalDebugResult], max_points: int) -> list[EvidencePoint]:
        company_terms = set().union(*(content_terms(r.company.name) | content_terms(r.company.ticker) for r in results))
        company_queries = company_clause_queries(query, {r.company.id: r.company for r in results}.values())
        anchors = required_evidence_patterns(query)
        # Preserve query word order for phrase matches: a "gross margin"
        # passage is stronger evidence than unrelated mentions of either word.
        def sequence(text: str) -> list[str]:
            return [next(iter(terms)) if terms else ""
                    for token in re.findall(r"[A-Za-z]+", text)
                    for terms in [content_terms(token)]]

        scope = question_scope(query)
        requested_periods = set(scope.periods)
        if scope.quarters and not any(quarter is not None for _, quarter in scope.periods):
            # A quarter can be separated from its year by the requested metric,
            # e.g. "Q1 revenue in fiscal 2024". Preserve an annual component only
            # when the question actually requests both annual and quarter data.
            requested_periods = {(year, quarter) for year, _ in scope.periods for quarter in scope.quarters}
            if re.search(r"\b(?:annual|full[- ]year|10[- ]?k)\b", query, re.I):
                requested_periods.update(scope.periods)
        scored: list[tuple[float, str, RetrievalDebugResult, set[str], set[tuple[str, ...]]]] = []
        table_labels: dict[tuple[UUID, str], str] = {}
        sections: dict[tuple[UUID, str], str] = {}
        for result in results:
            topic_query = company_queries[result.company.id]
            if topic_query == query:
                topic_query = source_clause_query(query, result.document.document_type)
            topics = query_topics(topic_query, excluded_terms=company_terms)
            query_terms = topics.terms
            query_sequence = sequence(topic_query)
            phrases = {(left, right) for left, right in zip(query_sequence, query_sequence[1:])
                       if left in query_terms and right in query_terms}
            facets = [content_terms(part) - company_terms for part in re.split(r"\band\b|\bor\b", topic_query, flags=re.I)]
            explanatory = bool(re.search(r"\b(?:why|drove|driver|drivers|factor|factors|discussion|commentary)\b", topic_query, re.I))
            qualitative_risk = bool(re.search(r"\bsupply\b|\binventory\b|\bcapacity\b", topic_query, re.I))
            # A fiscal-year release may contain both Q4 and full-year sections.
            # Use an explicit local heading when present; metadata alone cannot
            # distinguish those figures. Unknown section periods remain unknown.
            blocks = []
            local_period = None
            for block in re.split(r"\n\s*\n", result.chunk.text):
                text = " ".join(block.split())
                if len(text) < 160 and re.search(r"(?:highlights|results|performance)$|^guidance\b", text, re.I):
                    # Preserve bare-year headings such as "2031 Results".
                    # A quarter in the heading overrides this annual fallback.
                    year = re.search(r"\b20\d{2}\b", text)
                    if year:
                        local_period = (int(year[0]), None)
                    local_period = passage_period(text, local_period, result.document.fiscal_year)
                blocks.append((text, local_period))
            passages = evidence_passages(result.chunk.text)
            source_sections = [
                (" ".join(result.chunk.text[span["start"]:span["end"]].split()), span["section"])
                for span in result.chunk.metadata.get("section_spans", [])
            ]
            facts = result.chunk.metadata.get("financial_table_facts")
            if facts is None:  # Compatibility with documents ingested before table provenance was saved.
                facts = [asdict(fact) for fact in financial_table_facts(result.chunk.text)]
            table_candidates = table_passages(facts)
            for label, source_excerpt in table_candidates:
                if source_excerpt not in passages:
                    passages.append(source_excerpt)
                table_labels[(result.chunk.id, source_excerpt)] = label
            for cleaned in passages:
                # A table row's metric often lives in its verified caption.
                # Use that original header for relevance, never infer a metric
                # from an amount or an isolated geographic/business label.
                table_label = table_labels.get((result.chunk.id, cleaned))
                table_fact = table_label is not None
                sentence_terms = content_terms(table_label if table_fact else cleaned)
                containing_sections = {section for text, section in source_sections
                                       if " ".join(cleaned.split()) in text} if not table_fact else set()
                if len(containing_sections) > 1:
                    continue  # Identical prose in different segments has ambiguous scope.
                section = next(iter(containing_sections), "")
                if section:
                    sections[(result.chunk.id, cleaned)] = section
                leaf_terms = content_terms(" ".join(label.rsplit(": ", 1)[-1]
                    for label in table_label.splitlines())) if table_fact else sentence_terms
                direct_matched = (topics.matched_terms(leaf_terms) | (sentence_terms & query_terms)) & query_terms
                header = cleaned.split("\n...\n", 1)[0] if table_fact else ""
                context_matched = topics.matched_terms(content_terms(header)) & query_terms
                matched = direct_matched | context_matched
                if not (query_terms and (len(matched) >= min(2, len(query_terms))
                                        or any(facet and facet <= matched for facet in facets))
                        and all(re.search(pattern, cleaned, re.I) for pattern in anchors)):
                    continue
                literal_header_matches = content_terms(header) & query_terms
                overlap = (len(direct_matched) + len(literal_header_matches - direct_matched)
                           + 0.3 * len(context_matched - direct_matched - literal_header_matches)) / len(query_terms)
                words = sequence(table_label if table_fact else cleaned)
                phrase_overlap = len(phrases & set(zip(words, words[1:]))) / max(len(phrases), 1)
                score = result.score + 2 * overlap + 0.3 * phrase_overlap + 0.35 * topics.related_score(sentence_terms)
                score -= 0.2 if explanatory and "\n...\n" in cleaned else 0.0
                # Actual amounts and causal explanations answer financial
                # questions more directly than repeated accounting definitions.
                score += (0.08 if qualitative_risk else 0.25) * min(3, len(set(QUANTITY_RE.findall(cleaned)))) * max(0.25, overlap)
                if explanatory and CAUSE_RE.search(cleaned):
                    score += 0.2
                # Full-year releases can also contain their final quarter. Do
                # not treat an explicitly quarterly figure as a full-year total.
                normalized = " ".join(cleaned.split())
                local_period = next((period for text, period in blocks if normalized in text), None)
                if section:
                    local_period = passage_period(section, local_period, result.document.fiscal_year)
                    heading_year = re.match(r"^(20\d{2})\b", section)
                    if heading_year:
                        local_period = (int(heading_year[1]), None)
                # A release can switch to full-year results in prose without
                # introducing a new heading. The passage's explicit period is
                # stronger evidence than an inherited section label.
                local_period = passage_period(cleaned, local_period, result.document.fiscal_year)
                required_periods = result.chunk.metadata.get("retrieval_require_passage_period")
                if required_periods and (local_period is None or list(local_period) not in required_periods):
                    continue
                allowed_periods = requested_periods
                if (not allowed_periods and result.document.document_type.value in {"10-k", "annual_report"}
                        and not re.search(r"\b(?:historical|previous|prior|past)\b", query, re.I)):
                    allowed_periods = {(result.document.fiscal_year, None)}
                # An undated annual component has its own disclosed document
                # period; the separately named quarter does not override it.
                if (scope.quarters and not any(quarter is None for _, quarter in scope.periods)
                        and result.document.document_type.value in {"10-k", "annual_report"}
                        and re.search(r"\b(?:annual|full[- ]year|10[- ]?k)\b", query, re.I)):
                    allowed_periods = {(result.document.fiscal_year, None)}
                # Unknown scope must not conceal a conflicting explicit short
                # year or let a multi-period window pass as one requested period.
                short_years = re.findall(r"\bFY\s*(\d{2})(?!\d)", cleaned, re.I)
                if any(result.document.fiscal_year is None or result.document.fiscal_year % 100 != int(year)
                       for year in short_years):
                    continue
                stated_periods = explicit_periods(cleaned, result.document.fiscal_year)
                if allowed_periods and stated_periods and not stated_periods <= allowed_periods:
                    continue
                if allowed_periods and local_period:
                    if local_period not in allowed_periods:
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
                novelty = len(terms - covered) / max(1, len(terms))
                # A comparison must represent each eligible source before one
                # source can consume every answer slot. Rank evidence within
                # that requirement, rather than trading coverage for a bonus.
                coverage = (int(result.company.id not in {item[2].company.id for item in selected}),
                            int(not comparable)) if diversify else (0, 0)
                return (*coverage, score + 0.2 * novelty - 0.8 * redundancy)

            row = max(scored, key=priority)
            scored.remove(row)
            normalized = " ".join(row[1].lower().split())
            if any(item[2].document.id == row[2].document.id
                   and (normalized in " ".join(item[1].lower().split())
                        or " ".join(item[1].lower().split()) in normalized)
                   for item in selected):
                continue
            selected.append(row)
        evidence = []
        for _, text, result, _, _ in selected:
            section = sections.get((result.chunk.id, text))
            source_text = None
            if section and section not in text:
                source_text = section + "\n...\n" + text
                text = f"{section}: {text}"
            evidence.append(EvidencePoint(text, result, source_text))
        return evidence

    def _citation(self, result: RetrievalDebugResult, excerpt: str) -> Citation:
        document = result.document
        section = result.chunk.section_title
        spans = result.chunk.metadata.get("section_spans")
        if spans:
            sections = {span["section"] for span in spans}
            section = next(iter(sections)) if len(sections) == 1 else None
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
            section_title=section,
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
