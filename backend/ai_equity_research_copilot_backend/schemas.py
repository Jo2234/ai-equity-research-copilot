from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DocumentType(str, Enum):
    ten_k = "10-k"
    ten_q = "10-q"
    eight_k = "8-k"
    earnings_transcript = "earnings_transcript"
    investor_presentation = "investor_presentation"
    annual_report = "annual_report"
    manual_note = "manual_note"
    other = "other"


class DocumentStatus(str, Enum):
    uploaded = "uploaded"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class Confidence(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class CompanyCreate(BaseModel):
    ticker: str = Field(min_length=1, max_length=16)
    name: str = Field(min_length=1)
    exchange: str | None = None
    sector: str | None = None
    industry: str | None = None

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, value: str) -> str:
        return value.strip().upper()


class Company(CompanyCreate):
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class CompanyDetail(Company):
    document_count: int = 0
    ready_document_count: int = 0
    documents: list["Document"] = Field(default_factory=list)


class CompanyLookupResult(BaseModel):
    ticker: str
    name: str
    cik: int | None = None
    source: str = "local"
    local_company_id: UUID | None = None
    already_in_workspace: bool = False


class CompanyDiscoverRequest(BaseModel):
    query: str = Field(min_length=1)
    form_type: DocumentType = DocumentType.ten_k
    build_corpus: bool = True
    annual_limit: int = Field(default=1, ge=0, le=3)
    quarterly_limit: int = Field(default=4, ge=0, le=8)
    current_report_limit: int = Field(default=6, ge=0, le=20)
    proxy_limit: int = Field(default=1, ge=0, le=3)


class CompanyDiscoverResponse(BaseModel):
    company: CompanyDetail
    imported_document: "DocumentDetail"
    imported_documents: list["DocumentDetail"] = Field(default_factory=list)
    source: str = "sec"
    cik: int
    accession_number: str
    accession_numbers: list[str] = Field(default_factory=list)
    reused_existing_count: int = 0


class DocumentCreate(BaseModel):
    title: str = Field(min_length=1)
    document_type: DocumentType
    source_url: str | None = None
    filing_date: date | None = None
    period_end_date: date | None = None
    fiscal_year: int | None = None
    fiscal_quarter: int | None = Field(default=None, ge=1, le=4)


class Document(DocumentCreate):
    id: UUID = Field(default_factory=uuid4)
    company_id: UUID
    file_path: str | None = None
    status: DocumentStatus = DocumentStatus.uploaded
    parse_error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class DocumentDetail(Document):
    chunk_count: int = 0


class DocumentChunk(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    company_id: UUID
    chunk_index: int
    text: str
    embedding: list[float]
    page_start: int | None = None
    page_end: int | None = None
    section_title: str | None = None
    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class Citation(BaseModel):
    label: str
    document_id: UUID
    chunk_id: UUID
    excerpt: str
    score: float
    company_id: UUID | None = None
    title: str | None = None


class UsageMetadata(BaseModel):
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float = 0.0
    provider: str = "local"
    retrieval: dict[str, Any] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    conversation_id: UUID | None = None
    company_ids: list[UUID] = Field(min_length=1)
    question: str = Field(min_length=1)
    document_types: list[DocumentType] | None = None
    fiscal_years: list[int] | None = None
    top_k: int = Field(default=8, ge=1, le=25)


class ChatAnswerPayload(BaseModel):
    answer: str
    key_points: list[str]
    citations: list[Citation]
    confidence: Confidence
    limitations: list[str]


class ChatResponse(ChatAnswerPayload):
    message_id: UUID
    conversation_id: UUID
    usage: UsageMetadata


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1)
    company_ids: list[UUID] = Field(min_length=1)
    document_types: list[DocumentType] | None = None
    fiscal_years: list[int] | None = None
    top_k: int = Field(default=8, ge=1, le=25)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)


class RetrievedContext(BaseModel):
    query: str
    chunk_id: UUID
    document_id: UUID
    company_id: UUID
    ticker: str
    company_name: str
    document_title: str
    document_type: DocumentType
    fiscal_year: int | None = None
    fiscal_quarter: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_title: str | None = None
    excerpt: str
    token_count: int
    score: float
    keyword_score: float
    vector_score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    query: str
    total: int
    results: list[RetrievedContext]


class MemoRequest(BaseModel):
    company_id: UUID
    document_types: list[DocumentType] | None = None
    fiscal_years: list[int] | None = None
    top_k: int = Field(default=10, ge=3, le=30)


class MemoCompany(BaseModel):
    ticker: str
    name: str


class ResearchMemo(BaseModel):
    company: MemoCompany
    business_summary: str
    recent_performance: list[str]
    growth_drivers: list[str]
    margin_analysis: list[str]
    capital_allocation: list[str]
    risk_factors: list[str]
    management_commentary: list[str]
    bull_case: list[str]
    bear_case: list[str]
    open_questions: list[str]
    source_citations: list[Citation]
    limitations: list[str]
    usage: UsageMetadata


class CompareRequest(BaseModel):
    company_ids: list[UUID] = Field(min_length=2)
    question: str = Field(default="Compare operating metrics, growth drivers, and risks.", min_length=1)
    document_types: list[DocumentType] | None = None
    fiscal_years: list[int] | None = None
    top_k_per_company: int = Field(default=5, ge=1, le=15)


class CompanyComparison(BaseModel):
    company: MemoCompany
    summary: str
    key_points: list[str]
    citations: list[Citation]


class CompareResponse(BaseModel):
    question: str
    comparisons: list[CompanyComparison]
    limitations: list[str]
    usage: UsageMetadata


class Conversation(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class Message(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    conversation_id: UUID
    role: MessageRole
    content: str
    structured_payload: dict[str, Any] | None = None
    model: str | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    created_at: datetime = Field(default_factory=utcnow)


class StoredCitation(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    message_id: UUID
    document_chunk_id: UUID
    citation_label: str
    excerpt: str
    relevance_score: float
    created_at: datetime = Field(default_factory=utcnow)


class RetrievalDebugResult(BaseModel):
    query: str
    chunk: DocumentChunk
    document: Document
    company: Company
    score: float
    keyword_score: float
    vector_score: float


class MessageDetail(Message):
    citations: list[StoredCitation] = Field(default_factory=list)


class ConversationDetail(Conversation):
    messages: list[MessageDetail] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"
    data_dir: str
    companies: int
    documents: int
    chunks: int


for model in [
    Company,
    CompanyDetail,
    Document,
    DocumentDetail,
    DocumentChunk,
    Conversation,
    Message,
    StoredCitation,
]:
    model.model_config = ConfigDict(from_attributes=True)
