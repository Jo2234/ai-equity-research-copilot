from __future__ import annotations

import re
import shutil
from datetime import date
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings, get_settings
from .embeddings import HashingEmbedder
from .retrieval import RetrievalService
from .schemas import (
    ChatRequest,
    ChatResponse,
    Company,
    CompanyCreate,
    CompanyDiscoverRequest,
    CompanyDiscoverResponse,
    CompanyDetail,
    CompanyLookupResult,
    Conversation,
    ConversationDetail,
    CompareRequest,
    CompareResponse,
    Document,
    DocumentCreate,
    DocumentDetail,
    DocumentStatus,
    DocumentType,
    HealthResponse,
    MessageDetail,
    MemoRequest,
    RetrievedContext,
    RetrievalRequest,
    RetrievalResponse,
    ResearchMemo,
)
from .sec import SecEdgarClient, save_sec_filing_text
from .seed import seed_repository
from .services import IngestionService, ResearchService
from .storage import JsonRepository


def create_app(data_dir: str | Path | None = None, seed: bool = True) -> FastAPI:
    settings = get_settings(data_dir)
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    repo = JsonRepository(settings.state_file)
    embedder = HashingEmbedder(settings.embedding_dimensions)
    ingestion = IngestionService(repo, embedder, settings)
    retrieval = RetrievalService(repo, embedder, settings.retrieval_min_score)
    research = ResearchService(repo, retrieval, settings)
    sec_client = SecEdgarClient(settings)
    if seed:
        seed_repository(repo, ingestion, settings)

    api = FastAPI(
        title="AI Equity Research Copilot API",
        version="0.1.0",
        description="Local document-grounded equity research API with deterministic retrieval.",
    )
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    api.state.settings = settings
    api.state.repo = repo
    api.state.ingestion = ingestion
    api.state.retrieval = retrieval
    api.state.research = research
    api.state.sec_client = sec_client

    @api.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        counts = repo.health_counts()
        return HealthResponse(data_dir=str(settings.data_dir), **counts)

    @api.get("/companies", response_model=list[Company])
    def list_companies(search: str | None = None) -> list[Company]:
        return repo.list_companies(search=search)

    @api.get("/companies/search", response_model=list[CompanyLookupResult])
    def search_companies(q: str, limit: int = 8) -> list[CompanyLookupResult]:
        if limit < 1 or limit > 25:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 25")
        local = repo.list_companies(search=q)
        rows: list[CompanyLookupResult] = [
            CompanyLookupResult(
                ticker=company.ticker,
                name=company.name,
                source="local",
                local_company_id=company.id,
                already_in_workspace=True,
            )
            for company in local[:limit]
        ]
        seen = {row.ticker for row in rows}
        try:
            for match in sec_client.search_companies(q, limit=limit):
                local_match = next((company for company in repo.list_companies() if company.ticker == match.ticker), None)
                if match.ticker in seen and local_match:
                    continue
                rows.append(
                    CompanyLookupResult(
                        ticker=match.ticker,
                        name=match.name,
                        cik=match.cik,
                        source="sec",
                        local_company_id=local_match.id if local_match else None,
                        already_in_workspace=local_match is not None,
                    )
                )
                seen.add(match.ticker)
                if len(rows) >= limit:
                    break
        except RuntimeError as exc:
            if not rows:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
        return rows

    @api.post("/companies/discover", response_model=CompanyDiscoverResponse, status_code=201)
    def discover_company(payload: CompanyDiscoverRequest) -> CompanyDiscoverResponse:
        form_map = {
            DocumentType.ten_k: ["10-K", "10-K/A"],
            DocumentType.ten_q: ["10-Q", "10-Q/A"],
            DocumentType.eight_k: ["8-K", "8-K/A"],
        }
        if payload.form_type not in form_map:
            raise HTTPException(status_code=422, detail="SEC discovery currently supports 10-k, 10-q, and 8-k")
        matches = sec_client.search_companies(payload.query, limit=1)
        if not matches:
            raise HTTPException(status_code=404, detail=f"No SEC company match found for '{payload.query}'")
        match = matches[0]
        try:
            company = repo.upsert_company(sec_client.company_create_payload(match))
            filing = sec_client.latest_filing(match.cik, form_map[payload.form_type])
            document = next(
                (existing for existing in repo.list_documents(company.id) if existing.source_url == filing.source_url),
                None,
            )
            if document is None:
                text = sec_client.download_filing_text(filing)
                file_path = save_sec_filing_text(settings.raw_dir, company.id, filing, text)
                document = repo.create_document(company.id, sec_client.document_create_payload(match, filing), str(file_path))
                ingestion.ingest_document(document.id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        detail = get_company(company.id)
        return CompanyDiscoverResponse(
            company=detail,
            imported_document=_document_detail(repo, document.id),
            cik=match.cik,
            accession_number=filing.accession_number,
        )

    @api.post("/companies", response_model=Company, status_code=201)
    def create_company(payload: CompanyCreate) -> Company:
        try:
            return repo.create_company(payload)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @api.get("/companies/{company_id}", response_model=CompanyDetail)
    def get_company(company_id: UUID) -> CompanyDetail:
        company = repo.get_company(company_id)
        if not company:
            raise HTTPException(status_code=404, detail="Company not found")
        documents = repo.list_documents(company_id)
        return CompanyDetail(
            **company.model_dump(),
            document_count=len(documents),
            ready_document_count=sum(1 for document in documents if document.status == DocumentStatus.ready),
            documents=documents,
        )

    @api.post("/companies/{company_id}/documents", response_model=DocumentDetail, status_code=201)
    async def upload_document(company_id: UUID, request: Request) -> DocumentDetail:
        if not repo.get_company(company_id):
            raise HTTPException(status_code=404, detail="Company not found")
        form = await _parse_document_upload(request)
        payload = DocumentCreate(
            title=_required_field(form, "title"),
            document_type=DocumentType(_required_field(form, "document_type")),
            source_url=form.get("source_url") or None,
            filing_date=_optional_date(form.get("filing_date")),
            period_end_date=_optional_date(form.get("period_end_date")),
            fiscal_year=_optional_int(form.get("fiscal_year")),
            fiscal_quarter=_optional_int(form.get("fiscal_quarter")),
        )
        file_info = form.get("file")
        if not isinstance(file_info, dict):
            raise HTTPException(status_code=422, detail="Upload requires a file field")
        stored_path = _store_upload(settings.raw_dir, company_id, file_info)
        document = repo.create_document(company_id, payload, str(stored_path))
        ingestion.ingest_document(document.id)
        return _document_detail(repo, document.id)

    @api.get("/documents/{document_id}", response_model=DocumentDetail)
    def get_document(document_id: UUID) -> DocumentDetail:
        return _document_detail(repo, document_id)

    @api.get("/documents/{document_id}/chunks")
    def list_document_chunks(document_id: UUID, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        if not repo.get_document(document_id):
            raise HTTPException(status_code=404, detail="Document not found")
        chunks = repo.list_chunks(document_id=document_id)
        return {
            "document_id": document_id,
            "total": len(chunks),
            "offset": offset,
            "limit": limit,
            "items": chunks[offset : offset + limit],
        }

    @api.delete("/documents/{document_id}", status_code=204)
    def delete_document(document_id: UUID) -> Response:
        document = repo.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        deleted = repo.delete_document(document_id)
        if document.file_path:
            Path(document.file_path).unlink(missing_ok=True)
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found")
        return Response(status_code=204)

    @api.get("/conversations", response_model=list[Conversation])
    def list_conversations(limit: int = 50) -> list[Conversation]:
        if limit < 1 or limit > 200:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 200")
        return repo.list_conversations()[:limit]

    @api.get("/conversations/{conversation_id}", response_model=ConversationDetail)
    def get_conversation(conversation_id: UUID) -> ConversationDetail:
        conversation = repo.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        citations_by_message = {
            message.id: repo.list_citations(message.id)
            for message in repo.list_messages(conversation_id)
        }
        messages = [
            MessageDetail(**message.model_dump(), citations=citations_by_message.get(message.id, []))
            for message in repo.list_messages(conversation_id)
        ]
        return ConversationDetail(**conversation.model_dump(), messages=messages)

    @api.post("/research/retrieve", response_model=RetrievalResponse)
    def research_retrieve(payload: RetrievalRequest) -> RetrievalResponse:
        for company_id in payload.company_ids:
            if not repo.get_company(company_id):
                raise HTTPException(status_code=404, detail=f"Company '{company_id}' not found")
        results = retrieval.search(
            payload.query,
            company_ids=payload.company_ids,
            top_k=payload.top_k,
            document_types=payload.document_types,
            fiscal_years=payload.fiscal_years,
            min_score=payload.min_score,
        )
        return RetrievalResponse(
            query=payload.query,
            total=len(results),
            results=[_retrieved_context(result) for result in results],
        )

    @api.post("/research/chat", response_model=ChatResponse)
    def research_chat(payload: ChatRequest) -> ChatResponse:
        try:
            return research.chat(payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @api.post("/research/memo", response_model=ResearchMemo)
    def research_memo(payload: MemoRequest) -> ResearchMemo:
        try:
            return research.memo(payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @api.post("/research/compare", response_model=CompareResponse)
    def research_compare(payload: CompareRequest) -> CompareResponse:
        try:
            return research.compare(payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return api


async def _parse_document_upload(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "")
    body = await request.body()
    if content_type.startswith("application/json"):
        payload = await request.json()
        text = payload.pop("text", None)
        filename = payload.pop("filename", "manual-note.txt")
        if text is not None:
            payload["file"] = {
                "filename": filename,
                "content_type": "text/plain",
                "content": str(text).encode("utf-8"),
            }
        return payload
    if "multipart/form-data" not in content_type:
        raise HTTPException(status_code=415, detail="Use multipart/form-data or application/json")
    boundary_match = re.search(r'boundary="?([^";]+)"?', content_type)
    if not boundary_match:
        raise HTTPException(status_code=400, detail="Missing multipart boundary")
    boundary = boundary_match.group(1).encode("utf-8")
    return _parse_multipart(body, boundary)


def _parse_multipart(body: bytes, boundary: bytes) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    delimiter = b"--" + boundary
    for raw_part in body.split(delimiter):
        part = raw_part.strip()
        if not part or part == b"--":
            continue
        if part.endswith(b"--"):
            part = part[:-2].strip()
        header_blob, sep, content = part.partition(b"\r\n\r\n")
        if not sep:
            header_blob, sep, content = part.partition(b"\n\n")
        if not sep:
            continue
        headers = header_blob.decode("utf-8", errors="replace")
        disposition = next((line for line in headers.splitlines() if line.lower().startswith("content-disposition")), "")
        name_match = re.search(r'name="([^"]+)"', disposition)
        if not name_match:
            continue
        name = name_match.group(1)
        filename_match = re.search(r'filename="([^"]*)"', disposition)
        content = content.rstrip(b"\r\n")
        if filename_match:
            content_type = "application/octet-stream"
            for line in headers.splitlines():
                if line.lower().startswith("content-type"):
                    content_type = line.split(":", 1)[1].strip()
            parsed[name] = {
                "filename": Path(filename_match.group(1)).name or "upload.bin",
                "content_type": content_type,
                "content": content,
            }
        else:
            parsed[name] = content.decode("utf-8", errors="replace")
    return parsed


def _store_upload(raw_dir: Path, company_id: UUID, file_info: dict[str, Any]) -> Path:
    filename = Path(str(file_info["filename"])).name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".txt", ".md", ".text", ".pdf"}:
        raise HTTPException(status_code=422, detail="Only .txt, .md, and .pdf uploads are supported")
    company_dir = raw_dir / str(company_id)
    company_dir.mkdir(parents=True, exist_ok=True)
    destination = company_dir / filename
    counter = 1
    while destination.exists():
        destination = company_dir / f"{Path(filename).stem}-{counter}{suffix}"
        counter += 1
    destination.write_bytes(file_info["content"])
    return destination


def _document_detail(repo: JsonRepository, document_id: UUID) -> DocumentDetail:
    document = repo.get_document(document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentDetail(**document.model_dump(), chunk_count=len(repo.list_chunks(document_id=document_id)))


def _retrieved_context(result: Any) -> RetrievedContext:
    text = " ".join(result.chunk.text.split())
    excerpt = text[:500]
    if len(text) > 500:
        excerpt += "..."
    return RetrievedContext(
        query=result.query,
        chunk_id=result.chunk.id,
        document_id=result.document.id,
        company_id=result.company.id,
        ticker=result.company.ticker,
        company_name=result.company.name,
        document_title=result.document.title,
        document_type=result.document.document_type,
        fiscal_year=result.document.fiscal_year,
        fiscal_quarter=result.document.fiscal_quarter,
        page_start=result.chunk.page_start,
        page_end=result.chunk.page_end,
        section_title=result.chunk.section_title,
        excerpt=excerpt,
        token_count=result.chunk.token_count,
        score=result.score,
        keyword_score=result.keyword_score,
        vector_score=result.vector_score,
        metadata=result.chunk.metadata,
    )


def _required_field(form: dict[str, Any], key: str) -> str:
    value = form.get(key)
    if value is None or str(value).strip() == "":
        raise HTTPException(status_code=422, detail=f"Missing required field '{key}'")
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    return date.fromisoformat(str(value))


def copy_seed_file(source: Path, raw_dir: Path, company_id: UUID) -> Path:
    destination_dir = raw_dir / str(company_id)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    if not destination.exists():
        shutil.copyfile(source, destination)
    return destination


app = create_app()
