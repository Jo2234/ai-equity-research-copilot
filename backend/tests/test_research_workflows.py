from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from ai_equity_research_copilot_backend.config import get_settings
from ai_equity_research_copilot_backend.embeddings import HashingEmbedder
from ai_equity_research_copilot_backend.main import create_app
from ai_equity_research_copilot_backend.retrieval import RetrievalService
from ai_equity_research_copilot_backend.schemas import CompanyCreate, DocumentChunk, DocumentCreate, DocumentStatus, DocumentType
from ai_equity_research_copilot_backend.services import IngestionService
from ai_equity_research_copilot_backend.storage import JsonRepository


FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def _services(tmp_path: Path) -> tuple[JsonRepository, IngestionService, RetrievalService]:
    settings = get_settings(tmp_path)
    repo = JsonRepository(settings.state_file)
    embedder = HashingEmbedder(settings.embedding_dimensions)
    ingestion = IngestionService(repo, embedder, settings)
    retrieval = RetrievalService(repo, embedder, min_score=0.0)
    return repo, ingestion, retrieval


def test_ingestion_marks_empty_documents_failed_and_clears_stale_chunks(tmp_path: Path) -> None:
    repo, ingestion, _ = _services(tmp_path)
    embedder = HashingEmbedder(get_settings(tmp_path).embedding_dimensions)
    company = repo.create_company(CompanyCreate(ticker="NVDA", name="NVIDIA Corporation"))
    source = FIXTURE_DIR / "documents" / "empty_upload.txt"
    document = repo.create_document(
        company.id,
        DocumentCreate(title="Empty upload", document_type=DocumentType.manual_note),
        str(source),
    )
    repo.replace_chunks(
        document.id,
        [
            DocumentChunk(
                document_id=document.id,
                company_id=company.id,
                chunk_index=0,
                text="stale chunk that should be removed on failed ingestion",
                embedding=embedder.embed("stale chunk"),
                token_count=8,
            )
        ],
    )

    ingestion.ingest_document(document.id)

    failed_document = repo.get_document(document.id)
    assert failed_document is not None
    assert failed_document.status == DocumentStatus.failed
    assert failed_document.parse_error == "No text chunks were extracted"
    assert repo.list_chunks(document_id=document.id) == []


def test_retrieval_applies_document_type_year_and_ready_status_filters(tmp_path: Path) -> None:
    repo, ingestion, retrieval = _services(tmp_path)
    company = repo.create_company(CompanyCreate(ticker="NVDA", name="NVIDIA Corporation"))
    embedder = HashingEmbedder(get_settings(tmp_path).embedding_dimensions)

    ten_k_file = tmp_path / "nvda-10k.txt"
    ten_k_file.write_text((FIXTURE_DIR / "documents" / "multi_source_note.txt").read_text(encoding="utf-8"), encoding="utf-8")
    transcript_file = tmp_path / "nvda-transcript.txt"
    transcript_file.write_text(
        "PREPARED REMARKS\n\nManagement said Blackwell demand exceeded supply and customers were deploying systems for training and inference.",
        encoding="utf-8",
    )
    ten_k = repo.create_document(
        company.id,
        DocumentCreate(title="NVIDIA FY2025 10-K", document_type=DocumentType.ten_k, fiscal_year=2025),
        str(ten_k_file),
    )
    transcript = repo.create_document(
        company.id,
        DocumentCreate(
            title="NVIDIA FY2026 Q1 Transcript",
            document_type=DocumentType.earnings_transcript,
            fiscal_year=2026,
            fiscal_quarter=1,
        ),
        str(transcript_file),
    )
    draft = repo.create_document(
        company.id,
        DocumentCreate(title="Draft note", document_type=DocumentType.manual_note, fiscal_year=2026),
        str(tmp_path / "draft.txt"),
    )
    repo.replace_chunks(
        draft.id,
        [
            DocumentChunk(
                document_id=draft.id,
                company_id=company.id,
                chunk_index=0,
                text="Blackwell demand exceeded supply in this draft note, but the document is not ready.",
                embedding=embedder.embed("Blackwell demand exceeded supply in this draft note"),
                token_count=14,
            )
        ],
    )

    ingestion.ingest_document(ten_k.id)
    ingestion.ingest_document(transcript.id)

    results = retrieval.search(
        "What did management say about Blackwell demand and supply?",
        [company.id],
        top_k=5,
        document_types=[DocumentType.earnings_transcript],
        fiscal_years=[2026],
    )

    assert results
    assert {result.document.id for result in results} == {transcript.id}
    assert all(result.document.document_type == DocumentType.earnings_transcript for result in results)
    assert all(result.document.fiscal_year == 2026 for result in results)
    assert draft.id not in {result.document.id for result in results}


def test_chat_citations_include_page_ranges_truncated_excerpts_and_persisted_audit(tmp_path: Path) -> None:
    app = create_app(tmp_path, seed=False)
    client = TestClient(app)
    repo: JsonRepository = app.state.repo
    settings = get_settings(tmp_path)
    embedder = HashingEmbedder(settings.embedding_dimensions)

    company = repo.create_company(CompanyCreate(ticker="NVDA", name="NVIDIA Corporation"))
    document = repo.create_document(
        company.id,
        DocumentCreate(title="Q4 FY2025 Earnings Call Transcript", document_type=DocumentType.earnings_transcript, fiscal_year=2025),
        str(tmp_path / "transcript.txt"),
    )
    document.status = DocumentStatus.ready
    repo.update_document(document)
    supported_text = (
        "Gross margin expanded because Blackwell platform demand exceeded supply and data center customers deployed larger "
        "accelerated computing clusters for training and inference. Pricing remained favorable while networking attach rates "
        "increased as clusters scaled. Management also said transition costs and supply chain expenses could create quarter-to-quarter "
        "gross margin volatility, so analysts should verify platform ramp timing against later filings and earnings call commentary. "
        "This deliberately long excerpt keeps going so the citation drawer receives a truncated evidence snippet instead of a full chunk."
    )
    repo.replace_chunks(
        document.id,
        [
            DocumentChunk(
                document_id=document.id,
                company_id=company.id,
                chunk_index=0,
                text=supported_text,
                embedding=embedder.embed(supported_text),
                page_start=6,
                page_end=7,
                section_title="Prepared Remarks",
                token_count=95,
            )
        ],
    )

    response = client.post(
        "/research/chat",
        json={
            "company_ids": [str(company.id)],
            "question": "What drove gross margin expansion and what risk should I verify?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"].endswith("[1]")
    assert payload["citations"][0]["label"] == "NVDA Q4 FY2025 Earnings Call Transcript, pp. 6-7"
    assert payload["citations"][0]["excerpt"].endswith("...")
    assert len(payload["citations"][0]["excerpt"]) <= 423
    assert payload["citations"][0]["title"] == "Q4 FY2025 Earnings Call Transcript"
    assert payload["usage"]["retrieval"] == {"retrieved_chunks": 1, "cited_chunks": 1}

    state = json.loads(settings.state_file.read_text(encoding="utf-8"))
    assert [message["role"] for message in state["messages"]] == ["user", "assistant"]
    assert state["citations"][0]["message_id"] == payload["message_id"]
    assert state["citations"][0]["citation_label"] == payload["citations"][0]["label"]


def test_json_upload_and_api_validation_paths(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path, seed=False))
    company_response = client.post("/companies", json={"ticker": "AAPL", "name": "Apple Inc."})
    assert company_response.status_code == 201
    company_id = company_response.json()["id"]

    upload_response = client.post(
        f"/companies/{company_id}/documents",
        json={
            "title": "Apple Services Note",
            "document_type": "manual_note",
            "filename": "services-note.md",
            "text": "SERVICES\n\nServices revenue increased because subscriptions and cloud services grew across the installed base.",
            "filing_date": "2026-01-31",
            "fiscal_year": 2026,
            "fiscal_quarter": 1,
            "source_url": "https://example.com/apple-services-note",
        },
    )
    assert upload_response.status_code == 201
    document = upload_response.json()
    assert document["status"] == "ready"
    assert document["chunk_count"] == 1
    assert document["fiscal_quarter"] == 1

    company_detail = client.get(f"/companies/{company_id}").json()
    assert company_detail["document_count"] == 1
    assert company_detail["ready_document_count"] == 1

    unsupported_upload = client.post(
        f"/companies/{company_id}/documents",
        json={
            "title": "Bad CSV",
            "document_type": "manual_note",
            "filename": "bad.csv",
            "text": "Ticker,Revenue\nAAPL,100",
        },
    )
    assert unsupported_upload.status_code == 422
    assert "Only .txt, .md, and .pdf uploads are supported" in unsupported_upload.json()["detail"]

    compare_validation = client.post(
        "/research/compare",
        json={"company_ids": [company_id], "question": "Compare growth drivers."},
    )
    assert compare_validation.status_code == 422
