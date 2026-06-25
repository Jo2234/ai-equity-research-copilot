from __future__ import annotations

from pathlib import Path

from ai_equity_research_copilot_backend.config import get_settings
from ai_equity_research_copilot_backend.embeddings import HashingEmbedder
from ai_equity_research_copilot_backend.retrieval import RetrievalService
from ai_equity_research_copilot_backend.schemas import CompanyCreate, DocumentCreate, DocumentType
from ai_equity_research_copilot_backend.services import IngestionService
from ai_equity_research_copilot_backend.storage import JsonRepository


def test_ingestion_stores_embeddings_and_retrieval_filters(tmp_path: Path) -> None:
    settings = get_settings(tmp_path)
    repo = JsonRepository(settings.state_file)
    embedder = HashingEmbedder(settings.embedding_dimensions)
    ingestion = IngestionService(repo, embedder, settings)
    retrieval = RetrievalService(repo, embedder, min_score=0.0)

    nvda = repo.create_company(CompanyCreate(ticker="nvda", name="NVIDIA Corporation"))
    msft = repo.create_company(CompanyCreate(ticker="msft", name="Microsoft Corporation"))
    nvda_file = tmp_path / "nvda.txt"
    msft_file = tmp_path / "msft.txt"
    nvda_file.write_text(
        "MARGIN ANALYSIS\n\nGross margin expanded because product mix shifted toward accelerated computing platforms.",
        encoding="utf-8",
    )
    msft_file.write_text(
        "RISK FACTORS\n\nCloud capacity execution and cybersecurity incidents could pressure Azure growth.",
        encoding="utf-8",
    )
    nvda_doc = repo.create_document(
        nvda.id,
        DocumentCreate(title="NVIDIA Test 10-K", document_type=DocumentType.ten_k, fiscal_year=2025),
        str(nvda_file),
    )
    msft_doc = repo.create_document(
        msft.id,
        DocumentCreate(title="Microsoft Test 10-K", document_type=DocumentType.ten_k, fiscal_year=2025),
        str(msft_file),
    )

    ingestion.ingest_document(nvda_doc.id)
    ingestion.ingest_document(msft_doc.id)

    nvda_chunks = repo.list_chunks(document_id=nvda_doc.id)
    assert nvda_chunks
    assert len(nvda_chunks[0].embedding) == settings.embedding_dimensions
    assert repo.get_document(nvda_doc.id).status == "ready"

    results = retrieval.search("What drove gross margin expansion?", [nvda.id], top_k=3)
    assert results
    assert all(result.company.id == nvda.id for result in results)
    assert "gross margin" in results[0].chunk.text.lower()

    filtered = retrieval.search("cybersecurity cloud capacity", [nvda.id], top_k=3)
    assert all(result.document.id != msft_doc.id for result in filtered)
