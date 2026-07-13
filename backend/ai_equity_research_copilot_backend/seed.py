from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid5

from .config import Settings
from .schemas import CompanyCreate, DocumentCreate, DocumentType
from .storage import JsonRepository


SEED_NAMESPACE = UUID("8c88f4bc-24e3-4f56-9a12-81d774c21ff5")


def seed_repository(repo: JsonRepository, ingestion, settings: Settings) -> None:
    if repo.list_companies():
        return
    sample_dir = settings.seed_dir
    companies = [
        CompanyCreate(
            ticker="NVDA",
            name="NVIDIA Corporation",
            exchange="NASDAQ",
            sector="Information Technology",
            industry="Semiconductors",
        ),
        CompanyCreate(
            ticker="MSFT",
            name="Microsoft Corporation",
            exchange="NASDAQ",
            sector="Information Technology",
            industry="Software and Cloud Infrastructure",
        ),
    ]
    created = {
        company.ticker: repo.upsert_company(
            company,
            company_id=uuid5(SEED_NAMESPACE, f"company:{company.ticker}"),
        )
        for company in companies
    }
    documents = [
        (
            "NVDA",
            sample_dir / "nvda_fy2025_10k_excerpt.txt",
            DocumentCreate(
                title="FY2025 10-K Excerpt",
                document_type=DocumentType.ten_k,
                fiscal_year=2025,
                filing_date="2025-02-26",
            ),
        ),
        (
            "NVDA",
            sample_dir / "nvda_q1_fy2026_transcript_excerpt.txt",
            DocumentCreate(
                title="Q1 FY2026 Earnings Transcript Excerpt",
                document_type=DocumentType.earnings_transcript,
                fiscal_year=2026,
                fiscal_quarter=1,
                filing_date="2025-05-28",
            ),
        ),
        (
            "MSFT",
            sample_dir / "msft_fy2025_10k_excerpt.txt",
            DocumentCreate(
                title="FY2025 10-K Excerpt",
                document_type=DocumentType.ten_k,
                fiscal_year=2025,
                filing_date="2025-07-30",
            ),
        ),
        (
            "MSFT",
            sample_dir / "msft_q3_fy2026_transcript_excerpt.txt",
            DocumentCreate(
                title="Q3 FY2026 Earnings Transcript Excerpt",
                document_type=DocumentType.earnings_transcript,
                fiscal_year=2026,
                fiscal_quarter=3,
                filing_date="2026-04-24",
            ),
        ),
    ]
    for ticker, source, payload in documents:
        if not source.exists():
            continue
        company = created[ticker]
        if settings.demo_mode:
            document_path = source
        else:
            document_path = settings.raw_dir / str(company.id) / source.name
            document_path.parent.mkdir(parents=True, exist_ok=True)
            if not document_path.exists():
                document_path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        document = repo.create_document(
            company.id,
            payload,
            str(document_path),
            document_id=uuid5(SEED_NAMESPACE, f"document:{ticker}:{source.name}"),
        )
        ingestion.ingest_document(document.id)
