from __future__ import annotations

import html
import json
import re
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import Settings
from .schemas import CompanyCreate, DocumentCreate, DocumentType


COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

SUPPORTED_FORMS = {
    "10-K": DocumentType.ten_k,
    "10-K/A": DocumentType.ten_k,
    "10-Q": DocumentType.ten_q,
    "10-Q/A": DocumentType.ten_q,
    "8-K": DocumentType.eight_k,
    "8-K/A": DocumentType.eight_k,
}


@dataclass(frozen=True)
class SecCompanyMatch:
    ticker: str
    name: str
    cik: int
    source: str = "sec"

    @property
    def padded_cik(self) -> str:
        return str(self.cik).zfill(10)

    def to_payload(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "name": self.name,
            "cik": self.cik,
            "source": self.source,
        }


@dataclass(frozen=True)
class SecFiling:
    accession_number: str
    filing_date: date
    report_date: date | None
    form: str
    primary_document: str
    source_url: str


class SecEdgarClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._company_cache: list[SecCompanyMatch] | None = None
        self._last_request_at = 0.0

    def search_companies(self, query: str, limit: int = 8) -> list[SecCompanyMatch]:
        needle = query.strip().lower()
        if not needle:
            return []
        companies = self._load_companies()
        ranked: list[tuple[int, SecCompanyMatch]] = []
        for company in companies:
            ticker = company.ticker.lower()
            name = company.name.lower()
            score = 0
            if ticker == needle:
                score = 100
            elif ticker.startswith(needle):
                score = 80
            elif needle in ticker:
                score = 60
            elif name.startswith(needle):
                score = 50
            elif needle in name:
                score = 35
            if score:
                ranked.append((score, company))
        ranked.sort(key=lambda item: (-item[0], item[1].ticker))
        return [company for _, company in ranked[:limit]]

    def latest_filing(self, cik: int, forms: list[str]) -> SecFiling:
        wanted = {form.upper() for form in forms}
        payload = self._get_json(SUBMISSIONS_URL.format(cik=str(cik).zfill(10)))
        recent = payload.get("filings", {}).get("recent", {})
        rows = _transpose_recent_filings(recent)
        for row in rows:
            form = str(row.get("form") or "").upper()
            primary_document = str(row.get("primaryDocument") or "")
            accession_number = str(row.get("accessionNumber") or "")
            filing_date = _parse_date(row.get("filingDate"))
            if form in wanted and accession_number and primary_document and filing_date:
                source_url = ARCHIVE_URL.format(
                    cik=int(cik),
                    accession=accession_number.replace("-", ""),
                    document=primary_document,
                )
                report_date = _parse_date(row.get("reportDate"))
                return SecFiling(
                    accession_number=accession_number,
                    filing_date=filing_date,
                    report_date=report_date,
                    form=form,
                    primary_document=primary_document,
                    source_url=source_url,
                )
        raise LookupError(f"No recent {'/'.join(forms)} filing found for CIK {cik}")

    def download_filing_text(self, filing: SecFiling) -> str:
        raw = self._get_text(filing.source_url)
        text = html_to_text(raw)
        if len(text) > self.settings.sec_max_filing_chars:
            text = text[: self.settings.sec_max_filing_chars] + "\n\n[Truncated for local demo ingestion.]"
        return text

    def company_create_payload(self, match: SecCompanyMatch) -> CompanyCreate:
        return CompanyCreate(
            ticker=match.ticker,
            name=match.name.title() if match.name.isupper() else match.name,
            exchange=None,
            sector=None,
            industry=f"SEC CIK {match.cik}",
        )

    def document_create_payload(self, match: SecCompanyMatch, filing: SecFiling) -> DocumentCreate:
        document_type = SUPPORTED_FORMS.get(filing.form, DocumentType.other)
        fiscal_year = filing.report_date.year if filing.report_date else filing.filing_date.year
        return DocumentCreate(
            title=f"{match.ticker} {filing.form} filed {filing.filing_date.isoformat()}",
            document_type=document_type,
            source_url=filing.source_url,
            filing_date=filing.filing_date,
            period_end_date=filing.report_date,
            fiscal_year=fiscal_year,
        )

    def _load_companies(self) -> list[SecCompanyMatch]:
        if self._company_cache is not None:
            return self._company_cache
        payload = self._get_json(COMPANY_TICKERS_URL)
        companies = [
            SecCompanyMatch(
                ticker=str(row["ticker"]).upper(),
                name=str(row["title"]),
                cik=int(row["cik_str"]),
            )
            for row in payload.values()
            if row.get("ticker") and row.get("title") and row.get("cik_str")
        ]
        self._company_cache = companies
        return companies

    def _get_json(self, url: str) -> dict[str, Any]:
        return json.loads(self._get_text(url))

    def _get_text(self, url: str) -> str:
        self._throttle()
        request = Request(
            url,
            headers={
                "User-Agent": self.settings.sec_user_agent,
                "Accept-Encoding": "identity",
                "Accept": "application/json,text/html,text/plain,*/*",
            },
        )
        try:
            with urlopen(request, timeout=self.settings.sec_timeout_seconds) as response:
                return response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            raise RuntimeError(f"SEC request failed with HTTP {exc.code}: {url}") from exc
        except URLError as exc:
            raise RuntimeError(f"SEC request failed: {exc.reason}") from exc

    def _throttle(self) -> None:
        elapsed = time.perf_counter() - self._last_request_at
        if elapsed < 0.12:
            time.sleep(0.12 - elapsed)
        self._last_request_at = time.perf_counter()


def save_sec_filing_text(raw_dir: Path, company_id: Any, filing: SecFiling, text: str) -> Path:
    company_dir = raw_dir / str(company_id)
    company_dir.mkdir(parents=True, exist_ok=True)
    filename = f"sec-{filing.form.lower().replace('/', '-')}-{filing.accession_number}.txt"
    destination = company_dir / filename
    destination.write_text(text, encoding="utf-8")
    return destination


def html_to_text(raw: str) -> str:
    text = re.sub(r"(?is)<(script|style|ix:header).*?</\1>", " ", raw)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</(p|div|tr|table|section|article|h[1-6])>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def _transpose_recent_filings(recent: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not recent:
        return []
    keys = list(recent)
    length = max((len(recent[key]) for key in keys), default=0)
    return [
        {key: recent[key][idx] if idx < len(recent[key]) else None for key in keys}
        for idx in range(length)
    ]


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None
