from __future__ import annotations

import math
import re
import time
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI(title="AI Equity Research Copilot", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

COMPANIES: dict[str, dict[str, Any]] = {}
DOCUMENTS: dict[str, dict[str, Any]] = {}
CHUNKS: dict[str, dict[str, Any]] = {}
CONVERSATIONS: dict[str, dict[str, Any]] = {}
MESSAGES: dict[str, dict[str, Any]] = {}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-']+")
DOC_TYPES = {"10-k", "10-q", "8-k", "earnings_transcript", "investor_presentation", "annual_report", "manual_note", "other"}


class CompanyCreate(BaseModel):
    ticker: str = Field(min_length=1)
    name: str
    exchange: str | None = None
    sector: str | None = None
    industry: str | None = None


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    company_ids: list[str]
    question: str
    document_types: list[str] | None = None
    fiscal_years: list[int] | None = None
    top_k: int = Field(default=8, ge=1, le=20)


class MemoRequest(BaseModel):
    company_id: str
    top_k: int = 12


class CompareRequest(BaseModel):
    company_ids: list[str]
    question: str
    top_k: int = 10


def now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def vectorize(text: str) -> Counter[str]:
    return Counter(tokens(text))


def cosine(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    denom = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / denom if denom else 0.0


def parse_file(raw: bytes, filename: str) -> str:
    if filename.lower().endswith(".pdf"):
        try:
            import fitz  # type: ignore

            doc = fitz.open(stream=raw, filetype="pdf")
            return "\n".join(page.get_text() for page in doc)
        except Exception:
            return raw.decode("utf-8", errors="ignore")
    return raw.decode("utf-8", errors="ignore")


def chunk_text(text: str, target_words: int = 160) -> list[dict[str, Any]]:
    words = text.split()
    chunks: list[dict[str, Any]] = []
    for i in range(0, max(len(words), 1), target_words):
        body = " ".join(words[i : i + target_words]).strip()
        if not body:
            continue
        heading = body.split(".")[0][:90]
        chunks.append({"text": body, "section_title": heading, "page_start": 1 + i // 450, "page_end": 1 + i // 450})
    return chunks


def add_company(ticker: str, name: str, **extra: Any) -> str:
    cid = str(uuid4())
    COMPANIES[cid] = {"id": cid, "ticker": ticker.upper(), "name": name, "created_at": now(), "updated_at": now(), **extra}
    return cid


def add_document(company_id: str, title: str, document_type: str, text: str, fiscal_year: int | None = None) -> str:
    did = str(uuid4())
    DOCUMENTS[did] = {
        "id": did,
        "company_id": company_id,
        "title": title,
        "document_type": document_type,
        "source_url": None,
        "file_path": None,
        "filing_date": str(date.today()),
        "period_end_date": None,
        "fiscal_year": fiscal_year,
        "fiscal_quarter": None,
        "status": "ready",
        "parse_error": None,
        "created_at": now(),
        "updated_at": now(),
    }
    for idx, chunk in enumerate(chunk_text(text)):
        chunk_id = str(uuid4())
        CHUNKS[chunk_id] = {
            "id": chunk_id,
            "document_id": did,
            "company_id": company_id,
            "chunk_index": idx,
            "text": chunk["text"],
            "embedding": dict(vectorize(chunk["text"])),
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "section_title": chunk["section_title"],
            "token_count": len(tokens(chunk["text"])),
            "metadata": {},
            "created_at": now(),
        }
    return did


def seed() -> None:
    if COMPANIES:
        return
    nvda = add_company("NVDA", "NVIDIA Corporation", exchange="NASDAQ", sector="Technology", industry="Semiconductors")
    msft = add_company("MSFT", "Microsoft Corporation", exchange="NASDAQ", sector="Technology", industry="Software")
    add_document(
        nvda,
        "NVIDIA FY2025 10-K excerpt",
        "10-k",
        "Business overview. NVIDIA designs accelerated computing platforms. Data Center revenue growth was driven by demand for accelerated computing and generative AI. Gross margin expanded due to favorable product mix, higher Data Center scale, and supply chain execution. Risk factors include intense competition, export controls, supply constraints, customer concentration, and cyclicality in gaming demand. Management commentary emphasized AI factory demand, networking growth, and software opportunities.",
        fiscal_year=2025,
    )
    add_document(
        nvda,
        "NVIDIA Q1 FY2026 earnings transcript excerpt",
        "earnings_transcript",
        "Management discussion. The quarter benefited from hyperscale and enterprise demand for AI infrastructure. Networking and systems revenue contributed to growth. Management noted supply remains tight for some products and export controls may limit shipments to certain regions. Operating expenses rose as NVIDIA invested in research and go-to-market capacity.",
        fiscal_year=2026,
    )
    add_document(
        msft,
        "Microsoft FY2025 10-K excerpt",
        "10-k",
        "Business overview. Microsoft generates revenue from Productivity and Business Processes, Intelligent Cloud, and More Personal Computing. Azure growth was supported by cloud migration and AI services. Risks include cybersecurity threats, competition, data center capacity constraints, regulatory scrutiny, and foreign currency movements. Management commentary highlighted durable commercial cloud demand and disciplined operating expense growth.",
        fiscal_year=2025,
    )


def search_chunks(question: str, company_ids: list[str], top_k: int, document_types: list[str] | None = None, fiscal_years: list[int] | None = None) -> list[dict[str, Any]]:
    qv = vectorize(question)
    hits: list[dict[str, Any]] = []
    for chunk in CHUNKS.values():
        if chunk["company_id"] not in company_ids:
            continue
        doc = DOCUMENTS[chunk["document_id"]]
        if document_types and doc["document_type"] not in document_types:
            continue
        if fiscal_years and doc.get("fiscal_year") not in fiscal_years:
            continue
        score = cosine(qv, Counter(chunk["embedding"]))
        if score > 0:
            hits.append({**chunk, "score": round(score, 4), "document": doc, "company": COMPANIES[chunk["company_id"]]})
    return sorted(hits, key=lambda h: h["score"], reverse=True)[:top_k]


def citation(hit: dict[str, Any], idx: int) -> dict[str, Any]:
    company = hit["company"]
    doc = hit["document"]
    page = hit.get("page_start") or "n/a"
    return {
        "label": f"[{idx}] {company['ticker']} {doc['title']}, p. {page}",
        "document_id": doc["id"],
        "chunk_id": hit["id"],
        "excerpt": hit["text"][:320],
        "score": hit["score"],
        "section_title": hit.get("section_title"),
    }


def grounded_answer(question: str, hits: list[dict[str, Any]]) -> tuple[str, list[str], str, list[str]]:
    if not hits:
        return (
            "I do not have enough cited context to answer that. Upload relevant company documents or broaden the filters.",
            [],
            "low",
            ["No retrieved document chunks met the query and filter criteria."],
        )
    key_points = []
    for idx, hit in enumerate(hits[:4], start=1):
        sentence = re.split(r"(?<=[.!?])\s+", hit["text"])[0]
        key_points.append(f"{sentence} [{idx}]")
    answer = " ".join(key_points)
    if "price target" in question.lower() or "buy" in question.lower():
        answer += " I cannot provide a buy/sell recommendation or price target from this document-grounded workflow."
    return answer, key_points, "medium" if len(hits) < 3 else "high", []


seed()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/companies")
def companies(search: str | None = None) -> list[dict[str, Any]]:
    rows = list(COMPANIES.values())
    if search:
        s = search.lower()
        rows = [r for r in rows if s in r["ticker"].lower() or s in r["name"].lower()]
    return rows


@app.post("/companies", status_code=201)
def create_company(payload: CompanyCreate) -> dict[str, Any]:
    ticker = payload.ticker.upper()
    if any(c["ticker"] == ticker for c in COMPANIES.values()):
        raise HTTPException(409, "Ticker already exists")
    cid = add_company(ticker, payload.name, exchange=payload.exchange, sector=payload.sector, industry=payload.industry)
    return COMPANIES[cid]


@app.get("/companies/{company_id}")
def company_detail(company_id: str) -> dict[str, Any]:
    if company_id not in COMPANIES:
        raise HTTPException(404, "Company not found")
    docs = [d for d in DOCUMENTS.values() if d["company_id"] == company_id]
    return {**COMPANIES[company_id], "documents": docs, "document_count": len(docs)}


@app.post("/companies/{company_id}/documents", status_code=201)
async def upload_document(
    company_id: str,
    file: UploadFile = File(...),
    title: str = Form(...),
    document_type: str = Form("other"),
    filing_date: str | None = Form(None),
    fiscal_year: int | None = Form(None),
    fiscal_quarter: int | None = Form(None),
    source_url: str | None = Form(None),
) -> dict[str, Any]:
    if company_id not in COMPANIES:
        raise HTTPException(404, "Company not found")
    if document_type not in DOC_TYPES:
        raise HTTPException(422, "Unsupported document_type")
    raw = await file.read()
    path = UPLOAD_DIR / f"{uuid4()}_{file.filename}"
    path.write_bytes(raw)
    text = parse_file(raw, file.filename or "")
    did = add_document(company_id, title, document_type, text, fiscal_year=fiscal_year)
    DOCUMENTS[did].update({"file_path": str(path), "filing_date": filing_date, "fiscal_quarter": fiscal_quarter, "source_url": source_url})
    return DOCUMENTS[did]


@app.get("/documents/{document_id}")
def document_detail(document_id: str) -> dict[str, Any]:
    if document_id not in DOCUMENTS:
        raise HTTPException(404, "Document not found")
    count = sum(1 for c in CHUNKS.values() if c["document_id"] == document_id)
    return {**DOCUMENTS[document_id], "chunk_count": count}


@app.get("/documents/{document_id}/chunks")
def document_chunks(document_id: str) -> list[dict[str, Any]]:
    if document_id not in DOCUMENTS:
        raise HTTPException(404, "Document not found")
    return [c for c in CHUNKS.values() if c["document_id"] == document_id]


@app.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, Any]:
    if document_id not in DOCUMENTS:
        raise HTTPException(404, "Document not found")
    for cid in [c["id"] for c in CHUNKS.values() if c["document_id"] == document_id]:
        CHUNKS.pop(cid, None)
    DOCUMENTS.pop(document_id)
    return {"deleted": document_id}


@app.post("/research/chat")
def research_chat(payload: ChatRequest) -> dict[str, Any]:
    if not payload.company_ids:
        raise HTTPException(422, "company_ids is required")
    missing = [cid for cid in payload.company_ids if cid not in COMPANIES]
    if missing:
        raise HTTPException(404, f"Unknown company ids: {missing}")
    start = time.perf_counter()
    hits = search_chunks(payload.question, payload.company_ids, payload.top_k, payload.document_types, payload.fiscal_years)
    answer, key_points, confidence, limitations = grounded_answer(payload.question, hits)
    citations = [citation(hit, i) for i, hit in enumerate(hits[:6], start=1)]
    conversation_id = payload.conversation_id or str(uuid4())
    CONVERSATIONS.setdefault(conversation_id, {"id": conversation_id, "title": payload.question[:60], "created_at": now(), "updated_at": now()})
    message_id = str(uuid4())
    latency = int((time.perf_counter() - start) * 1000)
    MESSAGES[message_id] = {"id": message_id, "conversation_id": conversation_id, "role": "assistant", "content": answer, "created_at": now()}
    return {
        "message_id": message_id,
        "conversation_id": conversation_id,
        "answer": answer,
        "structured_payload": {"answer": answer, "key_points": key_points, "citations": citations, "confidence": confidence, "limitations": limitations},
        "citations": citations,
        "retrieval_debug": [{"chunk_id": h["id"], "score": h["score"], "document": h["document"]["title"]} for h in hits],
        "usage": {"model": "deterministic-local-retrieval-v1", "latency_ms": latency, "input_tokens": len(tokens(payload.question)), "output_tokens": len(tokens(answer)), "estimated_cost_usd": 0.0},
    }


@app.post("/research/memo")
def research_memo(payload: MemoRequest) -> dict[str, Any]:
    if payload.company_id not in COMPANIES:
        raise HTTPException(404, "Company not found")
    company = COMPANIES[payload.company_id]
    hits = search_chunks("business performance revenue growth margin risks management commentary", [payload.company_id], payload.top_k)
    cites = [citation(h, i) for i, h in enumerate(hits[:8], start=1)]
    return {
        "company": {"ticker": company["ticker"], "name": company["name"]},
        "business_summary": " ".join(h["text"].split(".")[0] for h in hits[:2]) or "No cited business summary available.",
        "recent_performance": [h["text"].split(".")[0] for h in hits[:2]],
        "growth_drivers": [h["text"].split(".")[0] for h in hits if "growth" in h["text"].lower()][:3],
        "margin_analysis": [h["text"].split(".")[0] for h in hits if "margin" in h["text"].lower()][:3],
        "capital_allocation": ["No capital allocation detail was present in the retrieved context."],
        "risk_factors": [h["text"].split(".")[0] for h in hits if "risk" in h["text"].lower()][:3],
        "management_commentary": [h["text"].split(".")[0] for h in hits if "management" in h["text"].lower()][:3],
        "bull_case": ["Potential upside depends on the cited growth drivers continuing."],
        "bear_case": ["Potential downside depends on the cited risk factors materializing."],
        "open_questions": ["Upload full filings to validate segment-level metrics and capital allocation details."],
        "source_citations": cites,
        "limitations": ["Deterministic local memo; not investment advice."],
    }


@app.post("/research/compare")
def compare(payload: CompareRequest) -> dict[str, Any]:
    if len(payload.company_ids) < 2:
        raise HTTPException(422, "At least two companies are required")
    hits = search_chunks(payload.question, payload.company_ids, payload.top_k)
    rows = []
    for company_id in payload.company_ids:
        company_hits = [h for h in hits if h["company_id"] == company_id]
        rows.append({
            "company": COMPANIES[company_id]["ticker"],
            "answer": " ".join(h["text"].split(".")[0] for h in company_hits[:3]) or "No cited context retrieved.",
            "citations": [citation(h, i + 1) for i, h in enumerate(company_hits[:3])],
        })
    return {"question": payload.question, "comparison": rows}
