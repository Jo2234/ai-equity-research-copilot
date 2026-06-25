# API Contract

This contract mirrors the project spec and the fixture payloads in `tests/fixtures/api`.

## `GET /companies/search`

Searches the local workspace first and then SEC EDGAR's company ticker mapping.

Request:

```text
GET /companies/search?q=AAPL&limit=8
```

Response:

```json
[
  {
    "ticker": "AAPL",
    "name": "Apple Inc.",
    "cik": 320193,
    "source": "sec",
    "local_company_id": null,
    "already_in_workspace": false
  }
]
```

## `POST /companies/discover`

Creates or reuses a company, fetches supported SEC filings, stores them as local text, and runs the ingestion pipeline so they become available to retrieval.

Request:

```json
{
  "query": "AAPL",
  "build_corpus": true,
  "annual_limit": 1,
  "quarterly_limit": 4,
  "current_report_limit": 6,
  "proxy_limit": 1
}
```

Response:

```json
{
  "company": {
    "ticker": "AAPL",
    "name": "Apple Inc.",
    "document_count": 1,
    "ready_document_count": 1,
    "documents": []
  },
  "imported_document": {
    "title": "AAPL 10-K filed 2025-10-31",
    "document_type": "10-k",
    "status": "ready",
    "chunk_count": 64
  },
  "imported_documents": [],
  "source": "sec",
  "cik": 320193,
  "accession_number": "0000320193-25-000079",
  "accession_numbers": ["0000320193-25-000079"],
  "reused_existing_count": 0
}
```

Default automated corpus imports: latest `10-k`, recent `10-q`, recent `8-k`, and recent `DEF 14A` where available.

## `POST /research/chat`

Request:

```json
{
  "conversation_id": null,
  "company_ids": ["11111111-1111-4111-8111-111111111111"],
  "question": "What drove NVIDIA data center revenue growth?",
  "document_types": ["10-k", "earnings_transcript"],
  "fiscal_years": [2025],
  "top_k": 8
}
```

Response:

```json
{
  "message_id": "90000000-0000-4000-8000-000000000001",
  "answer": "NVIDIA data center revenue growth was driven by compute demand for generative AI and accelerated computing, plus networking demand tied to large-scale AI clusters.",
  "key_points": [
    "Data center growth was tied to accelerated computing and generative AI demand.",
    "Networking demand increased as customers built larger AI clusters."
  ],
  "citations": [
    {
      "label": "NVIDIA FY2025 10-K, p. 42",
      "document_id": "21111111-1111-4111-8111-111111111111",
      "chunk_id": "31111111-1111-4111-8111-111111111111",
      "excerpt": "Data center revenue increased primarily from demand for compute products used in generative AI and accelerated computing workloads.",
      "score": 0.91
    }
  ],
  "confidence": "high",
  "limitations": [],
  "usage": {
    "model": "gemma3:4b",
    "latency_ms": 1420,
    "input_tokens": 1800,
    "output_tokens": 220,
    "estimated_cost_usd": 0.0042,
    "provider": "ollama",
    "retrieval": {
      "top_k": 8,
      "returned_chunks": 2
    }
  }
}
```

Validation expectations:

- `company_ids` is required and cannot be empty.
- `question` is required and should be non-empty after trimming.
- `top_k` defaults to `8` and should be bounded.
- Citations must reference chunks returned by retrieval.
- Unsupported answers should include low confidence and a limitation explaining missing context.

## `POST /research/memo`

Request:

```json
{
  "company_id": "11111111-1111-4111-8111-111111111111",
  "document_types": ["10-k", "earnings_transcript"],
  "fiscal_years": [2025],
  "top_k": 10
}
```

Response fields:

- `company`
- `business_summary`
- `recent_performance`
- `growth_drivers`
- `margin_analysis`
- `capital_allocation`
- `risk_factors`
- `management_commentary`
- `bull_case`
- `bear_case`
- `open_questions`
- `source_citations`
- `limitations`

## `POST /research/compare`

Request:

```json
{
  "company_ids": [
    "11111111-1111-4111-8111-111111111111",
    "22222222-2222-4222-8222-222222222222"
  ],
  "question": "Compare NVIDIA and Apple on growth drivers and margin risks.",
  "document_types": ["10-k"],
  "fiscal_years": [2025],
  "top_k_per_company": 5
}
```

Response should separate cited claims by company:

```json
{
  "question": "Compare NVIDIA and Apple on growth drivers and margin risks.",
  "comparisons": [
    {
      "company": {
        "ticker": "NVDA",
        "name": "NVIDIA Corporation"
      },
      "summary": "NVIDIA's growth drivers are tied to AI infrastructure demand.",
      "key_points": ["AI infrastructure demand supported data center revenue growth."],
      "citations": []
    },
    {
      "company": {
        "ticker": "AAPL",
        "name": "Apple Inc."
      },
      "summary": "Apple's growth drivers are tied to services expansion and installed base engagement.",
      "key_points": ["Services growth was supported by the installed base and subscriptions."],
      "citations": []
    }
  ],
  "limitations": [],
  "usage": {
    "model": "local-deterministic-grounded-v1",
    "latency_ms": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "estimated_cost_usd": 0.0,
    "provider": "local",
    "retrieval": {}
  }
}
```
