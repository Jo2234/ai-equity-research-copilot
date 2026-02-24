# Architecture Notes

## System Shape

The copilot is a retrieval-augmented research workflow with four main surfaces:

- Web app: company/document selection, chat, memo generation, comparison workspace, citation drawer.
- API: company/document CRUD, upload validation, retrieval, chat, memo, comparison, eval run endpoints.
- Worker: PDF/text parsing, chunking, embedding generation, ingestion status updates.
- Data layer: PostgreSQL with pgvector, Redis for jobs, local or object storage for raw files.

## Request Flow: Grounded Q&A

1. Client sends `POST /research/chat` with `company_ids`, `question`, optional filters, and `top_k`.
2. API validates company scope. Missing company context should return `422`.
3. Retrieval service embeds the query and searches chunks filtered by company, document type, fiscal year, and date.
4. API builds a grounded prompt from retrieved chunks only.
5. LLM returns a structured answer with citation references to chunk IDs.
6. API validates citation references against retrieved chunks.
7. API stores message, citations, retrieval metadata, usage, latency, and estimated cost.
8. Client renders answer and clickable citations.

## Request Flow: Ingestion

1. Client uploads a PDF or text document to `POST /companies/{company_id}/documents`.
2. API validates extension, MIME type, size, document type, and company ID.
3. API stores the raw file outside the public web directory and creates a `documents` row with `uploaded` status.
4. Worker extracts text and page metadata.
5. Worker chunks text into 600 to 1,000 token chunks with overlap.
6. Worker generates embeddings and stores chunk rows with page range, section title, token count, and metadata.
7. Worker marks the document `ready` or `failed` with a useful parse error.

## Data Contracts

Minimum tables:

- `companies`
- `documents`
- `document_chunks`
- `conversations`
- `messages`
- `citations`
- `eval_cases`
- `eval_runs`

Chunk metadata should be sufficient to render:

```text
NVIDIA FY2025 10-K, p. 42, Risk Factors
```

## Reliability Rules

- Retrieved context is the source of truth.
- Unsupported questions should be refused with a short explanation.
- Factual claims need citations.
- Company comparisons must label the company behind each claim.
- Model, latency, token counts, estimated cost, and retrieval IDs should be persisted for audit.
- Logs should include metadata, not secrets or raw API keys.

## Eval Loop

The eval harness should:

1. Load `evals/finance_qa_v1.jsonl`.
2. Seed or resolve fixture companies/documents/chunks.
3. Call `POST /research/chat` or `POST /research/compare`.
4. Score expected answer points, citation precision, citation recall, refusal correctness, unsupported claim count, latency, and cost.
5. Emit a JSON and Markdown report.

