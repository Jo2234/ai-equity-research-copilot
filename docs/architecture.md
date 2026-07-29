# Implemented Architecture

## Components

- **Web app:** Vite, React, and TypeScript under `frontend/`. It provides company discovery, document uploads, cited chat, comparisons, and research memos.
- **API:** FastAPI, created by `ai_equity_research_copilot_backend.main:create_app` under `backend/`. It constructs the repository, ingestion, retrieval, research, and SEC clients and seeds the bundled synthetic documents on startup unless seeding is disabled.
- **Storage:** `JsonRepository` in `storage.py` persists a JSON file, while source files live on disk. Docker Compose runs the API and web services and mounts `./data` into the API at `/app/data`.
- **Synthesis:** deterministic cited sentence selection, with optional local Ollama synthesis for chat. Embeddings always use the local `HashingEmbedder`.

There is no database server, vector database, job queue, or separate ingestion worker in the current application.

## Storage and Concurrency

`AIERC_DATA_DIR` defaults to the repository's `data/` directory. The repository stores six collections in `storage/state/store.json`: `companies`, `documents`, `chunks`, `conversations`, `messages`, and `citations`. Chunk records include their embedding, text, company/document IDs, page range, section title, token count, and source metadata.

Uploaded files and downloaded SEC filing text are stored under `storage/raw/<company_id>/`, outside the public web directory. `AIERC_SEED_DIR` selects the bundled seed corpus independently of the writable data directory.

Repository mutations read the state, update it under an instance-local `RLock`, and replace the JSON file through a temporary file. This is intended for a small local workspace: the lock does not coordinate separate API processes, and the store is rewritten on mutation. Multiple writers sharing the same directory are not a supported database-style deployment.

## Ingestion

1. `POST /companies/{company_id}/documents` validates the company, metadata, extension, MIME type, and upload size, then stores the source file and creates a document record.
2. The handler calls `IngestionService.ingest_document` before returning. It marks the record `processing`, extracts text, chunks it, and generates embeddings in the API process.
3. Text and Markdown files are read as text. PDFs use PyMuPDF, falling back to PyPDF2; the pipeline does not perform OCR.
4. Chunking preserves page and section metadata, with a default target of 800 estimated tokens and 80 tokens of overlap. `HashingEmbedder` creates normalized 128-dimensional vectors from token hashes by default; it does not call an embedding model service.
5. The service replaces the document's chunks and marks it `ready`. A parsing or ingestion failure clears its chunks, marks it `failed`, and records `parse_error`.

`POST /companies/discover` queries SEC EDGAR, creates or reuses a company, downloads filing text, and calls the same ingestion service. Corpus mode defaults to one annual filing, four quarterly filings, six current reports, and one proxy statement when available. Existing documents are reused by source URL. These downloads and ingestion steps complete within the request; there is no durable background job or retry queue.

## Retrieval and Grounded Chat

1. `POST /research/chat` accepts company IDs or ticker symbols, a question, optional document-type/fiscal-year filters, and `top_k`. The API resolves tickers and rejects unknown companies.
2. Retrieval loads one request-scoped corpus snapshot, selecting ready documents in the requested companies and preparing each chunk's term set once. Memo and comparison workflows also reuse a snapshot within their request; later requests reload storage.
3. The query is embedded locally. Eligible chunks are scanned and ranked by `0.75 × nonnegative cosine similarity + 0.25 × query-term overlap`. The default minimum score is `0.04`; results are sorted and limited to `top_k`. There is no approximate-nearest-neighbor index or date-range filter.
4. Chat synthesis receives the retrieved evidence. `AIERC_LLM_PROVIDER=local` uses deterministic cited synthesis; `auto` attempts Ollama and falls back when unavailable. `ollama` requires a responsive server and returns an explicit unavailable response if it cannot respond.
5. Ollama output must satisfy the structured schema and citation-number checks. Invalid or missing references fall back to deterministic synthesis, including in `ollama` mode. Those checks verify source membership and numbering, not whether every claim is entailed by the source.
6. Deterministic synthesis keeps each selected sentence associated with its supporting chunk and returns a low-confidence/refusal response when evidence is insufficient. Memos use deterministic section synthesis and label generic bull/bear prompts as analyst scenarios.
7. Chat persists user and assistant messages, structured answer content, citations, retrieval IDs/scores, provider/model, latency, estimated token counts, and estimated cost. The response exposes the cited excerpts and retrieval details for inspection. Memo and comparison responses are returned without creating conversation records.

`POST /research/retrieve` exposes retrieval results independently of answer synthesis. Conversation endpoints expose the saved chat history and citations. The frontend renders citation labels, source metadata, and excerpts alongside answers.

## Demo Boundaries

`AIERC_DEMO_MODE` disables company/document mutations and SEC discovery in the API. `VITE_PUBLIC_DEMO=true` selects the frontend's read-only presentation of the live seeded API. `VITE_BROWSER_DEMO=true` instead selects a separate synthetic browser adapter that sends no API requests. Neither mode turns failed live requests into fabricated successful responses.

## Evaluation and Limits

The repository keeps the 35 synthetic QA cases in `evals/finance_qa_v1.jsonl`, their scoring rubric, fixtures, and a documented evaluation runbook. `scripts/eval_smoke.py` validates dataset metadata and supporting JSON files; it does not run the full quality benchmark. `scripts/demo_smoke.py` exercises deterministic seeded research workflows and writes a JSON artifact. Backend tests cover ingestion, retrieval, citation integrity, and research/API behavior; frontend tests cover the UI contracts.

There is no eval-run API endpoint or persisted `eval_runs` collection. The quality metrics and pass criteria in [evaluation.md](evaluation.md) and [the runbook](../evals/runbook.md) describe the evaluation procedure, not a claim that a fresh model benchmark has passed.

The local hashed retrieval, synchronous ingestion, JSON persistence, and citation-membership validation define the current limits. Keep those limits visible when interpreting research output or sizing a deployment.
