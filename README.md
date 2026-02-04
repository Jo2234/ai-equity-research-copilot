# AI Equity Research Copilot

Document-grounded equity research assistant for public US companies. The app is designed around company profiles, uploaded filings/transcripts, retrieval over parsed document chunks, cited Q&A, company comparisons, and structured research memos.

This is research assistance software, not investment advice. It must not issue buy, sell, hold, or price-target recommendations unless the user supplies explicit valuation assumptions.

## Current Scope

- Backend API: FastAPI service under `backend/`.
- Frontend: Vite React TypeScript app under `frontend/`.
- Data services: current backend uses local JSON storage; Docker Compose also starts PostgreSQL with pgvector and Redis for the target architecture.
- Evals: curated finance QA cases under `evals/`.
- Fixtures: deterministic sample companies, documents, chunks, and API payloads under `tests/fixtures/`.

## API Contract Summary

The backend should expose these MVP endpoints:

- `GET /companies`
- `GET /companies/search?q=AAPL`
- `POST /companies`
- `POST /companies/discover`
- `GET /companies/{company_id}`
- `POST /companies/{company_id}/documents`
- `GET /documents/{document_id}`
- `GET /documents/{document_id}/chunks`
- `DELETE /documents/{document_id}`
- `POST /research/chat`
- `POST /research/memo`
- `POST /research/compare`

See [docs/api-contract.md](/Users/johanvaz/Documents/Portfolio/projects/ai-equity-research-copilot/docs/api-contract.md) for request and response shapes aligned with the project spec.

## SEC Company Discovery

The app can now move beyond the seeded local company list:

1. Search by ticker or company name in the sidebar.
2. Local matches filter immediately.
3. Press `SEC` to query SEC EDGAR's public company ticker mapping.
4. Choose `Import 10-K` to create the company, fetch the latest SEC filing metadata, download the primary filing document, convert it to text, chunk it, embed it, and add it to the RAG corpus.

Backend endpoints:

```bash
curl "http://localhost:8001/companies/search?q=AAPL"

curl -X POST "http://localhost:8001/companies/discover" \
  -H "Content-Type: application/json" \
  -d '{"query":"AAPL","form_type":"10-k"}'
```

SEC access uses official public EDGAR data endpoints and requires a declared user agent. Set this before running the API if you want your own contact string in requests:

```bash
export AIERC_SEC_USER_AGENT="Your Name your-email@example.com"
```

Currently supported automated imports are latest `10-k`, `10-q`, and `8-k` filings. Uploaded PDFs/text files are still supported for transcripts, investor decks, and custom notes.

## Local Setup

1. Copy the environment template:

```bash
cp .env.example .env
```

2. Fill in provider credentials as needed. Local tests and fixture-based eval dry runs should not require live LLM credentials.

3. Start local infrastructure:

```bash
docker compose up postgres redis
```

4. Run the full stack:

```bash
docker compose up --build
```

Expected local URLs:

- Web: `http://localhost:3000`
- API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- Postgres: `localhost:5432`
- Redis: `localhost:6379`

## Eval Dataset

The MVP eval set lives in [evals/finance_qa_v1.jsonl](/Users/johanvaz/Documents/Portfolio/projects/ai-equity-research-copilot/evals/finance_qa_v1.jsonl). It includes 35 cases:

- 12 single-document factual questions
- 10 multi-document synthesis questions
- 6 company comparison questions
- 7 unsupported or not-enough-information questions

Each case includes `id`, `company_ids`, `question`, `required_source_documents`, `expected_answer_points`, citation rules or acceptable chunk IDs, `must_not_include`, and `difficulty`.

Supporting eval docs:

- [evals/coverage_matrix.md](/Users/johanvaz/Documents/Portfolio/projects/ai-equity-research-copilot/evals/coverage_matrix.md)
- [evals/runbook.md](/Users/johanvaz/Documents/Portfolio/projects/ai-equity-research-copilot/evals/runbook.md)
- [evals/scoring_rubric.json](/Users/johanvaz/Documents/Portfolio/projects/ai-equity-research-copilot/evals/scoring_rubric.json)

Recommended MVP pass criteria:

- Answer accuracy: at least 80%
- Factual answers with citations: at least 90%
- Citation precision: at least 80%
- Severe hallucinations: 0 in smoke tests
- Median indexed-document Q&A latency: under 8 seconds

## Fixture Data

Fixtures and sample data are synthetic and intentionally small:

- [tests/fixtures/companies.json](/Users/johanvaz/Documents/Portfolio/projects/ai-equity-research-copilot/tests/fixtures/companies.json)
- [tests/fixtures/documents.json](/Users/johanvaz/Documents/Portfolio/projects/ai-equity-research-copilot/tests/fixtures/documents.json)
- [tests/fixtures/chunks/retrieval_chunks.json](/Users/johanvaz/Documents/Portfolio/projects/ai-equity-research-copilot/tests/fixtures/chunks/retrieval_chunks.json)
- [tests/fixtures/api/chat_request.json](/Users/johanvaz/Documents/Portfolio/projects/ai-equity-research-copilot/tests/fixtures/api/chat_request.json)
- [tests/fixtures/api/chat_response.json](/Users/johanvaz/Documents/Portfolio/projects/ai-equity-research-copilot/tests/fixtures/api/chat_response.json)
- [tests/fixtures/api/memo_response.json](/Users/johanvaz/Documents/Portfolio/projects/ai-equity-research-copilot/tests/fixtures/api/memo_response.json)
- [tests/fixtures/api/compare_response.json](/Users/johanvaz/Documents/Portfolio/projects/ai-equity-research-copilot/tests/fixtures/api/compare_response.json)
- [data/sample_documents/manifest.json](/Users/johanvaz/Documents/Portfolio/projects/ai-equity-research-copilot/data/sample_documents/manifest.json)

Use these fixtures for backend API tests, retrieval scoring smoke tests, and frontend contract mocks.

The sample document corpus covers NVDA, MSFT, AAPL, JPM, XOM, and TSLA with synthetic 10-K and earnings transcript excerpts. The canonical eval set uses NVDA, AAPL, JPM, XOM, and TSLA fixture IDs; MSFT remains useful for local seed/demo data.

## Useful Commands

Validate JSON fixtures:

```bash
python -m json.tool tests/fixtures/companies.json >/dev/null
python -m json.tool tests/fixtures/documents.json >/dev/null
python -m json.tool tests/fixtures/chunks/retrieval_chunks.json >/dev/null
```

Validate eval JSONL:

```bash
python - <<'PY'
import json
from pathlib import Path
path = Path("evals/finance_qa_v1.jsonl")
for line_no, line in enumerate(path.read_text().splitlines(), 1):
    if line.strip():
        json.loads(line)
print(f"valid jsonl: {path}")
PY
```

Validate eval support artifacts:

```bash
python - <<'PY'
import json
from pathlib import Path

for path in [
    Path("evals/scoring_rubric.json"),
    Path("data/sample_documents/manifest.json"),
    Path("data/evals/finance_qa_examples.json"),
]:
    json.loads(path.read_text())

print("valid eval support artifacts")
PY
```

Run backend tests:

```bash
cd backend
pytest
```

Run frontend checks:

```bash
cd frontend
npm run build
npm test
```

## Implementation Notes

- Keep raw uploaded files outside the public web directory.
- Store citation metadata at chunk level: company, document, page range, section title, excerpt, and score.
- All factual answer claims must map back to cited retrieved chunks.
- Refuse unsupported questions instead of answering from model memory.
- Log request metadata, retrieval IDs, model name, latency, token counts, and estimated cost, but not API keys or raw secrets.
