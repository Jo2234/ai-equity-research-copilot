# AI Equity Research Copilot

Document-grounded equity research assistant for public US companies. The app is designed around company profiles, uploaded filings/transcripts, retrieval over parsed document chunks, cited Q&A, company comparisons, and structured research memos.

This is research assistance software, not investment advice. It must not issue buy, sell, hold, or price-target recommendations unless the user supplies explicit valuation assumptions.

## Why This Matters

- Starts from company filings and uploaded source documents instead of market commentary.
- Makes retrieval visible through chunk-level citations, source metadata, and fallback behavior when evidence is weak.
- Includes a finance QA eval set so answer quality, citation precision, unsupported-question refusal, and hallucination risk can be tested instead of hand-waved.

## Quick Proof

- Corpus builder can discover a ticker through SEC EDGAR, fetch recent `10-K`, `10-Q`, `8-K`, and `DEF 14A` filings, convert them into chunks, and add them to the RAG corpus.
- Chat synthesis retrieves evidence first, then uses local Gemma/Ollama only when available; invalid or uncited model output falls back to deterministic cited synthesis.
- The eval suite has 35 finance QA cases covering factual extraction, multi-document synthesis, company comparison, and unsupported questions.
- Synthetic fixtures include companies, documents, chunks, chat responses, memo responses, and comparison responses for deterministic review.

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

The app can move beyond the seeded local company list and build a filings-first company corpus:

1. Search by ticker or company name in the sidebar.
2. Local matches filter immediately.
3. Press `SEC` to query SEC EDGAR's public company ticker mapping.
4. Choose `Build corpus` to create the company, fetch SEC filing metadata, download filings, convert them to text, chunk them, embed them, and add them to the RAG corpus.

Default corpus import:

- latest `10-K`
- latest 4 `10-Q` filings
- latest 6 `8-K` filings
- latest `DEF 14A` proxy statement where available

Backend endpoints:

```bash
curl "http://localhost:8001/companies/search?q=AAPL"

curl -X POST "http://localhost:8001/companies/discover" \
  -H "Content-Type: application/json" \
  -d '{"query":"AAPL","build_corpus":true}'
```

SEC access uses official public EDGAR data endpoints and requires a declared user agent. Set this before running the API if you want your own contact string in requests:

```bash
export AIERC_SEC_USER_AGENT="Your Name your-email@example.com"
```

Uploaded PDFs/text files are still supported for transcripts, investor decks, and custom notes.

## Local LLM Synthesis

The chat endpoint always retrieves evidence first. It then tries to synthesize the answer with a local Ollama model when available. The recommended default is a small 3-5B class Gemma model:

```bash
ollama pull gemma3:4b
export AIERC_LLM_PROVIDER=auto
export AIERC_OLLAMA_MODEL=gemma3:4b
export AIERC_OLLAMA_TIMEOUT_SECONDS=180
```

Provider modes:

- `AIERC_LLM_PROVIDER=auto`: use Ollama/Gemma if available, otherwise deterministic cited synthesis.
- `AIERC_LLM_PROVIDER=ollama`: require Ollama/Gemma and return a clear low-confidence response if unavailable.
- `AIERC_LLM_PROVIDER=local`: deterministic cited synthesis only.

The Ollama prompt is constrained to the retrieved filing excerpts and must return structured JSON with citation indices. If the model returns invalid or uncited output, the app falls back to deterministic synthesis.

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
