# Worker 04 Integration Notes

## Scope

- Reviewed `project-specs/01_ai_equity_research_copilot_spec.md`.
- Inspected the current `projects/ai-equity-research-copilot` implementation tree.
- Made one small compatibility fix so the existing Vite entrypoint resolves.

## Current Implementation Snapshot

The project currently contains a minimal frontend scaffold:

- `frontend/package.json`
- `frontend/index.html`
- `frontend/vite.config.ts`
- `frontend/tsconfig.json`
- `frontend/tsconfig.node.json`
- `frontend/.env.example`
- `frontend/src/setupTests.ts`
- `frontend/src/main.tsx`

The backend, evals, docs, shared tests, fixtures, and data sample directories are present but have no implementation files at review time.

## Spec Validation

### Present

- Frontend build tooling exists via Vite, React dependencies, TypeScript config, and Vitest config.
- API proxy is configured for `/api` to `VITE_API_BASE_URL` with a localhost fallback.
- Frontend test setup includes Testing Library Jest DOM matchers.
- `.env.example` documents the frontend API base URL.

### Missing Against MVP Scope

- No backend FastAPI application, API routes, Pydantic schemas, or service layer.
- No database schema or persistence layer for companies, documents, chunks, conversations, messages, citations, eval cases, or eval runs.
- No PostgreSQL/pgvector or Docker Compose setup.
- No Redis/background worker ingestion pipeline.
- No PDF/text parsing, chunking, embedding, vector search, retrieval filtering, reranking, or citation metadata pipeline.
- No LLM wrapper, grounded prompting, structured output validation, token/cost/latency tracking, or request metadata persistence.
- No company CRUD, document upload/status/chunk APIs, chat, memo, comparison, or eval endpoints.
- No dense analyst workstation UI for company/document selection, research chat, citation drawer, memo generation, comparison, retrieval debug, or evaluation summary.
- No citation click-through behavior or source excerpt drawer.
- No eval dataset with the required 30 finance QA examples.
- No backend, frontend component, integration, ingestion, retrieval, citation formatting, or API behavior tests.
- No README-level setup instructions, seed script, sample documents, or deployment instructions.

### Compatibility Fix Applied

- Added `frontend/src/main.tsx` because `frontend/index.html` imports `/src/main.tsx` and the file was missing.
- The added file is intentionally a small scaffold placeholder, not a product implementation.

## Remaining Gaps And Risks

- The implementation is far from MVP compliance; most required product, backend, RAG, eval, and testing surfaces are not implemented.
- The current frontend scaffold can only validate tooling basics, not spec behavior.
- The project uses Vite rather than the recommended Next.js stack. This may be acceptable if the team chooses Vite intentionally, but it should be documented.
- Frontend dependencies may require an install step before checks can run locally because no lockfile or `node_modules` state is guaranteed.
- No compatibility fixes were possible for backend behavior because there is no backend code to patch.

## Suggested Next Integration Checks

Run these after the relevant workers add implementation files:

```bash
cd projects/ai-equity-research-copilot/frontend
npm install
npm run build
npm test
```

```bash
cd projects/ai-equity-research-copilot/backend
pytest
```

```bash
cd projects/ai-equity-research-copilot
docker compose up --build
```

## Agent 03 Eval/Data/Docs Update

Scope respected: only evals, sample data, docs, README, and integration notes were changed.

Note: the earlier Worker 04 snapshot above appears to predate the current backend, frontend, eval, Docker, and fixture files now present in the tree. Treat this Agent 03 section as the current note for eval/data/docs scope only.

Added sample corpus coverage for the canonical eval companies that were missing local text excerpts:

- Apple FY2025 10-K and Q4 FY2025 transcript excerpts.
- JPMorgan Chase FY2025 10-K and Q4 FY2025 transcript excerpts.
- Exxon Mobil FY2025 10-K and Q4 FY2025 transcript excerpts.
- Tesla FY2025 10-K and Q4 FY2025 transcript excerpts.

Added `data/sample_documents/manifest.json` to describe the synthetic corpus and map applicable sample files to canonical eval document IDs.

Added eval support artifacts:

- `evals/coverage_matrix.md` for category, company, and workflow coverage.
- `evals/scoring_rubric.json` for metrics, quality gates, and hard-fail rules.
- `evals/runbook.md` for manual and automated eval execution.

Documentation updates:

- Expanded `evals/README.md` with supporting artifacts, grounded-answer expectations, and validation command.
- Expanded `docs/evaluation.md` with dataset counts, sample corpus scope, harness routing, hard-fail rules, and smoke targets.
- Updated `README.md` with eval support links, sample corpus description, and validation commands.
