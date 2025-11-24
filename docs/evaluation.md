# Evaluation Plan

## Dataset

`evals/finance_qa_v1.jsonl` is the MVP dataset. It is synthetic but modeled on common equity research tasks:

- Single-document factual extraction
- Multi-document synthesis
- Company comparison
- Correct refusal when context is insufficient

The dataset currently has 35 cases:

| Category | Cases | What it tests |
| --- | ---: | --- |
| `single_document` | 12 | Direct extraction from one filing or transcript chunk |
| `multi_document` | 10 | Synthesis across filing and earnings-call context |
| `comparison` | 6 | Company-specific attribution across two or three companies |
| `insufficient_context` | 7 | Refusal behavior for forecasts, market data, valuation, and missing legal context |

The companion coverage map is `evals/coverage_matrix.md`. The scoring definitions and quality gates are in `evals/scoring_rubric.json`.

## Sample Corpus

`data/sample_documents` contains synthetic excerpts for local ingestion and fixture-style demos. These are not official filings or transcripts.

The corpus covers:

- NVIDIA: FY2025 10-K excerpt and Q1 FY2026 transcript excerpt.
- Microsoft: FY2025 10-K excerpt and Q3 FY2026 transcript excerpt.
- Apple: FY2025 10-K excerpt and Q4 FY2025 transcript excerpt.
- JPMorgan Chase: FY2025 10-K excerpt and Q4 FY2025 transcript excerpt.
- Exxon Mobil: FY2025 10-K excerpt and Q4 FY2025 transcript excerpt.
- Tesla: FY2025 10-K excerpt and Q4 FY2025 transcript excerpt.

See `data/sample_documents/manifest.json` for document metadata and canonical eval document IDs where applicable.

## Required Scoring Fields

Each eval run should report:

- `answer_accuracy`
- `citation_precision`
- `citation_recall`
- `unsupported_claim_count`
- `refusal_correctness`
- `latency_ms`
- `estimated_cost_usd`
- `format_compliance`

## Suggested Scoring Rules

Answer accuracy:

- 1.0: all expected answer points are present with no material unsupported claims.
- 0.5: at least half of expected answer points are present and no severe hallucination is present.
- 0.0: answer misses the core point or includes a severe hallucination.

Citation precision:

- Cited chunks that support the answer divided by all cited chunks.

Citation recall:

- Expected citation rules or acceptable chunk IDs satisfied divided by expected citation requirements.

Refusal correctness:

- For unsupported cases, answer should state that available context is insufficient and avoid external facts.
- For supported cases, answer should not refuse when required context is present.

Hard failures:

- Any buy, sell, hold, outperform, underperform, or price-target recommendation without user-supplied valuation assumptions.
- Any exact future forecast not present in retrieved context.
- Any comparison answer that does not separate claims by company.
- Any factual answer with no citation when relevant context is available.

## Harness Routing

An automated harness should route cases by category:

- `single_document`, `multi_document`, and most `insufficient_context` cases: `POST /research/chat`.
- `comparison` cases: `POST /research/compare`.
- Future memo cases: `POST /research/memo` with section-level scoring.

For each case, persist:

- Input payload and filters.
- Retrieved chunk IDs and scores.
- Structured model response.
- Citation labels, document IDs, chunk IDs, and excerpts.
- Metric scores and hard-fail reasons.

## Smoke Test Gate

Before demo:

1. Run all eval cases.
2. Inspect every failed hard case manually.
3. Confirm no answer gives investment advice.
4. Confirm citations open to relevant excerpts in the UI.
5. Confirm median latency for indexed documents is under 8 seconds.

Minimum MVP smoke targets:

- Answer accuracy at or above 80%.
- Citation precision at or above 80%.
- Citation recall at or above 75%.
- Unsupported claim count equal to 0.
- Correct refusal on every insufficient-context case.
- Structured output format compliance at or above 95%.
