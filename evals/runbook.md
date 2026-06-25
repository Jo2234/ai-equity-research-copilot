# Eval Runbook

## Purpose

Use this runbook to evaluate whether the copilot answers equity research questions using only retrieved document context, keeps company claims separated, refuses unsupported requests, and returns usable citations.

## Inputs

- `finance_qa_v1.jsonl`: canonical eval cases.
- `schema.json`: JSON Schema for each eval case.
- `scoring_rubric.json`: metric definitions and quality gates.
- `../data/sample_documents/manifest.json`: synthetic sample corpus metadata.

## Local Validation

From `projects/ai-equity-research-copilot`:

```bash
python - <<'PY'
import json
from pathlib import Path

paths = [
    Path("evals/scoring_rubric.json"),
    Path("data/sample_documents/manifest.json"),
    Path("data/evals/finance_qa_examples.json"),
]
for path in paths:
    json.loads(path.read_text())

for line_no, line in enumerate(Path("evals/finance_qa_v1.jsonl").read_text().splitlines(), 1):
    if line.strip():
        json.loads(line)

print("eval artifacts are valid JSON/JSONL")
PY
```

## Manual Smoke Procedure

1. Seed or upload the sample documents.
2. Run at least one case from each category: single-document, multi-document, comparison, and insufficient-context.
3. Confirm factual claims cite source chunks.
4. Confirm unsupported cases do not use model memory.
5. Confirm no answer includes a buy, sell, hold, outperform, underperform, or price-target recommendation.
6. Record latency, model name, token counts, and estimated cost for each case.

## Automated Harness Expectations

The eventual harness should:

- Load every eval case and validate it against `schema.json`.
- Route single-company cases to `POST /research/chat`.
- Route multi-company comparison cases to `POST /research/compare`.
- Score answer points against `expected_answer_points`.
- Score citations against `acceptable_citation_chunk_ids` and `citation_rules`.
- Apply the hard-fail rules from `scoring_rubric.json`.
- Emit both JSON and Markdown reports with pass/fail status by case.

## Pass Criteria

Use the `mvp_smoke` quality gate in `scoring_rubric.json` before demo. Use `portfolio_demo` when the app is being shown as a polished project.
