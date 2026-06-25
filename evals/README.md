# Evals

`finance_qa_v1.jsonl` contains the MVP finance QA eval set for the AI Equity Research Copilot.

Supporting files:

- `schema.json`: strict JSON Schema for each JSONL row.
- `coverage_matrix.md`: human-readable coverage map by category, company, and workflow.
- `scoring_rubric.json`: metric definitions, quality gates, and hard-fail rules.
- `runbook.md`: manual and automated eval execution guidance.

## Coverage

- 35 total cases
- 12 single-document factual cases
- 10 multi-document synthesis cases
- 6 company comparison cases
- 7 unsupported or insufficient-context cases

## Case Format

Each JSONL row follows `schema.json` and includes:

- `id`
- `category`
- `company_ids`
- `question`
- `required_source_documents`
- `expected_answer_points`
- `acceptable_citation_chunk_ids`
- `citation_rules`
- `must_not_include`
- `difficulty`

The IDs map to fixtures under `tests/fixtures`.

## Expected Behavior

The evals enforce the spec's grounded-answer rules:

- Use retrieved context as source of truth.
- Cite factual claims with source document and chunk metadata.
- Keep company-specific claims separated in comparisons.
- Refuse exact forecasts, market data, price targets, and investment recommendations when source context is insufficient.
- Distinguish fact from interpretation in hard synthesis cases.

## Validation

```bash
python - <<'PY'
import json
from pathlib import Path

for line_no, line in enumerate(Path("evals/finance_qa_v1.jsonl").read_text().splitlines(), 1):
    if line.strip():
        json.loads(line)
print("eval jsonl is valid")
PY
```

Validate all JSON/JSONL eval artifacts:

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

for line_no, line in enumerate(Path("evals/finance_qa_v1.jsonl").read_text().splitlines(), 1):
    if line.strip():
        json.loads(line)

print("eval artifacts are valid")
PY
```
