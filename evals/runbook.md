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

## Live HTTP Collection

Install the backend dependencies into your Python environment (`python -m pip install -r backend/requirements.txt`). Collection launches the real local HTTP app with an isolated fixture store and routes all **35 synthetic canonical cases** to its existing chat/comparison endpoints. These excerpts are not SEC filings. No paid provider is used by this runner.

```bash
# Deterministic application baseline: no Ollama binary, model or server needed.
python scripts/run_live_eval.py --providers local --output evals/results/new-local-run

# Model configuration: requires a running Ollama server and the selected model.
python scripts/run_live_eval.py --providers ollama --model gemma3:4b --output evals/results/new-model-run

# Existing default: Ollama followed by the deterministic baseline.
python scripts/run_live_eval.py --output evals/results/new-comparison-run
```

The exact accepted selections are `local`, `ollama` and `both`; comma lists, duplicate selections and unknown names are rejected. `--port` selects the local app port, and `--ollama` selects the Ollama base URL. The port must be free: if another server answers health checks with a different data directory, collection fails instead of capturing it. Use a fresh output directory: existing captures are never overwritten.

Local-only makes **zero Ollama HTTP requests**, including metadata. Its model metadata fields are null with `ollama_metadata_status: not_requested`. Only selections containing Ollama query its version/model/tags; required Ollama preflight failure stops collection with explicit failed metadata. A later unavailable, timed-out or invalid model answer retains its unavailable/fallback flags; deterministic output is never counted as accepted model output. Comparisons and application refusal gates can bypass model inference even in the Ollama configuration.

CPU and physical memory observations are best effort: macOS uses sysctl, Linux uses `/proc/cpuinfo` and sysconf memory, and Windows uses PowerShell CIM queries. Missing commands, permission errors, invalid values or metadata timeouts produce null values plus collection-source/reason fields under `hardware_metadata`. They do not prevent evaluation. Windows/Linux branches have offline tests; validation on every physical platform is not implied.

Metadata retains commit, working-tree, runner and fixed-source hashes, and records selected providers, per-provider status/completed case counts, and response-file hashes. On a server, HTTP or response-schema failure, the command exits unsuccessfully; it retains metadata, server logs, fixture stores and any completed rows for inspection. The child API process is terminated/reaped, with kill escalation after 15 seconds. Partial captures are not complete benchmark results; rerun into a new directory after resolving the recorded error.

## Response-bound Scoring

Collection does not assign semantic grades. Review each selected configuration's complete responses against the existing rubric and create `review.json`, following the historical annotation structure. Each annotation must match its case ID and the SHA-256 of its exact response (`response_hash` in `scripts/score_live_eval.py`). Do not copy old annotations onto new responses or fabricate missing model scores.

```bash
python scripts/score_live_eval.py evals/results/new-local-run
```

The scorer uses `metadata.json`'s selected providers. Historical captures without that field use available local/Ollama response files, preserving Ollama-then-local order when both exist. `--providers local|ollama|both` explicitly limits aggregation, for example to a fully completed provider in an otherwise failed dual run. Every chosen provider still requires 35 unique rows, 35 matching response-bound annotations, complete recorded status and any recorded response-file hash. Missing/incomplete selected captures, missing/stale reviews or changed capture bytes fail before writing summary output. Unselected files/reviews are not required. Scores from a local-only run describe the deterministic baseline and are not a model comparison.

Published historical captures, scores and fixture definitions remain unchanged. Cross-configuration comparisons require the same cases, rubric and review method; portable metadata is not evidence of model quality.

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
