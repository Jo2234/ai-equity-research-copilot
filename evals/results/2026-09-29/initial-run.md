# Measured finance QA run — 29 September 2026

Neither configuration meets the repository's MVP quality gate. Gemma slightly improves graded answer coverage, but introduces citation errors and unsupported assertions. Conservative relevance checks refuse several answerable questions, and the annual-source default leaves some multi-document questions incomplete.

| Metric | Gemma configuration | Deterministic baseline |
| --- | ---: | ---: |
| Cases | 35 | 35 |
| Graded answer accuracy | 41.4% (14.5/35) | 40.0% (14/35) |
| Citation precision | 90.0% (27/30) | 100.0% (24/24) |
| Canonical citation recall | 43.4% (23/53) | 43.4% (23/53) |
| Graded refusal correctness | 92.9% (6.5/7) | 85.7% (6/7) |
| Fully correct refusals | 6/7 | 6/7 |
| Unsupported claims | 2 in 2 answers | 0 |
| Answers with unsupported claims | 5.7% (2/35) | 0% (0/35) |
| Accepted model answers | 11/35 | 0/35 |
| Validation fallback / model attempts | 35.3% (6/17) | N/A — no model calls |
| Validation fallback / all cases | 17.1% (6/35) | N/A — baseline by design |
| Structured response compliance | 100% | 100% |
| Median HTTP latency, all cases | 32.5 ms | 10.4 ms |
| Median HTTP latency, model attempts | 5,046.1 ms | N/A |

## Scope and provenance

- Code under evaluation: `b57109974e28eb7c9ca81c58659441e705350042`; collection/scoring scripts were added in this change. The collection runner's execution-time hash is in [metadata.json](metadata.json).
- Model: `gemma3:4b`, 4.3B parameters, `Q4_K_M`, model ID `a2af6cc3eb7f`; Ollama `0.34.4`. Full model metadata and source hashes are retained in [metadata.json](metadata.json).
- Hardware: Apple M5, 32 GiB unified memory, macOS 27.0, Metal acceleration; Python 3.12.13. Local inference has no billed provider fee; electricity and hardware costs are not estimated.
- Corpus: unchanged **synthetic** canonical fixtures: five companies, ten documents and twenty chunks in `tests/fixtures/`. These are not SEC filings. Text and canonical IDs are preserved, and the app's 128-dimensional hashing embeddings are regenerated.
- One sequential pass per configuration, Gemma first, with all 35 canonical cases and no excluded results. No benchmark-specific warmup; the model had been used for preflight. No concurrent inference was launched during this benchmark.
- Ordinary application retrieval, `top_k=8`, no explicit source/year filters, no gold answers or required-document IDs in requests. Six comparison cases use the documented `/research/compare` route, which is deterministic in both configurations. Refusal/evidence gates can bypass Ollama. **“Gemma configuration” does not mean 35 model-generated answers.**
- End-to-end HTTP latency includes retrieval and persistence, excludes server startup, and is measured locally. The all-case Gemma median is low because 18/35 cases bypassed the model. Provider-unavailable responses: zero.

## Scoring and caveats

The original repository supplied a [rubric](../../scoring_rubric.json), cases and metadata smoke check, but no executable semantic scorer. [review.json](review.json) applies that unchanged rubric with explicit per-case notes and response hashes. These qualitative annotations are **not blinded or independently validated**. Accuracy and refusal correctness use 0/0.5/1 grades; they are not binary success rates. Small, authored synthetic samples do not establish production accuracy on real filings.

Citation precision is micro-averaged over returned citation objects. A citation fails if a material claim attributed to it is unsupported, even when it supports another claim. Citation recall uses the rubric's acceptable canonical chunk-ID option. It does not independently grade every compound citation rule. Refusals without required citations contribute no precision denominator, so high precision can coexist with poor answer coverage.

Material review findings:

- Case 014: the model attributes record Services revenue to the growth chunk in a key point; the record claim belongs to the transcript chunk. The information exists in retrieved context, but the reference is wrong.
- Case 018: the model asserts a Blackwell-specific cost contribution using annual excerpts that only describe generic product transitions. The relevant transcript warning is absent from the retrieved context.
- Case 020: the model extends board approval to all capital uses; the source conditions only share repurchases on board approval.
- Case 032: the model refuses to quantify losses but changes “next year” to FY2025; the baseline gives allowance context without explicitly refusing the future-loss question.
- Several supported risk and comparison questions are refused. These failures remain in the denominator. No investment recommendation or exact future numerical forecast was observed.

## Reproduce

With backend dependencies installed and Ollama serving `gemma3:4b`, run from the repository root:

```bash
python scripts/run_live_eval.py --output evals/results/new-run
```

This starts isolated localhost HTTP servers, preserves complete requests/responses, and does not overwrite an existing run directory. It collects outcomes; it does **not** manufacture semantic scores using keyword matches. Review the new responses against the unchanged rubric, create response-hash-bound annotations following [review.json](review.json), then run:

```bash
python scripts/score_live_eval.py evals/results/new-run
```

To reproduce the aggregation of this recorded run exactly:

```bash
python scripts/score_live_eval.py evals/results/2026-09-29
```

Raw responses: [Gemma configuration](ollama.json), [baseline](local.json). Scores: [CSV](scores.csv), [summary JSON](summary.json). Runtime stores and server logs remain local and are excluded from Git.
