# Measured finance QA results — 29 September 2026

The final-code run below includes every case. Neither configuration meets the repository's MVP quality gate. Gemma modestly improves graded answer coverage and unsupported-question refusals, but loses citation precision and introduces one unsupported assertion.

| Metric | Gemma configuration | Deterministic baseline |
| --- | ---: | ---: |
| Cases | 35 | 35 |
| Graded answer accuracy | 41.4% (14.5/35) | 40.0% (14/35) |
| Citation precision | 96.4% (27/28) | 100.0% (24/24) |
| Canonical citation recall | 43.4% (23/53) | 43.4% (23/53) |
| Graded refusal correctness | 100% (7/7) | 85.7% (6/7) |
| Unsupported claims | 1 in 1 answer | 0 |
| Answers with unsupported claims | 2.9% (1/35) | 0% (0/35) |
| Accepted model answers | 13/35 | 0/35 |
| Validation fallback / model attempts | 23.5% (4/17) | N/A — no model calls |
| Validation fallback / all cases | 11.4% (4/35) | N/A — baseline by design |
| Structured response compliance | 100% | 100% |
| Median HTTP latency, all cases | 54.8 ms | 16.1 ms |
| Median HTTP latency, model attempts | 7,861.7 ms | N/A |

## Scope and provenance

- Evaluated commit: `2a1ca02` (full SHA in [metadata](bounded-structured-output/metadata.json)). Generation uses an explicit source-reference prompt, a JSON schema constraining source indices, and a 512-token generation cap. Citation validation remains enabled; structured decoding does not prove factual support.
- Model: Gemma 3 4B (`gemma3:4b`), 4.3B parameters, Q4_K_M, model ID `a2af6cc3eb7f`; Ollama 0.34.4, Metal acceleration. Hardware: Apple M5, 32 GiB unified memory, macOS 27.0; Python 3.12.13.
- Corpus: unchanged **synthetic** canonical fixtures, five companies, ten documents, twenty chunks in `tests/fixtures/`. These are separate from the video's real SEC corpus. Canonical IDs/text are preserved; hashing embeddings are regenerated.
- All 35 cases, one sequential pass per configuration, Gemma first. `top_k=8`; no gold source IDs, answer hints, or explicit source/year filters. Six comparison cases use the deterministic comparison route in both configurations. Other evidence/refusal gates can bypass inference. **13 accepted model answers + 4 validation fallbacks + 18 deterministic bypasses = 35.** No unavailable-provider response occurred.
- Latency is end-to-end local HTTP time, including retrieval and persistence, excluding backend startup. The model server was restarted before this final run, so the first inference includes loading. No concurrent model inference; ordinary desktop activity and a backend test run mean this is not an isolated hardware benchmark. The all-case median is low because more than half the cases bypass the model.
- Local provider cost is $0 billed, excluding electricity and hardware. Metadata omits local model-file paths and unrelated installed models.

## Scoring and limitations

The original repository supplied cases, an unchanged [scoring rubric](../../scoring_rubric.json), and a metadata smoke check, but no live-backend semantic scorer. [Response-bound review notes](bounded-structured-output/review.json) apply the rubric with 0/0.5/1 grades. These annotations are **not blinded or independently validated**. Accuracy is a graded mean, not a binary pass rate. The 35 authored synthetic cases are a development set, not a held-out estimate of real-filing accuracy.

Citation precision is micro-averaged over returned citation objects. A citation fails when a material assertion attributed to it is unsupported. Citation recall measures the rubric's acceptable canonical chunk-ID coverage, not independent satisfaction of every compound citation rule. Refusals without required citations do not add a precision denominator. High precision therefore coexists with incomplete answers.

Case 018 infers a Blackwell-specific cost contribution from annual excerpts that discuss only generic transitions. This is one unsupported assertion and a failed citation. Several answerable risk/comparison questions are refused; multi-document answers often miss transcript details. The baseline does not explicitly refuse the future-credit-loss question (032). All these failures remain in the denominator. No buy/sell recommendation or exact future numerical forecast was observed.

## Raw files and reproduction

Final run: [Gemma responses](bounded-structured-output/ollama.json), [baseline responses](bounded-structured-output/local.json), [CSV scores](bounded-structured-output/scores.csv), [summary JSON](bounded-structured-output/summary.json), [metadata and hashes](bounded-structured-output/metadata.json), [per-case review](bounded-structured-output/review.json).

With backend dependencies installed and Ollama serving `gemma3:4b`, collect a fresh run from the repository root:

```bash
python scripts/run_live_eval.py --output evals/results/new-run
```

Review its complete responses against the existing rubric and create hash-bound annotations following `review.json`, then aggregate with `python scripts/score_live_eval.py evals/results/new-run`. Collection does not fabricate semantic grades from keyword matches. To reproduce the published aggregation: `python scripts/score_live_eval.py evals/results/2026-09-29/bounded-structured-output`.

## Development-run history

All completed runs and the interrupted diagnostic are retained. The headline uses the last complete run on the final implementation, not the best per-case result across runs. Prompt/schema changes were informed by these development cases; no holdout claim is made.

| Run | Commit | Accuracy | Citation precision | Fallbacks / attempts |
| --- | --- | ---: | ---: | ---: |
| [Initial](initial-run.md) | b571099 | 41.4% | 90.0% | 6/17 |
| [Explicit source references](explicit-source-references/) | 9082053 | 40.0% | 93.1% | 4/17 |
| [Schema diagnostic, incomplete](structured-output/README.md) | e2c1f34 | Not scored | Not scored | Not scored |
| [Bounded structured output, final](bounded-structured-output/) | 2a1ca02 | 41.4% | 96.4% | 4/17 |

Initial-run JSON/CSV remain at this directory's root for historical reproducibility. Runtime stores and server logs remain local and are excluded from Git.
