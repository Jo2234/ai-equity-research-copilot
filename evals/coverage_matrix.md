# Eval Coverage Matrix

The canonical eval file is `finance_qa_v1.jsonl`. It contains 35 cases and is designed to exercise the project-spec MVP workflows: document-grounded Q&A, cross-document synthesis, company comparison, research memo inputs, and unsupported-question refusal.

## Summary

| Dimension | Coverage |
| --- | ---: |
| Total cases | 35 |
| Single-document factual extraction | 12 |
| Multi-document synthesis | 10 |
| Company comparison | 6 |
| Insufficient-context refusal | 7 |
| Companies covered | 5 |
| Source documents covered | 10 |
| Acceptable citation chunks covered | 20 |

## Company Coverage

| Ticker | Company | Primary task coverage |
| --- | --- | --- |
| NVDA | NVIDIA Corporation | Data center growth, gross margin, risks, Blackwell transition, valuation refusal |
| AAPL | Apple Inc. | Services growth, gross margin mix, risk factors, capital allocation, legal-impact refusal |
| JPM | JPMorgan Chase & Co. | Net interest income, credit normalization, capital management, CRE-loss refusal |
| XOM | Exxon Mobil Corporation | Upstream drivers, commodity risks, capital allocation, oil-price refusal |
| TSLA | Tesla, Inc. | Automotive margin, energy storage, autonomy framing, robotaxi-date refusal |

## Workflow Coverage

| Spec workflow | Eval case IDs |
| --- | --- |
| Single filing or transcript Q&A | `finance_qa_v1_001` to `finance_qa_v1_012` |
| Cross-document analysis | `finance_qa_v1_013` to `finance_qa_v1_022` |
| Company comparison | `finance_qa_v1_023` to `finance_qa_v1_028` |
| Unsupported or insufficient context | `finance_qa_v1_029` to `finance_qa_v1_035` |

## Coverage Gaps

These are intentional gaps for the current MVP eval version:

- No live SEC EDGAR ingestion scoring. The sample corpus is synthetic and local.
- No financial table extraction scoring.
- No valuation model scoring.
- No PDF rendering or page-coordinate validation beyond page labels in chunk metadata.
- No multi-provider model comparison scoring.

## Next Dataset Additions

Good next additions after the backend eval endpoint stabilizes:

- Memo-generation cases with section-level expected answer points.
- Retrieval-only cases that score top-k chunk ranking before generation.
- Citation-clickthrough cases that verify document ID, chunk ID, page label, and excerpt text.
- Regression cases for malformed uploads, failed parsing, and empty company scope.
