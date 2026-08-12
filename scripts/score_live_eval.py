"""Aggregate response-bound rubric annotations without pretending lexical overlap is correctness."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def response_hash(response):
    return hashlib.sha256(json.dumps(response, sort_keys=True).encode()).hexdigest()


def score(directory):
    reviews = json.loads((directory / "review.json").read_text())
    summary = {}
    flat = []
    for mode in ("ollama", "local"):
        rows = json.loads((directory / f"{mode}.json").read_text())
        annotations = reviews[mode]
        assert len(rows) == len(annotations) == 35
        for row, review in zip(rows, annotations):
            assert row["id"] == review["id"]
            assert response_hash(row["response"]) == review["response_sha256"], "Stale review"
            assert review["answer_accuracy"] in (0, .5, 1)
            assert 0 <= review["supported_citations"] <= review["citation_count"]
            assert 0 <= review["citation_requirements_met"] <= review["citation_requirements"]
            flat.append({"configuration": mode, "id": row["id"], "category": row["category"],
                         "latency_ms": row["latency_ms"], "accepted_model_output": row["accepted_model_output"],
                         "validation_fallback": row["validation_fallback"],
                         **{k: v for k, v in review.items() if k not in ("id", "response_sha256")}})
        unsupported = [x for x, r in zip(annotations, rows) if r["category"] == "insufficient_context"]
        cites = sum(x["citation_count"] for x in annotations)
        requirements = sum(x["citation_requirements"] for x in annotations)
        accepted = sum(x["accepted_model_output"] for x in rows)
        fallback = sum(x["validation_fallback"] for x in rows)
        unavailable = sum(x["model_unavailable"] for x in rows)
        summary[mode] = {
            "cases": len(rows), "answer_accuracy": statistics.mean(x["answer_accuracy"] for x in annotations),
            "answer_score_sum": sum(x["answer_accuracy"] for x in annotations),
            "citation_precision": sum(x["supported_citations"] for x in annotations) / cites,
            "supported_citations": sum(x["supported_citations"] for x in annotations), "citations": cites,
            "citation_recall": sum(x["citation_requirements_met"] for x in annotations) / requirements,
            "citation_requirements_met": sum(x["citation_requirements_met"] for x in annotations),
            "citation_requirements": requirements,
            "refusal_correctness": statistics.mean(x["refusal_correctness"] for x in unsupported),
            "fully_correct_refusals": sum(x["refusal_correctness"] == 1 for x in unsupported),
            "unsupported_cases": len(unsupported),
            "unsupported_claim_count": sum(x["unsupported_claim_count"] for x in annotations),
            "answers_with_unsupported_claims": sum(x["unsupported_claim_count"] > 0 for x in annotations),
            "accepted_model_outputs": accepted, "validation_fallbacks": fallback,
            "model_attempts": accepted + fallback + unavailable,
            "fallback_rate_among_model_attempts": fallback / (accepted + fallback + unavailable) if accepted + fallback + unavailable else None,
            "fallback_rate_all_cases": fallback / len(rows),
            "model_unavailable": unavailable,
            "format_compliance": statistics.mean(x["format_compliance"] for x in rows),
            "median_latency_ms": statistics.median(x["latency_ms"] for x in rows),
            "median_model_attempt_latency_ms": statistics.median(x["latency_ms"] for x in rows if x["accepted_model_output"] or x["validation_fallback"] or x["model_unavailable"]) if accepted + fallback + unavailable else None,
            "estimated_provider_cost_usd": sum(x["response"]["usage"]["estimated_cost_usd"] for x in rows),
        }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (directory / "scores.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    score(parser.parse_args().directory)
