from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "evals" / "finance_qa_v1.jsonl"
SUPPORT_FILES = [
    ROOT / "evals" / "schema.json",
    ROOT / "evals" / "scoring_rubric.json",
    ROOT / "data" / "sample_documents" / "manifest.json",
    ROOT / "data" / "evals" / "finance_qa_examples.json",
]
REQUIRED_FIELDS = {
    "id",
    "category",
    "company_ids",
    "question",
    "required_source_documents",
    "expected_answer_points",
    "citation_rules",
    "must_not_include",
    "difficulty",
}


def load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    for path in SUPPORT_FILES:
        load_json(path)

    rows: list[dict[str, object]] = []
    with EVAL_PATH.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED_FIELDS - set(row)
            if missing:
                raise AssertionError(f"{EVAL_PATH}:{line_no} missing fields: {sorted(missing)}")
            if not str(row["question"]).strip():
                raise AssertionError(f"{EVAL_PATH}:{line_no} has an empty question")
            if not isinstance(row["expected_answer_points"], list) or not row["expected_answer_points"]:
                raise AssertionError(f"{EVAL_PATH}:{line_no} needs expected_answer_points")
            rows.append(row)

    categories = {str(row["category"]) for row in rows}
    unsupported = [row for row in rows if row["category"] in {"unsupported", "insufficient_context"}]
    if len(rows) < 10:
        raise AssertionError(f"expected a meaningful eval set, found {len(rows)} cases")
    if len(categories) < 3:
        raise AssertionError(f"expected at least 3 eval categories, found {sorted(categories)}")
    if not unsupported:
        raise AssertionError("expected at least one unsupported/refusal eval case")

    print(
        json.dumps(
            {
                "status": "ok",
                "eval_cases": len(rows),
                "categories": sorted(categories),
                "support_files": [str(path.relative_to(ROOT)) for path in SUPPORT_FILES],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
