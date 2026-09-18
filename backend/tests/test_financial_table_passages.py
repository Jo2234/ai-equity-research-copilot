from dataclasses import asdict

from ai_equity_research_copilot_backend.financial_table_passages import (
    MAX_BUNDLE_CHARACTERS,
    table_passages,
)
from ai_equity_research_copilot_backend.financial_tables import financial_table_facts


def _facts(text):
    return [asdict(fact) for fact in financial_table_facts(text)]


def _record(label, row, *, page=1, header=(0, 20), header_text="(in millions)\n2032 2031"):
    return {
        "label": label,
        "source_excerpt": header_text + "\n...\n" + label + " 20 19",
        "header_span": header, "row_span": (row, row + 5), "source_page_index": page,
    }


def test_columns_stay_together_and_section_bundles_never_mix_metrics():
    text = """(in millions)
2032 2031
North division
Revenue 100 90
Operating income 40 30
South division
Revenue 80 70
Operating income 20 10
"""
    facts = _facts(text)
    passages = table_passages(facts)
    assert len(passages) == 6  # Four complete rows, two common-metric bundles.
    revenue = [(label, excerpt) for label, excerpt in passages
               if label == "North division: Revenue\nSouth division: Revenue"]
    assert len(revenue) == 1
    label, excerpt = revenue[0]
    assert "North division\n...\nRevenue 100 90" in excerpt
    assert "South division\n...\nRevenue 80 70" in excerpt
    assert "Operating income" not in label and "Operating income" not in excerpt
    assert all(period in excerpt for period in ("2032", "2031"))
    assert facts[0]["source_excerpt"] in excerpt
    assert facts[2]["source_excerpt"].split("\n...\n", 1)[1] in excerpt
    operating = [(label, excerpt) for label, excerpt in passages
                 if label == "North division: Operating income\nSouth division: Operating income"]
    assert len(operating) == 1
    assert "Revenue" not in operating[0][0] and "Revenue" not in operating[0][1]


def test_small_simple_table_retains_individual_regions_and_complete_bundle():
    facts = _facts("(in millions)\n2032 2031\nNorth 100 90\nSouth 80 70\nEast 60 50")
    passages = table_passages(facts)
    assert len(passages) == 4
    assert all(region in passages[-1][0] for region in ("North", "South", "East"))
    assert passages[-1][1].count("2032") == 1
    assert passages[-1][1].count("2031") == 1


def test_conflicting_page_header_offsets_and_units_are_separate_tables():
    cases = [
        [_record("North: Revenue", 30), _record("South: Revenue", 40, page=2)],
        [_record("Revenue", 30), _record("Revenue", 30, page=2)],
        [_record("North: Revenue", 30), _record("South: Revenue", 80, header=(50, 70))],
        [_record("North: Revenue", 30), _record("South: Revenue", 40, header_text="(in billions)\n2032 2031")],
    ]
    for facts in cases:
        assert len(table_passages(facts)) == 2


def test_repeated_label_is_not_treated_as_distinct_sections():
    facts = [_record("North: Revenue", 30), _record("North: Revenue", 40)]
    assert len(table_passages(facts)) == 2  # Keep both rows but do not bundle them.


def test_bundles_are_not_truncated_to_fit_row_or_character_limits():
    facts = [_record(f"Region {index}", 30 + index * 10) for index in range(9)]
    assert len(table_passages(facts)) == 9
    facts = [_record("North", 30), _record("South", 40)]
    facts[0]["source_excerpt"] += " detail" * (MAX_BUNDLE_CHARACTERS // 7)
    passages = table_passages(facts)
    assert len(passages) == 2
    assert passages[0] == ("North", facts[0]["source_excerpt"])


def test_duplicate_columns_are_removed_without_losing_distinct_period_values():
    facts = _facts("(in millions)\n2032 2031\nRevenue 100 90")
    passages = table_passages(facts + facts)
    assert len(passages) == 1
    assert passages[0][0] == "Revenue"
    assert passages[0][1].count("2032") == 1
    assert passages[0][1].count("2031") == 1


def test_invalid_offsets_and_missing_source_are_not_eligible_for_bundling():
    valid = _record("North", 30)
    invalid = [dict(valid, row_span=(10, 12)), dict(valid, row_span=(-1, 10)),
               dict(valid, header_span=[True, 20]), dict(valid, source_excerpt=""), {}]
    assert table_passages(invalid) == []
    assert len(table_passages([valid, *invalid])) == 1


def test_verified_row_presentation_keeps_exact_currency_caption_and_negative_notation():
    source = "Results (USD millions)\n2034 2033\nOperating income (17) 28\nNet income (23) 21"
    passages = table_passages(_facts(source))
    assert len(passages) == 3
    for label, excerpt in passages:
        assert "Results (USD millions)\n2034 2033" in excerpt
        assert "2034" not in label
    assert passages[0][0] == "Operating income"
    assert "Operating income (17) 28" in passages[0][1]
    assert "Net income (23) 21" in passages[1][1]


def test_legacy_cell_metadata_still_groups_into_one_source_row():
    record = _record("Revenue", 30)
    cells = [dict(record, period="2032", amount="20", normalized_text="Revenue was 20 million."),
             dict(record, period="2031", amount="19", normalized_text="Revenue was 19 million.")]
    assert table_passages(cells) == [("Revenue", record["source_excerpt"])]


def test_bundle_keeps_shared_header_once_without_deduplicating_section_transitions():
    header = "RESULTS " + "Header context " * 70 + "\n(in millions)\n2032 2031"
    records = [_record("North: Revenue", 1200, header=(0, len(header)), header_text=header),
               _record("South: Revenue", 1300, header=(0, len(header)), header_text=header)]
    records[0]["source_excerpt"] = header + "\n...\nNorth\n...\nOpening cash $ 20 $ 19\n...\nRevenue 20 19"
    records[1]["source_excerpt"] = header + "\n...\nSouth\n...\nOpening cash $ 20 $ 19\n...\nRevenue 18 17"
    passages = table_passages(records)
    assert len(passages) == 3
    label, excerpt = passages[-1]
    assert label == "North: Revenue\nSouth: Revenue"
    assert excerpt.count(header) == 1
    assert excerpt.count("Opening cash $ 20 $ 19") == 2
    assert "North\n...\nOpening cash $ 20 $ 19\n...\nRevenue 20 19" in excerpt
    assert "South\n...\nOpening cash $ 20 $ 19\n...\nRevenue 18 17" in excerpt
    assert len(excerpt) <= MAX_BUNDLE_CHARACTERS
    assert "Opening cash" not in label
