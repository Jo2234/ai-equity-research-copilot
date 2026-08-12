import pytest

from ai_equity_research_copilot_backend.chunking import ParsedPage, chunk_pages, detect_heading


@pytest.mark.parametrize("heading", [
    "2030 Upstream Earnings Driver Analysis",
    "2029 Energy Products Earnings Driver Analysis (1)",
    "2030 Financial Results:",
    "Consumer & Community Banking",
    "Corporate & Investment Bank",
    "Asset and Wealth Management",
    "Upstream",
    "RISK FACTORS:",
    "Liquidity and Capital Resources",
    "Capital Returns",
    "Cash Flows from Investing Activities",
    "Cash Flows from Financing Activities:",
    "Results of Operations",
    "Credit and Investment Risk Management",
    "Financial Position",
    "Upstream Operational Results",
    "Upstream Additional Information",
    "Downstream Operational Results",
])
def test_explicit_financial_headings_retain_original_punctuation(heading):
    assert detect_heading(heading) == heading


@pytest.mark.parametrize("text", [
    "Consumer & Community Banking reported higher revenue.",
    "2030 Upstream earnings increased by 12 percent.",
    "REVENUE 1200 1100",
    "Operating income $120 $110",
    "Consumer deposits 1200 1100",
    "Net Income",
    "Operating Income",
    "2029 compared with 2028",
    "Our liquidity is sufficient",
    "Our Liquidity Is Sufficient",
    "We Increased Capital Returns",
    "Liquidity Improved Capital Resources",
    "Capital Returns $12 $11",
    "Cash Flows from Investing Activities 2030 2029",
    "Results of Operations increased during the year.",
    "Cash flows from investing activities were negative",
    "Community Outreach Activities",
])
def test_sentences_and_metric_rows_are_not_promoted_to_sections(text):
    assert detect_heading(text) is None


def test_unit_spans_preserve_mixed_sections_inside_one_chunk():
    units = (
        "Opening paragraph without a known section.",
        "2030 Upstream Earnings Driver Analysis",
        "Lower crude realizations reduced earnings during the year.",
        "2029 Energy Products Earnings Driver Analysis (1)",
        "Refining margins declined as new industry supply became available.",
    )
    chunk, = chunk_pages([ParsedPage(1, "", paragraphs=units)], target_tokens=800)
    spans = chunk.metadata["section_spans"]
    assert [chunk.text[s["start"]:s["end"]] for s in spans] == list(units)
    assert [s["section"] for s in spans] == [None, units[1], units[1], units[3], units[3]]
    assert chunk.section_title == units[3]  # Legacy field retained for callers.


def test_inherited_section_survives_chunk_and_page_boundaries_with_overlap():
    heading = "Consumer & Community Banking"
    first = "Consumer loans remained resilient during the reporting period. " * 14
    second = "Deposit balances increased during the reporting period. " * 14
    pages = [ParsedPage(1, "", paragraphs=(heading, first)),
             ParsedPage(2, "", paragraphs=(second,))]
    chunks = chunk_pages(pages, target_tokens=80, overlap_tokens=10)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.metadata["section_spans"]
        assert all(span["section"] == heading for span in chunk.metadata["section_spans"])
        for span in chunk.metadata["section_spans"]:
            assert 0 <= span["start"] < span["end"] <= len(chunk.text)
    assert any(second.strip() in chunk.text and heading not in chunk.text for chunk in chunks)


def test_span_offsets_account_for_outer_whitespace_and_internal_newlines():
    units = ("  Consumer & Community Banking  ", "  Deposit balances\nincreased during the year. \n")
    chunk, = chunk_pages([ParsedPage(1, "", paragraphs=units)])
    spans = chunk.metadata["section_spans"]
    assert chunk.text[spans[0]["start"]:spans[0]["end"]] == units[0].lstrip()
    assert chunk.text[spans[1]["start"]:spans[1]["end"]] == units[1].rstrip()
    assert all(span["section"] == "Consumer & Community Banking" for span in spans)
    assert chunk.text[spans[0]["end"]:spans[1]["start"]] == "\n\n"


def test_missing_heading_is_left_unknown_instead_of_guessed_from_metric():
    text = "The provision for credit losses was $11.5 billion."
    chunk, = chunk_pages([ParsedPage(1, text)])
    assert chunk.metadata["section_spans"] == [{"start": 0, "end": len(text), "section": None}]


def test_capital_allocation_fact_inherits_exact_financial_heading():
    heading = "Liquidity and Capital Resources"
    fact = "During the quarter, we repurchased shares and paid cash dividends."
    chunk, = chunk_pages([ParsedPage(1, "", paragraphs=(heading, fact))])
    span = chunk.metadata["section_spans"][1]
    assert span["section"] == heading
    assert chunk.text[span["start"]:span["end"]] == fact


def test_undated_operating_sections_end_previous_driver_year_scope():
    units = (
        "2029 Upstream Earnings Driver Analysis",
        "Lower realizations reduced earnings during the year.",
        "Upstream Operational Results",
        "Net production increased during the reporting period.",
        "Upstream Additional Information",
        "2030 versus 2029\nProduction increased as new projects came online.",
    )
    chunks = chunk_pages([ParsedPage(1, "", paragraphs=units)], target_tokens=80, overlap_tokens=0)
    observed = {chunk.text[span["start"]:span["end"]]: span["section"]
                for chunk in chunks for span in chunk.metadata["section_spans"]}
    assert observed[units[1]] == units[0]
    assert observed[units[3]] == units[2]
    assert observed[units[5]] == units[4]
    assert "2029" not in observed[units[5]]


def test_unrecognized_structural_title_clears_inherited_financial_scope():
    units = ("2029 Upstream Earnings Driver Analysis", "Environmental Review",
             "Emissions decreased as operational efficiency improved.")
    assert detect_heading(units[1]) is None
    chunk, = chunk_pages([ParsedPage(1, "", paragraphs=units)])
    assert [span["section"] for span in chunk.metadata["section_spans"]] == [units[0], None, None]


@pytest.mark.parametrize("unit", [
    "Environmental Review 30 29", "Our Results", "(millions of dollars)",
    "We Reported Improved Results", "Environmental results improved during the year.",
])
def test_metric_rows_and_prose_do_not_clear_inherited_section(unit):
    heading = "Liquidity and Capital Resources"
    chunk, = chunk_pages([ParsedPage(1, "", paragraphs=(heading, unit))])
    assert chunk.metadata["section_spans"][-1]["section"] == heading


def test_verified_table_category_clears_scope_without_changing_source_units():
    units = ("Intelligent Cloud", "The segment provides hosted computing services.",
             "(in millions)", "2030 2029", "More Personal Computing",
             "Revenue 130 120", "Operating income 30 25",
             "Research spending increased as the company invested in engineering.")
    chunk, = chunk_pages([ParsedPage(1, "", paragraphs=units)])
    spans = chunk.metadata["section_spans"]
    assert [chunk.text[s["start"]:s["end"]] for s in spans] == list(units)
    assert spans[1]["section"] == units[0]
    assert all(s["section"] is None for s in spans[4:])


def test_unsupported_numeric_table_category_clears_scope():
    units = ("Liquidity and Capital Resources", "25,682", "More Personal Computing",
             "62,363", "Engineering investment increased across the company.")
    chunk, = chunk_pages([ParsedPage(1, "", paragraphs=units)])
    assert all(s["section"] is None for s in chunk.metadata["section_spans"][2:])


def test_real_segment_heading_after_table_reestablishes_scope():
    units = ("(in millions)", "2030 2029", "More Personal Computing",
             "Revenue 130 120", "The preceding table summarizes segment revenue.",
             "More Personal Computing", "Revenue increased because customer demand grew.")
    chunk, = chunk_pages([ParsedPage(1, "", paragraphs=units)])
    spans = chunk.metadata["section_spans"]
    assert spans[2]["section"] is None
    assert spans[-1]["section"] == "More Personal Computing"


@pytest.mark.parametrize("title", ["Credit Losses", "Net Income"])
def test_ambiguous_title_ends_previous_section_without_guessing_new_scope(title):
    units = ("FORWARD-LOOKING STATEMENTS", title,
             "The allowance increased because credit conditions deteriorated.")
    chunk, = chunk_pages([ParsedPage(1, "", paragraphs=units)])
    assert [s["section"] for s in chunk.metadata["section_spans"]] == [units[0], None, None]


def test_table_introduction_is_not_inherited_as_a_prose_heading():
    introduction = "The following table presents the components of interest income and interest expense:"
    assert detect_heading(introduction) is None
    units = (introduction, "Credit Losses", "The reserve increased during the year.")
    chunk, = chunk_pages([ParsedPage(1, "", paragraphs=units)])
    assert all(s["section"] is None for s in chunk.metadata["section_spans"])


def test_unit_caption_preserves_explicit_year_scope():
    heading = "2029 Energy Products Earnings Driver Analysis"
    units = (heading, "(millions of dollars)", "Lower margins reduced earnings.")
    chunk, = chunk_pages([ParsedPage(1, "", paragraphs=units)])
    assert all(s["section"] == heading for s in chunk.metadata["section_spans"])
