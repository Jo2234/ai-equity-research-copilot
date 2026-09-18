import hashlib
import json
from dataclasses import replace

from ai_equity_research_copilot_backend.chunking import (
    ChunkDraft,
    ParsedPage,
    chunk_pages,
)
from ai_equity_research_copilot_backend.table_context import (
    MAX_FACTS_PER_CHUNK,
    MAX_METADATA_BYTES_PER_CHUNK,
    attach_table_context,
)


def _draft(text, page=1):
    return ChunkDraft(0, text, page, page, None, 10, {"retained": True})


def _facts(draft):
    return draft.metadata.get("financial_table_facts", [])


def test_small_chunks_keep_original_table_header_and_row_provenance():
    text = "RESULTS\n(in millions)\n2032 2031\n" + "\n".join(
        f"Business unit {chr(65 + index)} revenue {100 + index} {90 + index}"
        for index in range(20)
    )
    page = ParsedPage(1, text)
    originals = chunk_pages([page], target_tokens=80, overlap_tokens=0)
    drafts = attach_table_context([page], originals)
    last = drafts[-1]
    assert "(in millions)" not in last.text
    assert _facts(last)
    for fact in _facts(last):
        assert "(in millions)" in fact["header_text"]
        assert fact["header_text"] == text[slice(*fact["header_span"])]
        assert fact["row_text"] == text[slice(*fact["row_span"])]
        assert fact["row_text"] in last.text
        assert fact["source_page"] == 1 and fact["source_page_index"] == 0
        assert fact["source_page_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert [draft.text for draft in drafts] == [draft.text for draft in originals]
    assert all("financial_table_facts" not in draft.metadata for draft in originals)


def test_multiline_row_matches_only_complete_source_row_with_whitespace_changes():
    text = "(in billions)\n2032\n2031\nRevenue\n$\n124\n$\n98\n"
    page = ParsedPage(1, text)
    draft = _draft("Revenue\n\n$  124\n\n$\t98")
    facts = _facts(attach_table_context([page], [draft])[0])
    assert len(facts) == 1
    assert facts[0]["row_text"] == "Revenue\n$\n124\n$\n98"
    assert not _facts(attach_table_context([page], [_draft("$ 124 $ 98")])[0])
    assert not _facts(attach_table_context([page], [_draft("Revenue $ 124")])[0])


def test_identical_values_different_pages_keep_their_own_headers_and_units():
    pages = [ParsedPage(1, "(in millions)\n2032 2031\nRevenue 124 98"),
             ParsedPage(2, "(in billions)\n2030 2029\nRevenue 124 98")]
    drafts = attach_table_context(pages, [_draft("Revenue 124 98", 1), _draft("Revenue 124 98", 2)])
    first, = _facts(drafts[0])
    second, = _facts(drafts[1])
    assert first["header_text"] == "(in millions)\n2032 2031"
    assert second["header_text"] == "(in billions)\n2030 2029"
    assert first["row_text"] == second["row_text"] == "Revenue 124 98"


def test_repeated_row_on_same_page_is_not_assigned_to_arbitrary_table():
    text = "(in millions)\n2032 2031\nRevenue 124 98\n(in billions)\n2030 2029\nRevenue 124 98"
    assert not _facts(attach_table_context([ParsedPage(1, text)], [_draft("Revenue 124 98")])[0])
    distinguishable = _draft("(in billions)\n2030 2029\nRevenue 124 98")
    facts = _facts(attach_table_context([ParsedPage(1, text)], [distinguishable])[0])
    assert len(facts) == 1 and facts[0]["header_text"] == "(in billions)\n2030 2029"


def test_header_never_spills_across_pages_or_onto_unrelated_same_value_row():
    pages = [ParsedPage(1, "(in millions)\n2032 2031\nRevenue 124 98"),
             ParsedPage(2, "Operating cash 124 98")]
    assert not _facts(attach_table_context(pages, [_draft("Operating cash 124 98", 2)])[0])
    assert not _facts(attach_table_context(pages, [_draft("Revenue 124 98 unrelated", 1)])[0])


def test_missing_page_numbers_are_safe_when_source_text_is_unique():
    page = ParsedPage(None, "(in millions)\n2032 2031\nRevenue 124 98")
    draft = _draft("Revenue 124 98", None)
    assert len(_facts(attach_table_context([page], [draft])[0])) == 1
    assert not _facts(attach_table_context([page, page], [draft])[0])


def test_chunk_crossing_pages_retains_each_rows_original_page_context():
    pages = [ParsedPage(1, "(in millions)\n2032 2031\nRevenue 124 98"),
             ParsedPage(2, "(in billions)\n2030 2029\nCash 88 76")]
    draft = replace(_draft("Revenue 124 98\n(in billions)\n2030 2029\nCash 88 76"), page_end=2)
    facts = _facts(attach_table_context(pages, [draft])[0])
    assert len(facts) == 2
    assert [(fact["source_page"], fact["header_text"]) for fact in facts] == [
        (1, "(in millions)\n2032 2031"), (2, "(in billions)\n2030 2029")]


def test_metadata_is_bounded_json_serializable_and_does_not_keep_stale_facts():
    text = "(in millions)\n2032 2031\n" + "\n".join(
        f"Revenue segment {chr(65 + index // 26)}{chr(65 + index % 26)} 124 98" for index in range(100)
    )
    draft = _draft(text)
    enriched = attach_table_context([ParsedPage(1, text)], [draft])[0]
    facts = _facts(enriched)
    assert 0 < len(facts) <= MAX_FACTS_PER_CHUNK
    assert len(json.dumps(facts, ensure_ascii=False).encode()) <= MAX_METADATA_BYTES_PER_CHUNK
    assert enriched.metadata["retained"] is True
    assert all(not {"amount", "unit", "period", "normalized_text", "currency"} & fact.keys() for fact in facts)
    assert len({(tuple(fact["row_span"]), fact["label"]) for fact in facts}) == len(facts)
    stale = replace(enriched, text="No table here")
    assert "financial_table_facts" not in attach_table_context([ParsedPage(1, text)], [stale])[0].metadata
