from __future__ import annotations

from ai_equity_research_copilot_backend.chunking import ParsedPage, chunk_pages


def test_chunking_preserves_page_and_section_metadata() -> None:
    pages = [
        ParsedPage(
            page_number=1,
            text="RISK FACTORS\n\nExport controls may reduce demand conversion. " * 10,
        ),
        ParsedPage(
            page_number=2,
            text="MARGIN ANALYSIS\n\nGross margin improved because mix shifted toward software-rich platforms. " * 10,
        ),
    ]

    chunks = chunk_pages(pages, target_tokens=35, overlap_tokens=5)

    assert len(chunks) >= 2
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2
    assert any(chunk.section_title in {"RISK FACTORS", "MARGIN ANALYSIS"} for chunk in chunks)
    assert all(chunk.token_count > 0 for chunk in chunks)
