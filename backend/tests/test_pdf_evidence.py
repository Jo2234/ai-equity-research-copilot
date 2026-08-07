import fitz
import pytest

from ai_equity_research_copilot_backend.chunking import chunk_pages
from ai_equity_research_copilot_backend.evidence import evidence_passages, usable_prose
from ai_equity_research_copilot_backend.parsing import parse_pdf


def test_pdf_wrapped_sentence_retains_figure_and_growth(tmp_path):
    path = tmp_path / "release.pdf"
    first = "The company achieved revenue of $7.42 billion in the quarter,"
    second = "representing 16 percent growth from the same period last year."
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((50, 50), "Quarterly financial highlights")
        page.insert_text((50, 100), first + "\n" + second)
        page.insert_text((50, 180), "The board approved a new distribution agreement.")
        pdf.new_page().insert_text((50, 50), "Separate page with accounting policies.")
        pdf.save(path)
    pages = parse_pdf(path)
    assert [page.page_number for page in pages] == [1, 2]
    assert first + "\n" + second in pages[0].text
    assert "highlights\n\n" in pages[0].text
    passages = [text for chunk in chunk_pages(pages) for text in evidence_passages(chunk.text)]
    assert first + " " + second in passages
    assert all(not (first in text and "distribution agreement" in text) for text in passages)


def test_single_block_pdf_does_not_split_wrapped_lines(tmp_path):
    path = tmp_path / "one-block.pdf"
    first = "The company achieved revenue of $7.42 billion in the quarter,"
    second = "representing 16 percent growth from the same period last year."
    with fitz.open() as pdf:
        pdf.new_page().insert_text((50, 100), first + "\n" + second)
        pdf.save(path)
    passages = [text for chunk in chunk_pages(parse_pdf(path)) for text in evidence_passages(chunk.text)]
    assert first + " " + second in passages


@pytest.mark.parametrize("verb", ["achieved", "generated", "reported", "recorded"])
def test_financial_prose_with_multiple_figures_is_not_discarded(verb):
    assert usable_prose(
        f"The company {verb} revenue of $7.42 billion, representing 16 percent growth."
    )


def test_unlabelled_numeric_rows_remain_excluded():
    assert not usable_prose("Subscription revenue and related services $7,420 $6,396 16%")


def test_section_label_is_not_an_answer_passage():
    assert not usable_prose("Cash flows from operating activities:")
