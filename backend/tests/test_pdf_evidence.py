import fitz
import pytest

from ai_equity_research_copilot_backend.chunking import chunk_pages
from ai_equity_research_copilot_backend.evidence import evidence_passages, usable_prose
from ai_equity_research_copilot_backend.parsing import _pdf_paragraphs, parse_pdf


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


def test_pdf_producer_with_one_block_per_wrapped_line(tmp_path):
    path = tmp_path / "split-blocks.pdf"
    first = "Operating cash flow was $9.4 billion"
    second = "during the fiscal year, up 17 percent."
    with fitz.open() as pdf:
        page = pdf.new_page()
        page.insert_text((90, 100), first, fontsize=12)
        page.insert_text((54, 121), second, fontsize=12)
        pdf.save(path)
    passages = [text for chunk in chunk_pages(parse_pdf(path)) for text in evidence_passages(chunk.text)]
    assert first + " " + second in passages


def test_pdf_continuations_do_not_cross_columns_or_completed_sentences():
    blocks = [
        (50, 50, 250, 62, "The company generated revenue from several different products", 0, 0),
        (300, 70, 500, 82, "during the prior fiscal year, up 14 percent.", 1, 0),
        (50, 100, 250, 112, "The company reported revenue of $2.4 billion.", 2, 0),
        (50, 120, 250, 132, "other amounts are presented in the following table.", 3, 0),
    ]
    assert len(_pdf_paragraphs(blocks)) == 4


@pytest.mark.parametrize("verb", ["achieved", "generated", "reported", "recorded"])
def test_financial_prose_with_multiple_figures_is_not_discarded(verb):
    assert usable_prose(
        f"The company {verb} revenue of $7.42 billion, representing 16 percent growth."
    )


def test_unlabelled_numeric_rows_remain_excluded():
    assert not usable_prose("Subscription revenue and related services $7,420 $6,396 16%")


def test_section_label_is_not_an_answer_passage():
    assert not usable_prose("Cash flows from operating activities:")
