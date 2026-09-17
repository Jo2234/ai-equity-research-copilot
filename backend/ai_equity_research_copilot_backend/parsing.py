from __future__ import annotations

from pathlib import Path
import re

from .chunking import ParsedPage


TEXT_SUFFIXES = {".txt", ".md", ".text"}


def _pdf_paragraphs(blocks: list[tuple]) -> tuple[str, ...]:
    paragraphs: list[str] = []
    previous = None
    for block in blocks:
        text = block[4].strip()
        if block[6] != 0 or not text:
            continue
        # Some PDF producers emit one block per wrapped line. Join only a
        # nearby lowercase continuation of unfinished prose in the same column.
        # Keep table rows, bullets, completed sentences and column jumps apart.
        continuation = False
        if previous is not None and paragraphs:
            gap = block[1] - previous[3]
            line_height = min(previous[3] - previous[1], block[3] - block[1], 18)
            overlap = min(block[2], previous[2]) - max(block[0], previous[0])
            continuation = (
                text[0].islower()
                and paragraphs[-1][-1] not in ".!?:;"
                and len(re.findall(r"[A-Za-z]+", paragraphs[-1])) >= 5
                and 0 <= gap <= line_height
                and overlap > 40
                and block[0] <= previous[0] + 5
            )
        if continuation:
            paragraphs[-1] += "\n" + text
        else:
            paragraphs.append(text)
        previous = block
    return tuple(paragraphs)


def parse_document(path: Path) -> list[ParsedPage]:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return [ParsedPage(page_number=1, text=path.read_text(encoding="utf-8", errors="replace"))]
    if suffix == ".pdf":
        return parse_pdf(path)
    raise ValueError(f"Unsupported document type '{suffix}'. Upload .txt, .md, or .pdf files.")


def parse_pdf(path: Path) -> list[ParsedPage]:
    errors: list[str] = []
    try:
        import fitz  # type: ignore

        pages: list[ParsedPage] = []
        with fitz.open(path) as doc:
            for idx, page in enumerate(doc, start=1):
                # Preserve layout paragraph boundaries. Plain page text loses
                # them, causing the chunker to treat every wrapped PDF line as
                # a separate paragraph and truncate financial sentences.
                paragraphs = _pdf_paragraphs(page.get_text("blocks"))
                pages.append(ParsedPage(page_number=idx, text="\n\n".join(paragraphs), paragraphs=paragraphs))
        if any(page.text.strip() for page in pages):
            return pages
        errors.append("PyMuPDF extracted no text")
    except Exception as exc:  # pragma: no cover - depends on optional parser internals
        errors.append(f"PyMuPDF failed: {exc}")

    try:
        from PyPDF2 import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        pages = [
            ParsedPage(page_number=idx, text=page.extract_text() or "")
            for idx, page in enumerate(reader.pages, start=1)
        ]
        if any(page.text.strip() for page in pages):
            return pages
        errors.append("PyPDF2 extracted no text")
    except Exception as exc:  # pragma: no cover - depends on optional parser internals
        errors.append(f"PyPDF2 failed: {exc}")

    detail = "; ".join(errors) if errors else "no PDF parser is installed"
    raise RuntimeError(f"PDF parsing failed: {detail}")
