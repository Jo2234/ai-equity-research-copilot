from __future__ import annotations

from pathlib import Path

from .chunking import ParsedPage


TEXT_SUFFIXES = {".txt", ".md", ".text"}


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
                pages.append(ParsedPage(page_number=idx, text=page.get_text("text")))
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
