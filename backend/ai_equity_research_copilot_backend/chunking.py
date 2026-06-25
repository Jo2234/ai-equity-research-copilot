from __future__ import annotations

import re
from dataclasses import dataclass

from .embeddings import estimate_tokens, tokenize


HEADING_RE = re.compile(r"^(item\s+\d+[a-z]?\.?|[A-Z][A-Z0-9 ,&/-]{6,}|[A-Z][A-Za-z ]{3,}:)\s*$")


@dataclass(frozen=True)
class ParsedPage:
    page_number: int | None
    text: str


@dataclass(frozen=True)
class ChunkDraft:
    chunk_index: int
    text: str
    page_start: int | None
    page_end: int | None
    section_title: str | None
    token_count: int
    metadata: dict[str, object]


def detect_heading(line: str) -> str | None:
    cleaned = line.strip()
    if not cleaned or len(cleaned) > 120:
        return None
    if HEADING_RE.match(cleaned):
        return cleaned.rstrip(":")
    return None


def chunk_pages(
    pages: list[ParsedPage],
    target_tokens: int = 800,
    overlap_tokens: int = 80,
) -> list[ChunkDraft]:
    target_tokens = max(80, target_tokens)
    overlap_tokens = max(0, min(overlap_tokens, target_tokens // 3))
    units: list[tuple[str, int | None, str | None]] = []
    current_section: str | None = None

    for page in pages:
        for paragraph in _paragraphs(page.text):
            heading = detect_heading(paragraph)
            if heading:
                current_section = heading
            units.append((paragraph, page.page_number, current_section))

    chunks: list[ChunkDraft] = []
    current: list[tuple[str, int | None, str | None]] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = estimate_tokens(unit[0])
        if current and current_tokens + unit_tokens > target_tokens:
            chunks.append(_build_chunk(len(chunks), current))
            current = _overlap_units(current, overlap_tokens)
            current_tokens = sum(estimate_tokens(text) for text, _, _ in current)
        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append(_build_chunk(len(chunks), current))
    return chunks


def _paragraphs(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [block.strip() for block in re.split(r"\n\s*\n", normalized) if block.strip()]
    if len(blocks) <= 1:
        lines = [line.strip() for line in normalized.split("\n") if line.strip()]
        return lines or ([normalized.strip()] if normalized.strip() else [])
    return blocks


def _overlap_units(units: list[tuple[str, int | None, str | None]], overlap_tokens: int) -> list[tuple[str, int | None, str | None]]:
    if overlap_tokens <= 0:
        return []
    selected: list[tuple[str, int | None, str | None]] = []
    total = 0
    for unit in reversed(units):
        selected.append(unit)
        total += len(tokenize(unit[0]))
        if total >= overlap_tokens:
            break
    return list(reversed(selected))


def _build_chunk(index: int, units: list[tuple[str, int | None, str | None]]) -> ChunkDraft:
    text = "\n\n".join(unit[0] for unit in units).strip()
    pages = [unit[1] for unit in units if unit[1] is not None]
    section_title = next((unit[2] for unit in reversed(units) if unit[2]), None)
    return ChunkDraft(
        chunk_index=index,
        text=text,
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        section_title=section_title,
        token_count=estimate_tokens(text),
        metadata={"unit_count": len(units), "section_title": section_title},
    )
