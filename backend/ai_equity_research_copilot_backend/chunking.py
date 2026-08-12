from __future__ import annotations

import re
from dataclasses import dataclass

from .embeddings import estimate_tokens, tokenize
from .financial_tables import financial_table_facts


HEADING_RE = re.compile(r"^(item\s+\d+[a-z]?\.?|[A-Z][A-Z0-9 ,&/-]{6,}:?)\s*$")
FINANCIAL_YEAR_HEADING_RE = re.compile(
    r"^(?:19|20)\d{2}\s+(?:[A-Z][A-Za-z&/-]*\s+){0,7}"
    r"(?:Earnings(?:\s+Driver)?\s+Analysis|Financial\s+(?:Results|Highlights)|"
    r"Results(?:\s+of\s+Operations)?|Performance)(?:\s*\(\d+\))?:?$"
)
SEGMENT_HEADING_RE = re.compile(
    r"^(?:(?:[A-Z][A-Za-z/-]*|&|and|of|the)\s+){1,7}"
    r"(?:Bank|Banking|Management|Services|Products|Solutions|Computing|Cloud|Processes):?$"
)
FINANCIAL_SECTION_RE = re.compile(
    r"^(?:(?:[A-Z][A-Za-z/-]*|&|and|of|the|from|for|to)\s+){1,9}"
    r"(?:Resources|Returns|Activities|Operations|Management|Liquidity|Capital|Position|Results|Information):?$"
)
FINANCIAL_SECTION_TOPIC_RE = re.compile(
    r"\b(?:Liquidity|Capital|Cash|Financial|Credit|Investment|Risk|Results|"
    r"Upstream|Downstream|Energy|Earnings|Revenue|Sales|Income|Profit)\b"
)
HEADING_PROSE_RE = re.compile(
    r"\b(?:our|we|they|their|is|are|was|were|has|have|had|will|would|can|could|"
    r"increased?|decreased?|improved?|reported?|remains?|sufficient)\b", re.IGNORECASE
)
UNKNOWN_SECTION_BOUNDARY_RE = re.compile(
    r"^(?:(?:[A-Z][A-Za-z/-]*|&|and|of|the|from|for|to)\s+){1,9}"
    r"[A-Z][A-Za-z/-]*:?$"
)


@dataclass(frozen=True)
class ParsedPage:
    page_number: int | None
    text: str
    paragraphs: tuple[str, ...] | None = None


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
    # Recognize explicit, standalone financial titles without treating a
    # sentence or a numeric row as a section. Retain source punctuation.
    normalized = " ".join(cleaned.split())
    if FINANCIAL_YEAR_HEADING_RE.fullmatch(normalized):
        return cleaned
    if (not re.search(r"[\d$€£%]|[.!?]", normalized)
            and (SEGMENT_HEADING_RE.fullmatch(normalized)
                 or (FINANCIAL_SECTION_RE.fullmatch(normalized)
                     and FINANCIAL_SECTION_TOPIC_RE.search(normalized)
                     and not HEADING_PROSE_RE.search(normalized))
                 or normalized.rstrip(":") in {"Upstream", "Downstream"})):
        return cleaned
    if HEADING_RE.match(cleaned):
        if re.search(r"\d", cleaned) and not re.match(r"(?i)^item\s+\d", cleaned):
            return None
        return cleaned
    return None


def _unknown_section_boundary(text: str) -> bool:
    # A short standalone title can invalidate an old
    # scope even when its financial meaning is unknown. Do not infer a new one.
    normalized = " ".join(text.strip().split())
    return bool(
        len(normalized) <= 120
        and UNKNOWN_SECTION_BOUNDARY_RE.fullmatch(normalized)
        and not HEADING_PROSE_RE.search(normalized)
    )


def _numeric_table_neighbor(text: str) -> bool:
    cleaned = text.strip()
    return bool(cleaned and re.search(r"\d", cleaned)
                and re.fullmatch(r"[\d\s$€£,.()%+−/—–-]+", cleaned))


def _ambiguous_table_heading(
    paragraphs: list[tuple[str, int | None]], index: int, start: int,
    table_spans: set[tuple[int, int]],
) -> bool:
    # A table's category label is not a prose section. Verified row/context
    # offsets take precedence; numeric cells on both sides also expose a label
    # in an unsupported table without requiring guessed period columns.
    if any(left <= start < right for left, right in table_spans):
        return True
    heading = paragraphs[index][0]
    if not SEGMENT_HEADING_RE.fullmatch(" ".join(heading.split())):
        return False
    return bool(index > 0 and index + 1 < len(paragraphs)
                and _numeric_table_neighbor(paragraphs[index - 1][0])
                and _numeric_table_neighbor(paragraphs[index + 1][0]))


def chunk_pages(
    pages: list[ParsedPage],
    target_tokens: int = 800,
    overlap_tokens: int = 80,
) -> list[ChunkDraft]:
    target_tokens = max(80, target_tokens)
    overlap_tokens = max(0, min(overlap_tokens, target_tokens // 3))
    units: list[tuple[str, int | None, str | None]] = []
    current_section: str | None = None

    paragraphs = [(paragraph, page.page_number) for page in pages
                  for paragraph in (page.paragraphs if page.paragraphs is not None else _paragraphs(page.text))]
    source = "\n\n".join(paragraph for paragraph, _ in paragraphs)
    table_spans = {span for fact in financial_table_facts(source)
                   for span in (fact.context_span, fact.row_span) if span is not None}
    offset = 0
    for index, (paragraph, page_number) in enumerate(paragraphs):
        heading = detect_heading(paragraph)
        if heading:
            current_section = (None if _ambiguous_table_heading(paragraphs, index, offset, table_spans)
                               else heading)
        elif _unknown_section_boundary(paragraph):
            current_section = None
        units.append((paragraph, page_number, current_section))
        offset += len(paragraph) + 2

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
    joined = "\n\n".join(unit[0] for unit in units)
    text = joined.strip()
    leading = len(joined) - len(joined.lstrip())
    trailing = len(joined.rstrip())
    section_spans: list[dict[str, object]] = []
    offset = 0
    for source, _, section in units:
        start, end = max(offset, leading), min(offset + len(source), trailing)
        if start < end:
            section_spans.append({"start": start - leading, "end": end - leading, "section": section})
        offset += len(source) + 2
    pages = [unit[1] for unit in units if unit[1] is not None]
    section_title = next((unit[2] for unit in reversed(units) if unit[2]), None)
    return ChunkDraft(
        chunk_index=index,
        text=text,
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        section_title=section_title,
        token_count=estimate_tokens(text),
        metadata={"unit_count": len(units), "section_title": section_title,
                  "section_spans": section_spans},
    )
