"""Carry verified table headers across chunk boundaries without changing text.

Facts are parsed separately on each complete page. They belong to a chunk only
when its complete text maps unambiguously to the source and contains the complete
row. This intentionally omits partial rows and ambiguous repeated source text.
"""
from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_left
from dataclasses import dataclass, replace

from .chunking import ChunkDraft, ParsedPage
from .financial_tables import financial_table_facts

MAX_FACTS_PER_CHUNK = 128
MAX_METADATA_BYTES_PER_CHUNK = 131_072


@dataclass(frozen=True)
class _PageSource:
    page: ParsedPage
    index: int
    start: int
    normalized: str
    word_starts: list[int]
    word_ends: list[int]
    normalized_starts: list[int]

    def span(self, original: tuple[int, int]) -> tuple[int, int] | None:
        start, end = original
        if not 0 <= start < end <= len(self.page.text):
            return None
        first = bisect_left(self.word_ends, start + 1)
        last = bisect_left(self.word_starts, end) - 1
        if first > last or last < 0:
            return None
        # A parser span must contain whole words, not a numeric substring.
        if self.word_starts[first] < start or self.word_ends[last] > end:
            return None
        left = self.normalized_starts[first]
        right = self.normalized_starts[last] + self.word_ends[last] - self.word_starts[last]
        return self.start + left, self.start + right


def _sources(pages: list[ParsedPage]) -> tuple[str, list[_PageSource]]:
    sources = []
    start = 0
    for index, page in enumerate(pages):
        words = list(re.finditer(r"\S+", page.text))
        offsets = []
        offset = 0
        for word in words:
            offsets.append(offset)
            offset += len(word.group()) + 1
        normalized = " ".join(word.group() for word in words)
        sources.append(_PageSource(page, index, start, normalized,
            [word.start() for word in words], [word.end() for word in words], offsets))
        start += len(normalized) + 1
    return " ".join(source.normalized for source in sources), sources


def _chunk_span(text: str, draft: ChunkDraft, sources: list[_PageSource]) -> tuple[int, int] | None:
    needle = " ".join(draft.text.split())
    if not needle:
        return None
    result = None
    cursor = 0
    while (start := text.find(needle, cursor)) >= 0:
        end = start + len(needle)
        cursor = start + 1
        if start and not text[start - 1].isspace() or end < len(text) and not text[end].isspace():
            continue
        numbers = [source.page.page_number for source in sources
                   if source.start < end and source.start + len(source.normalized) > start
                   and source.page.page_number is not None]
        if draft.page_start is not None and (not numbers or min(numbers) != draft.page_start):
            continue
        if draft.page_end is not None and (not numbers or max(numbers) != draft.page_end):
            continue
        if result is not None:
            return None
        result = start, end
    return result


def attach_table_context(pages: list[ParsedPage], drafts: list[ChunkDraft]) -> list[ChunkDraft]:
    """Return drafts with bounded, JSON-safe ``financial_table_facts`` metadata.

    Offsets are character positions in the original individual page, never in
    normalized text. ``source_page_index`` disambiguates missing/repeated page
    numbers; ``source_page_sha256`` fingerprints that original page. Excerpts
    are parser-produced exact source pieces with explicit omission markers.
    Text, embeddings, chunk indices and existing metadata remain unchanged.
    """
    text, sources = _sources(pages)
    facts = []
    for source in sources:
        digest = hashlib.sha256(source.page.text.encode("utf-8")).hexdigest()
        for fact in financial_table_facts(source.page.text):
            row = source.span(fact.row_span)
            header = source.span(fact.header_span)
            context = source.span(fact.context_span) if fact.context_span is not None else None
            if row is None or header is None or fact.context_span is not None and context is None:
                continue
            if header[1] > row[0] or context is not None and context[1] > row[0]:
                continue
            record = {
                "label": fact.label,
                "source_excerpt": fact.source_excerpt,
                "row_text": source.page.text[slice(*fact.row_span)],
                "header_text": source.page.text[slice(*fact.header_span)],
                "row_span": list(fact.row_span), "header_span": list(fact.header_span),
                "context_span": list(fact.context_span) if fact.context_span is not None else None,
                "source_page": source.page.page_number, "source_page_index": source.index,
                "source_page_sha256": digest,
            }
            size = len(json.dumps(record, ensure_ascii=False).encode("utf-8")) + 2
            facts.append((row, record, size))
    result = []
    for draft in drafts:
        metadata = {key: value for key, value in draft.metadata.items() if key != "financial_table_facts"}
        span = _chunk_span(text, draft, sources)
        selected = []
        used_bytes = 2
        seen = set()
        if span is not None:
            for row, record, size in facts:
                identity = (record["source_page_index"], tuple(record["header_span"]),
                            row, record["label"], record["source_excerpt"])
                if span[0] <= row[0] < row[1] <= span[1] and identity not in seen:
                    if len(selected) >= MAX_FACTS_PER_CHUNK or used_bytes + size > MAX_METADATA_BYTES_PER_CHUNK:
                        continue
                    selected.append(record)
                    seen.add(identity)
                    used_bytes += size
        if selected:
            metadata["financial_table_facts"] = selected
        result.append(replace(draft, metadata=metadata))
    return result
