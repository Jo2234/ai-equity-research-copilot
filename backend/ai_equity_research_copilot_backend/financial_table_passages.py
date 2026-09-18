"""Build question-independent row and metric passages from verified table facts."""
from __future__ import annotations

from collections import defaultdict

MAX_BUNDLE_CHARACTERS = 2400
MAX_BUNDLE_ROWS = 8
_OMISSION = "\n...\n"


def _span(value: object) -> tuple[int, int] | None:
    if (isinstance(value, (list, tuple)) and len(value) == 2
            and all(isinstance(offset, int) and not isinstance(offset, bool) for offset in value)
            and 0 <= value[0] < value[1]):
        return value[0], value[1]
    return None


def _join_unique(values: list[str], separator: str) -> str:
    return separator.join(dict.fromkeys(values))


def table_passages(facts: list[dict]) -> list[tuple[str, str]]:
    """Return ``(row_label, source_excerpt)`` row and multirow passages.

    Each input must be a verified parser fact with label, source_excerpt,
    header_span and row_span. Page metadata is included in group
    identity when supplied. Facts without page metadata must come from a single
    parsed source (such as one chunk); callers must never concatenate anonymous
    records from separate pages or documents.

    Individual rows retain every supplied column. Additional bundles contain
    either the same metric across sections or an entire small unsectioned table.
    Bundles never cross headers/pages or truncate rows to meet the size limit.
    Labels support relevance scoring independently of citation context. Excerpts
    retain original units, accounting notation and precision.
    Nonadjacent excerpts use explicit omission markers.
    """
    tables: dict[tuple, dict[tuple, list[dict]]] = defaultdict(dict)
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        if any(not isinstance(fact.get(key), str) or not fact[key].strip()
               for key in ("label", "source_excerpt")):
            continue
        header = _span(fact.get("header_span"))
        row = _span(fact.get("row_span"))
        if header is None or row is None or header[1] > row[0]:
            continue
        # The exact header source supplements offsets: different extracts can
        # have identical offsets but different units or accounting columns.
        header_source = fact.get("header_text")
        if not isinstance(header_source, str) or not header_source:
            header_source = fact["source_excerpt"].split(_OMISSION, 1)[0]
        page_identity = tuple(str(fact.get(key, "")) for key in (
            "document_id", "source_page_index", "source_page", "source_page_sha256"))
        table_key = (page_identity, header, header_source)
        row_key = (row, fact["label"])
        tables[table_key].setdefault(row_key, []).append(fact)

    result: list[tuple[str, str]] = []
    for (_, _, header_source), rows in tables.items():
        passages = []
        for (_, label), columns in sorted(rows.items(), key=lambda entry: entry[0][0]):
            excerpt = _join_unique([column["source_excerpt"] for column in columns], _OMISSION)
            passage = (label, excerpt)
            result.append(passage)
            passages.append((label, passage))

        def bundle(selected: list[tuple[str, tuple[str, str]]], header_source: str = header_source) -> None:
            if not 2 <= len(selected) <= MAX_BUNDLE_ROWS:
                return
            excerpts = [passage[1] for _, passage in selected]
            # The table identity verifies one shared header. Keep it once, but
            # retain every following section/currency block in its original order.
            prefix = header_source + _OMISSION
            pieces = [excerpts[0]] + [excerpt.removeprefix(prefix) for excerpt in excerpts[1:]]
            excerpt = _OMISSION.join(pieces)
            if len(excerpt) <= MAX_BUNDLE_CHARACTERS:
                result.append(("\n".join(label for label, _ in selected), excerpt))

        if any(": " in label for label, _ in passages):
            metrics: dict[str, list[tuple[str, tuple[str, str]]]] = defaultdict(list)
            for label, passage in passages:
                metrics[label.rsplit(": ", 1)[-1]].append((label, passage))
            for selected in metrics.values():
                if selected[0][0].rsplit(": ", 1)[-1].lower() in {"other", "total", "net", "adjustments", "balance"}:
                    continue
                # Repeated identical labels may represent different sub-tables
                # with missing section captions. Do not imply distinct sections.
                if len({label for label, _ in selected}) == len(selected):
                    bundle(selected)
        elif len({label for label, _ in passages}) == len(passages):
            bundle(passages)
    return result
