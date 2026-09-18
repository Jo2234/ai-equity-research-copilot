"""Conservative, source-linked extraction of explicitly labelled table rows.

This parser never supplies missing periods, numbers or units from document names
or outside knowledge. Unsupported headers and mismatched rows are omitted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

_MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
_PERIOD = re.compile(rf"{_MONTH}\s+\d{{1,2}},?\s+(?:19|20)\d{{2}}|(?:19|20)\d{{2}}|[1-4]Q\d{{2}}", re.IGNORECASE)
_COLUMN = re.compile(rf"{_PERIOD.pattern}|(?:Percentage\s+)?Change|Capital ratio requirements", re.IGNORECASE)
_UNIT_CAPTION = re.compile(r"\((?:(?:in|USD)\s+)?(?:thousands|millions|billions|percent(?:age)?s?)(?:,[^)]*)?\)", re.IGNORECASE)
_APPROACH_GROUPS = re.compile(r"(?:Standardized|Advanced)(?:\s+(?:Standardized|Advanced))*", re.IGNORECASE)
_VALUE = re.compile(r"(?P<currency>[$€£])?\s*(?P<number>\(\s*-?\d[\d,]*(?:\.\d+)?\s*\)|[-+−]?\d[\d,]*(?:\.\d+)?|[—–])\s*(?P<percent>%?)")
_FOOTNOTE = re.compile(r"\([a-z]\)", re.IGNORECASE)
_SCALE = re.compile(r"\b(?:in|dollars in|USD)\s+(thousands|millions|billions)\b", re.IGNORECASE)
_PER_SHARE = re.compile(r"\bper(?:[ -]+(?:common|ordinary|preferred|diluted|basic))?[ -]+share\b", re.IGNORECASE)
_PROSE = re.compile(r"\b(?:was|were|increased|decreased|grew|reported|compared|during)\b", re.IGNORECASE)


@dataclass(frozen=True)
class FinancialTableRow:
    label: str
    source_excerpt: str
    header_span: tuple[int, int]
    row_span: tuple[int, int]
    context_span: tuple[int, int] | None = None


def _values(text: str) -> list[tuple[str | None, str, str]] | None:
    cleaned = _FOOTNOTE.sub("", text)
    matches = list(_VALUE.finditer(cleaned))
    if not matches or _VALUE.sub("", cleaned).strip():
        return None
    values = []
    for match in matches:
        number = match["number"]
        # Preserve the original sign, precision and punctuation. Dashes remain
        # missing cells for width validation, never converted into zero.
        values.append((None if number in {"—", "–"} else number,
                       match["currency"] or "", match["percent"]))
    return values


def _split_row(text: str) -> tuple[str, str] | None:
    # A numeric digit within a label (e.g. Tier 1) is not a cell: only accept
    # the trailing suffix if it consists entirely of values and footnotes.
    for match in _VALUE.finditer(text):
        label = text[:match.start()].strip()
        suffix = text[match.start():]
        # Currency figures inside an allowance/other note are part of the
        # label, not data columns. Retain a wrapped label for its next cells.
        if re.search(r"[$€£]|\b(?:and|or|of|at|for)\s*$", label, re.IGNORECASE):
            continue
        if label and re.search(r"[A-Za-z]", label) and _values(suffix):
            return _FOOTNOTE.sub("", label).strip(), suffix
    return None


def _header_columns(text: str, groups: tuple[str, ...] = ()) -> list[str] | None:
    text = _UNIT_CAPTION.sub("", _FOOTNOTE.sub("", text))
    matches = list(_COLUMN.finditer(text))
    if len([m for m in matches if _PERIOD.fullmatch(m.group())]) < 2:
        return None
    residue = _COLUMN.sub("", text)
    residue = re.sub(rf"{_MONTH}\s+\d{{1,2}},?", "", residue, flags=re.IGNORECASE)
    if residue.strip(" \t,|:"):
        return None
    columns = [" ".join(m.group().split()) for m in matches]
    periods = [column.lower() for column in columns if _PERIOD.fullmatch(column)]
    # Repeated years can belong to different durations or accounting approaches.
    # Without those grouping labels, assigning either value would be ambiguous.
    if groups:
        if len(groups) != len({group.lower() for group in groups}) or len(columns) % len(groups):
            return None
        width = len(columns) // len(groups)
        first = columns[:width]
        if any(columns[offset:offset + width] != first for offset in range(0, len(columns), width)):
            return None
        first_periods = [column.lower() for column in first if _PERIOD.fullmatch(column)]
        if len(first_periods) != len(set(first_periods)):
            return None
    elif len(periods) != len(set(periods)):
        return None
    return columns


def _header_piece(text: str) -> bool:
    # A date may wrap between its month/day and its year. Only known header
    # vocabulary is allowed; arbitrary captions cannot become columns.
    residue = _COLUMN.sub("", _UNIT_CAPTION.sub("", _FOOTNOTE.sub("", text)))
    residue = re.sub(rf"{_MONTH}\s+\d{{1,2}},?", "", residue, flags=re.IGNORECASE)
    residue = re.sub(r"\bPercentage\b", "", residue, flags=re.IGNORECASE)
    return bool(text) and not residue.strip(" \t,|:")


def financial_table_facts(text: str) -> list[FinancialTableRow]:
    """Return one source record per matching period-labelled row.

    Spans index the original input. Each excerpt contains the exact header and
    row, separated by an explicit omission marker when they are nonadjacent.
    Values, units, signs and grouped columns remain exactly as printed. No
    arithmetic, currency conversion or generated financial sentences are used.
    """
    lines: list[tuple[str, int, int]] = []
    offset = 0
    for raw in text.splitlines(keepends=True):
        cleaned = " ".join(raw.split())
        if cleaned:
            lines.append((cleaned, offset, offset + len(raw.rstrip("\r\n"))))
        offset += len(raw)
    # Some PDF/HTML extracts put currency symbols in their own cells/lines.
    # Join only a standalone symbol with the immediately following numeric cell.
    merged: list[tuple[str, int, int]] = []
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        if line[0] in {"$", "€", "£"} and line_index + 1 < len(lines) and _values(lines[line_index + 1][0]):
            following = lines[line_index + 1]
            merged.append((line[0] + " " + following[0], line[1], following[2]))
            line_index += 2
        else:
            merged.append(line)
            line_index += 1
    lines = merged
    facts: list[FinancialTableRow] = []
    currency_context = {}
    index = 0
    while index < len(lines):
        if not _header_piece(lines[index][0]):
            index += 1
            continue
        end = index
        while end < min(len(lines), index + 12) and _header_piece(lines[end][0]):
            end += 1
        groups: tuple[str, ...] = ()
        if index > 0 and _APPROACH_GROUPS.fullmatch(lines[index - 1][0]):
            groups = tuple(lines[index - 1][0].split())
        columns = _header_columns(" ".join(line[0] for line in lines[index:end]), groups)
        if columns is None:
            index = max(index + 1, end)
            continue
        caption_start = index - 1 if groups else index
        # Include adjacent explicit units/period captions and a compact title.
        for back in range(index - 1, max(-1, index - 5), -1):
            line = lines[back][0]
            if len(line) > 240 or _PROSE.search(line) or _values(line):
                break
            if _SCALE.search(line) or re.search(r"\b(?:ended|ending|statements|results|highlights)\b", line, re.IGNORECASE) or caption_start < index and len(line) < 100 and not re.search(r"\d", line):
                caption_start = back
            else:
                break
        header_span = (lines[caption_start][1], lines[end - 1][2])
        caption = text[header_span[0]:header_span[1]]
        section = ""
        section_index: int | None = None
        pending: list[tuple[str, int]] = []
        cursor = end
        while cursor < min(len(lines), end + 180):
            line = lines[cursor][0]
            if _header_piece(line) or _SCALE.search(line) or re.fullmatch(r"\(?\s*(?:in\s+)?percent(?:age)?s?\s*\)?", line, re.IGNORECASE) or len(line) > 280 or _PROSE.search(line) or line.lower().startswith(("see accompanying", "note ")):
                break
            split = _split_row(line)
            if split:
                label, suffix = split
                row_start = cursor
                if pending:
                    section, section_index = pending[-1]
                pending = []
            elif _values(line) and pending:
                label, row_start = pending[-1]
                if len(pending) > 1:
                    section, section_index = pending[-2]
                pending = []
                suffix = line
            elif _values(line):
                # No label: never attach orphan values to the preceding row.
                cursor += 1
                continue
            else:
                pending.append((_FOOTNOTE.sub("", line).strip(), cursor))
                pending = pending[-2:]
                cursor += 1
                continue
            row_end = cursor
            values = _values(suffix)
            while values is not None and len(values) < len(columns) and row_end + 1 < len(lines):
                more = lines[row_end + 1][0]
                if not _values(more):
                    break
                suffix += " " + more
                row_end += 1
                values = _values(suffix)
            if (values is not None and len(values) == len(columns)
                    and any(number is not None for number, _, _ in values)):
                changes = [i for i, column in enumerate(columns) if "change" in column.lower()]
                percents = [i for i, (_, _, percent) in enumerate(values) if percent]
                # A leading Change caption can be visually reordered during
                # extraction. Retain the row only when one explicit % cell
                # disambiguates it; never reorder or rewrite the source cells.
                if changes and changes[0] == 0 and not (len(changes) == len(percents) == 1):
                    cursor = row_end + 1
                    continue
                row_span = (lines[row_start][1], lines[row_end][2])
                source_row = text[row_span[0]:row_span[1]]
                per_share_section = bool(_PER_SHARE.search(section))
                if per_share_section and not re.fullmatch(r"(?:Net income:\s*)?(?:Basic|Diluted)", label, re.IGNORECASE):
                    section = ""
                    section_index = None
                contextual_label = f"{section.rstrip(':')}: {label}" if section and section != label else label
                context_span = ((lines[section_index][1], lines[section_index][2])
                                if section_index is not None else None)
                excerpt = caption + "\n...\n"
                if context_span is not None and context_span[0] < row_span[0]:
                    excerpt += text[context_span[0]:context_span[1]] + "\n...\n"
                excerpt += source_row
                # A percentage-exception caption does not identify an unmarked
                # margin/growth measure as money or percent. Omit ambiguous rows.
                percentage_exception = re.search(r"except[^)]*percent", caption, re.IGNORECASE)
                ambiguous_measure = re.search(r"\b(?:margin|growth|return on|yield)\b", label, re.IGNORECASE)
                if percentage_exception and ambiguous_measure and not all(currency or percent for _, currency, percent in values):
                    cursor = row_end + 1
                    continue
                fact = FinancialTableRow(contextual_label, excerpt, header_span, row_span, context_span)
                facts.append(fact)
                # A currency-bearing first row can provide source context for
                # later monetary rows. Exceptions never borrow that context.
                monetary = bool(_SCALE.search(caption)) and not (
                    re.search(r"\bdollars\b", caption, re.IGNORECASE)
                    or _PER_SHARE.search(contextual_label)
                    or re.search(r"\b(?:ratios?|rate|employees|headcount|employee count|shares)\b",
                                 contextual_label, re.IGNORECASE)
                    or re.match(r"(?:percentage|percent)\b", section, re.IGNORECASE)
                    or re.match(r"(?:percentage|percent)\b", label, re.IGNORECASE)
                    or re.search(r"\b(?:percentage|percent)$", label, re.IGNORECASE)
                )
                currency_context.setdefault(header_span, []).append((fact, values, monetary))
            cursor = row_end + 1
        index = max(index + 1, cursor)
    return _attach_currency_source(facts, currency_context, text)


def _attach_currency_source(facts: list[FinancialTableRow], tables: dict, text: str) -> list[FinancialTableRow]:
    """Retain the original first-row currency evidence, without inferring values."""
    contexts = {}
    headers = sorted(tables)
    for index, header in enumerate(headers):
        limit = headers[index + 1][0] if index + 1 < len(headers) else len(text)
        # Include unparsed rows in the conflict check. Unsupported layouts must
        # not silently hide a different currency within the same table region.
        if len(set(re.findall(r"[$€£]", text[header[0]:limit]))) != 1:
            continue
        first, first_cells, _ = tables[header][0]
        for fact, cells, monetary in tables[header][1:]:
            if monetary and any(number is not None and not currency and not percent and first_cell[1]
                                for (number, currency, percent), first_cell in zip(cells, first_cells, strict=True)):
                spans = {fact.header_span, first.row_span, fact.row_span}
                if fact.context_span is not None:
                    spans.add(fact.context_span)
                contexts[fact.row_span] = "\n...\n".join(text[slice(*span)] for span in sorted(spans))
    return [replace(fact, source_excerpt=contexts[fact.row_span]) if fact.row_span in contexts else fact
            for fact in facts]
