"""Bounded source passages shared by retrieval and extractive answers."""

from __future__ import annotations

import re


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
YEAR_HEADER = re.compile(r"\b(?:19|20)\d{2}\b")


def evidence_passages(text: str) -> list[str]:
    passages: list[str] = []
    # Keep paragraph boundaries: flattening an entire chunk can attach a table
    # to its first sentence and cause useful prose to exceed the size limit.
    for paragraph in re.split(r"\n\s*\n", text):
        for sentence in SENTENCE_RE.split(" ".join(paragraph.split())):
            if usable_prose(sentence):
                passages.append(sentence)

    lines = text.splitlines()
    header: int | None = None
    header_end = 0
    for index, line in enumerate(lines):
        cleaned = " ".join(line.split())
        if (len(YEAR_HEADER.findall(cleaned)) >= 2 and len(cleaned) < 160
                and not re.search(r"\b(?:was|were|from|to|increased|decreased)\b", cleaned, re.I)):
            # Include the adjacent period/unit caption, when present.
            header = index
            header_end = index
            while header > max(0, index - 6):
                previous = lines[header - 1].strip()
                if len(previous) > 120 or re.search(r"\d", previous):
                    break
                header -= 1
        elif header is not None and index - header <= 25:
            numbers = re.findall(r"(?<!\w)\(?-?\d[\d,.]*\)?\s*%?", cleaned)
            # Only quote a compact table with its original column labels. Never
            # infer missing headers, currencies, signs or column assignments.
            if (len(numbers) >= 2 and re.search(r"[A-Za-z]", cleaned)
                    and len(cleaned) < 180 and not usable_prose(cleaned)):
                caption = "\n".join(lines[header:header_end + 1]).strip()
                excerpt = caption + "\n...\n" + line.strip()
                if len(excerpt) <= 1800:
                    passages.append(excerpt)
        if usable_prose(cleaned):
            header = None
            passages.extend(s for s in SENTENCE_RE.split(cleaned) if usable_prose(s))
    return list(dict.fromkeys(passages))


def usable_prose(sentence: str) -> bool:
    if not 35 <= len(sentence) <= 650 or "table of contents" in sentence.lower():
        return False
    if sentence.count("▪") + sentence.count("•") > 2 or sentence.count(";") > 5:
        return False
    # Rows without known column headings must not masquerade as prose.
    if len(re.findall(r"(?<!\w)\(?-?\d[\d,.]*\)?", sentence)) >= 2 and not re.search(
        r"\b(?:was|were|is|are|grew|rose|fell|increased|decreased|declined|totaled|reached)\b|\b(?:in|for|during)\s*(?:FY)?20\d{2}\b", sentence, re.I
    ):
        return False
    return sum(char.isalpha() for char in sentence) / len(sentence) > 0.55
