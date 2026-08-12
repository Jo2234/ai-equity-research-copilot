"""Bounded source passages shared by retrieval and extractive answers."""

from __future__ import annotations

import re


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
QUANTITY_RE = re.compile(r"(?:[$€£]\s*\d[\d,.]*|\d[\d,.]*\s*(?:%|percent\b|million\b|billion\b|trillion\b))", re.I)
CAUSE_RE = re.compile(r"\b(?:driven by|due to|because|reflect\w*|primarily|as a result|offset by)\b", re.I)
HEADLINE_RE = re.compile(
    r"^[•▪*-]?\s*(?P<label>[A-Za-z0-9][A-Za-z0-9 ()/&+.-]{3,110}?)\s+"
    r"[$€£]\s*\d[\d,]*(?:\.\d+)?(?:\s+(?:thousand|million|billion|trillion))?"
    r"\s*,?\s+(?:up|down)\s+\d+(?:\.\d+)?\s*(?:%|percent)"
    r"(?:\s+(?:year[- ]over[- ]year|sequentially|in\s+(?:USD|EUR|GBP)))?[.!]?$", re.I,
)
HEADLINE_METRIC_RE = re.compile(
    r"\b(?:revenue|revenues|sales|income|profit|cash\s+flows?|backlog|"
    r"remaining\s+performance\s+obligations|earnings\s+per\s+share|EPS)\b", re.I,
)
FRAGMENT_END_RE = re.compile(
    r"\b(?:and|or|but|for|from|with|of|in|to|by|as|than|between|during|at|the|a|an)\s*$"
    r"|[,;/]\s*$|\w-\s*$", re.I,
)


def evidence_passages(text: str) -> list[str]:
    passages: list[str] = []
    # Keep paragraph boundaries: flattening an entire chunk can attach a table
    # to its first sentence and cause useful prose to exceed the size limit.
    for paragraph in re.split(r"\n\s*\n", text):
        sentences = SENTENCE_RE.split(" ".join(paragraph.split()))
        for index, sentence in enumerate(sentences):
            if usable_prose(sentence):
                passages.append(sentence)
            # Preserve a result and its nearby explanation, without treating a
            # numeric row or unfinished sentence as prose through concatenation.
            for width in (2, 3):
                window = sentences[index:index + width]
                if len(window) != width or not all(
                    usable_prose(part) and part.endswith((".", "!", "?"))
                    and not re.search(r"[•▪]|^[-*]\s", part) for part in window
                ):
                    continue
                context = " ".join(window)
                continuation = bool(CAUSE_RE.search(" ".join(window[1:]))
                                    or re.match(r"(?:this|these|such|where|therefore|accordingly)\b", window[1], re.I)
                                    or re.search(r"\b(?:these|those|such)\b", window[1], re.I))
                if ((continuation or width == 2 and QUANTITY_RE.search(context))
                        and (width == 2 or QUANTITY_RE.search(context))
                        and usable_prose(context, max_chars=650 if width == 2 else 1000)):
                    passages.append(context)
    # A PDF highlights block may contain several independent labelled lines.
    # Tables are handled once by the dedicated source-linked table parser.
    passages.extend(sentence for line in text.splitlines()
                    for sentence in SENTENCE_RE.split(" ".join(line.split())) if usable_prose(sentence))

    return list(dict.fromkeys(passages))


def usable_prose(sentence: str, *, max_chars: int = 650) -> bool:
    headline = HEADLINE_RE.fullmatch(sentence.strip())
    labelled_headline = bool(headline and HEADLINE_METRIC_RE.search(headline["label"]))
    if (not (25 if labelled_headline else 35) <= len(sentence) <= max_chars or "table of contents" in sentence.lower()
            or sentence.rstrip().endswith(":")):
        return False
    # Physical PDF lines can end before the sentence does. Joined paragraph
    # candidates are evaluated separately; never present these fragments alone.
    if FRAGMENT_END_RE.search(sentence):
        return False
    if sentence.count("▪") + sentence.count("•") > 2 or sentence.count(";") > 5:
        return False
    # Rows without known column headings must not masquerade as prose.
    if not labelled_headline and len(re.findall(r"(?<!\w)\(?-?\d[\d,.]*\)?", sentence)) >= 2 and not re.search(
        r"\b(?:was|were|is|are|expect|grew|rose|fell|increased|decreased|declined|totaled|reached|achieved|generated|reported|recorded|represented|repurchased|spent|paid|returned|amounted)\b|\b(?:in|for|during)\s*(?:FY)?20\d{2}\b", sentence, re.I
    ):
        return False
    return labelled_headline or sum(char.isalpha() for char in sentence) / len(sentence) > 0.55
