#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Text processing utilities for sentence boundary detection and tag parsing.

This module provides utilities for natural language text processing including
sentence boundary detection, email and number pattern handling, and XML-style
tag parsing for structured text content.

Sentence boundaries use sentencex's embedded language rules without external data.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache

# The native extension does not publish Python type stubs.
from sentencex import segment  # pyright: ignore[reportAttributeAccessIssue]

_QUOTED_WORD_END = re.compile(r"(?<!\w)([\"'“‘«‹„‟「『]+)[\w.]+\.$")


def resolve_sentence_tokenizer_language(language: str | None) -> str:
    """Normalize a language code for sentencex's language fallback map.

    Args:
        language: Language code, optionally with a region. None, empty, and
            automatic language selections use English.

    Returns:
        A lowercase base language code. Sentencex resolves unsupported codes.
    """
    if not isinstance(language, str):
        return "en"
    normalized = language.strip().lower().replace("_", "-").split("-")[0]
    return "en" if normalized in ("", "auto") else normalized


@cache
def _sent_tokenizer(language: str = "en") -> Callable[[str], list[str]]:
    """Return a sentence splitter that preserves source offsets before whitespace.

    Paragraph separators belong to the following sentence. Trailing whitespace
    stays in the aggregation buffer until that sentence is emitted.
    """
    code = resolve_sentence_tokenizer_language(language)

    def tokenize(text: str) -> list[str]:
        sentences = []
        pending = ""
        for span in segment(code, text):
            pending += span
            if span.strip():
                sentence = pending.rstrip()
                sentences.append(sentence)
                pending = pending[len(sentence) :]
        return sentences

    return tokenize


SENTENCE_ENDING_PUNCTUATION: frozenset[str] = frozenset(
    {
        # Latin script punctuation (most European languages, Filipino, etc.)
        ".",
        "!",
        "?",
        ";",
        "…",
        # East Asian punctuation (Chinese (Traditional & Simplified), Japanese, Korean)
        "。",  # Ideographic full stop
        "？",  # Full-width question mark
        "！",  # Full-width exclamation mark
        "；",  # Full-width semicolon
        "．",  # Full-width period
        "｡",  # Halfwidth ideographic period
        # Indic scripts punctuation (Hindi, Sanskrit, Marathi, Nepali, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Punjabi, Oriya, Assamese)
        "।",  # Devanagari danda (single vertical bar)
        "॥",  # Devanagari double danda (double vertical bar)
        # Arabic script punctuation (Arabic, Persian, Urdu, Pashto)
        "؟",  # Arabic question mark
        "؛",  # Arabic semicolon
        "۔",  # Urdu full stop
        "؏",  # Arabic sign misra (classical texts)
        # Myanmar/Burmese
        "၊",  # Myanmar sign little section
        "။",  # Myanmar sign section
        # Khmer
        "។",  # Khmer sign khan
        "៕",  # Khmer sign bariyoosan
        # Lao
        "໌",  # Lao cancellation mark (used as period)
        # Tibetan
        "།",  # Tibetan mark intersyllabic tsheg
        "༎",  # Tibetan mark delimiter tsheg bstar
        # Armenian
        "։",  # Armenian full stop
        "՜",  # Armenian exclamation mark
        "՞",  # Armenian question mark
        # Ethiopic script (Amharic)
        "።",  # Ethiopic full stop
        "፧",  # Ethiopic question mark
        "፨",  # Ethiopic paragraph separator
    }
)

# Latin punctuation that sentencex handles well — these need sentencex's disambiguation
# because "." can appear in abbreviations, decimals, etc.
_LATIN_SENTENCE_ENDING_PUNCTUATION: frozenset[str] = frozenset({".", "!", "?", ";", "…"})

# Non-Latin sentence-ending punctuation that is always unambiguous and never needs
# the tokenizer's disambiguation logic. Used for punctuation outside its rules.
UNAMBIGUOUS_SENTENCE_ENDING_PUNCTUATION: frozenset[str] = (
    SENTENCE_ENDING_PUNCTUATION - _LATIN_SENTENCE_ENDING_PUNCTUATION
)

StartEndTags = tuple[str, str]


def replace_match(text: str, match: re.Match, old: str, new: str) -> str:
    """Replace occurrences of a substring within a matched section of text.

    Args:
        text: The input text in which replacements will be made.
        match: A regex match object representing the section of text to modify.
        old: The substring to be replaced.
        new: The substring to replace `old` with.

    Returns:
        The modified text with the specified replacements made within the matched section.
    """
    start = match.start()
    end = match.end()
    replacement = text[start:end].replace(old, new)
    text = text[:start] + replacement + text[end:]
    return text


def match_endofsentence(text: str, *, language: str | None = None) -> int:
    """Find the position of the end of a sentence in the provided text.

    This function uses sentencex's sentence tokenizer to detect sentence boundaries
    in the input text, combined with punctuation verification to ensure that
    single tokens without proper sentence endings aren't considered complete sentences.

    Args:
        text: The input text in which to find the end of the sentence.
        language: Language code. Defaults to English.

    Returns:
        The position of the end of the sentence if found, otherwise 0.
    """
    text = text.rstrip()

    if not text:
        return 0

    sentences = _sent_tokenizer(resolve_sentence_tokenizer_language(language))(text)

    if not sentences:
        return 0

    first_sentence = sentences[0]

    tokenizer_text = text
    # With an unfinished quote such as 'She said, "Dr. S', sentencex can split
    # after '"Dr.'. Recheck a proposed split after a quoted word ending in a
    # period, letting sentencex decide whether that word is an abbreviation.
    if len(sentences) > 1 and (quote := _QUOTED_WORD_END.search(first_sentence)):
        # Group 1 contains only the opening quote characters. Replace them with
        # equal-width spaces so sentencex sees the abbreviation without the quote.
        # Boundary offsets still refer to the original text, which keeps its quotes.
        start, end = quote.span(1)
        tokenizer_text = text[:start] + " " * (end - start) + text[end:]
        sentences = _sent_tokenizer(resolve_sentence_tokenizer_language(language))(tokenizer_text)
        first_sentence = sentences[0]

    # A single span can be an incomplete fragment; require terminal punctuation.
    if len(sentences) == 1 and first_sentence == tokenizer_text:
        if text and text[-1] in SENTENCE_ENDING_PUNCTUATION:
            return len(text)
        # Additional punctuation can delimit sentences outside the selected rules.
        for i, ch in enumerate(text):
            if ch in UNAMBIGUOUS_SENTENCE_ENDING_PUNCTUATION:
                return i + 1
        return 0

    if len(sentences) > 1:
        return len(first_sentence)

    return 0


def parse_start_end_tags(
    text: str,
    tags: Sequence[StartEndTags],
    current_tag: StartEndTags | None,
    current_tag_index: int,
) -> tuple[StartEndTags | None, int]:
    """Parse text to identify start and end tag pairs.

    If a start tag was previously found (i.e., current_tag is valid), wait for
    the corresponding end tag. Otherwise, wait for a start tag.

    This function returns the index in the text where parsing should continue
    in the next call and the current or new tags.

    Args:
        text: The text to be parsed.
        tags: List of tuples containing start and end tags.
        current_tag: The currently active tags, if any.
        current_tag_index: The current index in the text.

    Returns:
        A tuple containing None or the current tag and the index of the text.
    """
    # If we are already inside a tag, check if the end tag is in the text.
    if current_tag:
        _, end_tag = current_tag
        if end_tag in text[current_tag_index:]:
            return (None, len(text))
        return (current_tag, current_tag_index)

    # Check if any start tag appears in the text
    for start_tag, end_tag in tags:
        start_tag_count = text[current_tag_index:].count(start_tag)
        end_tag_count = text[current_tag_index:].count(end_tag)
        if start_tag_count == 0 and end_tag_count == 0:
            # This tag pair does not appear in the text; keep scanning the
            # remaining pairs before deciding nothing is open.
            continue
        elif start_tag_count > end_tag_count:
            return ((start_tag, end_tag), len(text))
        elif start_tag_count == end_tag_count:
            return (None, len(text))

    return (None, current_tag_index)


def longest_trailing_partial_match(text: str, candidates: Sequence[str]) -> int:
    """Find the length of the longest suffix of text that is a proper prefix of a candidate.

    Used to detect a delimiter (e.g., an XML-style start tag) that has been
    split across two text chunks: the trailing partial delimiter can be held
    back from output until the next chunk completes it, rather than being
    flushed as plain text and losing the delimiter.

    Args:
        text: The text to check for a trailing partial match.
        candidates: Full strings to match a proper prefix of against the end of text.

    Returns:
        The length of the longest matching suffix, or 0 if no candidate has a
        proper prefix matching the end of text.
    """
    longest = 0
    for candidate in candidates:
        max_len = min(len(text), len(candidate) - 1)
        for length in range(max_len, longest, -1):
            if text[-length:] == candidate[:length]:
                longest = length
                break
    return longest


@dataclass
class TextPartForConcatenation:
    """Class representing a part of text for concatenation with concatenate_aggregated_text.

    Parameters:
        text: The text content.
        includes_inter_part_spaces: Whether any necessary inter-frame
            (leading/trailing) spaces are already included in the text.
    """

    text: str
    includes_inter_part_spaces: bool

    def __str__(self):
        return f"{type(self).__name__}(text: [{self.text}], includes_inter_part_spaces: {self.includes_inter_part_spaces})"


def concatenate_aggregated_text(text_parts: list[TextPartForConcatenation]) -> str:
    """Concatenate a list of text parts into a single string.

    This function joins the provided list of text parts into a single string,
    taking into account whether or not the parts already contain spacing.

    This function is useful for aggregating text segments received from LLMs or
    transcription services.

    Args:
        text_parts: A list of text parts to concatenate.

    Returns:
        A single concatenated string.
    """
    result = ""
    last_includes_inter_part_spaces = False

    if not text_parts:
        return result

    def append_part(part: TextPartForConcatenation):
        nonlocal result
        nonlocal last_includes_inter_part_spaces
        result += part.text
        last_includes_inter_part_spaces = part.includes_inter_part_spaces

    for part in text_parts:
        # Part is empty.
        # Skip.
        if not part.text:
            continue

        # Result is as yet empty.
        # Just append.
        if not result:
            append_part(part)
            continue

        if part.includes_inter_part_spaces and last_includes_inter_part_spaces:
            # This part is part of an ongoing run that has spaces already included.
            # Just append.
            append_part(part)
        elif not part.includes_inter_part_spaces and not last_includes_inter_part_spaces:
            # This part is part of an ongoing run that has no spaces included.
            # Add a space before appending.
            result += " "
            append_part(part)
        else:
            # This part represents a transition to a new run (spaces -> no spaces, or vice versa).
            # Add a space if needed, before appending.
            if not result[-1].isspace() and not part.text[0].isspace():
                result += " "
            append_part(part)

    # NOTE: the above logic assumes that runs of text parts with
    # includes_inter_part_spaces=True are well-formed, i.e. they're not
    # actually multiple separate runs with a space-less boundary, like
    # "hello ", "world.", "goodnight ", "moon."

    # Clean up any excessive whitespace
    result = result.strip()

    return result
