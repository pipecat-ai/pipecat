#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Text processing utilities for sentence boundary detection and tag parsing.

This module provides utilities for natural language text processing including
sentence boundary detection, email and number pattern handling, and XML-style
tag parsing for structured text content.

Dependencies:
    This module uses NLTK (Natural Language Toolkit) for robust sentence
    tokenization. NLTK is licensed under the Apache License 2.0.
    See: https://www.nltk.org/
    Source: https://www.nltk.org/api/nltk.tokenize.punkt.html
"""

import re
from typing import FrozenSet, Optional, Sequence, Tuple

import nltk
from nltk.tokenize import sent_tokenize

# Ensure punkt_tab tokenizer data is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

SENTENCE_ENDING_PUNCTUATION: FrozenSet[str] = frozenset(
    {
        # Latin script punctuation (most European languages, Filipino, etc.)
        ".",
        "!",
        "?",
        ";",
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
        # Thai
        "।",  # Thai uses Devanagari-style punctuation in some contexts
        # Myanmar/Burmese
        "၊",  # Myanmar sign little section
        "။",  # Myanmar sign section
        # Khmer
        "។",  # Khmer sign khan
        "៕",  # Khmer sign bariyoosan
        # Lao
        "໌",  # Lao cancellation mark (used as period)
        "༎",  # Tibetan mark delimiter tsheg bstar (also used in Lao contexts)
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

StartEndTags = Tuple[str, str]


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


def match_endofsentence(text: str) -> int:
    """Find the position of the end of a sentence in the provided text.

    This function uses NLTK's sentence tokenizer to detect sentence boundaries
    in the input text, combined with punctuation verification to ensure that
    single tokens without proper sentence endings aren't considered complete sentences.

    Args:
        text: The input text in which to find the end of the sentence.

    Returns:
        The position of the end of the sentence if found, otherwise 0.
    """
    text = text.rstrip()

    if not text:
        return 0

    # Use NLTK's sentence tokenizer to find sentence boundaries
    sentences = sent_tokenize(text)

    if not sentences:
        return 0

    first_sentence = sentences[0]

    # If there's only one sentence that equals the entire text,
    # verify it actually ends with sentence-ending punctuation.
    # This is required as NLTK may return a single sentence for
    # text that's a single word. In the case of LLM tokens, it's
    # common for text to be single words, so we need to ensure
    # sentence-ending punctuation is present.
    if len(sentences) == 1 and first_sentence == text:
        return len(text) if text and text[-1] in SENTENCE_ENDING_PUNCTUATION else 0

    # If there are multiple sentences, the first one is complete by definition
    # (NLTK found a boundary, so there must be proper punctuation)
    if len(sentences) > 1:
        return len(first_sentence)

    # Single sentence that doesn't equal the full text means incomplete
    return 0


def parse_start_end_tags(
    text: str,
    tags: Sequence[StartEndTags],
    current_tag: Optional[StartEndTags],
    current_tag_index: int,
) -> Tuple[Optional[StartEndTags], int]:
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
            return (None, current_tag_index)
        elif start_tag_count > end_tag_count:
            return ((start_tag, end_tag), len(text))
        elif start_tag_count == end_tag_count:
            return (None, len(text))

    return (None, current_tag_index)
