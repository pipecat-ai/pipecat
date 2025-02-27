#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import re

ENDOFSENTENCE_PATTERN_STR = r"""
    (?<![A-Z])       # Negative lookbehind: not preceded by an uppercase letter (e.g., "U.S.A.")
    (?<!\d\.\d)      # Not preceded by a decimal number (e.g., "3.14159")
    (?<!^\d\.)       # Not preceded by a numbered list item (e.g., "1. Let's start")
    (?<!\d\s[ap])    # Negative lookbehind: not preceded by time (e.g., "3:00 a.m.")
    (?<!Mr|Ms|Dr)    # Negative lookbehind: not preceded by Mr, Ms, Dr (combined bc. length is the same)
    (?<!Mrs)         # Negative lookbehind: not preceded by "Mrs"
    (?<!Prof)        # Negative lookbehind: not preceded by "Prof"
    (\.\s*\.\s*\.|[\.\?\!;])|   # Match a period, question mark, exclamation point, or semicolon
    (\。\s*\。\s*\。|[。？！；।])  # the full-width version (mainly used in East Asian languages such as Chinese, Hindi)
    $                # End of string
"""

ENDOFSENTENCE_PATTERN = re.compile(ENDOFSENTENCE_PATTERN_STR, re.VERBOSE)

EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")

NUMBER_PATTERN = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")


def replace_match(text: str, match: re.Match, old: str, new: str) -> str:
    start = match.start()
    end = match.end()
    replacement = text[start:end].replace(old, new)
    text = text[:start] + replacement + text[end:]
    return text


def match_endofsentence(text: str) -> int:
    text = text.rstrip()

    # Replace email dots by ampersands so we can find the end of sentence.
    emails = list(EMAIL_PATTERN.finditer(text))
    for email_match in emails:
        text = replace_match(text, email_match, ".", "&")

    # Replace number dots by ampersands so we can find the end of sentence.
    numbers = list(NUMBER_PATTERN.finditer(text))
    for number_match in numbers:
        text = replace_match(text, number_match, ".", "&")

    # Match against the new text.
    match = ENDOFSENTENCE_PATTERN.search(text)

    return match.end() if match else 0
