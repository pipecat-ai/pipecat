#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import re

ENDOFSENTENCE_PATTERN_STR = r"""
    (?<![A-Z])       # Negative lookbehind: not preceded by an uppercase letter (e.g., "U.S.A.")
    (?<!\d)          # Negative lookbehind: not preceded by a digit (e.g., "1. Let's start")
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


def match_endofsentence(text: str) -> int:
    text = text.rstrip()

    # Find all emails.
    emails = list(EMAIL_PATTERN.finditer(text))

    # Replace email dots by ampersands so we can find the end of sentence.
    for email_match in emails:
        start = email_match.start()
        end = email_match.end()
        new_email = text[start:end].replace(".", "&")
        text = text[:start] + new_email + text[end:]

    # Match against the new text.
    match = ENDOFSENTENCE_PATTERN.search(text)

    return match.end() if match else 0
