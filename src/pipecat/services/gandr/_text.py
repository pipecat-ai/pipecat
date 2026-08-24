#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Transcript handling with no dependency on Pipecat or the network.

Kept separate so it can be unit-tested on its own.
"""

from __future__ import annotations

from typing import List

#: The API caps a single request's transcript at this many characters.
MAX_REQUEST_CHARS = 2000


def split_for_request(text: str, limit: int = MAX_REQUEST_CHARS) -> List[str]:
    """Split *text* into pieces the API will accept, on the cleanest boundary.

    Prefers a sentence end, falls back to a word boundary, and only cuts inside
    a word when a single word is longer than the limit. The split is lossless:
    the pieces rejoin to the input, apart from whitespace at the seams.

    Args:
        text: The transcript to split.
        limit: Maximum characters per piece.

    Returns:
        A list of non-empty pieces, in order. Empty or whitespace-only input
        yields an empty list.

    Raises:
        ValueError: If ``limit`` is not positive.
    """
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")

    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    pieces: List[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
        if cut != -1:
            cut += 1  # keep the terminator with the piece it ends
        else:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = limit  # one unbroken token longer than the limit
        piece = remaining[:cut].strip()
        if piece:
            pieces.append(piece)
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


__all__ = ["split_for_request", "MAX_REQUEST_CHARS"]
