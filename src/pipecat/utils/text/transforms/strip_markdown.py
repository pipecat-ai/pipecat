#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Strip Markdown formatting symbols before TTS synthesis."""

import re

from pipecat.frames.frames import AggregationType


async def strip_markdown(text: str, aggregation_type: str | AggregationType) -> str:
    """Remove Markdown formatting symbols that have no spoken equivalent.

    Strips bold/italic markers, backtick code spans, fenced code blocks, ATX
    headers, and blockquote markers. Does not modify link or image syntax since
    those may carry meaningful text (the link label is preserved verbatim).

    This transformer is alphanumeric-preserving: the normalized alnum sequence of
    the output is identical to the input, so the :class:`WordCompletionTracker`
    requires no segment-map overhead.

    Args:
        text: Input text, potentially containing Markdown formatting.
        aggregation_type: Aggregation type of the text frame (unused).

    Returns:
        Text with Markdown formatting symbols removed.

    Example::

        result = await strip_markdown("**Hello** and _world_", "*")
        # "Hello and world"
    """
    # Fenced code blocks (``` or ~~~)
    text = re.sub(r"```[\s\S]*?```", "", text)
    text = re.sub(r"~~~[\s\S]*?~~~", "", text)
    # Inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Bold+italic ***text*** or ___text___
    text = re.sub(r"\*{3}(.+?)\*{3}", r"\1", text)
    text = re.sub(r"_{3}(.+?)_{3}", r"\1", text)
    # Bold **text** or __text__
    text = re.sub(r"\*{2}(.+?)\*{2}", r"\1", text)
    text = re.sub(r"_{2}(.+?)_{2}", r"\1", text)
    # Italic *text* or _text_
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"\b_(.+?)_\b", r"\1", text)
    # ATX headers (# ## ### etc.)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Blockquotes
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    # Horizontal rules
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    return text
