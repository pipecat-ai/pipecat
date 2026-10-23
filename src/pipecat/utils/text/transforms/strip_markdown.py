#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Strip Markdown formatting symbols before TTS synthesis."""

import re

from pipecat.frames.frames import TextType

# Stands in for an asterisk that is content rather than emphasis while the
# emphasis rules run. A private-use character, so it cannot occur in the text.
_STAR = ""


class StripMarkdown:
    r"""Transform that removes Markdown formatting symbols with no spoken equivalent.

    Strips bold/italic markers, backtick code spans, fenced code blocks, ATX
    headers, list bullets, blockquote markers and horizontal rules. Struck-through
    text (``~~like this~~``) is removed along with its markers, since it is text
    the reply takes back. A link or image is reduced to its label or alt text, so
    the URL is not read out, unless ``strip_urls`` is off.

    An asterisk that is content stays: one between digits (``2*3``), one escaped
    as ``\*``, or one standing alone between spaces (``Press * then #``).

    Outside of fenced code, struck-through text and link URLs, this transformer
    is alphanumeric-preserving: the normalized alnum sequence of the output is
    identical to the input, so the :class:`WordCompletionTracker` requires no
    segment-map overhead.

    :data:`strip_markdown` is an instance with the default options, for registering
    directly.

    Example::

        tts = CartesiaTTSService(
            text_transforms=[("*", StripMarkdown(strip_urls=False))],
        )
    """

    def __init__(self, *, strip_urls: bool = True):
        """Initialize the transform.

        Args:
            strip_urls: Reduce ``[label](url)`` and ``![alt](url)`` to the label or
                alt text. Enabled by default.
        """
        self._strip_urls = strip_urls

    async def __call__(self, text: str, text_type: str | TextType) -> str:
        """Remove Markdown formatting symbols from the text.

        Args:
            text: Input text, potentially containing Markdown formatting.
            text_type: Text type of the text frame (unused).

        Returns:
            Text with Markdown formatting symbols removed.
        """
        # Fenced code blocks (``` or ~~~)
        text = re.sub(r"```[\s\S]*?```", "", text)
        text = re.sub(r"~~~[\s\S]*?~~~", "", text)
        # Images and links, keeping the alt text or label
        if self._strip_urls:
            text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
        # Inline code
        text = re.sub(r"`([^`]+)`", r"\1", text)
        # Strikethrough, with the space before it so the words around it still join
        text = re.sub(r"[ \t]?~~(?=\S).+?(?<=\S)~~", "", text)
        # Horizontal rules (---, ***, * * *), before list bullets so a spaced one
        # isn't taken for an item
        text = re.sub(r"^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$\n?", "", text, flags=re.MULTILINE)
        # List bullets (-, +, *) at any indent, since nested items are indented
        text = re.sub(r"^([ \t]*)[-+*][ \t]+(?=\S)", r"\1", text, flags=re.MULTILINE)
        # Content asterisks: escaped (\*) or between digits (2*3, 2**3)
        text = re.sub(r"\\\*|(?<=\d)\*+(?=\d)", lambda m: _STAR * m.group(0).count("*"), text)
        # Bold+italic ***text*** or ___text___
        text = re.sub(r"\*{3}(.+?)\*{3}", r"\1", text)
        text = re.sub(r"_{3}(.+?)_{3}", r"\1", text)
        # Bold **text** or __text__
        text = re.sub(r"\*{2}(.+?)\*{2}", r"\1", text)
        text = re.sub(r"_{2}(.+?)_{2}", r"\1", text)
        # Italic *text* or _text_. The markers hug the text, so a standalone
        # asterisk ("Press * then #") is not an opener or a closer.
        text = re.sub(r"\*(?=\S)(.+?)(?<=\S)\*", r"\1", text)
        text = re.sub(r"\b_(.+?)_\b", r"\1", text)
        # ATX headers (# ## ### etc.)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        # Blockquotes
        text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
        return text.replace(_STAR, "*")


strip_markdown = StripMarkdown()
"""A :class:`StripMarkdown` with the default options.

Example::

    result = await strip_markdown("**Hello** and _world_", "*")
    # "Hello and world"

    tts = CartesiaTTSService(
        text_transforms=[("*", strip_markdown)],
    )
"""
