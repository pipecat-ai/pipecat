#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Self-closing tag aggregator for handling XML-style self-closing tags.

This module provides a generic text aggregator that can handle any self-closing
XML-style tags (e.g., <break time="0.1s"/>, <pause duration="500ms"/>, etc.)
that should prevent sentence boundary detection when incomplete during streaming.
"""

import re
from typing import List, Optional

from pipecat.utils.string import match_endofsentence
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator


class SelfClosingTagAggregator(BaseTextAggregator):
    r"""Aggregator that handles self-closing XML-style tags during streaming.

    This aggregator is designed to handle any self-closing tags that might appear
    in streaming text and could be split inappropriately when incomplete.
    It prevents sentence boundary detection only when tags are incomplete.

    The aggregator works by:

    1. Detecting incomplete self-closing tags during streaming (e.g., '<break time="0.')
    2. Buffering text until all tags are complete (e.g., '<break time="0.1s"/>')
    3. Applying normal sentence boundary detection once all tags are complete
    4. Supporting configurable tag patterns for different use cases

    Example usage::

        # For Cartesia break tags
        aggregator = SelfClosingTagAggregator(['break'])

        # For multiple tag types
        aggregator = SelfClosingTagAggregator(['break', 'pause', 'emphasis'])

        # For custom patterns
        aggregator = SelfClosingTagAggregator(
            patterns=[r'<break\\s+time="[^"]*"\\s*/?>', r'<pause\\s+duration="[^"]*"\\s*/>']
        )
    """

    def __init__(
        self,
        tags: Optional[List[str]] = None,
        patterns: Optional[List[str]] = None,
    ):
        """Initialize the self-closing tag aggregator.

        Args:
            tags: List of tag names to handle (e.g., ['break', 'pause']).
                 Will generate patterns like <break .../>
            patterns: List of custom regex patterns for complete tags.
                     Takes precedence over tags if provided.

        Raises:
            ValueError: If neither tags nor patterns are provided.
        """
        self._text = ""

        if patterns:
            # Use custom patterns
            self._complete_patterns = [re.compile(pattern) for pattern in patterns]
            # Generate incomplete patterns from complete ones
            self._incomplete_patterns = []
            for pattern in patterns:
                # Convert complete pattern to incomplete by making the closing part optional
                # This is a simple heuristic - for complex patterns, users should provide both
                incomplete = pattern.replace(r"\s*/?>", r"[^>]*$").replace(r"/>", r"[^>]*$")
                self._incomplete_patterns.append(re.compile(incomplete))
        elif tags:
            # Generate patterns from tag names
            self._complete_patterns = []
            self._incomplete_patterns = []

            for tag_name in tags:
                # Pattern for complete self-closing tags: <tagname .../>
                complete_pattern = rf"<{re.escape(tag_name)}\s+[^>]*\s*/?>"
                self._complete_patterns.append(re.compile(complete_pattern))

                # Pattern for incomplete tags: <tagname ... (without closing)
                incomplete_pattern = rf"<{re.escape(tag_name)}\s+[^>]*$"
                self._incomplete_patterns.append(re.compile(incomplete_pattern))
        else:
            raise ValueError("Must provide either 'tags' or 'patterns' parameter")

    @property
    def text(self) -> str:
        """Get the currently buffered text.

        Returns:
            The current text buffer content that hasn't been processed yet.
        """
        return self._text

    def _has_incomplete_tags(self, text: str) -> bool:
        """Check if the text ends with incomplete self-closing tags.

        Args:
            text: The text to check.

        Returns:
            True if the text ends with any incomplete tag patterns.
        """
        for pattern in self._incomplete_patterns:
            if pattern.search(text):
                return True
        return False

    async def aggregate(self, text: str) -> Optional[str]:
        """Aggregate text while being aware of self-closing tags.

        This method adds the new text to the buffer and checks for sentence
        boundaries. If tags are incomplete, it continues buffering until
        they are complete. Once all tags are complete, normal sentence
        detection applies.

        Args:
            text: New text to add to the buffer.

        Returns:
            Processed text up to a sentence boundary, or None if incomplete
            tags are present and more text is needed.
        """
        # Add new text to buffer
        self._text += text

        # Check if we have incomplete tags - if so, keep buffering
        if self._has_incomplete_tags(self._text):
            return None

        # No incomplete tags, use normal sentence detection
        eos_marker = match_endofsentence(self._text)
        if eos_marker:
            result = self._text[:eos_marker]
            self._text = self._text[eos_marker:]
            return result

        # No sentence boundary found yet
        return None

    async def handle_interruption(self):
        """Handle interruptions by clearing the buffer.

        Called when an interruption occurs in the processing pipeline,
        to reset the state and discard any partially aggregated text.
        """
        self._text = ""

    async def reset(self):
        """Clear the internally aggregated text.

        Resets the aggregator to its initial state, discarding any
        buffered text.
        """
        self._text = ""
