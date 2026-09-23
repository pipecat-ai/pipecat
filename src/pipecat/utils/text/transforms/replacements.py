#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User-defined text replacements for TTS preprocessing."""

import re
from collections.abc import Awaitable, Callable

from pipecat.frames.frames import AggregationType


class _ReplacementTransform:
    """Callable replacement transform with optional token-stream lookback metadata."""

    def __init__(self, replacements: list[tuple[str, str]], lookback_max_chars: int):
        self._compiled = [
            (re.compile(pattern), replacement) for pattern, replacement in replacements
        ]
        self._lookback_max_chars = lookback_max_chars

    @property
    def lookback_max_chars(self) -> int:
        """Return the configured maximum cross-token source match span."""
        return self._lookback_max_chars

    def safe_prefix_length(self, text: str, max_prefix_length: int) -> int:
        """Find a prefix whose replacement output is stable for the current text.

        Args:
            text: Buffered source text, including the trailing lookback.
            max_prefix_length: Largest prefix that leaves the configured lookback.

        Returns:
            A source prefix length that does not split a currently visible replacement.
        """
        transformed = self._apply(text)
        min_prefix_length = max(0, max_prefix_length - self._lookback_max_chars)
        for prefix_length in range(max_prefix_length, min_prefix_length - 1, -1):
            if transformed.startswith(self._apply(text[:prefix_length])):
                return prefix_length
        return 0

    def _apply(self, text: str) -> str:
        for pattern, replacement in self._compiled:
            text = pattern.sub(replacement, text)
        return text

    async def __call__(self, text: str, aggregation_type: str | AggregationType) -> str:
        return self._apply(text)


def replace_text(
    replacements: list[tuple[str, str]],
    *,
    lookback_max_chars: int = 0,
) -> Callable[[str, str | AggregationType], Awaitable[str]]:
    r"""Return a transform that applies a list of find-and-replace rules.

    Each rule is a ``(pattern, replacement)`` tuple. Patterns are treated as
    regular expressions; use ``re.escape(pattern)`` for literal string matching.

    Rules are applied in the order provided. Whether the resulting transform is
    alphanumeric-preserving depends on the replacements supplied.

    Patterns are compiled at construction time so invalid regex patterns raise
    :exc:`re.error` immediately rather than during live TTS processing.

    Args:
        replacements: Ordered list of ``(regex_pattern, replacement_string)`` pairs.
        lookback_max_chars: Maximum source match span to support across token
            boundaries when the transform is used with token-mode TTS. Set this to
            at least the longest source span that should match. ``0`` (default)
            preserves immediate per-token transformation.

    Returns:
        An async transform callable compatible with ``text_transforms``.

    Example::

        transform = replace_text([
            (r"\bDr\.", "Doctor"),
            (r"\bSt\.", "Street"),
            (r"\bvs\b", "versus"),
        ], lookback_max_chars=4)
        tts = CartesiaTTSService(text_transforms=[("*", transform)])
    """
    if lookback_max_chars < 0:
        raise ValueError("lookback_max_chars must be greater than or equal to 0")

    return _ReplacementTransform(replacements, lookback_max_chars)
