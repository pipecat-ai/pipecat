#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User-defined text replacements for TTS preprocessing."""

import re
from collections.abc import Callable

from pipecat.frames.frames import AggregationType


def replace_text(
    replacements: list[tuple[str, str]],
) -> Callable[[str, str | AggregationType], object]:
    r"""Return a transform that applies a list of find-and-replace rules.

    Each rule is a ``(pattern, replacement)`` tuple. Patterns are treated as
    regular expressions; use ``re.escape(pattern)`` for literal string matching.

    Rules are applied in the order provided. Whether the resulting transform is
    alphanumeric-preserving depends on the replacements supplied.

    Patterns are compiled at construction time so invalid regex patterns raise
    :exc:`re.error` immediately rather than during live TTS processing.

    Args:
        replacements: Ordered list of ``(regex_pattern, replacement_string)`` pairs.

    Returns:
        An async transform callable compatible with ``text_transforms``.

    Example::

        transform = replace_text([
            (r"\bDr\.", "Doctor"),
            (r"\bSt\.", "Street"),
            (r"\bvs\b", "versus"),
        ])
        tts = CartesiaTTSService(text_transforms=[("*", transform)])
    """
    compiled = [(re.compile(pattern), replacement) for pattern, replacement in replacements]

    async def _transform(text: str, aggregation_type: str | AggregationType) -> str:
        for pattern, replacement in compiled:
            text = pattern.sub(replacement, text)
        return text

    return _transform
