#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MiniMax expression adapter for Pipecat."""

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.expressive.expression import (
    FAST_TONE,
    SLOW_TONE,
    SPEED_TONES,
    ExpressionCapabilities,
    ExpressionTerm,
    ToneScope,
)

# The speed each speed tone asks for.
MINIMAX_SPEEDS = {SLOW_TONE: 0.75, FAST_TONE: 1.25}


class MiniMaxExpressionAdapter(BaseExpressionAdapter):
    """Performs expression tags with MiniMax speech-2.8 models.

    Part of expressive conversation (alpha). Tones are MiniMax's emotions and
    speeds, sent with each request, so a tone can change only between
    sentences, and a sentence written in several emotions is spoken in the
    first. Events are its
    interjections, such as ``(laughs)``, and pauses are ``<#s#>`` markers. Its
    vocabulary is fixed: an interjection outside it is read aloud.
    """

    def capabilities(self) -> ExpressionCapabilities:
        """MiniMax's emotions, interjections and pauses."""
        return ExpressionCapabilities(
            events=[
                ExpressionTerm(name)
                for name in (
                    "laughs chuckle coughs clear-throat groans breath pant inhale exhale gasps "
                    "sniffs sighs snorts burps lip-smacking humming hissing emm sneezes"
                ).split()
            ],
            tones=[
                ExpressionTerm(name)
                for name in ("happy", "sad", "angry", "fearful", "disgusted", "surprised", "calm")
            ]
            + SPEED_TONES,
            pause_range_ms=(10, 99990),
            tone_scope=ToneScope.SENTENCE,
        )

    def format_event(self, name: str) -> str:
        """Format an interjection."""
        return f"({name})"

    def format_pause(self, duration_ms: int) -> str:
        """Format a pause marker."""
        return f"<#{duration_ms / 1000:g}#>"
