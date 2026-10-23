#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MiniMax expression adapter for Pipecat."""

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.expressive.expression import ExpressionCapabilities, ExpressionTerm, ModeScope


class MiniMaxExpressionAdapter(BaseExpressionAdapter):
    """Performs expression tags with MiniMax speech-2.8 models.

    Part of expressive conversation (alpha). Modes are MiniMax's emotions, sent
    with each request, so a mode can change only between sentences, and a
    sentence written in several modes is spoken in the first. Events are its
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
            modes=[
                ExpressionTerm(name)
                for name in ("happy", "sad", "angry", "fearful", "disgusted", "surprised", "calm")
            ],
            pause_range_ms=(10, 99990),
            mode_scope=ModeScope.SENTENCE,
        )

    def render_event(self, name: str) -> str:
        """Render an interjection."""
        return f"({name})"

    def render_pause(self, duration_ms: int) -> str:
        """Render a pause marker."""
        return f"<#{duration_ms / 1000:g}#>"
