#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MiniMax expression adapter for Pipecat."""

from typing import Any

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.expressive.expression import (
    FAST_DELIVERY,
    SLOW_DELIVERY,
    SPEED_DELIVERY,
    DeliveryScope,
    ExpressionCapabilities,
    ExpressionTerm,
)

# The speed each speed delivery asks for.
MINIMAX_SPEEDS = {SLOW_DELIVERY: 0.75, FAST_DELIVERY: 1.25}


class MiniMaxExpressionAdapter(BaseExpressionAdapter):
    """Performs expression tags with MiniMax speech-2.8 models.

    Part of expressive conversation (alpha). Its deliveries are MiniMax's emotions and
    speeds, sent with each request, so a delivery can change only between
    sentences, and a sentence written in several emotions is spoken in the
    first. Sounds are its
    interjections, such as ``(laughs)``, and pauses are ``<#s#>`` markers. Its
    vocabulary is fixed: an interjection outside it is read aloud.
    """

    def capabilities(self) -> ExpressionCapabilities:
        """MiniMax's emotions, interjections and pauses."""
        return ExpressionCapabilities(
            sounds=[
                ExpressionTerm(name)
                for name in (
                    "laughs chuckle coughs clear-throat groans breath pant inhale exhale gasps "
                    "sniffs sighs snorts burps lip-smacking humming hissing emm sneezes"
                ).split()
            ],
            delivery=[
                ExpressionTerm(name)
                for name in ("happy", "sad", "angry", "fearful", "disgusted", "surprised", "calm")
            ]
            + SPEED_DELIVERY,
            pause_range_ms=(10, 99990),
            delivery_scope=DeliveryScope.SENTENCE,
        )

    def request_settings(self, delivery: list[str]) -> dict[str, Any]:
        """The emotion and speed a request in this delivery asks for, merged into its ``voice_setting``."""
        settings: dict[str, Any] = {}
        emotion = next((name for name in delivery if name not in MINIMAX_SPEEDS), None)
        if emotion is not None:
            settings["emotion"] = emotion
        speed = next((MINIMAX_SPEEDS[name] for name in delivery if name in MINIMAX_SPEEDS), None)
        if speed is not None:
            settings["speed"] = speed
        return settings

    def format_sound(self, name: str) -> str:
        """Format an interjection."""
        return f"({name})"

    def format_pause(self, duration_ms: int) -> str:
        """Format a pause marker."""
        return f"<#{duration_ms / 1000:g}#>"
