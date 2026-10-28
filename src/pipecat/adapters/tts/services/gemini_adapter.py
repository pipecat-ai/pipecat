#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gemini TTS expression adapter for Pipecat."""

from typing import Any

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter, delivery_tags
from pipecat.expressive.expression import (
    EMPHASIS_DELIVERY,
    FAST_DELIVERY,
    SLOW_DELIVERY,
    SPEED_DELIVERY,
    DeliveryScope,
    ExpressionCapabilities,
    ExpressionDelivery,
    ExpressionSound,
    ExpressionTerm,
)

# How each speed delivery is described in a request's style.
_SPEED_STYLES = {SLOW_DELIVERY: "speaking slowly", FAST_DELIVERY: "speaking quickly"}

# The audio tags in Gemini 3.8's speech generation guide that a voice makes,
# leaving out interjections, such as <phew>, and those naming a delivery, such as
# <whispering>.
_SOUNDS = [
    ExpressionTerm(name)
    for name in (
        "laugh",
        "chuckle",
        "giggle",
        "cackle",
        "sigh",
        "breath",
        "heavy breath",
        "exhales",
        "gasp",
        "pant",
        "cough",
        "throat-clearing",
        "sneeze",
        "yawn",
        "snort",
        "groan",
        "grunt",
        "moan",
        "growl",
        "hiss",
        "sob",
        "cry",
        "whimper",
        "shriek",
    )
]


class Gemini38ExpressionAdapter(BaseExpressionAdapter):
    """Performs expression tags with Gemini 3.8 TTS models.

    Part of expressive conversation (alpha). Gemini 3.8 reads its text as a
    verbatim transcript and takes delivery as a free-text style with each
    request, so it performs any delivery described in a few words, such as
    ``calm, speaking slowly``, and a delivery can change only between sentences. Sounds and pauses
    are its inline audio tags, such as ``<laugh>`` and ``<short pause>``, and
    emphasis is written in capitals.
    """

    def capabilities(self) -> ExpressionCapabilities:
        """Free-text delivery, Gemini's audio tags, emphasis and approximate pauses."""
        return ExpressionCapabilities(
            sounds=list(_SOUNDS),
            delivery=list(SPEED_DELIVERY),
            allows_free_text_delivery=True,
            pause_range_ms=(100, 2000),
            emphasis=True,
            delivery_scope=DeliveryScope.SENTENCE,
            service_vocabulary={
                ExpressionSound.LIGHT_BREATH: ["breath"],
                ExpressionSound.DEEP_BREATH: ["heavy breath"],
                ExpressionSound.EXHALE: ["exhales"],
                ExpressionSound.CLEAR_THROAT: ["throat-clearing"],
            },
        )

    def format_text(
        self, text: str, delivery: list[str], previous_delivery: list[str] | None
    ) -> str:
        """Capitalize emphasis; the delivery goes in the request's style."""
        return text.upper() if EMPHASIS_DELIVERY in delivery else text

    def request_settings(self, delivery: list[str]) -> dict[str, Any]:
        """The style a request in this delivery asks for, sent as its ``speech_metadata``."""
        styles = delivery_tags(delivery, _SPEED_STYLES)
        return {"style": ", ".join(styles)} if styles else {}

    def format_sound(self, name: str) -> str:
        """Format an audio tag."""
        return f" <{name}> "

    def format_pause(self, duration_ms: int) -> str:
        """Format a short pause under a second, otherwise a long pause."""
        return " <short pause> " if duration_ms < 1000 else " <long pause> "
