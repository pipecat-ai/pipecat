#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Soniox expression adapter for Pipecat."""

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter, delivery_tags
from pipecat.expressive.expression import (
    EMPHASIS_DELIVERY,
    FAST_DELIVERY,
    SLOW_DELIVERY,
    SPEED_DELIVERY,
    ExpressionCapabilities,
    ExpressionDelivery,
    ExpressionSound,
    ExpressionTerm,
)

# The tag each speed delivery is written as.
_SPEED_TAGS = {SLOW_DELIVERY: "slowly", FAST_DELIVERY: "quickly"}
# The tag that sets the delivery back. A tag lasts past the sentence it is sent
# in, since a stream carries a whole response.
_NEUTRAL_TAG = "[neutral]"

# The tags in Soniox's emotion and tone reference, leaving out those that
# change over the words they cover, such as "getting louder", and the pitch
# and voice quality tags, such as "raspy".
_DELIVERY = [
    ExpressionTerm(name)
    for name in (
        "happy sad angry excited nervous fearful surprised annoyed relieved disappointed "
        "curious delighted calm warm stern serious playful sarcastic flirty deadpan "
        "sincerely reassuringly dramatically mockingly whispering softly loudly shouting "
        "muttering hesitantly"
    ).split()
]

_SOUNDS = [
    ExpressionTerm(name)
    for name in (
        "laughs",
        "chuckles",
        "giggles",
        "sobs",
        "sniffles",
        "whimpers",
        "sighs",
        "exhales",
        "gasps",
        "grunts",
        "groans",
        "clears throat",
        "coughs",
        "yawns",
    )
]


class SonioxExpressionAdapter(BaseExpressionAdapter):
    """Formats expression tags as Soniox TTS audio tags.

    Part of expressive conversation (alpha). Delivery and sounds are both tags,
    such as ``[excited]`` or ``[laughs]``, from Soniox's emotion and tone
    reference; Soniox may read a tag outside it aloud. A tag lasts past the
    sentence it is sent in, so a return to the voice's baseline within a
    sentence writes ``[neutral]``, and so does the end of a sentence spoken in a
    delivery. Slow and fast are written as ``[slowly]`` and ``[quickly]``,
    emphasis as ``*word*``, and a pause as a pause or a long pause, which is
    approximate.
    """

    def capabilities(self) -> ExpressionCapabilities:
        """Soniox's tags, emphasis and approximate pauses."""
        return ExpressionCapabilities(
            sounds=list(_SOUNDS),
            delivery=[*_DELIVERY, *SPEED_DELIVERY],
            pause_range_ms=(100, 2000),
            emphasis=True,
            service_vocabulary={
                ExpressionDelivery.SYMPATHETIC: ["warm"],
                ExpressionDelivery.APOLOGETIC: ["sincerely"],
                ExpressionDelivery.GRATEFUL: ["warm"],
                ExpressionDelivery.FLIRTATIOUS: ["flirty"],
                ExpressionDelivery.SINCERE: ["sincerely"],
                ExpressionDelivery.REASSURING: ["reassuringly"],
                ExpressionDelivery.DRAMATIC: ["dramatically"],
                ExpressionDelivery.MOCKING: ["mockingly"],
                ExpressionDelivery.SOFT: ["softly"],
                ExpressionDelivery.LOUD: ["loudly"],
                ExpressionDelivery.HESITANT: ["hesitantly"],
                ExpressionSound.LAUGH: ["laughs"],
                ExpressionSound.CHUCKLE: ["chuckles"],
                ExpressionSound.GIGGLE: ["giggles"],
                ExpressionSound.SOB: ["sobs"],
                ExpressionSound.SNIFFLE: ["sniffles"],
                ExpressionSound.WHIMPER: ["whimpers"],
                ExpressionSound.SIGH: ["sighs"],
                ExpressionSound.BREATH: ["exhales"],
                ExpressionSound.EXHALE: ["exhales"],
                ExpressionSound.GASP: ["gasps"],
                ExpressionSound.GRUNT: ["grunts"],
                ExpressionSound.GROAN: ["groans"],
                ExpressionSound.CLEAR_THROAT: ["clears throat"],
                ExpressionSound.COUGH: ["coughs"],
                ExpressionSound.YAWN: ["yawns"],
            },
        )

    def format_text(
        self, text: str, delivery: list[str], previous_delivery: list[str] | None
    ) -> str:
        """Prefix tags where the delivery changes, and mark emphasis."""
        if EMPHASIS_DELIVERY in delivery:
            text = f"*{text}*"
        tags = delivery_tags(delivery, _SPEED_TAGS)
        previous = delivery_tags(previous_delivery, _SPEED_TAGS)
        if tags == previous:
            return text
        if tags:
            return " ".join(f"[{tag}]" for tag in tags) + " " + text
        return f"{_NEUTRAL_TAG} {text}"

    def format_end(self, delivery: list[str]) -> str:
        """Set the delivery back after a sentence spoken in one."""
        return f" {_NEUTRAL_TAG}" if delivery_tags(delivery, _SPEED_TAGS) else ""

    def format_sound(self, name: str) -> str:
        """Format a tag, set apart so its timestamps time it as a word of its own."""
        return f" [{name}] "

    def format_pause(self, duration_ms: int) -> str:
        """Format a pause under a second, otherwise a long pause, set apart like a sound."""
        return " [pause] " if duration_ms < 1000 else " [long pause] "
