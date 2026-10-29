#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Fish Audio expression adapter for Pipecat."""

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
_SPEED_TAGS = {SLOW_DELIVERY: "speaking slowly", FAST_DELIVERY: "speaking quickly"}
# The tag that sets the pace back. A pace a tag sets lasts past the input it
# is sent in, until another tag changes it.
_NORMAL_PACE_TAG = "[normal pace]"

# The tags in Fish Audio's S2 emotion reference, sorted into sounds made at one
# point and deliveries that last. Multi-word deliveries, such as "soft tone",
# aren't listed; written in a tag, they're passed on as free text.
_S2_SOUNDS = [
    ExpressionTerm("laughing"),
    ExpressionTerm("chuckling"),
    ExpressionTerm("sobbing"),
    ExpressionTerm("crying loudly"),
    ExpressionTerm("sighing"),
    ExpressionTerm("groaning"),
    ExpressionTerm("panting"),
    ExpressionTerm("gasping"),
    ExpressionTerm("yawning"),
    ExpressionTerm("snoring"),
    ExpressionTerm("clear throat"),
    ExpressionTerm("audience laughing", "sound effect"),
    ExpressionTerm("background laughter", "sound effect"),
    ExpressionTerm("crowd laughing", "sound effect"),
]

_S2_DELIVERY = [
    ExpressionTerm(name)
    for name in (
        "happy sad angry excited calm nervous confident surprised satisfied delighted "
        "scared worried upset frustrated depressed empathetic embarrassed disgusted moved "
        "proud relaxed grateful curious sarcastic disdainful unhappy anxious hysterical "
        "indifferent uncertain doubtful confused disappointed regretful guilty ashamed "
        "jealous envious hopeful optimistic pessimistic nostalgic lonely bored contemptuous "
        "sympathetic compassionate determined resigned shouting screaming whispering"
    ).split()
]


class FishAudioS2ExpressionAdapter(BaseExpressionAdapter):
    """Formats expression tags as Fish Audio S2 bracket tags.

    Part of expressive conversation (alpha). Delivery and sounds are both tags,
    such as ``[excited]`` or ``[laughing]``, written in free text: the terms
    listed are Fish Audio's S2 emotion reference, and others described in a
    few words work too. A tag applies within the input it is sent in, so each
    sentence states its delivery from its start. There is no tag for the voice's
    baseline, so a return to it writes none. Slow and fast are written as
    ``[speaking slowly]`` and ``[speaking quickly]``, and since a pace lasts
    until another tag changes it, a ``[normal pace]`` tag ends them, within a
    sentence or at the end of one. Emphasis is an ``[emphasis]`` tag
    before the stressed words, and a pause is a break or a long break, which
    is approximate.
    """

    def capabilities(self) -> ExpressionCapabilities:
        """Free-text tags, emphasis and approximate pauses."""
        return ExpressionCapabilities(
            sounds=list(_S2_SOUNDS),
            delivery=[*_S2_DELIVERY, *SPEED_DELIVERY],
            allows_free_text_sounds=True,
            allows_free_text_delivery=True,
            pause_range_ms=(100, 1500),
            emphasis=True,
            service_vocabulary={
                ExpressionDelivery.CALM: ["calm", "relaxed"],
                ExpressionDelivery.SYMPATHETIC: ["sympathetic", "empathetic", "compassionate"],
                ExpressionDelivery.APOLOGETIC: ["regretful"],
                ExpressionDelivery.WARM: ["happy", "relaxed"],
                ExpressionDelivery.EXCITED: ["excited", "delighted"],
                ExpressionSound.LAUGH: ["laughing"],
                ExpressionSound.CHUCKLE: ["chuckling"],
                ExpressionSound.SIGH: ["sighing"],
                ExpressionSound.BREATH: ["inhaling"],
                ExpressionSound.LIGHT_BREATH: ["inhaling"],
                ExpressionSound.EXHALE: ["exhaling"],
                ExpressionSound.SOB: ["sobbing", "crying loudly"],
                ExpressionSound.GROAN: ["groaning"],
                ExpressionSound.PANT: ["panting"],
                ExpressionSound.GASP: ["gasping"],
                ExpressionSound.YAWN: ["yawning"],
                ExpressionSound.SNORE: ["snoring"],
            },
        )

    def format_text(
        self, text: str, delivery: list[str], previous_delivery: list[str] | None
    ) -> str:
        """Prefix tags where the delivery changes, and emphasis before stressed words."""
        tags = delivery_tags(delivery, _SPEED_TAGS)
        previous = delivery_tags(previous_delivery, _SPEED_TAGS)
        prefix = "".join(f"[{tag}]" for tag in tags) if tags and tags != previous else ""
        if _has_speed(previous_delivery or []) and not _has_speed(delivery):
            prefix = _NORMAL_PACE_TAG + prefix
        if EMPHASIS_DELIVERY in delivery:
            prefix += "[emphasis]"
        return f"{prefix} {text}" if prefix else text

    def format_end(self, delivery: list[str]) -> str:
        """Set the pace back after a request that ends in a speed delivery."""
        return f" {_NORMAL_PACE_TAG}" if _has_speed(delivery) else ""

    def format_sound(self, name: str) -> str:
        """Format a tag."""
        return f"[{name}]"

    def format_pause(self, duration_ms: int) -> str:
        """Format a break under a second, otherwise a long break."""
        return "[break]" if duration_ms < 1000 else "[long-break]"


def _has_speed(delivery: list[str]) -> bool:
    return any(name in _SPEED_TAGS for name in delivery)
