#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Inworld expression adapter for Pipecat."""

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter, delivery_tags
from pipecat.expressive.expression import (
    EMPHASIS_DELIVERY,
    SPEED_DELIVERY,
    ExpressionCapabilities,
    ExpressionDelivery,
    ExpressionSound,
    ExpressionTerm,
)

# The tag that sets the delivery back to the voice's baseline. An instruction lasts
# past the sentence it is sent in, until another tag changes it.
_RESET_TAG = "[reset]"

# The non-verbal tags in Inworld's audio markup reference that a voice makes,
# leaving out those a conversation has no use for, such as [beatbox].
_SOUNDS = [
    ExpressionTerm(name)
    for name in (
        "breathe",
        "sigh",
        "gasp",
        "pant",
        "huff",
        "grunt",
        "groan",
        "moan",
        "laugh",
        "chuckle",
        "giggle",
        "cackle",
        "snort",
        "scoff",
        "cry",
        "sob",
        "whimper",
        "sniffle",
        "sniff",
        "shriek",
        "clear throat",
        "cough",
        "sneeze",
        "hiccup",
        "yawn",
        "burp",
        "snore",
        "swallow",
        "gulp",
        "tongue click",
        "lip smack",
        "shush",
        "whistle",
        "growl",
    )
]

_SOUND_VOCABULARY: dict[ExpressionDelivery | ExpressionSound, list[str]] = {
    ExpressionSound.BREATH: ["breathe"],
    ExpressionSound.LIGHT_BREATH: ["breathe"],
}


class InworldFlashExpressionAdapter(BaseExpressionAdapter):
    """Formats expression tags as Inworld TTS-2 Flash audio markup.

    Part of expressive conversation (alpha). Inworld TTS-2 Flash performs
    non-verbal tags, such as ``[laugh]``, and ``<break>`` pauses, and ignores
    instructions, so it has no delivery.
    """

    def capabilities(self) -> ExpressionCapabilities:
        """Inworld's non-verbal sounds and pauses."""
        return ExpressionCapabilities(
            sounds=list(_SOUNDS),
            pause_range_ms=(100, 10000),
            service_vocabulary=dict(_SOUND_VOCABULARY),
        )

    def format_sound(self, name: str) -> str:
        """Format a non-verbal tag, set apart so its alignment times it as a word of its own."""
        return f" [{name}] "

    def format_pause(self, duration_ms: int) -> str:
        """Format a break, set apart from the words."""
        return f' <break time="{duration_ms}ms"/> '


class InworldExpressionAdapter(InworldFlashExpressionAdapter):
    """Formats expression tags as Inworld TTS-2 audio markup.

    Part of expressive conversation (alpha). A delivery is an instruction tag,
    which Inworld TTS-2 reads as free text, so it performs any delivery
    described in a few words: ``<<calm>>`` is written ``[calm]``, and several
    deliveries at once, such as a slow span in a calm sentence, ``[calm and
    slow]``. An instruction lasts past the sentence it is sent in, so a return
    to the voice's baseline within a sentence writes ``[reset]``, and so does
    the end of a sentence spoken in a delivery. Emphasis is written in capitals.
    Sounds are its non-verbal tags, and pauses are ``<break>`` tags.
    """

    def capabilities(self) -> ExpressionCapabilities:
        """Free-text delivery, Inworld's non-verbal sounds, emphasis and pauses."""
        return ExpressionCapabilities(
            sounds=list(_SOUNDS),
            delivery=list(SPEED_DELIVERY),
            allows_free_text_delivery=True,
            pause_range_ms=(100, 10000),
            emphasis=True,
            service_vocabulary=dict(_SOUND_VOCABULARY),
        )

    def format_text(
        self, text: str, delivery: list[str], previous_delivery: list[str] | None
    ) -> str:
        """Prefix an instruction where the delivery changes, and capitalize emphasis."""
        if EMPHASIS_DELIVERY in delivery:
            text = text.upper()
        tags = delivery_tags(delivery, {})
        previous = delivery_tags(previous_delivery, {})
        if tags == previous:
            return text
        if tags:
            return f"[{' and '.join(tags)}] {text}"
        return f"{_RESET_TAG} {text}"

    def format_end(self, delivery: list[str]) -> str:
        """Set the delivery back after a sentence spoken in one."""
        return f" {_RESET_TAG}" if delivery_tags(delivery, {}) else ""
