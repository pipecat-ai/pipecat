#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Mistral expression adapter for Pipecat."""

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.expressive.expression import (
    FAST_DELIVERY,
    SLOW_DELIVERY,
    SPEED_DELIVERY,
    ExpressionCapabilities,
    ExpressionDelivery,
    ExpressionSound,
    ExpressionTerm,
)

# The pace each speed delivery is written as, and the pace that sets it back.
_PACES = {SLOW_DELIVERY: "slow", FAST_DELIVERY: "fast"}
_NORMAL_PACE = "moderate"

# The emotions in Mistral's emotion tag reference, leaving out "critical", which
# names no delivery of its own.
_EMOTIONS = [
    ExpressionTerm(name)
    for name in (
        "amused angry anxious apologetic appreciative assertive bored calm confident confused "
        "curious determined disappointed disgusted excited frustrated grateful happy hopeful "
        "playful proud reflective resigned sad sarcastic skeptical somber surprised tense "
        "tired urgent warm wistful"
    ).split()
]

_SOUNDS = [
    ExpressionTerm(name)
    for name in (
        "um",
        "uh",
        "laugh",
        "laugh hard",
        "chuckle",
        "breath",
        "sigh",
        "gasp",
        "sniff",
        "cough",
        "throat clearing",
        "yawn",
    )
]


class MistralExpressionAdapter(BaseExpressionAdapter):
    """Formats expression tags as Voxtral 3 TTS tags.

    Part of expressive conversation (alpha). A delivery is an emotion tag, such
    as ``[emotion=warm]``, from Mistral's fixed list, and slow and fast are pace
    tags, ``[pace=slow]`` and ``[pace=fast]``. A tag lasts until the next tag of
    its kind or the end of the request, so a sentence states its delivery from its
    start. Voxtral 3 has no emotion for the voice's baseline, so a return to it
    within a sentence writes none, while a return to the normal pace writes
    ``[pace=moderate]``. A request speaks one emotion at a time, so a delivery of
    several is spoken in the first. Sounds are tags such as ``[laugh]``. Voxtral 3
    has no pauses or emphasis.
    """

    def capabilities(self) -> ExpressionCapabilities:
        """Voxtral 3's emotions, paces and sounds."""
        return ExpressionCapabilities(
            sounds=list(_SOUNDS),
            delivery=[*_EMOTIONS, *SPEED_DELIVERY],
            service_vocabulary={
                ExpressionDelivery.SYMPATHETIC: ["warm"],
                ExpressionDelivery.GRATEFUL: ["grateful", "appreciative"],
                ExpressionDelivery.CONFIDENT: ["confident", "assertive"],
                ExpressionDelivery.THOUGHTFUL: ["reflective"],
                ExpressionDelivery.CONTEMPLATIVE: ["reflective"],
                ExpressionDelivery.NERVOUS: ["tense"],
                ExpressionDelivery.WORRIED: ["anxious"],
                ExpressionSound.UM: ["um", "uh"],
                ExpressionSound.LAUGH_HARDER: ["laugh hard"],
                ExpressionSound.LIGHT_BREATH: ["breath"],
                ExpressionSound.CLEAR_THROAT: ["throat clearing"],
            },
        )

    def format_text(
        self, text: str, delivery: list[str], previous_delivery: list[str] | None
    ) -> str:
        """Prefix an emotion tag and a pace tag where each changes."""
        emotion, pace = _emotion(delivery), _pace(delivery)
        previous = previous_delivery or []
        tags = ""
        if emotion is not None and emotion != _emotion(previous):
            tags += f"[emotion={emotion}]"
        if pace != _pace(previous):
            tags += f"[pace={pace or _NORMAL_PACE}]"
        return f"{tags} {text}" if tags else text

    def format_sound(self, name: str) -> str:
        """Format a tag."""
        return f"[{name}]"


def _emotion(delivery: list[str]) -> str | None:
    """The first name in the delivery that is an emotion, not a pace."""
    return next((name for name in delivery if name not in _PACES), None)


def _pace(delivery: list[str]) -> str | None:
    """The pace of the delivery's first speed, or None for the normal pace."""
    return next((_PACES[name] for name in delivery if name in _PACES), None)
