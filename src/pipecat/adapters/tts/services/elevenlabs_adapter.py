#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""ElevenLabs expression adapter for Pipecat."""

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

# The audio tag each speed delivery is written as.
_SPEED_TAGS = {SLOW_DELIVERY: "slowly", FAST_DELIVERY: "quickly"}

# The audio tags in ElevenLabs' prompting guide for Eleven v3 and v4, and a
# short breath, a light one beside the guide's deep one, sorted into sounds made
# at one point and deliveries that last. Their guide doesn't draw that line.
# Multi-word deliveries, such as accents, aren't listed; written in a tag,
# they're passed on as free text.
_AUDIO_TAG_SOUNDS = [
    ExpressionTerm("laughs"),
    ExpressionTerm("laughs harder"),
    ExpressionTerm("starts laughing"),
    ExpressionTerm("laughing"),
    ExpressionTerm("wheezing", "a wheezing laugh"),
    ExpressionTerm("giggles"),
    ExpressionTerm("giggling"),
    ExpressionTerm("chuckles"),
    ExpressionTerm("snorts"),
    ExpressionTerm("sighs"),
    ExpressionTerm("sighing"),
    ExpressionTerm("exhales"),
    ExpressionTerm("exhales sharply"),
    ExpressionTerm("inhales deeply"),
    ExpressionTerm("short breath"),
    ExpressionTerm("clears throat"),
    ExpressionTerm("gulps"),
    ExpressionTerm("swallows"),
    ExpressionTerm("woo", "an excited whoop"),
    ExpressionTerm("applause", "sound effect"),
    ExpressionTerm("clapping", "sound effect"),
    ExpressionTerm("gunshot", "sound effect"),
    ExpressionTerm("explosion", "sound effect"),
    ExpressionTerm("fart", "sound effect"),
]

_AUDIO_TAG_DELIVERY = [
    ExpressionTerm("happy"),
    ExpressionTerm("excited"),
    ExpressionTerm("delighted"),
    ExpressionTerm("amazed"),
    ExpressionTerm("impressed"),
    ExpressionTerm("surprised"),
    ExpressionTerm("curious"),
    ExpressionTerm("questioning"),
    ExpressionTerm("thoughtful"),
    ExpressionTerm("warmly"),
    ExpressionTerm("sympathetic"),
    ExpressionTerm("reassuring"),
    ExpressionTerm("professional"),
    ExpressionTerm("cautiously"),
    ExpressionTerm("nervously"),
    ExpressionTerm("sheepishly"),
    ExpressionTerm("sad"),
    ExpressionTerm("crying", "speaking while crying"),
    ExpressionTerm("desperately"),
    ExpressionTerm("panicking"),
    ExpressionTerm("frustrated"),
    ExpressionTerm("annoyed"),
    ExpressionTerm("angry"),
    ExpressionTerm("appalled"),
    ExpressionTerm("dismissive"),
    ExpressionTerm("sarcastic"),
    ExpressionTerm("deadpan"),
    ExpressionTerm("mischievously"),
    ExpressionTerm("dramatically"),
    ExpressionTerm("cute"),
    ExpressionTerm("whispers", "speaking in a whisper"),
    ExpressionTerm("muttering", "speaking under one's breath"),
    ExpressionTerm("singing"),
]


class ElevenLabsAudioTagExpressionAdapter(BaseExpressionAdapter):
    """Formats expression tags as Eleven v3 and v4 audio tags.

    Part of expressive conversation (alpha). Delivery and sounds are both audio
    tags, such as ``[excited]`` or ``[laughs]``, written in free text: the
    terms listed are the ones in ElevenLabs' prompting guide, and others
    described in a few words work too. Emphasis is written in capitals. A delivery
    tag is written where the delivery changes; Eleven v3 and v4 have no tag for
    the voice's baseline, so a return to it writes none. Slow and fast are
    written as ``[slowly]`` and ``[quickly]``. They don't support
    ``<break>``, so a pause is written as a short or long pause cue, which is
    approximate.
    """

    def capabilities(self) -> ExpressionCapabilities:
        """Free-text audio tags, emphasis and approximate pauses."""
        return ExpressionCapabilities(
            sounds=list(_AUDIO_TAG_SOUNDS),
            delivery=[*_AUDIO_TAG_DELIVERY, *SPEED_DELIVERY],
            allows_free_text_sounds=True,
            allows_free_text_delivery=True,
            pause_range_ms=(100, 3000),
            emphasis=True,
            service_vocabulary={
                ExpressionDelivery.CALM: ["calm", "reassuring"],
                ExpressionDelivery.APOLOGETIC: ["apologetic", "sheepishly"],
                ExpressionDelivery.WARM: ["warmly"],
                ExpressionDelivery.EXCITED: ["excited", "delighted"],
                ExpressionDelivery.GRATEFUL: ["grateful", "warmly"],
                ExpressionDelivery.CONFIDENT: ["confident", "professional"],
                ExpressionSound.LAUGH: ["laughs", "laughing"],
                ExpressionSound.CHUCKLE: ["chuckles"],
                ExpressionSound.SIGH: ["sighs", "sighing"],
                ExpressionSound.BREATH: ["quick breath"],
                # A policy that leaves out a kind's own name gets its next best.
                ExpressionSound.LIGHT_BREATH: ["quick breath", "short breath"],
                ExpressionSound.DEEP_BREATH: ["deep breath", "inhales deeply"],
                ExpressionSound.EXHALE: ["exhales softly", "exhales"],
                ExpressionDelivery.CAUTIOUS: ["cautiously"],
                ExpressionDelivery.NERVOUS: ["nervously"],
                ExpressionDelivery.SHEEPISH: ["sheepishly"],
                ExpressionDelivery.DESPERATE: ["desperately"],
                ExpressionDelivery.PANICKED: ["panicking"],
                ExpressionDelivery.MISCHIEVOUS: ["mischievously"],
                ExpressionDelivery.DRAMATIC: ["dramatically"],
                ExpressionDelivery.WHISPERING: ["whispers"],
                ExpressionSound.LAUGH_HARDER: ["laughs harder"],
                ExpressionSound.START_LAUGHING: ["starts laughing"],
                ExpressionSound.WHEEZE: ["wheezing"],
                ExpressionSound.GIGGLE: ["giggles", "giggling"],
                ExpressionSound.SNORT: ["snorts"],
                ExpressionSound.SHARP_EXHALE: ["exhales sharply"],
                ExpressionSound.CLEAR_THROAT: ["clears throat"],
                ExpressionSound.GULP: ["gulps"],
                ExpressionSound.SWALLOW: ["swallows"],
            },
        )

    def format_text(
        self, text: str, delivery: list[str], previous_delivery: list[str] | None
    ) -> str:
        """Prefix audio tags where the delivery changes, and capitalize emphasis."""
        if EMPHASIS_DELIVERY in delivery:
            text = text.upper()
        tags = delivery_tags(delivery, _SPEED_TAGS)
        previous = delivery_tags(previous_delivery, _SPEED_TAGS)
        if tags and tags != previous:
            text = " ".join(f"[{tag}]" for tag in tags) + " " + text
        return text

    def format_sound(self, name: str) -> str:
        """Format an audio tag, set apart so its alignment times it as a word of its own."""
        return f" [{name}] "

    def format_pause(self, duration_ms: int) -> str:
        """Format a short pause under a second, otherwise a long one, set apart like a sound."""
        return " [short pause] " if duration_ms < 1000 else " [long pause] "
