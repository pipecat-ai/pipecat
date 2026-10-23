#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""ElevenLabs expression adapter for Pipecat."""

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.expressive.expression import EMPHASIS_MODE, ExpressionCapabilities, ExpressionTerm

# The audio tags in ElevenLabs' prompting guide for Eleven v3 and v4, sorted
# into sounds made at one point and deliveries that last. Their guide doesn't
# draw that line. Modes are single words, since a mode tag separates modes by
# spaces, so multi-word deliveries such as accents aren't listed.
_AUDIO_TAG_EVENTS = [
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

_AUDIO_TAG_MODES = [
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

    Part of expressive conversation (alpha). Modes and events are both audio
    tags, such as ``[excited]`` or ``[laughs]``, written in free text: the
    terms listed are the ones in ElevenLabs' prompting guide, and others
    described in a few words work too. Emphasis is written in capitals. A mode
    tag is written where the modes change; Eleven v3 and v4 have no tag for
    the voice's baseline, so a return to it writes none. They don't support
    ``<break>``, so a pause is written as a short or long pause cue, which is
    approximate.
    """

    def capabilities(self) -> ExpressionCapabilities:
        """Free-text audio tags, emphasis and approximate pauses."""
        return ExpressionCapabilities(
            events=list(_AUDIO_TAG_EVENTS),
            modes=list(_AUDIO_TAG_MODES),
            free_text_events=True,
            free_text_modes=True,
            pause_range_ms=(100, 3000),
            emphasis=True,
        )

    def format_text(self, text: str, modes: list[str], previous_modes: list[str] | None) -> str:
        """Prefix audio tags where the modes change, and capitalize emphasis."""
        if EMPHASIS_MODE in modes:
            text = text.upper()
        tags = [mode for mode in modes if mode != EMPHASIS_MODE]
        previous = [mode for mode in previous_modes or [] if mode != EMPHASIS_MODE]
        if tags and tags != previous:
            text = " ".join(f"[{tag}]" for tag in tags) + " " + text
        return text

    def format_event(self, name: str) -> str:
        """Format an audio tag, set apart so its alignment times it as a word of its own."""
        return f" [{name}] "

    def format_pause(self, duration_ms: int) -> str:
        """Format a short pause under a second, otherwise a long one, set apart like an event."""
        return " [short pause] " if duration_ms < 1000 else " [long pause] "
