#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Fish Audio expression adapter for Pipecat."""

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.expressive.expression import (
    EMPHASIS_TONE,
    FAST_TONE,
    SLOW_TONE,
    SPEED_TONES,
    ExpressionCapabilities,
    ExpressionTerm,
)

# The tag each speed tone is written as.
_SPEED_TAGS = {SLOW_TONE: "speaking slowly", FAST_TONE: "speaking quickly"}
# The tag that sets the pace back. A pace a tag sets lasts past the input it
# is sent in, until another tag changes it.
_NORMAL_PACE_TAG = "[normal pace]"

# The tags in Fish Audio's S2 emotion reference, sorted into sounds made at one
# point and deliveries that last. Tones are single words, since a tone tag
# separates tones by spaces, so multi-word tones such as "soft tone" aren't
# listed.
_S2_EVENTS = [
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

_S2_TONES = [
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

    Part of expressive conversation (alpha). Tones and events are both tags,
    such as ``[excited]`` or ``[laughing]``, written in free text: the terms
    listed are Fish Audio's S2 emotion reference, and others described in a
    few words work too. A tag applies within the input it is sent in, so each
    sentence states its tones from its start. There is no tag for the voice's
    baseline, so a return to it writes none. The speed tones are written as
    ``[speaking slowly]`` and ``[speaking quickly]``, and since a pace lasts
    until another tag changes it, a ``[normal pace]`` tag ends them, within a
    sentence or at the end of one. Emphasis is an ``[emphasis]`` tag
    before the stressed words, and a pause is a break or a long break, which
    is approximate.
    """

    def capabilities(self) -> ExpressionCapabilities:
        """Free-text tags, emphasis and approximate pauses."""
        return ExpressionCapabilities(
            events=list(_S2_EVENTS),
            tones=[*_S2_TONES, *SPEED_TONES],
            free_text_events=True,
            free_text_tones=True,
            pause_range_ms=(100, 1500),
            emphasis=True,
        )

    def format_text(self, text: str, tones: list[str], previous_tones: list[str] | None) -> str:
        """Prefix tags where the tones change, and emphasis before stressed words."""
        tags = [_SPEED_TAGS.get(tone, tone) for tone in tones if tone != EMPHASIS_TONE]
        previous = [
            _SPEED_TAGS.get(tone, tone) for tone in previous_tones or [] if tone != EMPHASIS_TONE
        ]
        prefix = "".join(f"[{tag}]" for tag in tags) if tags and tags != previous else ""
        if _has_speed(previous_tones or []) and not _has_speed(tones):
            prefix = _NORMAL_PACE_TAG + prefix
        if EMPHASIS_TONE in tones:
            prefix += "[emphasis]"
        return f"{prefix} {text}" if prefix else text

    def format_end(self, tones: list[str]) -> str:
        """Set the pace back after a request that ends in a speed tone."""
        return f" {_NORMAL_PACE_TAG}" if _has_speed(tones) else ""

    def format_event(self, name: str) -> str:
        """Format a tag."""
        return f"[{name}]"

    def format_pause(self, duration_ms: int) -> str:
        """Format a break under a second, otherwise a long break."""
        return "[break]" if duration_ms < 1000 else "[long-break]"


def _has_speed(tones: list[str]) -> bool:
    return any(tone in _SPEED_TAGS for tone in tones)
