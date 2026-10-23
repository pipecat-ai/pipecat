#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Fish Audio expression adapter for Pipecat."""

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.expressive.expression import EMPHASIS_TONE, ExpressionCapabilities, ExpressionTerm

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
    baseline, so a return to it writes none. Emphasis is an ``[emphasis]`` tag
    before the stressed words, and a pause is a break or a long break, which
    is approximate.
    """

    def capabilities(self) -> ExpressionCapabilities:
        """Free-text tags, emphasis and approximate pauses."""
        return ExpressionCapabilities(
            events=list(_S2_EVENTS),
            tones=list(_S2_TONES),
            free_text_events=True,
            free_text_tones=True,
            pause_range_ms=(100, 1500),
            emphasis=True,
        )

    def format_text(self, text: str, tones: list[str], previous_tones: list[str] | None) -> str:
        """Prefix tags where the tones change, and emphasis before stressed words."""
        tags = [tone for tone in tones if tone != EMPHASIS_TONE]
        previous = [tone for tone in previous_tones or [] if tone != EMPHASIS_TONE]
        prefix = "".join(f"[{tag}]" for tag in tags) if tags and tags != previous else ""
        if EMPHASIS_TONE in tones:
            prefix += "[emphasis]"
        return f"{prefix} {text}" if prefix else text

    def format_event(self, name: str) -> str:
        """Format a tag."""
        return f"[{name}]"

    def format_pause(self, duration_ms: int) -> str:
        """Format a break under a second, otherwise a long break."""
        return "[break]" if duration_ms < 1000 else "[long-break]"
