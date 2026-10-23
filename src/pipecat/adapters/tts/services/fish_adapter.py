#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Fish Audio expression adapter for Pipecat."""

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.expressive.expression import ExpressionCapabilities, ExpressionTerm
from pipecat.expressive.parser import EMPHASIS_MODE

# The tags in Fish Audio's S2 emotion reference, sorted into sounds made at one
# point and deliveries that last. Modes are single words, since a mode tag
# separates modes by spaces, so multi-word tones such as "soft tone" aren't
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

_S2_MODES = [
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
    """Renders expression tags as Fish Audio S2 bracket tags.

    Part of expressive conversation (alpha). Modes and events are both tags,
    such as ``[excited]`` or ``[laughing]``, written in free text: the terms
    listed are Fish Audio's S2 emotion reference, and others described in a
    few words work too. A tag applies within the input it is sent in, so each
    sentence states its modes from its start. There is no tag for the voice's
    baseline, so a return to it writes none. Emphasis is an ``[emphasis]`` tag
    before the stressed words, and a pause is a break or a long break, which
    is approximate.
    """

    def capabilities(self) -> ExpressionCapabilities:
        """Free-text tags, emphasis and approximate pauses."""
        return ExpressionCapabilities(
            events=list(_S2_EVENTS),
            modes=list(_S2_MODES),
            free_text_events=True,
            free_text_modes=True,
            pause_range_ms=(100, 1500),
            emphasis=True,
        )

    def render_text(self, text: str, modes: list[str], previous_modes: list[str] | None) -> str:
        """Prefix tags where the modes change, and emphasis before stressed words."""
        tags = [mode for mode in modes if mode != EMPHASIS_MODE]
        previous = [mode for mode in previous_modes or [] if mode != EMPHASIS_MODE]
        prefix = "".join(f"[{tag}]" for tag in tags) if tags and tags != previous else ""
        if EMPHASIS_MODE in modes:
            prefix += "[emphasis]"
        return f"{prefix} {text}" if prefix else text

    def render_event(self, name: str) -> str:
        """Render a tag."""
        return f"[{name}]"

    def render_pause(self, duration_ms: int) -> str:
        """Render a break under a second, otherwise a long break."""
        return "[break]" if duration_ms < 1000 else "[long-break]"
