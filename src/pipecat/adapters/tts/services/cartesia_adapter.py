#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Cartesia expression adapter for Pipecat."""

from enum import StrEnum

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.expressive.expression import ExpressionCapabilities, ExpressionTerm, ModeScope


class CartesiaEmotion(StrEnum):
    """Predefined Emotions supported by Cartesia."""

    # Primary emotions supported by Cartesia
    NEUTRAL = "neutral"
    ANGRY = "angry"
    EXCITED = "excited"
    CONTENT = "content"
    SAD = "sad"
    SCARED = "scared"
    # Additional emotions supported by Cartesia
    HAPPY = "happy"
    ENTHUSIASTIC = "enthusiastic"
    ELATED = "elated"
    EUPHORIC = "euphoric"
    TRIUMPHANT = "triumphant"
    AMAZED = "amazed"
    SURPRISED = "surprised"
    FLIRTATIOUS = "flirtatious"
    JOKING_COMEDIC = "joking/comedic"
    CURIOUS = "curious"
    PEACEFUL = "peaceful"
    SERENE = "serene"
    CALM = "calm"
    GRATEFUL = "grateful"
    AFFECTIONATE = "affectionate"
    TRUST = "trust"
    SYMPATHETIC = "sympathetic"
    ANTICIPATION = "anticipation"
    MYSTERIOUS = "mysterious"
    MAD = "mad"
    OUTRAGED = "outraged"
    FRUSTRATED = "frustrated"
    AGITATED = "agitated"
    THREATENED = "threatened"
    DISGUSTED = "disgusted"
    CONTEMPT = "contempt"
    ENVIOUS = "envious"
    SARCASTIC = "sarcastic"
    IRONIC = "ironic"
    DEJECTED = "dejected"
    MELANCHOLIC = "melancholic"
    DISAPPOINTED = "disappointed"
    HURT = "hurt"
    GUILTY = "guilty"
    BORED = "bored"
    TIRED = "tired"
    REJECTED = "rejected"
    NOSTALGIC = "nostalgic"
    WISTFUL = "wistful"
    APOLOGETIC = "apologetic"
    HESITANT = "hesitant"
    INSECURE = "insecure"
    CONFUSED = "confused"
    RESIGNED = "resigned"
    ANXIOUS = "anxious"
    PANICKED = "panicked"
    ALARMED = "alarmed"
    PROUD = "proud"
    CONFIDENT = "confident"
    DISTANT = "distant"
    SKEPTICAL = "skeptical"
    CONTEMPLATIVE = "contemplative"
    DETERMINED = "determined"


class CartesiaExpressionAdapter(BaseExpressionAdapter):
    """Renders expression tags as Cartesia's inline markup.

    Part of expressive conversation (alpha). Modes are Cartesia's emotions,
    rendered as ``<emotion>`` tags; Cartesia supports them in English only.
    Cartesia changes emotion reliably only between generations, so a mode can
    change only between sentences, and a sentence written in several modes is
    spoken in the first. The one event is ``[laughter]``, and pauses are
    ``<break>`` tags. Cartesia has no emphasis.
    """

    def capabilities(self) -> ExpressionCapabilities:
        """Cartesia's emotions, laughter and pauses."""
        return ExpressionCapabilities(
            events=[ExpressionTerm("laughter")],
            modes=[
                ExpressionTerm(emotion.value)
                for emotion in CartesiaEmotion
                if emotion not in (CartesiaEmotion.NEUTRAL, CartesiaEmotion.JOKING_COMEDIC)
            ],
            pause_range_ms=(100, 5000),
            mode_scope=ModeScope.SENTENCE,
        )

    def render_text(self, text: str, modes: list[str], previous_modes: list[str] | None) -> str:
        """Prefix an emotion tag, set apart from the words, where the emotion changes."""
        emotion = modes[0] if modes else None
        previous = previous_modes[0] if previous_modes else None
        if emotion == previous:
            return text
        return f'<emotion value="{emotion or CartesiaEmotion.NEUTRAL.value}"/> {text}'

    def render_event(self, name: str) -> str:
        """Render laughter, set apart so Cartesia times it as a word of its own."""
        return f" [{name}] "

    def render_pause(self, duration_ms: int) -> str:
        """Render a break, set apart from the words."""
        return f' <break time="{duration_ms}ms"/> '
