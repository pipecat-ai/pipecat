#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Cartesia expression adapter for Pipecat."""

from enum import StrEnum
from typing import Any

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter
from pipecat.expressive.expression import (
    FAST_DELIVERY,
    SLOW_DELIVERY,
    SPEED_DELIVERY,
    DeliveryScope,
    ExpressionCapabilities,
    ExpressionDelivery,
    ExpressionSound,
    ExpressionTerm,
)

# The speed each speed delivery asks for. Cartesia takes speed as guidance, so the
# steps are large.
CARTESIA_SPEEDS = {SLOW_DELIVERY: 0.7, FAST_DELIVERY: 1.3}


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
    """Formats expression tags as Cartesia's inline markup.

    Part of expressive conversation (alpha). Its deliveries are Cartesia's emotions,
    formatted as ``<emotion>`` tags; Cartesia supports them in English only.
    Cartesia changes emotion reliably only between generations, so a delivery can
    change only between sentences, and a sentence written in several emotions
    is spoken in the first. Slow and fast set each request's speed (see
    :data:`CARTESIA_SPEEDS`). The one sound is ``[laughter]``, and pauses are
    ``<break>`` tags. Cartesia has no emphasis.
    """

    def capabilities(self) -> ExpressionCapabilities:
        """Cartesia's emotions, laughter and pauses."""
        return ExpressionCapabilities(
            sounds=[ExpressionTerm("laughter")],
            delivery=[
                *(
                    ExpressionTerm(emotion.value)
                    for emotion in CartesiaEmotion
                    if emotion not in (CartesiaEmotion.NEUTRAL, CartesiaEmotion.JOKING_COMEDIC)
                ),
                *SPEED_DELIVERY,
            ],
            pause_range_ms=(100, 5000),
            delivery_scope=DeliveryScope.SENTENCE,
            service_vocabulary={
                ExpressionDelivery.CALM: ["calm", "peaceful", "serene"],
                ExpressionDelivery.WARM: ["content", "happy"],
                ExpressionDelivery.EXCITED: ["excited", "enthusiastic"],
                ExpressionDelivery.SAD: ["sad", "disappointed"],
                ExpressionSound.LAUGH: ["laughter"],
                ExpressionSound.CHUCKLE: ["laughter"],
                ExpressionDelivery.TRUSTING: ["trust"],
                ExpressionDelivery.ANTICIPATING: ["anticipation"],
                ExpressionDelivery.CONTEMPTUOUS: ["contempt"],
            },
        )

    def format_text(
        self, text: str, delivery: list[str], previous_delivery: list[str] | None
    ) -> str:
        """Prefix an emotion tag, set apart from the words, where the emotion changes."""
        emotion = _emotion(delivery)
        previous = _emotion(previous_delivery or [])
        if emotion == previous:
            return text
        return f'<emotion value="{emotion or CartesiaEmotion.NEUTRAL.value}"/> {text}'

    def request_settings(self, delivery: list[str]) -> dict[str, Any]:
        """The speed a request in this delivery asks for, merged into its ``generation_config``."""
        speed = next((CARTESIA_SPEEDS[name] for name in delivery if name in CARTESIA_SPEEDS), None)
        return {"speed": speed} if speed is not None else {}

    def format_sound(self, name: str) -> str:
        """Format laughter, set apart so Cartesia times it as a word of its own."""
        return f" [{name}] "

    def format_pause(self, duration_ms: int) -> str:
        """Format a break, set apart from the words."""
        return f' <break time="{duration_ms}ms"/> '


def _emotion(delivery: list[str]) -> str | None:
    """The first name in the delivery that is an emotion, not a speed."""
    return next((name for name in delivery if name not in CARTESIA_SPEEDS), None)
