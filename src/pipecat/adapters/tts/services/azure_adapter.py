#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Azure expression adapter for Pipecat."""

from pipecat.adapters.tts.base_expression_adapter import BaseExpressionAdapter, delivery_tags
from pipecat.expressive.expression import (
    SPEED_DELIVERY,
    ExpressionCapabilities,
    ExpressionDelivery,
    ExpressionSound,
    ExpressionTerm,
)

# The marker that returns to the voice's default style.
_NEUTRAL_MARKER = "[Neutral]"

# The styles in Azure's HD voice reference, leaving out those that name a sound
# or an effort rather than a delivery, such as "laughing" and "struggling", and
# a few with no universal name, such as "defiant". "fast" and "slow" are the
# speed deliveries.
_STYLES = [
    ExpressionTerm(name)
    for name in (
        "amazed amused angry annoyed anxious appreciative calm cautious concerned confident "
        "confused curious defeated determined disappointed disgusted doubtful ecstatic "
        "encouraging excited fearful frustrated happy hesitant hurt impressed joking "
        "optimistic panicked pleading proud quiet reassuring reflective relieved remorseful "
        "resigned sad sarcastic serious shocked shouting shy skeptical surprised sympathetic "
        "terrified upset urgent whispering"
    ).split()
]

_SOUNDS = [
    ExpressionTerm(name)
    for name in ("laughter", "coughing", "throat_clearing", "breathing", "sighing", "yawning")
]


class AzureHDExpressionAdapter(BaseExpressionAdapter):
    """Formats expression tags as Azure HD voice style markers.

    Part of expressive conversation (alpha). A delivery is a style marker in the
    text, such as ``[excited]``, from Azure's fixed list for HD voices, and slow
    and fast are its ``[slow]`` and ``[fast]`` styles. A voice speaks one style
    at a time, so a delivery of several is spoken in the one added last, such as
    the slow of a slow span in a calm sentence. A marker lasts until the next one
    or the end of its sentence, and ``[Neutral]`` returns to the voice's default
    style. Sounds are its paralinguistic markers, such as ``[laughter]``. Styles
    are performed on English text only. HD voices have no emphasis, and their
    pauses are SSML, which isn't written into the text.
    """

    def capabilities(self) -> ExpressionCapabilities:
        """Azure's HD voice styles and paralinguistic sounds."""
        return ExpressionCapabilities(
            sounds=list(_SOUNDS),
            delivery=[*_STYLES, *SPEED_DELIVERY],
            service_vocabulary={
                ExpressionDelivery.APOLOGETIC: ["remorseful"],
                ExpressionDelivery.REGRETFUL: ["remorseful"],
                ExpressionDelivery.WARM: ["happy"],
                ExpressionDelivery.GRATEFUL: ["appreciative"],
                ExpressionDelivery.WORRIED: ["concerned"],
                ExpressionDelivery.DEJECTED: ["defeated"],
                ExpressionDelivery.ELATED: ["ecstatic"],
                ExpressionDelivery.EUPHORIC: ["ecstatic"],
                ExpressionDelivery.PLAYFUL: ["joking"],
                ExpressionDelivery.DESPERATE: ["pleading"],
                ExpressionDelivery.SOFT: ["quiet"],
                ExpressionDelivery.THOUGHTFUL: ["reflective"],
                ExpressionDelivery.CONTEMPLATIVE: ["reflective"],
                ExpressionDelivery.ALARMED: ["shocked"],
                ExpressionDelivery.SHEEPISH: ["shy"],
                ExpressionDelivery.SCARED: ["terrified"],
                ExpressionSound.LAUGH: ["laughter"],
                ExpressionSound.CHUCKLE: ["laughter"],
                ExpressionSound.COUGH: ["coughing"],
                ExpressionSound.CLEAR_THROAT: ["throat_clearing"],
                ExpressionSound.BREATH: ["breathing"],
                ExpressionSound.LIGHT_BREATH: ["breathing"],
                ExpressionSound.SIGH: ["sighing"],
                ExpressionSound.YAWN: ["yawning"],
            },
        )

    def format_text(
        self, text: str, delivery: list[str], previous_delivery: list[str] | None
    ) -> str:
        """Prefix a style marker where the style changes."""
        style = _style(delivery)
        previous = _style(previous_delivery or [])
        if style == previous:
            return text
        return f"[{style}] {text}" if style else f"{_NEUTRAL_MARKER} {text}"

    def format_sound(self, name: str) -> str:
        """Format a paralinguistic marker, set apart from the words."""
        return f" [{name}] "


def _style(delivery: list[str]) -> str | None:
    """The style a voice speaks a delivery in: its name added last."""
    tags = delivery_tags(delivery, {})
    return tags[-1] if tags else None
