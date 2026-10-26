#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Expression capabilities: what a TTS service can perform.

Part of expressive conversation, which lets an agent hear how something was
said and answer in kind. An LLM writes expression into its reply as inline
tags: ``[[name]]`` for a sound or silence at one point, ``<<name>>`` for the
delivery of the words after it, ``<<name: words>>`` for the delivery of just
those words, and ``^^word^^`` to stress a word. A delivery lasts until the next
delivery tag or the end of the response, and ``<<neutral>>`` returns to the
voice's baseline.

For example::

    <<cheerful>> It's [[laughs]] ^^back^^ up. [[pause 300]] Your code is <<slow: A B 7>>.

A TTS service declares which of these it can perform with
:class:`ExpressionCapabilities`, and broadcasts them at start in a
:class:`~pipecat.frames.frames.TTSMetadataFrame`, so the LLM can be told which
tags to write.

**Alpha.** The tag syntax and these types are likely to change between releases.
"""

from dataclasses import dataclass, field
from enum import Enum

# The delivery that returns to the voice's baseline. Part of the protocol, so it is
# a delivery whatever a service declares.
NEUTRAL_DELIVERY = "neutral"

# The delivery of stressed words, written ^^words^^.
EMPHASIS_DELIVERY = "emphasis"

# The deliveries that slow the words down and speed them up. Part of the protocol,
# so every service that performs speed names them the same.
SLOW_DELIVERY = "slow"
FAST_DELIVERY = "fast"


class DeliveryScope(Enum):
    """Where in a response a TTS service can change delivery.

    In ``"<<calm>> Sure, <<excited>> it worked!"``, a service with
    ``ANYWHERE`` speaks "it worked!" excited. One with ``SENTENCE`` speaks the
    whole sentence calm, the delivery its words start in, and the next sentence
    excited. A span's delivery is the exception: ``SENTENCE`` speaks the whole of
    ``"Your code is <<slow: A B 7>>."`` slowly, so the span's words still are.

    Parameters:
        ANYWHERE: A delivery takes effect where its tag appears.
        SENTENCE: A delivery can change only between requests, for a service
            that takes delivery as a setting of each request, such as MiniMax,
            or that changes it reliably only between generations, such as
            Cartesia. A sentence is one request; when the service streams
            tokens, a request is a token, so a change can take effect within
            a sentence.
    """

    ANYWHERE = "anywhere"
    SENTENCE = "sentence"


@dataclass
class ExpressionTerm:
    """One sound or delivery a TTS service can perform.

    Parameters:
        name: The word written inside the tag, such as ``"laugh"`` or
            ``"cheerful"``.
        description: A short line on when to use it, or empty.
    """

    name: str
    description: str = ""


@dataclass
class ExpressionCapabilities:
    """The expression tags a TTS service can perform.

    The LLM is taught these, and tags outside them are left out of the
    service's speech. A service that can laugh, speak in two moods, pause and
    stress words declares::

        ExpressionCapabilities(
            sounds=[ExpressionTerm("laughs")],
            delivery=[
                ExpressionTerm("cheerful"),
                ExpressionTerm("calm", "to reassure a worried caller"),
            ],
            pause_range_ms=(100, 3000),
            emphasis=True,
        )

    Parameters:
        sounds: Sounds the service can make at one point, written ``[[name]]``.
        delivery: Deliveries the service can apply to the words that follow,
            written ``<<name>>``.
        free_text_sounds: Whether the service also accepts sounds outside
            ``sounds``, described in a few words.
        free_text_delivery: Whether the service also accepts deliveries outside
            ``delivery``, described in a few words.
        pause_range_ms: The shortest and longest pause the service can make,
            in milliseconds, or None when it can't pause.
        emphasis: Whether the service can stress words marked ``^^word^^``.
        delivery_scope: Where in a response the service can change delivery.
    """

    sounds: list[ExpressionTerm] = field(default_factory=list)
    delivery: list[ExpressionTerm] = field(default_factory=list)
    free_text_sounds: bool = False
    free_text_delivery: bool = False
    pause_range_ms: tuple[int, int] | None = None
    emphasis: bool = False
    delivery_scope: DeliveryScope = DeliveryScope.ANYWHERE


# Slow and fast, as a service that performs them declares them.
SPEED_DELIVERY = [
    ExpressionTerm(
        SLOW_DELIVERY, "to say something carefully, such as a number, a code or an address"
    ),
    ExpressionTerm(FAST_DELIVERY, "to say something in passing"),
]
