#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Expression capabilities: the delivery a TTS service can perform.

Part of expressive conversation, which lets an agent hear how something was
said and answer in kind. An LLM writes delivery into its reply as inline tags:
``[[event]]`` for a sound or silence at one point, ``<<mode>>`` for how the
words after it are spoken, and ``^^word^^`` to stress a word. A mode lasts until
the next mode tag or the end of the response, and ``<<neutral>>`` returns to
the voice's baseline.

For example::

    <<cheerful>> It's [[laughs]] ^^back^^ up. [[pause 300]] <<neutral>> Anything else?

A TTS service declares which of these it can perform with
:class:`ExpressionCapabilities`, and broadcasts them at start in a
:class:`~pipecat.frames.frames.TTSMetadataFrame`, so the LLM can be told which
tags to write.

**Alpha.** The tag syntax and these types are likely to change between releases.
"""

from dataclasses import dataclass, field
from enum import Enum

from pipecat.utils.text.base_text_aggregator import TextType

# The mode that returns to the voice's baseline. Part of the protocol, so it is
# a mode whatever a service declares.
NEUTRAL_MODE = "neutral"

# The mode of stressed words, written ^^words^^.
EMPHASIS_MODE = "emphasis"


class ModeScope(Enum):
    """Where in a response a TTS service can change mode.

    In ``"<<calm>> Sure, <<excited>> it worked!"``, a service with
    ``ANYWHERE`` speaks "it worked!" excited. One with ``SENTENCE`` speaks the
    whole sentence calm, the modes its words start in, and the next sentence
    excited.

    Parameters:
        ANYWHERE: A mode takes effect where its tag appears.
        SENTENCE: A mode can change only between requests, for a service
            that takes modes as a setting of each request, such as MiniMax,
            or that changes them reliably only between generations, such as
            Cartesia. A sentence is one request; when the service streams
            tokens, a request is a token, so a change can take effect within
            a sentence.
    """

    ANYWHERE = "anywhere"
    SENTENCE = "sentence"


@dataclass
class ExpressionTerm:
    """One event or mode a TTS service can perform.

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
            events=[ExpressionTerm("laughs")],
            modes=[
                ExpressionTerm("cheerful"),
                ExpressionTerm("calm", "to reassure a worried caller"),
            ],
            pause_range_ms=(100, 3000),
            emphasis=True,
        )

    Parameters:
        events: Sounds the service can make at one point, written ``[[name]]``.
        modes: Deliveries the service can apply to the words that follow,
            written ``<<name>>``.
        free_text_events: Whether the service also accepts events outside
            ``events``, described in a few words.
        free_text_modes: Whether the service also accepts modes outside
            ``modes``, described in a few words.
        pause_range_ms: The shortest and longest pause the service can make,
            in milliseconds, or None when it can't pause.
        emphasis: Whether the service can stress words marked ``^^word^^``.
        mode_scope: Where in a response the service can change mode.
    """

    events: list[ExpressionTerm] = field(default_factory=list)
    modes: list[ExpressionTerm] = field(default_factory=list)
    free_text_events: bool = False
    free_text_modes: bool = False
    pause_range_ms: tuple[int, int] | None = None
    emphasis: bool = False
    mode_scope: ModeScope = ModeScope.ANYWHERE
