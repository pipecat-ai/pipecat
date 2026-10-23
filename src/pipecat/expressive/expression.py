#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Expression capabilities: the delivery a TTS service can perform.

Part of expressive conversation, which lets an agent hear how something was
said and answer in kind. An LLM writes delivery into its reply as inline tags:
``[[event]]`` for a sound or silence at one point, ``<<mode>>`` for how the
words after it are spoken, and ``^word^`` to stress a word. A mode lasts until
the next mode tag or the end of the response, and ``<<neutral>>`` returns to
the voice's baseline.

A TTS service declares which of these it can perform with
:class:`ExpressionCapabilities`, and broadcasts them at start in a
:class:`~pipecat.frames.frames.TTSMetadataFrame`, so the LLM can be told which
tags to write.

**Alpha.** The tag syntax and these types are likely to change between releases.
"""

from dataclasses import dataclass, field
from enum import Enum

from pipecat.utils.text.base_text_aggregator import TextType


class ModeScope(Enum):
    """Where in a response a TTS service can change mode.

    Parameters:
        ANYWHERE: A mode takes effect where its tag appears.
        SENTENCE: A mode can change only between sentences, for a service
            that takes modes with each request rather than as markup. When
            the service streams tokens, a request is a token, so a change can
            take effect within a sentence.
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
        emphasis: Whether the service can stress words marked ``^word^``.
        mode_scope: Where in a response the service can change mode.
    """

    events: list[ExpressionTerm] = field(default_factory=list)
    modes: list[ExpressionTerm] = field(default_factory=list)
    free_text_events: bool = False
    free_text_modes: bool = False
    pause_range_ms: tuple[int, int] | None = None
    emphasis: bool = False
    mode_scope: ModeScope = ModeScope.ANYWHERE
