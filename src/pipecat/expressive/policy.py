#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The developer's policy for expressive conversation.

Part of expressive conversation (alpha). An LLM is taught expression tags by
their universal names, those the active TTS service can perform. The policy says
which it is offered and when to use them.

**Alpha.** The tag syntax and these types are likely to change between releases.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

from pipecat.expressive.expression import ExpressionDelivery, ExpressionSound


class ExpressionGuidance(StrEnum):
    """Guidance presets for :class:`ExpressionPolicy`.

    Each preset is a short paragraph on one aspect of delivery, written to
    combine with other presets and with the developer's own lines. A preset
    is a string, so it joins with them as written, but stays a member of this
    enum in a policy, apart from a line of the developer's with the same text.
    Presets are about writing tags, so an LLM taught none, such as one whose
    replies a tagger marks up, isn't given them.

    Parameters:
        MATCH_STYLE: Answer in the user's mood and energy, but stay calm with
            hostility.
        SLOW_FOR_DETAILS: Say just what the user may need to write down
            slowly, in a span. For a service that performs the ``slow``
            delivery.
        BREATHE: Take an audible breath before a long or difficult sentence
            now and then, at most once a reply. For a service that performs a
            breath. A tagger reads it as a switch for breaths of its own
            choosing.
    """

    MATCH_STYLE = (
        "Let your voice match the user's mood and energy: be excited for their good news, "
        "sympathetic with their bad news, and laugh when they say something funny. If they "
        "are hostile, stay calm instead of matching it. Most sentences need no tag."
    )

    SLOW_FOR_DETAILS = (
        "When you give the user something to write down, such as a code, a phone number or an "
        "email address, say just that slowly: Your code is <<slow: A B 7 4 9>>. Anything else?"
    )

    BREATHE = (
        "Now and then, take an audible breath before a sentence that is long, carries hard news "
        "or starts a careful explanation, as a person on the phone does. At most one breath a "
        "reply, and most replies need none: rarely before a short or routine sentence, never by "
        "habit at the start of every reply, and never right after another sound."
    )


class ExpressionPolicy(BaseModel):
    """Which expression tags an LLM is offered, and when to use them.

    Parameters:
        guidance: Lines on when and how much to express, such as
            ``ExpressionGuidance.MATCH_STYLE`` or the developer's own. They are
            joined in order, so a later line can refine an earlier one. The
            developer's lines reach the LLM whenever it has a part to play in
            expressive conversation: tags to write, or a reading of the user
            to act on. The presets are about writing tags, so they go to
            whoever writes them: the LLM, or a tagger in its place.
        delivery: The deliveries to offer, by their universal names
            (:class:`~pipecat.expressive.expression.ExpressionDelivery`), or
            None for the core ones. A name that isn't universal, such as an
            accent, is offered as written. Those the TTS service can't perform
            are left out, with a warning.
        sounds: The sounds to offer, as ``delivery`` offers deliveries
            (:class:`~pipecat.expressive.expression.ExpressionSound`). None
            offers the core ones, and a breath only with
            ``ExpressionGuidance.BREATHE``.
    """

    guidance: list[ExpressionGuidance | str] = Field(default_factory=list)
    delivery: list[ExpressionDelivery | str] | None = None
    sounds: list[ExpressionSound | str] | None = None
