#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The developer's policy for expressive conversation.

Part of expressive conversation (alpha). An LLM is taught the expression tags
the active TTS service can perform. The policy narrows which of them it is
offered and says when to use them.

**Alpha.** The tag syntax and these types are likely to change between releases.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class ExpressionGuidance(StrEnum):
    """Guidance presets for :class:`ExpressionPolicy`.

    Each preset is a short paragraph on one aspect of delivery, written to
    combine with other presets and with the developer's own lines. A preset
    is a string, so it joins with them as written, but stays a member of this
    enum in a policy, apart from a line of the developer's with the same text.
    An LLM taught no tags, such as one whose replies a tagger marks up, is
    given the developer's lines without the presets.

    Parameters:
        MATCH_STYLE: Answer in the user's mood and energy, but stay calm with
            hostility.
        SLOW_FOR_DETAILS: Say just what the user may need to write down
            slowly, in a span. For a service that performs the ``slow``
            delivery.
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


class ExpressionPolicy(BaseModel):
    """Which expression tags an LLM is offered, and when to use them.

    Parameters:
        guidance: Lines on when and how much to express, such as
            ``ExpressionGuidance.MATCH_STYLE`` or the developer's own. They are
            joined in order, so a later line can refine an earlier one.
        delivery: The deliveries to offer, or None for every delivery the TTS
            service declares. Deliveries a service with a fixed list lacks are
            left out; a service that accepts any delivery is offered these as
            written.
        sounds: The sounds to offer, or None for every sound the TTS service
            declares, narrowed the same way as ``delivery``.
    """

    guidance: list[ExpressionGuidance | str] = Field(default_factory=list)
    delivery: list[str] | None = None
    sounds: list[str] | None = None
