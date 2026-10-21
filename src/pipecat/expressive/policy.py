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

from dataclasses import dataclass, field


class ExpressionGuidance:
    """Guidance presets for :class:`ExpressionPolicy`.

    Each preset is a short paragraph on one aspect of delivery, written to
    combine with other presets and with the developer's own lines.

    Parameters:
        MATCH_STYLE: Answer in the user's mood and energy, but stay calm with
            hostility.
        SLOW_FOR_DETAILS: Say anything the user may need to write down slowly,
            in a sentence of its own. For a service that performs the
            ``slow`` tone.
    """

    MATCH_STYLE = (
        "Let your voice match the user's mood and energy: be excited for their good news, "
        "sympathetic with their bad news, and laugh when they say something funny. If they "
        "are hostile, stay calm instead of matching it. Most sentences need no tag."
    )

    SLOW_FOR_DETAILS = (
        "When you give the user something to write down, such as a code, a phone number or an "
        "email address, say it slowly in a sentence of its own, then start the next sentence "
        "with the tone you want: <<slow>> It's A B 7 4 9. <<neutral>> Anything else?"
    )


@dataclass
class ExpressionPolicy:
    """Which expression tags an LLM is offered, and when to use them.

    Parameters:
        guidance: Lines on when and how much to express, such as
            ``ExpressionGuidance.MATCH_STYLE`` or the developer's own. They are
            joined in order, so a later line can refine an earlier one.
        tones: The tones to offer, or None for every tone the TTS service
            declares. Tones a service with a fixed list lacks are left out; a
            service that accepts any tone is offered these as written.
        events: The events to offer, or None for every event the TTS service
            declares, narrowed the same way as ``tones``.
    """

    guidance: list[str] = field(default_factory=list)
    tones: list[str] | None = None
    events: list[str] | None = None
