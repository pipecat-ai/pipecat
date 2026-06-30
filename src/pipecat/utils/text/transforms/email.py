#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Transform email addresses into spoken form for TTS."""

import re

from pipecat.frames.frames import AggregationType

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")


def _email_to_spoken(match: re.Match) -> str:
    email = match.group(0)
    local, domain = email.split("@", 1)
    local_spoken = (
        local.replace(".", " dot ")
        .replace("_", " underscore ")
        .replace("-", " dash ")
        .replace("+", " plus ")
    )
    domain_spoken = domain.replace(".", " dot ").replace("-", " dash ")
    return f"{local_spoken} at {domain_spoken}"


async def email_to_speech(text: str, aggregation_type: str | AggregationType) -> str:
    """Transform email addresses into their spoken form.

    Args:
        text: Input text possibly containing email addresses.
        aggregation_type: Aggregation type of the text frame (unused).

    Returns:
        Text with email addresses replaced by spoken equivalents.

    Example::

        result = await email_to_speech("Contact user@example.com today", "*")
        # "Contact user at example dot com today"
    """
    return _EMAIL_RE.sub(_email_to_spoken, text)
