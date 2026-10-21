#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Internal descriptions of the errors Deepgram reports when refusing a connection."""

from websockets.exceptions import InvalidStatus


def describe_refusal(exception: Exception) -> str | None:
    """Describe Deepgram refusing a websocket handshake, if that is what failed.

    Deepgram explains a refusal (an invalid setting, a bad API key, a model the
    project can't use) in the ``dg-error`` response header, which the
    exception's own message leaves out.

    Args:
        exception: The exception raised by the connection attempt.

    Returns:
        A description naming the HTTP status and Deepgram's reason, or None
        when the exception isn't a refusal carrying one.
    """
    if isinstance(exception, InvalidStatus):
        reason = exception.response.headers.get("dg-error")
        if reason:
            return (
                f"Deepgram refused the connection (HTTP {exception.response.status_code}): {reason}"
            )
    return None
