#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""RTVI pipeline frame definitions."""

from dataclasses import dataclass
from typing import Any, Optional

from pipecat.frames.frames import DataFrame, SystemFrame


@dataclass
class RTVIServerMessageFrame(SystemFrame):
    """A frame for sending server messages to the client.

    Parameters:
        data: The message data to send to the client.
    """

    data: Any

    def __str__(self):
        """String representation of the RTVI server message frame."""
        return f"{self.name}(data: {self.data})"


@dataclass
class RTVIClientMessageFrame(SystemFrame):
    """A frame for sending messages from the client to the RTVI server.

    This frame is meant for custom messaging from the client to the server
    and expects a server-response message.
    """

    msg_id: str
    type: str
    data: Optional[Any] = None


@dataclass
class RTVIServerResponseFrame(SystemFrame):
    """A frame for responding to a client RTVI message.

    This frame should be sent in response to an RTVIClientMessageFrame
    and include the original RTVIClientMessageFrame to ensure the response
    is properly attributed to the original request. To respond with an error,
    set the `error` field to a string describing the error. This will result
    in the client receiving an `error-response` message instead of a
    `server-response` message.
    """

    client_msg: RTVIClientMessageFrame
    data: Optional[Any] = None
    error: Optional[str] = None
