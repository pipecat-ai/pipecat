#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bookkeeping for resuming a Soniox real-time stream on a new connection.

See https://docs.soniox.com/stt/rt/stream-reconnect.
"""

import itertools
import json
import time
from collections import deque
from collections.abc import Callable
from typing import Any

# How long a stream waits for a reconnect after its connection drops: Soniox's
# default idle timeout.
RECONNECT_WINDOW_S = 20.0

# How often to ping, how late a pong may be, and how long a reconnect attempt
# may take: each a quarter of the reconnect window, as Soniox recommends.
PING_INTERVAL_S = RECONNECT_WINDOW_S / 4

RECONNECT_TIMEOUT_S = RECONNECT_WINDOW_S / 4

Message = str | bytes


class SonioxReconnectRefusedError(Exception):
    """Soniox refused to resume a stream, which has to be started anew."""


class SonioxStreamReconnect:
    """Tracks what a Soniox stream needs to be resumed on a new connection.

    Soniox numbers both sides of a stream. Its transcription responses carry a
    ``seq``, and it counts the client's messages after the start request, all
    but pings. A reconnect sends the ``seq`` of the last response received;
    Soniox answers with how many messages it received, and the client resends
    the ones after those. This class holds each message until a pong confirms
    Soniox has it, and does no I/O itself.

    A stream can be resumed once Soniox offers it a reconnect, which it does
    before any transcription response. A stream that responds without one, or
    that waits out the reconnect window, can't be: it stops holding messages.
    """

    def __init__(
        self,
        *,
        reconnect_window: float = RECONNECT_WINDOW_S,
        clock: Callable[[], float] = time.monotonic,
    ):
        """Initialize the stream's bookkeeping.

        Args:
            reconnect_window: How long, in seconds, the stream waits for a
                reconnect after its connection drops.
            clock: Monotonic time source, in seconds.
        """
        self._reconnect_window = reconnect_window
        self._clock = clock
        self._url: str | None = None
        self._token: str | None = None
        self._last_seq = 0
        self._numbered = 0
        self._held: deque[tuple[int, Message]] = deque()
        # Each unanswered ping's id, with the count of messages before it and
        # when it was sent.
        self._pings: dict[int, tuple[int, float]] = {}
        self._ping_ids = itertools.count(1)
        self._dropped_at: float | None = None
        self._closed = False

    @property
    def url(self) -> str | None:
        """Where to reconnect, as Soniox gave it."""
        return self._url

    @property
    def resumable(self) -> bool:
        """Whether the stream can carry on over a new connection."""
        self._expire_if_due()
        return not self._closed and bool(self._token)

    def hold(self, message: Message):
        """Number a message sent on the stream and hold it until Soniox confirms it.

        Args:
            message: Any client message after the start request, except a ping.
        """
        self._expire_if_due()
        if self._closed:
            return
        self._numbered += 1
        self._held.append((self._numbered, message))

    def held_after(self, number: int) -> list[tuple[int, Message]]:
        """The held messages numbered after ``number``, in order.

        Args:
            number: The last message number not wanted.

        Returns:
            Each message with its number.
        """
        return [(n, message) for n, message in self._held if n > number]

    def handle_message(self, content: dict[str, Any]) -> bool:
        """Track the stream from a message Soniox sent.

        Args:
            content: The decoded message.

        Returns:
            True if the message is part of the reconnect protocol, with nothing
            else to handle.
        """
        message_type = content.get("type")
        if message_type == "reconnect":
            self._url = content["reconnect_url"]
            self._token = content["reconnect_token"]
            return True
        if message_type == "pong":
            self._handle_pong(content.get("id"))
            return True
        if (seq := content.get("seq")) is not None:
            self._last_seq = seq
        elif not self._token and "error_code" not in content:
            self._close()
        return False

    def ping(self) -> str:
        """Record a ping, so its pong confirms the messages sent before it.

        Returns:
            The ping message to send.
        """
        ping_id = next(self._ping_ids)
        self._pings[ping_id] = (self._numbered, self._clock())
        return json.dumps({"type": "ping", "id": ping_id})

    def ping_overdue(self, timeout: float) -> bool:
        """Whether a ping has gone unanswered for ``timeout`` seconds or more.

        Args:
            timeout: How long, in seconds, a pong may take.

        Returns:
            True if the connection should count as lost.
        """
        now = self._clock()
        return any(now - sent_at >= timeout for _, sent_at in self._pings.values())

    def connection_lost(self):
        """Start the reconnect window, unless it is already running."""
        if self._dropped_at is None:
            self._dropped_at = self._clock()

    def reconnect_request(self) -> str:
        """The first message to send on a reconnect connection.

        Returns:
            The reconnect message, carrying the newest token.
        """
        request = {
            "type": "reconnect",
            "reconnect_token": self._token,
            "last_seq": self._last_seq,
        }
        return json.dumps(request)

    def resumed(self, reply: dict[str, Any]) -> int:
        """Take Soniox's answer to a reconnect request.

        Args:
            reply: The first message Soniox sent on the reconnect connection.

        Returns:
            How many messages Soniox received; the held messages after that
            many are to be resent.

        Raises:
            SonioxReconnectRefusedError: If Soniox refused to resume the stream.
            ConnectionError: If the reconnect failed for a reason worth
                retrying, with the same token.
        """
        if reply.get("type") != "reconnected":
            error = f"{reply.get('error_type')}: {reply.get('error_message')}"
            if reply.get("error_type") == "reconnect_failed":
                raise SonioxReconnectRefusedError(error)
            raise ConnectionError(f"Soniox did not resume the stream ({error})")
        self._token = reply["reconnect_token"]
        self._dropped_at = None
        # Pongs lost with the old connection are never sent.
        self._pings.clear()
        received = reply["last_request"]
        self._confirm(received)
        return received

    def _handle_pong(self, ping_id: Any):
        ping = self._pings.pop(ping_id, None)
        if not ping:
            return
        # Pongs come back in order, so the pings before it are answered too.
        count, sent_at = ping
        self._pings = {i: p for i, p in self._pings.items() if p[1] > sent_at}
        self._confirm(count)

    def _confirm(self, count: int):
        while self._held and self._held[0][0] <= count:
            self._held.popleft()

    def _expire_if_due(self):
        if (
            self._dropped_at is not None
            and self._clock() - self._dropped_at > self._reconnect_window
        ):
            self._close()

    def _close(self):
        self._closed = True
        self._held.clear()
        self._pings.clear()
