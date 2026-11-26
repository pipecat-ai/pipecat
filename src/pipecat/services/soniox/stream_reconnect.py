#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Soniox stream reconnect state.

See https://docs.soniox.com/stt/rt/stream-reconnect.
"""

import itertools
import json
import time
from collections import deque
from collections.abc import Callable
from typing import Any

# How long Soniox keeps a dropped stream open for a reconnect (its default idle timeout).
RECONNECT_WINDOW_S = 20.0

# Soniox recommends a quarter of the reconnect window for each of these.
PING_INTERVAL_S = RECONNECT_WINDOW_S / 4
RECONNECT_TIMEOUT_S = RECONNECT_WINDOW_S / 4

ClientMessage = str | bytes


class SonioxReconnectRefusedError(Exception):
    """Soniox refused the reconnect, so the stream must be restarted."""


class SonioxReconnectTracker:
    """Tracks the state needed to reconnect to a Soniox stream.

    Soniox counts the client messages it receives (all but pings) and numbers
    its responses with ``seq``. On reconnect, the client sends its last
    ``seq`` and Soniox replies with its received count, so the client resends
    only the messages Soniox is missing. Sent messages are kept until a pong
    acknowledges them. This class does no I/O.
    """

    def __init__(
        self,
        *,
        reconnect_window: float = RECONNECT_WINDOW_S,
        clock: Callable[[], float] = time.monotonic,
    ):
        """Initialize the tracker.

        Args:
            reconnect_window: Seconds Soniox keeps a dropped stream open.
            clock: Monotonic time source, in seconds.
        """
        self._reconnect_window = reconnect_window
        self._clock = clock
        self._reconnect_url: str | None = None
        self._reconnect_token: str | None = None
        self._last_seq = 0
        self._sent_count = 0
        self._unacknowledged: deque[tuple[int, ClientMessage]] = deque()
        # Ping id -> (sent count when pinged, time pinged).
        self._pending_pings: dict[int, tuple[int, float]] = {}
        self._ping_ids = itertools.count(1)
        self._disconnected_at: float | None = None
        self._disabled = False

    @property
    def reconnect_url(self) -> str | None:
        """The URL Soniox gave for reconnecting."""
        return self._reconnect_url

    @property
    def can_reconnect(self) -> bool:
        """Whether Soniox offered a reconnect and the reconnect window is still open."""
        self._disable_if_window_expired()
        return not self._disabled and self._reconnect_token is not None

    def track_sent(self, message: ClientMessage):
        """Number a sent message and keep it until Soniox acknowledges it.

        Args:
            message: A client message other than a ping.
        """
        self._disable_if_window_expired()
        if self._disabled:
            return
        self._sent_count += 1
        self._unacknowledged.append((self._sent_count, message))

    def unacknowledged_after(self, count: int) -> list[tuple[int, ClientMessage]]:
        """Return the kept messages numbered above ``count``, oldest first.

        Args:
            count: Number of messages Soniox already has.

        Returns:
            ``(number, message)`` pairs.
        """
        return [(number, message) for number, message in self._unacknowledged if number > count]

    def handle_server_message(self, content: dict[str, Any]) -> bool:
        """Update the state from a Soniox message.

        Args:
            content: The decoded message.

        Returns:
            True if the message was a reconnect offer or a pong and needs no
            further handling.
        """
        message_type = content.get("type")
        if message_type == "reconnect":
            self._reconnect_url = content["reconnect_url"]
            self._reconnect_token = content["reconnect_token"]
            return True
        if message_type == "pong":
            self._handle_pong(content.get("id"))
            return True
        if content.get("error_code") or content.get("finished"):
            # The stream has ended.
            self._disable()
        elif self._reconnect_token is None:
            # A response before any reconnect offer: reconnect is unavailable.
            self._disable()
        elif (seq := content.get("seq")) is not None:
            self._last_seq = seq
        return False

    def create_ping(self) -> str:
        """Create a ping whose pong acknowledges every message sent before it.

        Returns:
            The ping message.
        """
        ping_id = next(self._ping_ids)
        self._pending_pings[ping_id] = (self._sent_count, self._clock())
        return json.dumps({"type": "ping", "id": ping_id})

    def is_pong_overdue(self, timeout: float) -> bool:
        """Whether any ping has waited ``timeout`` seconds or more for its pong.

        Args:
            timeout: Seconds a pong may take.

        Returns:
            True if the connection should be treated as lost.
        """
        now = self._clock()
        return any(now - pinged_at >= timeout for _, pinged_at in self._pending_pings.values())

    def mark_disconnected(self):
        """Start the reconnect window, unless already started."""
        if self._disconnected_at is None:
            self._disconnected_at = self._clock()

    def create_reconnect_request(self) -> str:
        """Create the first message for a reconnect connection.

        Returns:
            The reconnect message.
        """
        request = {
            "type": "reconnect",
            "reconnect_token": self._reconnect_token,
            "last_seq": self._last_seq,
        }
        return json.dumps(request)

    def handle_reconnect_reply(self, reply: dict[str, Any]) -> int:
        """Apply Soniox's reply to a reconnect request.

        Args:
            reply: The first message on the reconnect connection.

        Returns:
            Number of messages Soniox received; resend the ones after it.

        Raises:
            SonioxReconnectRefusedError: If Soniox refused the reconnect.
            ConnectionError: If the reconnect failed and can be retried.
        """
        if reply.get("type") != "reconnected":
            error = f"{reply.get('error_type')}: {reply.get('error_message')}"
            if reply.get("error_type") == "reconnect_failed":
                self._disable()
                raise SonioxReconnectRefusedError(error)
            raise ConnectionError(f"Soniox reconnect failed ({error})")
        self._reconnect_token = reply["reconnect_token"]
        self._disconnected_at = None
        # Soniox never sends pongs lost with the old connection.
        self._pending_pings.clear()
        received_count = reply["last_request"]
        self._acknowledge(received_count)
        return received_count

    def _handle_pong(self, ping_id: Any):
        ping = self._pending_pings.pop(ping_id, None)
        if not ping:
            return
        sent_count, pinged_at = ping
        # Pongs arrive in order, so earlier pings are answered too.
        self._pending_pings = {
            ping_id: pending
            for ping_id, pending in self._pending_pings.items()
            if pending[1] > pinged_at
        }
        self._acknowledge(sent_count)

    def _acknowledge(self, count: int):
        while self._unacknowledged and self._unacknowledged[0][0] <= count:
            self._unacknowledged.popleft()

    def _disable_if_window_expired(self):
        if (
            self._disconnected_at is not None
            and self._clock() - self._disconnected_at > self._reconnect_window
        ):
            self._disable()

    def _disable(self):
        self._disabled = True
        self._unacknowledged.clear()
        self._pending_pings.clear()
