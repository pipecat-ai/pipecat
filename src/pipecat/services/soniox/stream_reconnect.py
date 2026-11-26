#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Soniox stream reconnect: a connection that resumes its stream after a drop.

See https://docs.soniox.com/stt/rt/stream-reconnect.
"""

import asyncio
import itertools
import json
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK
from websockets.protocol import State

# How long Soniox keeps a dropped stream open for a reconnect (its default idle timeout).
RECONNECT_WINDOW_S = 20.0

# Soniox recommends a quarter of the reconnect window for each of these.
PING_INTERVAL_S = RECONNECT_WINDOW_S / 4
RECONNECT_TIMEOUT_S = RECONNECT_WINDOW_S / 4

# Wait between failed reconnect attempts, which retry until the window closes.
RECONNECT_RETRY_DELAY_S = 1.0

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


class SonioxWebSocket:
    """A Soniox stream connection that reconnects to the same stream after a drop.

    Provides :class:`~pipecat.services.websocket_service.WebsocketConnection`,
    so the Soniox STT service uses it like the connection it wraps. Underneath it follows
    Soniox's stream reconnect protocol: it numbers and keeps every message it
    sends, pings Soniox from the receive side, and, when the connection drops or
    a pong is late, opens a new connection to the same stream and resends what
    Soniox is missing. Reconnect messages and pongs never reach the caller.

    While reconnecting, ``state`` stays ``OPEN`` and sent messages are kept for
    the resend. When the stream can't be reconnected (Soniox refused, or the
    reconnect window closed), receiving raises the error that dropped the
    connection, as the wrapped connection would.
    """

    def __init__(
        self,
        websocket: ClientConnection,
        connect: Callable[[str], Awaitable[ClientConnection]],
        *,
        ping_interval: float = PING_INTERVAL_S,
        reconnect_timeout: float = RECONNECT_TIMEOUT_S,
        retry_delay: float = RECONNECT_RETRY_DELAY_S,
        clock: Callable[[], float] = time.monotonic,
    ):
        """Initialize the connection.

        Args:
            websocket: The stream's connection, after its configuration message.
            connect: Opens a connection to a URL, authenticated with the API key
                that started the stream.
            ping_interval: Seconds between pings.
            reconnect_timeout: Seconds a pong, or a reconnect attempt, may take.
            retry_delay: Seconds to wait between failed reconnect attempts.
            clock: Monotonic time source, in seconds.
        """
        self._websocket: ClientConnection | None = websocket
        self._connect = connect
        self._ping_interval = ping_interval
        self._reconnect_timeout = reconnect_timeout
        self._retry_delay = retry_delay
        self._clock = clock
        self._tracker = SonioxReconnectTracker(clock=clock)
        self._next_ping_at = clock() + ping_interval
        self._closing = False

    @property
    def state(self) -> State:
        """The stream's state: ``OPEN`` while connected or reconnecting."""
        websocket = self._websocket
        if websocket is None:
            return State.CLOSING if self._closing else State.OPEN
        if websocket.state is not State.OPEN and not self._closing and self._tracker.can_reconnect:
            # Dropped, and the receive side has yet to reconnect.
            return State.OPEN
        return websocket.state

    async def send(self, message: ClientMessage):
        """Send a message, keeping it for a resend after a reconnect.

        Args:
            message: The message to send.

        Raises:
            ConnectionClosed: If the connection is closed and the stream can't
                be reconnected.
        """
        self._tracker.track_sent(message)
        websocket = self._websocket
        if websocket is None:
            # Reconnecting: resent once the new connection is up.
            return
        try:
            await websocket.send(message)
        except ConnectionClosed:
            if self._closing or not self._tracker.can_reconnect:
                raise

    async def recv(self) -> ClientMessage:
        """Receive the next Soniox response, reconnecting if the connection drops.

        Returns:
            The next message for the caller.

        Raises:
            ConnectionClosed: If the connection closed and the stream can't be
                reconnected.
        """
        while True:
            websocket = self._websocket
            assert websocket, "recv() called while another recv() is reconnecting"
            try:
                message = await self._recv_with_pings(websocket)
            except ConnectionClosed as e:
                if self._closing or not self._tracker.can_reconnect:
                    raise
                await self._reconnect(websocket, e)
                continue
            if not self._consume(message):
                return message

    def __aiter__(self) -> "SonioxWebSocket":
        """Iterate over Soniox responses until the connection closes normally."""
        return self

    async def __anext__(self) -> ClientMessage:
        """Receive the next Soniox response.

        Returns:
            The next message for the caller.

        Raises:
            StopAsyncIteration: When the connection closed normally.
        """
        try:
            return await self.recv()
        except ConnectionClosedOK:
            raise StopAsyncIteration from None

    async def ping(self) -> Awaitable[float]:
        """Send a websocket protocol ping on the current connection.

        Returns:
            A future that completes when the matching protocol pong arrives.

        Raises:
            ConnectionError: If reconnecting, with no connection to ping.
        """
        if self._websocket is None:
            raise ConnectionError("Soniox stream is reconnecting")
        return await self._websocket.ping()

    async def close(self):
        """Close the connection with status 1000, which ends the stream."""
        self._closing = True
        websocket = self._websocket
        if websocket is None:
            return
        try:
            await websocket.close()
        finally:
            if websocket.state is not State.CLOSED:
                websocket.transport.abort()

    async def wait_closed(self):
        """Wait until the current connection is closed."""
        if self._websocket:
            await self._websocket.wait_closed()

    async def _recv_with_pings(self, websocket: ClientConnection) -> ClientMessage:
        """Receive a message, sending a ping whenever one is due."""
        while True:
            delay = self._next_ping_at - self._clock()
            if delay <= 0:
                self._next_ping_at = self._clock() + self._ping_interval
                await self._ping_or_abort(websocket)
                continue
            # Canceling `recv()` loses no data.
            try:
                async with asyncio.timeout(delay):
                    return await websocket.recv()
            except TimeoutError:
                continue

    async def _ping_or_abort(self, websocket: ClientConnection):
        """Ping Soniox, or drop the connection if a pong is late."""
        if not self._tracker.can_reconnect:
            return
        if self._tracker.is_pong_overdue(self._reconnect_timeout):
            logger.warning("Soniox pong overdue, reconnecting")
            # The pending `recv()` then fails and reconnects.
            websocket.transport.abort()
            return
        try:
            await websocket.send(self._tracker.create_ping())
        except ConnectionClosed:
            # The pending `recv()` fails the same way and reconnects.
            pass

    def _consume(self, message: ClientMessage) -> bool:
        """Apply a Soniox message to the reconnect state.

        Returns:
            True if the message is for the reconnect protocol only.
        """
        if not isinstance(message, str):
            return False
        try:
            content = json.loads(message)
        except json.JSONDecodeError:
            return False
        if not isinstance(content, dict):
            return False
        if content.get("error_type") == "connection_replaced":
            # Another connection took the stream over; ignore this one's errors.
            logger.debug("Soniox connection replaced by a reconnect")
            return True
        return self._tracker.handle_server_message(content)

    async def _reconnect(self, lost: ClientConnection, error: ConnectionClosed):
        """Reconnect to the stream, retrying until Soniox refuses or the window closes.

        Args:
            lost: The connection that dropped.
            error: Why it dropped, raised if the stream can't be reconnected.
        """
        self._tracker.mark_disconnected()
        self._websocket = None
        # Abort rather than close: a close with status 1000 ends the stream.
        lost.transport.abort()
        logger.debug("Reconnecting to the Soniox stream")
        while True:
            try:
                self._websocket = await self._reconnect_once()
            except SonioxReconnectRefusedError as e:
                logger.warning(f"Soniox refused the reconnect ({e})")
                self._websocket = lost
                raise error from e
            except Exception as e:
                if not self._tracker.can_reconnect:
                    logger.warning(f"Soniox reconnect window closed ({e})")
                    self._websocket = lost
                    raise error from e
                logger.warning(f"Soniox reconnect failed ({e}), retrying")
                await asyncio.sleep(self._retry_delay)
                continue
            except BaseException:
                self._websocket = lost
                raise
            self._next_ping_at = self._clock() + self._ping_interval
            logger.debug("Reconnected to the Soniox stream")
            return

    async def _reconnect_once(self) -> ClientConnection:
        """Open a connection to the stream and resend what Soniox is missing.

        Messages sent meanwhile are kept, and resent after the missing ones.

        Returns:
            The new connection.

        Raises:
            SonioxReconnectRefusedError: If Soniox refused the reconnect.
            Exception: If the attempt failed and can be retried.
        """
        url = self._tracker.reconnect_url
        assert url
        websocket: ClientConnection | None = None
        try:
            async with asyncio.timeout(self._reconnect_timeout):
                websocket = await self._connect(url)
                await websocket.send(self._tracker.create_reconnect_request())
                reply = json.loads(await websocket.recv())
                received_count = self._tracker.handle_reconnect_reply(reply)
            while missing := self._tracker.unacknowledged_after(received_count):
                for received_count, message in missing:
                    await websocket.send(message)
        except BaseException:
            if websocket:
                # Abort rather than close: a close with status 1000 ends the stream.
                websocket.transport.abort()
            raise
        return websocket
