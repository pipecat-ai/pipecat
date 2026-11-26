#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.protocol import State

from pipecat.services.soniox.stream_reconnect import (
    SonioxReconnectRefusedError,
    SonioxReconnectTracker,
    SonioxWebSocket,
)

RECONNECT_OFFER = {"type": "reconnect", "reconnect_url": "wss://soniox/r", "reconnect_token": "t1"}


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _tracker_with_offer(clock=None) -> SonioxReconnectTracker:
    tracker = SonioxReconnectTracker(clock=clock or _Clock())
    assert tracker.handle_server_message(RECONNECT_OFFER)
    return tracker


def _pong(ping: str) -> dict:
    return {"type": "pong", "id": json.loads(ping)["id"]}


def test_can_reconnect_once_soniox_offers_a_reconnect():
    tracker = SonioxReconnectTracker()
    assert not tracker.can_reconnect

    tracker.handle_server_message(RECONNECT_OFFER)

    assert tracker.can_reconnect
    assert tracker.reconnect_url == "wss://soniox/r"


def test_a_response_before_any_offer_disables_reconnect():
    tracker = SonioxReconnectTracker()
    tracker.track_sent(b"audio")

    assert not tracker.handle_server_message({"tokens": []})
    tracker.track_sent(b"audio")

    assert not tracker.can_reconnect
    assert tracker.unacknowledged_after(0) == []


def test_a_seq_response_before_any_offer_disables_reconnect():
    tracker = SonioxReconnectTracker()

    tracker.handle_server_message({"tokens": [], "seq": 1})
    tracker.track_sent(b"audio")

    assert tracker.unacknowledged_after(0) == []


@pytest.mark.parametrize(
    "response",
    [
        {"tokens": [], "error_code": 408, "error_type": "request_timeout", "seq": 2},
        {"tokens": [], "finished": True, "seq": 2},
    ],
)
def test_an_ended_stream_cannot_be_reconnected(response):
    tracker = _tracker_with_offer()

    assert not tracker.handle_server_message(response)

    assert not tracker.can_reconnect


def test_a_refused_reconnect_disables_reconnect():
    tracker = _tracker_with_offer()

    with pytest.raises(SonioxReconnectRefusedError):
        tracker.handle_reconnect_reply({"tokens": [], "error_type": "reconnect_failed"})

    assert not tracker.can_reconnect


def test_sent_messages_are_numbered_from_one_in_send_order():
    tracker = _tracker_with_offer()

    for message in (b"a", '{"type": "finalize"}', b"b"):
        tracker.track_sent(message)

    assert tracker.unacknowledged_after(1) == [(2, '{"type": "finalize"}'), (3, b"b")]


def test_a_pong_acknowledges_the_messages_sent_before_its_ping():
    tracker = _tracker_with_offer()
    tracker.track_sent(b"a")
    tracker.track_sent(b"b")
    ping = tracker.create_ping()
    tracker.track_sent(b"c")

    assert tracker.handle_server_message(_pong(ping))

    assert tracker.unacknowledged_after(0) == [(3, b"c")]


def test_a_pong_also_answers_earlier_pings():
    clock = _Clock()
    tracker = _tracker_with_offer(clock)
    tracker.create_ping()
    clock.now = 1
    later = tracker.create_ping()

    tracker.handle_server_message(_pong(later))
    clock.now = 10

    assert not tracker.is_pong_overdue(5)


def test_pong_is_overdue_after_the_timeout():
    clock = _Clock()
    tracker = _tracker_with_offer(clock)
    tracker.create_ping()

    clock.now = 4.9
    assert not tracker.is_pong_overdue(5)
    clock.now = 5
    assert tracker.is_pong_overdue(5)


def test_reconnect_request_carries_the_newest_token_and_last_seq():
    tracker = _tracker_with_offer()
    tracker.handle_server_message({"tokens": [], "seq": 7})

    assert json.loads(tracker.create_reconnect_request()) == {
        "type": "reconnect",
        "reconnect_token": "t1",
        "last_seq": 7,
    }


def test_reconnect_reply_acknowledges_received_messages_and_updates_the_token():
    tracker = _tracker_with_offer()
    for message in (b"a", b"b", b"c"):
        tracker.track_sent(message)

    received_count = tracker.handle_reconnect_reply(
        {"type": "reconnected", "reconnect_token": "t2", "last_request": 2}
    )

    assert received_count == 2
    assert tracker.unacknowledged_after(0) == [(3, b"c")]
    assert json.loads(tracker.create_reconnect_request())["reconnect_token"] == "t2"


def test_reconnect_reply_clears_pending_pings():
    clock = _Clock()
    tracker = _tracker_with_offer(clock)
    tracker.create_ping()

    tracker.handle_reconnect_reply(
        {"type": "reconnected", "reconnect_token": "t2", "last_request": 0}
    )
    clock.now = 10

    assert not tracker.is_pong_overdue(5)


@pytest.mark.parametrize(
    "reply,error",
    [
        ({"error_type": "reconnect_failed", "error_code": 400}, SonioxReconnectRefusedError),
        ({"error_type": "service_unavailable", "error_code": 503}, ConnectionError),
    ],
)
def test_failed_reconnect_reply_raises(reply, error):
    tracker = _tracker_with_offer()

    with pytest.raises(error):
        tracker.handle_reconnect_reply({"tokens": [], **reply})


def test_reconnect_is_disabled_after_the_window_expires():
    clock = _Clock()
    tracker = _tracker_with_offer(clock)
    tracker.track_sent(b"a")
    tracker.mark_disconnected()

    clock.now = 20
    assert tracker.can_reconnect
    clock.now = 20.1
    tracker.track_sent(b"b")

    assert not tracker.can_reconnect
    assert tracker.unacknowledged_after(0) == []


def test_reconnect_window_starts_at_the_first_disconnect():
    clock = _Clock()
    tracker = _tracker_with_offer(clock)
    tracker.mark_disconnected()
    clock.now = 15
    tracker.mark_disconnected()
    clock.now = 21

    assert not tracker.can_reconnect


_DROPPED = object()
_CLOSED = object()


class _FakeConnection:
    """A websocket connection whose responses a test feeds, and that can drop."""

    def __init__(self, *responses):
        self.state = State.OPEN
        self.sent = []
        self._queue: asyncio.Queue = asyncio.Queue()
        self.transport = MagicMock()
        self.transport.abort.side_effect = self.drop
        for response in responses:
            self.feed(response)

    def feed(self, response):
        self._queue.put_nowait(json.dumps(response) if isinstance(response, dict) else response)

    def drop(self):
        if self.state is State.OPEN:
            self.state = State.CLOSED
            self._queue.put_nowait(_DROPPED)

    async def send(self, message):
        if self.state is not State.OPEN:
            raise ConnectionClosedError(None, None)
        self.sent.append(message)

    async def recv(self):
        message = await self._queue.get()
        if message is _DROPPED:
            self._queue.put_nowait(_DROPPED)
            raise ConnectionClosedError(None, None)
        if message is _CLOSED:
            self._queue.put_nowait(_CLOSED)
            raise ConnectionClosedOK(None, None)
        return message

    async def close(self):
        self.state = State.CLOSED
        self._queue.put_nowait(_CLOSED)


def _reconnected(last_request: int, token: str = "t2") -> dict:
    return {"type": "reconnected", "reconnect_token": token, "last_request": last_request}


def _soniox_websocket(first, *reconnects, **kwargs):
    """A stream on ``first`` that Soniox offered a reconnect, and its reconnect URLs."""
    urls = []
    remaining = list(reconnects)

    async def connect(url):
        urls.append(url)
        connection = remaining.pop(0)
        if isinstance(connection, Exception):
            raise connection
        return connection

    kwargs.setdefault("ping_interval", 60)
    kwargs.setdefault("retry_delay", 0)
    first.feed(RECONNECT_OFFER)
    return SonioxWebSocket(first, connect, **kwargs), urls


async def _recv(websocket):
    return await asyncio.wait_for(websocket.recv(), timeout=1)


@pytest.mark.asyncio
async def test_reconnect_messages_never_reach_the_caller():
    first = _FakeConnection()
    websocket, _ = _soniox_websocket(first)
    first.feed({"type": "pong", "id": 1})
    first.feed({"tokens": [], "seq": 1})

    assert json.loads(await _recv(websocket)) == {"tokens": [], "seq": 1}


@pytest.mark.asyncio
async def test_a_dropped_connection_reconnects_and_resends_the_missing_messages():
    first, second = _FakeConnection(), _FakeConnection(_reconnected(last_request=1))
    websocket, urls = _soniox_websocket(first, second)
    first.feed({"tokens": [], "seq": 1})
    assert json.loads(await _recv(websocket))["seq"] == 1
    for message in (b"a", b"b", b"c"):
        await websocket.send(message)

    first.drop()
    second.feed({"tokens": [], "seq": 2})

    assert json.loads(await _recv(websocket))["seq"] == 2
    assert urls == ["wss://soniox/r"]
    request, *resent = second.sent
    assert json.loads(request) == {"type": "reconnect", "reconnect_token": "t1", "last_seq": 1}
    assert resent == [b"b", b"c"]


@pytest.mark.asyncio
async def test_messages_sent_while_reconnecting_follow_the_resent_ones():
    first, second = _FakeConnection(), _FakeConnection(_reconnected(last_request=0))
    websocket, _ = _soniox_websocket(first, second)
    await websocket.send(b"a")
    original_send = second.send

    async def send_and_interleave(message):
        await original_send(message)
        if message == b"a":
            await websocket.send(b"b")

    second.send = send_and_interleave

    first.drop()
    second.feed({"tokens": []})
    await _recv(websocket)

    assert second.sent[1:] == [b"a", b"b"]


@pytest.mark.asyncio
async def test_state_stays_open_while_a_dropped_connection_waits_to_reconnect():
    first = _FakeConnection()
    websocket, _ = _soniox_websocket(first)
    await _recv_nothing(websocket)

    first.drop()

    assert websocket.state is State.OPEN
    await websocket.send(b"kept")


async def _recv_nothing(websocket):
    """Let the receive side take the pending reconnect offer."""
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(websocket.recv(), timeout=0.01)


@pytest.mark.asyncio
async def test_a_failed_reconnect_attempt_is_retried():
    first, second = _FakeConnection(), _FakeConnection(_reconnected(last_request=0))
    websocket, urls = _soniox_websocket(first, OSError("unreachable"), second)

    first.drop()
    second.feed({"tokens": []})
    await _recv(websocket)

    assert urls == ["wss://soniox/r", "wss://soniox/r"]


@pytest.mark.asyncio
async def test_a_refused_reconnect_raises_the_drop_and_closes():
    first = _FakeConnection()
    refusing = _FakeConnection({"tokens": [], "error_code": 400, "error_type": "reconnect_failed"})
    websocket, _ = _soniox_websocket(first, refusing)

    first.drop()

    with pytest.raises(ConnectionClosedError):
        await _recv(websocket)
    refusing.transport.abort.assert_called_once()
    assert websocket.state is State.CLOSED


@pytest.mark.asyncio
async def test_a_drop_after_the_stream_finished_is_not_reconnected():
    first = _FakeConnection()
    websocket, urls = _soniox_websocket(first)
    first.feed({"tokens": [], "finished": True, "seq": 1})
    await _recv(websocket)

    first.drop()

    with pytest.raises(ConnectionClosedError):
        await _recv(websocket)
    assert urls == []


@pytest.mark.asyncio
async def test_a_late_pong_drops_the_connection_and_reconnects():
    first, second = _FakeConnection(), _FakeConnection(_reconnected(last_request=0))
    websocket, urls = _soniox_websocket(first, second, ping_interval=0.01, reconnect_timeout=0.02)
    second.feed({"tokens": []})

    await _recv(websocket)

    assert any(json.loads(message)["type"] == "ping" for message in first.sent)
    assert first.state is State.CLOSED
    assert urls == ["wss://soniox/r"]


@pytest.mark.asyncio
async def test_iteration_stops_when_the_stream_is_closed():
    first = _FakeConnection()
    websocket, urls = _soniox_websocket(first)
    first.feed({"tokens": [], "seq": 1})

    await websocket.close()
    received = [message async for message in websocket]

    assert websocket.state is State.CLOSED
    assert urls == []
    assert len(received) == 1
