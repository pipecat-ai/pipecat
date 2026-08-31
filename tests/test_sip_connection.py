#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the SIP connection layer."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

# The ``sip`` extra is optional; skip the whole module when baresip-python
# isn't installed, matching CI environments that don't pull it.
pytest.importorskip("baresip")

from baresip import CallBusy, CallState  # noqa: E402
from baresip.events import Event, StackEvent  # noqa: E402

import pipecat.transports.sip.connection as sip_connection  # noqa: E402
from pipecat.transports.sip.connection import SIPConnection, _SharedRuntime  # noqa: E402

EVENT_TIMEOUT = 2.0


def make_fake_call(handle=0x1, peer="sip:2002@example.com", headers=None):
    call = Mock()
    call.handle = handle
    call.peer = peer
    call.call_id = "abc123"
    call.headers = headers or {}
    call.state = CallState.OUTGOING
    call.final_stats = None
    call.answer = AsyncMock()
    call.reject = AsyncMock()
    call.hangup = AsyncMock()
    call.wait_established = AsyncMock()
    call.send_dtmf = AsyncMock()
    call.hold = AsyncMock()
    call.resume = AsyncMock()
    call.transfer = AsyncMock()
    call.attended_transfer = AsyncMock()
    call.listeners = []
    call.on = Mock(side_effect=call.listeners.append)
    call.off = Mock()
    call.on_dtmf = Mock()
    call.off_dtmf = Mock()
    call.on_audio_warning = Mock()
    call.off_audio_warning = Mock()
    return call


def make_fake_ua():
    ua = Mock()
    ua.register = AsyncMock()
    ua.dial = AsyncMock()
    ua.incoming_callbacks = []
    ua.on_incoming = Mock(side_effect=ua.incoming_callbacks.append)
    return ua


class Env:
    """A fresh shared runtime plus mocked binding entry points."""

    def __init__(self, monkeypatch):
        self.runtime = Mock()
        self.runtime.start = AsyncMock()
        self.runtime.close = AsyncMock()
        self.runtime_constructions = 0

        def make_runtime(*args, **kwargs):
            self.runtime_constructions += 1
            return self.runtime

        self.ua = make_fake_ua()
        fake_user_agent = Mock()
        fake_user_agent.create = AsyncMock(return_value=self.ua)

        monkeypatch.setattr(sip_connection, "_SHARED", _SharedRuntime())
        monkeypatch.setattr(sip_connection, "Runtime", make_runtime)
        monkeypatch.setattr(sip_connection, "UserAgent", fake_user_agent)


@pytest.fixture
def env(monkeypatch):
    return Env(monkeypatch)


def make_connection(**kwargs):
    args = dict(user="1001", domain="example.com", password="secret")
    args.update(kwargs)
    return SIPConnection(**args)


def capture(connection, event_name):
    """Collect an event's payloads and an asyncio.Event set on arrival."""
    payloads = []
    arrived = asyncio.Event()

    @connection.event_handler(event_name)
    async def handler(connection, *args):
        payloads.append(args[0] if args else None)
        arrived.set()

    return payloads, arrived


@pytest.mark.asyncio
async def test_connect_registers_and_emits_registered(env):
    connection = make_connection()
    payloads, arrived = capture(connection, "registered")

    await connection.connect()

    env.runtime.start.assert_awaited_once()
    env.ua.register.assert_awaited_once()
    await asyncio.wait_for(arrived.wait(), EVENT_TIMEOUT)
    assert payloads == ["sip:1001@example.com"]
    assert connection.is_connected


@pytest.mark.asyncio
async def test_trunk_mode_skips_registration(env):
    connection = make_connection(reg_interval=0)

    await connection.connect()

    env.ua.register.assert_not_awaited()
    assert connection.is_connected


@pytest.mark.asyncio
async def test_two_connections_share_one_runtime(env):
    first = make_connection()
    second = make_connection(user="1002")

    await first.connect()
    await second.connect()
    assert env.runtime_constructions == 1

    await first.disconnect()
    env.runtime.close.assert_not_awaited()
    await second.disconnect()
    env.runtime.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_conflicting_runtime_settings_raise(env):
    first = make_connection()
    second = make_connection(user="1002", net_interface="127.0.0.1")

    await first.connect()
    with pytest.raises(ValueError):
        await second.connect()


@pytest.mark.asyncio
async def test_shared_owner_pairing_runs_work_once(env):
    connection = make_connection()

    # Both transport halves call connect()/disconnect(); the body runs once.
    await connection.connect()
    await connection.connect()
    env.ua.register.assert_awaited_once()

    await connection.disconnect()
    env.runtime.close.assert_not_awaited()
    await connection.disconnect()
    env.runtime.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_incoming_routed_to_idle_connection(env):
    first = make_connection()
    second = make_connection()
    await first.connect()
    await second.connect()

    first_payloads, first_arrived = capture(first, "incoming")
    second_payloads, second_arrived = capture(second, "incoming")

    route = env.ua.incoming_callbacks[0]
    call_a = make_fake_call(handle=0xA)
    call_b = make_fake_call(handle=0xB)
    call_c = make_fake_call(handle=0xC)

    route(call_a)
    await asyncio.wait_for(first_arrived.wait(), EVENT_TIMEOUT)
    assert first.has_active_call
    assert first_payloads[0]["sessionId"] == str(0xA)
    assert first_payloads[0]["sipFrom"] == "sip:2002@example.com"

    route(call_b)
    await asyncio.wait_for(second_arrived.wait(), EVENT_TIMEOUT)
    assert second.has_active_call
    assert second_payloads[0]["sessionId"] == str(0xB)

    route(call_c)
    for _ in range(10):
        await asyncio.sleep(0)
    call_c.reject.assert_awaited_once()


@pytest.mark.asyncio
async def test_dial_returns_session_id_and_busy_raises(env):
    connection = make_connection()
    await connection.connect()
    call = make_fake_call(handle=0x7)
    env.ua.dial.return_value = call

    session_id = await connection.dial("sip:9196@example.com")

    assert session_id == str(0x7)
    assert connection.session_id == str(0x7)
    with pytest.raises(RuntimeError):
        await connection.dial("sip:9197@example.com")


@pytest.mark.asyncio
async def test_dial_failure_emits_call_failed(env):
    connection = make_connection()
    await connection.connect()
    call = make_fake_call()
    call.wait_established = AsyncMock(side_effect=CallBusy("busy here"))
    env.ua.dial.return_value = call
    payloads, arrived = capture(connection, "call_failed")

    await connection.dial("sip:9196@example.com")

    await asyncio.wait_for(arrived.wait(), EVENT_TIMEOUT)
    assert payloads[0]["error"] == "CallBusy"
    assert "busy" in payloads[0]["message"]


@pytest.mark.asyncio
async def test_call_events_relay_established_and_closed(env):
    connection = make_connection()
    await connection.connect()
    call = make_fake_call()
    env.ua.dial.return_value = call
    established_payloads, established = capture(connection, "call_established")
    closed_payloads, closed = capture(connection, "call_closed")

    await connection.dial("sip:9196@example.com")
    listener = call.listeners[0]

    listener(StackEvent(event=Event.CALL_ESTABLISHED, call=call.handle))
    await asyncio.wait_for(established.wait(), EVENT_TIMEOUT)
    assert established_payloads[0]["destination"] == "sip:2002@example.com"
    assert established_payloads[0]["sipCallId"] == "abc123"
    assert established_payloads[0]["direction"] == "out"

    listener(StackEvent(event=Event.CALL_CLOSED, call=call.handle, text="hangup"))
    await asyncio.wait_for(closed.wait(), EVENT_TIMEOUT)
    assert closed_payloads[0]["reason"] == "hangup"
    assert closed_payloads[0]["established"] is True
    assert not connection.has_active_call


@pytest.mark.asyncio
async def test_disconnect_hangs_up_active_call(env):
    connection = make_connection()
    await connection.connect()
    call = make_fake_call()
    env.ua.dial.return_value = call
    await connection.dial("sip:9196@example.com")

    await connection.disconnect()

    call.hangup.assert_awaited_once()
    env.runtime.close.assert_awaited_once()
    assert not connection.has_active_call
    assert not connection.is_connected


@pytest.mark.asyncio
async def test_answer_passes_video_and_headers_through(env):
    connection = make_connection()
    await connection.connect()
    call = make_fake_call()
    route = env.ua.incoming_callbacks[0]
    route(call)

    await connection.answer(video=True, headers={"X-Agent": "pipecat"})

    call.answer.assert_awaited_once_with(video=True, headers={"X-Agent": "pipecat"})


@pytest.mark.asyncio
async def test_media_taps_without_call_are_quiet(env):
    connection = make_connection()

    assert connection.read_audio(320) == b""
    assert connection.write_audio(b"\x00" * 320) == 0
    assert connection.read_video_frame() is None
    assert connection.write_video_frame(b"\x00") is False
    assert connection.audio_info() is None
    await connection.request_keyframe()


@pytest.mark.asyncio
async def test_register_failure_releases_runtime(env):
    env.ua.register = AsyncMock(side_effect=RuntimeError("401"))
    connection = make_connection()

    with pytest.raises(RuntimeError):
        await connection.connect()

    env.runtime.close.assert_awaited_once()
    assert not connection.is_connected

    # A failed connect is terminal: the same error re-raises, unretried.
    env.ua.register = AsyncMock()
    with pytest.raises(RuntimeError):
        await connection.connect()
    env.ua.register.assert_not_awaited()


@pytest.mark.asyncio
async def test_ua_create_failure_releases_runtime(env, monkeypatch):
    failing_user_agent = Mock()
    failing_user_agent.create = AsyncMock(side_effect=RuntimeError("alloc failed"))
    monkeypatch.setattr(sip_connection, "UserAgent", failing_user_agent)
    connection = make_connection()

    with pytest.raises(RuntimeError):
        await connection.connect()

    env.runtime.close.assert_awaited_once()
    assert not connection.is_connected
