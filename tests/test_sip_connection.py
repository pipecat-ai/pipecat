#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the SIP connection layer."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

# The ``sip`` extra is optional; skip the whole module when baresip-python
# isn't installed, matching CI environments that don't pull it.
pytest.importorskip("baresip")

from baresip import BaresipError, CallBusy, CallFailed, CallState, Config  # noqa: E402
from baresip.events import Event, StackEvent  # noqa: E402

import pipecat.transports.sip.connection as sip_connection  # noqa: E402
from pipecat.transports.sip.connection import SIPConnection, _SharedRuntime  # noqa: E402
from pipecat.transports.sip.media_encryption import SIPMediaEncryption  # noqa: E402

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
    call.on_info = Mock()
    call.off_info = Mock()
    call.on_audio_warning = Mock()
    call.off_audio_warning = Mock()
    return call


def make_fake_ua():
    ua = Mock()
    ua.register = AsyncMock()
    ua.unregister = AsyncMock()
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
async def test_rtp_timeout_defaults_to_thirty_seconds(env):
    connection = make_connection()

    await connection.connect()

    (config,) = env.runtime.start.await_args.args
    assert config.rtp_timeout == 30


@pytest.mark.asyncio
async def test_runtime_starts_with_config_object(env):
    # The object form is load-bearing: expose_headers and instance_id are
    # applied by Runtime.start() only when it receives a Config — rendered
    # text silently drops them.
    connection = make_connection(
        net_interface="127.0.0.1",
        expose_headers=("X-Case",),
        rtp_timeout=30,
        instance_id="0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0",
    )

    await connection.connect()

    (config,) = env.runtime.start.await_args.args
    assert isinstance(config, Config)
    assert config.net_interface == "127.0.0.1"
    assert config.expose_headers == ("X-Case",)
    assert config.rtp_timeout == 30
    assert config.instance_id == "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0"


@pytest.mark.asyncio
async def test_debug_logging_settings_reach_config(env):
    connection = make_connection(native_log_level="debug", sip_trace=True)

    await connection.connect()

    (config,) = env.runtime.start.await_args.args
    assert config.native_log_level == "debug"
    assert config.sip_trace is True
    # The stdlib gate must open far enough for the DEBUG-level SIP trace.
    assert logging.getLogger("baresip").level == logging.DEBUG


@pytest.mark.asyncio
async def test_jitter_buffer_renders_into_config(env):
    connection = make_connection(jitter_buffer_mode="adaptive", jitter_buffer_ms=(20, 100))

    await connection.connect()

    (config,) = env.runtime.start.await_args.args
    assert config.extra_config_text == (
        "audio_jitter_buffer_type adaptive\naudio_jitter_buffer_ms 20-100\n"
    )


@pytest.mark.asyncio
async def test_jitter_buffer_off_renders_the_type_alone(env):
    connection = make_connection(jitter_buffer_mode="off")

    await connection.connect()

    (config,) = env.runtime.start.await_args.args
    assert config.extra_config_text == "audio_jitter_buffer_type off\n"


@pytest.mark.asyncio
async def test_jitter_buffer_unset_renders_the_transport_default(env):
    # Unset means the transport's own default, rendered explicitly — not the
    # stack's compiled setting, which is 60 ms deeper.
    connection = make_connection()

    await connection.connect()

    (config,) = env.runtime.start.await_args.args
    assert connection._settings.jitter_buffer_mode == "fixed"
    assert connection._settings.jitter_buffer_ms == (40, 60)
    assert config.extra_config_text == (
        "audio_jitter_buffer_type fixed\naudio_jitter_buffer_ms 40-60\n"
    )


@pytest.mark.asyncio
async def test_stack_default_is_reachable_explicitly(env):
    connection = make_connection(jitter_buffer_ms=(100, 200))

    await connection.connect()

    (config,) = env.runtime.start.await_args.args
    assert config.extra_config_text == (
        "audio_jitter_buffer_type fixed\naudio_jitter_buffer_ms 100-200\n"
    )


def test_jitter_buffer_range_alone_means_fixed():
    connection = make_connection(jitter_buffer_ms=(40, 60))

    assert connection._settings.jitter_buffer_mode == "fixed"
    assert connection._settings.jitter_buffer_ms == (40, 60)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"jitter_buffer_mode": "fixed"},  # a range is required
        {"jitter_buffer_mode": "off", "jitter_buffer_ms": (20, 40)},
        {"jitter_buffer_mode": "sometimes", "jitter_buffer_ms": (20, 40)},
        {"jitter_buffer_ms": (60, 40)},  # min must not exceed max
        {"jitter_buffer_ms": (0, 40)},
        {"jitter_buffer_ms": (20,)},
    ],
)
def test_jitter_buffer_rejects_inconsistent_arguments(kwargs):
    with pytest.raises(ValueError):
        make_connection(**kwargs)


@pytest.mark.asyncio
async def test_conflicting_jitter_buffers_raise(env):
    # The buffer belongs to the process-wide stack, like net_interface.
    first = make_connection(jitter_buffer_ms=(40, 60))
    second = make_connection(user="1002", jitter_buffer_ms=(20, 40))

    await first.connect()
    with pytest.raises(ValueError):
        await second.connect()


@pytest.mark.asyncio
async def test_sip_port_renders_into_config(env):
    connection = make_connection(reg_interval=0, sip_port=5060)

    await connection.connect()

    (config,) = env.runtime.start.await_args.args
    assert config.extra_config_text == (
        "sip_listen 0.0.0.0:5060\naudio_jitter_buffer_type fixed\naudio_jitter_buffer_ms 40-60\n"
    )


@pytest.mark.parametrize("port", [1, 5060, 65535])
def test_sip_port_accepts_valid_ports(port):
    connection = make_connection(sip_port=port)

    assert connection._settings.sip_port == port


@pytest.mark.parametrize("port", [0, -1, 65536, "5060", 5060.0, True])
def test_sip_port_rejects_invalid_values(port):
    with pytest.raises(ValueError):
        make_connection(sip_port=port)


@pytest.mark.asyncio
async def test_conflicting_sip_port_raise(env):
    # One stack per process owns the listen socket.
    first = make_connection(reg_interval=0, sip_port=5060)
    second = make_connection(user="1002", reg_interval=0, sip_port=5070)

    await first.connect()
    with pytest.raises(ValueError):
        await second.connect()


@pytest.mark.asyncio
async def test_extra_config_unset_leaves_the_config_text_as_before(env):
    connection = make_connection(sip_port=5060)

    await connection.connect()

    (config,) = env.runtime.start.await_args.args
    assert config.extra_config_text == (
        "sip_listen 0.0.0.0:5060\naudio_jitter_buffer_type fixed\naudio_jitter_buffer_ms 40-60\n"
    )


@pytest.mark.asyncio
async def test_extra_config_lines_follow_the_transports_own(env):
    connection = make_connection(
        sip_port=5060, extra_config=("sip_verify_server no", "rtp_ports 20000-20100")
    )

    await connection.connect()

    (config,) = env.runtime.start.await_args.args
    assert config.extra_config_text == (
        "sip_listen 0.0.0.0:5060\naudio_jitter_buffer_type fixed\naudio_jitter_buffer_ms 40-60\n"
        "sip_verify_server no\nrtp_ports 20000-20100\n"
    )


@pytest.mark.parametrize(
    "extra_config", ["sip_verify_server no", ("",), ("a b\nc d",), ("a\x00b",), (5,)]
)
def test_extra_config_rejects_malformed_values(extra_config):
    with pytest.raises(ValueError, match="extra_config"):
        make_connection(extra_config=extra_config)


@pytest.mark.asyncio
async def test_conflicting_extra_config_raises(env):
    first = make_connection()
    await first.connect()

    second = make_connection(user="1002", extra_config=("sip_verify_server no",))
    with pytest.raises(ValueError, match="extra_config"):
        await second.connect()


def test_extra_params_render_into_the_account_aor():
    connection = make_connection(
        extra_params=("medianat=ice", "stunserver=stun:stun.example.com:3478")
    )

    assert connection._account.extra_params == (
        "medianat=ice",
        "stunserver=stun:stun.example.com:3478",
    )
    aor = connection._account.aor()
    assert "medianat=ice" in aor
    assert "stunserver=stun:stun.example.com:3478" in aor


def test_media_encryption_defaults_to_none_and_leaves_the_account_as_before():
    connection = make_connection()

    assert connection._account.media_encryption == "none"
    assert connection._account.requires_srtp is False
    assert "mediaenc" not in connection._account.aor()
    assert connection._account_kwargs["media_encryption"] is SIPMediaEncryption.NONE


@pytest.mark.parametrize(
    "domain, transport, registrar",
    [
        ("example.com", "udp", "sip:example.com;transport=udp"),
        ("example.com", "tls", "sip:example.com;transport=tls"),
        ("example.com:5061", "tls", "sip:example.com:5061;transport=tls"),
    ],
)
def test_registrar_defaults_to_the_domain_of_a_registering_account(domain, transport, registrar):
    connection = make_connection(domain=domain, transport=transport)

    assert connection._account.registrar == registrar
    assert f'outbound="{registrar}"' in connection._account.aor()


def test_registrar_is_not_defaulted_for_an_account_that_does_not_register():
    connection = make_connection(reg_interval=0)

    assert connection._account.registrar is None
    assert "outbound" not in connection._account.aor()


def test_registrar_given_is_used_as_given():
    connection = make_connection(registrar="sbc.example.com;transport=tcp")

    assert connection._account.registrar == "sbc.example.com;transport=tcp"
    # An address given is used in trunk mode too.
    trunk = make_connection(registrar="sbc.example.com", reg_interval=0)
    assert trunk._account.registrar == "sbc.example.com"


@pytest.mark.parametrize("param", ['outbound="sip:sbc.example.com"', "outbound1=sip:a.example.com"])
def test_registrar_is_not_defaulted_over_an_outbound_in_extra_params(param):
    connection = make_connection(extra_params=("medianat=stun", param))

    assert connection._account.registrar is None
    assert connection._account.aor().count("outbound") == 1


@pytest.mark.asyncio
async def test_unregistered_sibling_shares_a_registering_account(env):
    # The consult leg of a transfer does not register; it has to agree with
    # the registered connection whose user agent it places its call on.
    connection = make_connection()
    await connection.connect()

    sibling = connection.consult_connection()
    assert sibling._account_kwargs == connection._account_kwargs
    await sibling.connect()


@pytest.mark.parametrize("value", [SIPMediaEncryption.SRTP, "srtp", "SRTP"])
def test_media_encryption_srtp_reaches_the_account(value):
    connection = make_connection(media_encryption=value)

    assert connection._account.media_encryption == "srtp"
    assert connection._account.requires_srtp is True
    assert connection._account_kwargs["media_encryption"] is SIPMediaEncryption.SRTP
    # The consult leg of a transfer is on the same account, so it encrypts too.
    assert connection.consult_connection()._account.media_encryption == "srtp"


@pytest.mark.parametrize("value", ["dtls", "", None, 0, True])
def test_media_encryption_rejects_an_unknown_value(value):
    with pytest.raises(ValueError, match="media encryption must be one of 'none', 'srtp'"):
        make_connection(media_encryption=value)


@pytest.mark.asyncio
async def test_conflicting_media_encryption_on_one_account_raises(env):
    first = make_connection()
    await first.connect()

    second = make_connection(media_encryption=SIPMediaEncryption.SRTP)
    with pytest.raises(ValueError, match="media_encryption"):
        await second.connect()


@pytest.mark.asyncio
async def test_media_encrypted_follows_the_active_call(env):
    connection = make_connection()
    await connection.connect()
    assert connection.media_encrypted is False  # no call

    call = await _answered_outbound_call(env, connection)
    call.media_encrypted = False
    assert connection.media_encrypted is False
    call.media_encrypted = True
    assert connection.media_encrypted is True


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


def test_consult_connection_shares_account_and_is_registrationless():
    connection = make_connection(transport="tls", extra_params=("medianat=stun",))

    consult = connection.consult_connection()

    assert isinstance(consult, SIPConnection)
    assert consult._account.user == "1001"
    assert consult._account.domain == "example.com"
    assert consult._account.transport == "tls"
    assert consult._account.extra_params == ("medianat=stun",)
    # A consult leg never registers; it reuses the primary's registered UA.
    assert consult._account.reg_interval == 0
    # ...and stays out of inbound routing so a call never lands on it while idle.
    assert consult._route_inbound is False


@pytest.mark.asyncio
async def test_consult_leg_is_not_routed_inbound(env):
    # A consult leg opts out of inbound routing, so an incoming call is never
    # handed to it while it sits idle between connect and dial — even when the
    # primary is busy, the call is rejected rather than landing on the consult.
    primary = make_connection()
    await primary.connect()
    primary.take_incoming(make_fake_call(handle=0xA1))  # primary is now busy
    assert primary.has_active_call

    consult = primary.consult_connection()
    await consult.connect()
    assert not consult.has_active_call

    route = env.ua.incoming_callbacks[0]
    incoming = make_fake_call(handle=0xB2)
    route(incoming)
    for _ in range(10):
        await asyncio.sleep(0)

    assert not consult.has_active_call  # the consult leg never claimed it
    incoming.reject.assert_awaited_once()  # every routed connection was busy


@pytest.mark.asyncio
async def test_consult_connection_connects_against_configured_runtime(env):
    # A sibling must present the primary's runtime-wide settings, or acquiring
    # the already-running runtime raises on a mismatch. Exercise it with a
    # non-default setting and a live connect.
    connection = make_connection(rtp_timeout=30)
    await connection.connect()

    consult = connection.consult_connection()
    await consult.connect()  # must not raise on a settings mismatch

    assert env.runtime_constructions == 1  # sibling reuses the running runtime
    assert consult.is_connected
    env.ua.register.assert_awaited_once()  # only the primary registered


@pytest.mark.asyncio
async def test_wait_established_delegates_to_call(env):
    connection = make_connection()
    await connection.connect()
    call = make_fake_call(handle=0x7)
    env.ua.dial.return_value = call
    await connection.dial("sip:9196@example.com")

    await connection.wait_established(timeout=5.0)

    # dial() also arms an establishment watcher that awaits wait_established();
    # assert our explicit-timeout call landed among the awaits.
    call.wait_established.assert_any_await(5.0)


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
async def test_local_hangup_of_ringing_dial_does_not_emit_call_failed(env):
    # hangup() of a not-yet-established dial-out closes the call, which makes
    # the binding's wait_established() raise; that close is a requested stop,
    # not a failure, so the establishment watcher stays quiet.
    connection = make_connection()
    await connection.connect()
    call = make_fake_call()
    hung_up = asyncio.Event()

    async def wait_established(timeout=None):
        await hung_up.wait()
        raise CallFailed("hangup")

    call.wait_established = wait_established
    call.hangup = AsyncMock(side_effect=lambda: hung_up.set())
    env.ua.dial.return_value = call
    payloads, arrived = capture(connection, "call_failed")

    await connection.dial("sip:9196@example.com")
    await connection.hangup()

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(arrived.wait(), 0.2)
    assert not payloads


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

    # Connect fires on the later of answered + audio RTP, not the SIP handshake.
    listener(StackEvent(event=Event.CALL_ANSWERED, call=call.handle))
    listener(StackEvent(event=Event.CALL_RTPESTAB, call=call.handle, text="audio"))
    await asyncio.wait_for(established.wait(), EVENT_TIMEOUT)
    assert established_payloads[0]["destination"] == "sip:2002@example.com"
    assert established_payloads[0]["sipCallId"] == "abc123"
    assert established_payloads[0]["direction"] == "out"

    listener(StackEvent(event=Event.CALL_CLOSED, call=call.handle, text="hangup"))
    await asyncio.wait_for(closed.wait(), EVENT_TIMEOUT)
    assert closed_payloads[0]["reason"] == "hangup"
    assert closed_payloads[0]["transferred"] is False  # a hangup is not a transfer
    assert closed_payloads[0]["established"] is True
    assert not connection.has_active_call


@pytest.mark.asyncio
async def test_call_closed_by_the_audio_timeout_reports_rtp_timeout(env):
    connection = make_connection()
    await connection.connect()
    call = make_fake_call()
    env.ua.dial.return_value = call
    closed_payloads, closed = capture(connection, "call_closed")

    await connection.dial("sip:9196@example.com")
    listener = call.listeners[0]

    listener(StackEvent(event=Event.CALL_CLOSED, call=call.handle, text="rtp stream error"))
    await asyncio.wait_for(closed.wait(), EVENT_TIMEOUT)
    assert closed_payloads[0]["reason"] == "rtp timeout"
    assert closed_payloads[0]["transferred"] is False


@pytest.mark.asyncio
async def test_call_closed_flags_transfer_from_reason(env):
    # The connection derives a `transferred` flag from the stack's exact
    # transfer-success close reason, so a completed transfer (blind or Replaces)
    # is told apart from a hangup on the close.
    connection = make_connection()
    await connection.connect()
    call = make_fake_call()
    env.ua.dial.return_value = call
    closed_payloads, closed = capture(connection, "call_closed")

    await connection.dial("sip:9196@example.com")
    listener = call.listeners[0]

    listener(StackEvent(event=Event.CALL_CLOSED, call=call.handle, text="Call transfered"))
    await asyncio.wait_for(closed.wait(), EVENT_TIMEOUT)
    assert closed_payloads[0]["reason"] == "Call transfered"
    assert closed_payloads[0]["transferred"] is True


@pytest.mark.asyncio
async def test_connect_requires_answered_and_audio_rtp(env):
    connection = make_connection()
    await connection.connect()
    call = make_fake_call()
    env.ua.dial.return_value = call
    payloads, established = capture(connection, "call_established")

    await connection.dial("sip:9196@example.com")
    listener = call.listeners[0]

    # Answered alone does not connect — media is not up yet.
    listener(StackEvent(event=Event.CALL_ANSWERED, call=call.handle))
    await asyncio.sleep(0.05)
    assert not established.is_set()

    # Audio RTP completes the media path → connect fires exactly once.
    listener(StackEvent(event=Event.CALL_RTPESTAB, call=call.handle, text="audio"))
    await asyncio.wait_for(established.wait(), EVENT_TIMEOUT)
    assert len(payloads) == 1


@pytest.mark.asyncio
async def test_connect_audio_rtp_before_answered(env):
    # Early media: RTP can arrive before the 200 OK; connect still waits for
    # answered so the bot does not greet over ringback.
    connection = make_connection()
    await connection.connect()
    call = make_fake_call()
    env.ua.dial.return_value = call
    payloads, established = capture(connection, "call_established")

    await connection.dial("sip:9196@example.com")
    listener = call.listeners[0]

    listener(StackEvent(event=Event.CALL_RTPESTAB, call=call.handle, text="audio"))
    await asyncio.sleep(0.05)
    assert not established.is_set()

    listener(StackEvent(event=Event.CALL_ANSWERED, call=call.handle))
    await asyncio.wait_for(established.wait(), EVENT_TIMEOUT)
    assert len(payloads) == 1


@pytest.mark.asyncio
async def test_video_rtp_does_not_connect(env):
    # RTPESTAB fires per stream; a voice bot connects on audio, not video.
    connection = make_connection()
    await connection.connect()
    call = make_fake_call()
    env.ua.dial.return_value = call
    payloads, established = capture(connection, "call_established")

    await connection.dial("sip:9196@example.com")
    listener = call.listeners[0]

    listener(StackEvent(event=Event.CALL_ANSWERED, call=call.handle))
    listener(StackEvent(event=Event.CALL_RTPESTAB, call=call.handle, text="video"))
    await asyncio.sleep(0.05)
    assert not established.is_set()

    listener(StackEvent(event=Event.CALL_RTPESTAB, call=call.handle, text="audio"))
    await asyncio.wait_for(established.wait(), EVENT_TIMEOUT)
    assert len(payloads) == 1


@pytest.mark.asyncio
async def test_close_before_connect_reports_not_established(env):
    # A call that never reaches media-up reports established=False on close
    # (and never emitted call_established).
    connection = make_connection()
    await connection.connect()
    call = make_fake_call()
    env.ua.dial.return_value = call
    established_payloads, established = capture(connection, "call_established")
    closed_payloads, closed = capture(connection, "call_closed")

    await connection.dial("sip:9196@example.com")
    listener = call.listeners[0]

    listener(StackEvent(event=Event.CALL_ANSWERED, call=call.handle))  # no audio RTP
    listener(StackEvent(event=Event.CALL_CLOSED, call=call.handle, text="timeout"))
    await asyncio.wait_for(closed.wait(), EVENT_TIMEOUT)

    assert closed_payloads[0]["established"] is False
    assert not established.is_set()


@pytest.mark.asyncio
async def test_renegotiated_fires_only_once_established(env):
    connection = make_connection()
    await connection.connect()
    call = make_fake_call()
    env.ua.dial.return_value = call
    payloads, arrived = capture(connection, "renegotiated")

    await connection.dial("sip:9196@example.com")
    listener = call.listeners[0]

    # The initial answer's REMOTE_SDP is negotiation, not an update.
    listener(StackEvent(event=Event.CALL_REMOTE_SDP, call=call.handle, text="answer"))
    listener(StackEvent(event=Event.CALL_ANSWERED, call=call.handle))
    listener(StackEvent(event=Event.CALL_RTPESTAB, call=call.handle, text="audio"))
    listener(StackEvent(event=Event.CALL_REMOTE_SDP, call=call.handle, text="offer"))

    await asyncio.wait_for(arrived.wait(), EVENT_TIMEOUT)
    assert len(payloads) == 1
    assert payloads[0]["sdp"] == "offer"


@pytest.mark.asyncio
async def test_disconnect_unregisters_the_account(env):
    connection = make_connection()
    await connection.connect()

    await connection.disconnect()

    env.ua.unregister.assert_awaited_once()
    env.runtime.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_shared_account_unregisters_after_its_last_connection(env):
    first = make_connection()
    second = make_connection()
    await first.connect()
    await second.connect()

    await first.disconnect()
    env.ua.unregister.assert_not_awaited()

    await second.disconnect()
    env.ua.unregister.assert_awaited_once()


@pytest.mark.asyncio
async def test_unregister_failure_does_not_block_disconnect(env):
    env.ua.unregister = AsyncMock(side_effect=BaresipError("registrar gone"))
    connection = make_connection()
    await connection.connect()

    await connection.disconnect()

    env.ua.unregister.assert_awaited_once()
    env.runtime.close.assert_awaited_once()


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


async def _answered_outbound_call(env, connection):
    """Dial and report the far end's answer; returns the fake call."""
    call = make_fake_call()
    env.ua.dial.return_value = call
    await connection.dial("sip:9196@example.com")
    call.listeners[0](StackEvent(event=Event.CALL_ANSWERED, call=call.handle))
    return call


@pytest.mark.asyncio
async def test_hangup_without_headers_calls_the_binding_bare(env):
    connection = make_connection()
    await connection.connect()
    call = await _answered_outbound_call(env, connection)

    await connection.hangup()

    call.hangup.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_hangup_sends_the_headers_set_for_it(env):
    connection = make_connection()
    await connection.connect()
    call = await _answered_outbound_call(env, connection)

    connection.set_hangup_headers({"X-RouteType": "queue", "X-RouteValue": 7})
    await connection.hangup()

    call.hangup.assert_awaited_once_with(headers={"X-RouteType": "queue", "X-RouteValue": "7"})


@pytest.mark.asyncio
async def test_disconnect_sends_the_headers_set_for_hangup(env):
    connection = make_connection()
    await connection.connect()
    call = await _answered_outbound_call(env, connection)

    connection.set_hangup_headers({"X-RouteType": "queue"})
    await connection.disconnect()

    call.hangup.assert_awaited_once_with(headers={"X-RouteType": "queue"})


@pytest.mark.asyncio
async def test_hangup_headers_are_replaced_and_cleared(env):
    connection = make_connection()
    await connection.connect()
    call = await _answered_outbound_call(env, connection)

    connection.set_hangup_headers({"X-RouteType": "queue"})
    connection.set_hangup_headers({"X-RouteType": "agent"})
    assert connection._hangup_headers == {"X-RouteType": "agent"}

    connection.set_hangup_headers(None)
    await connection.hangup()

    call.hangup.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_hangup_headers_need_an_answered_call(env):
    connection = make_connection()
    await connection.connect()
    with pytest.raises(RuntimeError, match="no active call"):
        connection.set_hangup_headers({"X-RouteType": "queue"})

    env.ua.dial.return_value = make_fake_call()
    await connection.dial("sip:9196@example.com")
    with pytest.raises(RuntimeError, match="answered"):
        connection.set_hangup_headers({"X-RouteType": "queue"})


@pytest.mark.parametrize(
    "headers",
    [
        {"": "v"},
        {"X Bad": "v"},
        {"X-Bad:": "v"},
        {"X-Bad\n": "v"},
        {"X-Ok": "a\r\nInjected: 1"},
        {"X-Ok": None},
    ],
)
@pytest.mark.asyncio
async def test_hangup_headers_are_checked_when_set(env, headers):
    connection = make_connection()
    await connection.connect()
    await _answered_outbound_call(env, connection)

    with pytest.raises(ValueError):
        connection.set_hangup_headers(headers)
    assert connection._hangup_headers is None


@pytest.mark.asyncio
async def test_hangup_headers_too_long_for_the_stack_are_refused_when_set(env):
    connection = make_connection()
    await connection.connect()
    await _answered_outbound_call(env, connection)

    # "X-Fits: <value>\r\n" is the value plus ten bytes; 2048 is one too many.
    with pytest.raises(ValueError, match="less than 2048 bytes, got 2048"):
        connection.set_hangup_headers({"X-Fits": "a" * 2038})
    assert connection._hangup_headers is None

    connection.set_hangup_headers({"X-Fits": "a" * 2037})
    assert connection._hangup_headers == {"X-Fits": "a" * 2037}


@pytest.mark.asyncio
async def test_disconnect_finishes_when_the_stack_refuses_the_hangup_headers(env):
    connection = make_connection()
    await connection.connect()
    call = await _answered_outbound_call(env, connection)
    connection.set_hangup_headers({"X-RouteType": "queue"})

    async def hangup(headers=None):
        if headers:
            raise ValueError("headers must fit in 2048 bytes")

    call.hangup = AsyncMock(side_effect=hangup)

    await connection.disconnect()

    # Refused with the headers, then ended without them; teardown completed.
    assert [c.kwargs for c in call.hangup.await_args_list] == [
        {"headers": {"X-RouteType": "queue"}},
        {},
    ]
    env.ua.unregister.assert_awaited_once()
    env.runtime.close.assert_awaited_once()
    assert not connection.has_active_call


@pytest.mark.asyncio
async def test_hangup_headers_do_not_reach_the_next_call(env):
    connection = make_connection()
    await connection.connect()
    first = await _answered_outbound_call(env, connection)
    connection.set_hangup_headers({"X-RouteType": "queue"})
    first.listeners[0](StackEvent(event=Event.CALL_CLOSED, call=first.handle, text="hangup"))

    second = await _answered_outbound_call(env, connection)
    await connection.hangup()

    second.hangup.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_sip_info_is_emitted_with_its_content_type_and_body(env):
    connection = make_connection()
    await connection.connect()
    call = await _answered_outbound_call(env, connection)
    payloads, arrived = capture(connection, "info")

    (on_info,) = call.on_info.call_args.args
    on_info(SimpleNamespace(content_type="application/json", body=b'{"event":"hold"}'))

    await asyncio.wait_for(arrived.wait(), EVENT_TIMEOUT)
    assert payloads == [
        {"sessionId": "1", "contentType": "application/json", "body": b'{"event":"hold"}'}
    ]


@pytest.mark.asyncio
async def test_sip_info_listener_is_removed_when_the_call_closes(env):
    connection = make_connection()
    await connection.connect()
    call = await _answered_outbound_call(env, connection)
    (on_info,) = call.on_info.call_args.args

    call.listeners[0](StackEvent(event=Event.CALL_CLOSED, call=call.handle, text="hangup"))

    call.off_info.assert_called_once_with(on_info)


@pytest.mark.asyncio
async def test_transfer_without_headers_calls_the_binding_bare(env):
    connection = make_connection()
    await connection.connect()
    call = await _answered_outbound_call(env, connection)

    await connection.transfer("sip:9197@example.com")

    call.transfer.assert_awaited_once_with("sip:9197@example.com")


@pytest.mark.asyncio
async def test_transfer_passes_headers_through(env):
    connection = make_connection()
    await connection.connect()
    call = await _answered_outbound_call(env, connection)

    await connection.transfer("sip:9197@example.com", headers={"User-to-User": "56a3;encoding=hex"})

    call.transfer.assert_awaited_once_with(
        "sip:9197@example.com", headers={"User-to-User": "56a3;encoding=hex"}
    )


@pytest.mark.asyncio
async def test_reject_defaults_to_busy_here(env):
    connection = make_connection()
    await connection.connect()
    call = make_fake_call()
    env.ua.incoming_callbacks[0](call)

    await connection.reject()

    call.reject.assert_awaited_once_with(status=486, reason=None, headers=None)


@pytest.mark.asyncio
async def test_reject_passes_status_reason_and_headers_through(env):
    connection = make_connection()
    await connection.connect()
    call = make_fake_call()
    env.ua.incoming_callbacks[0](call)

    await connection.reject(503, reason="Try Elsewhere", headers={"Retry-After": "30"})

    call.reject.assert_awaited_once_with(
        status=503, reason="Try Elsewhere", headers={"Retry-After": "30"}
    )


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
    assert connection.flush_tx() is None
    await connection.request_keyframe()
    # Direction changes are call-level: without a call they refuse.
    with pytest.raises(RuntimeError):
        await connection.set_video_direction("sendrecv")


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
