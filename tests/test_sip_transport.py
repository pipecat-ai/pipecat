#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the SIP transport."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

# The ``sip`` extra is optional; skip the whole module when baresip-python
# isn't installed, matching CI environments that don't pull it.
pytest.importorskip("baresip")

from pipecat.frames.frames import (  # noqa: E402
    ClientConnectedFrame,
    InputDTMFFrame,
    OutputAudioRawFrame,
)
from pipecat.transports.sip.connection import SIPConnection  # noqa: E402
from pipecat.transports.sip.transport import (  # noqa: E402
    SIPOutputTransport,
    SIPParams,
    SIPTransport,
)


def make_connection(**kwargs):
    args = dict(user="1001", domain="example.com", password="secret")
    args.update(kwargs)
    connection = SIPConnection(**args)
    connection.connect = AsyncMock()
    connection.disconnect = AsyncMock()
    connection.dial = AsyncMock(return_value="7")
    connection.answer = AsyncMock()
    connection.hangup = AsyncMock()
    connection.send_dtmf = AsyncMock()
    connection.transfer = AsyncMock()
    return connection


def make_transport(connection=None, **params_kwargs):
    connection = connection or make_connection()
    params = SIPParams(audio_in_enabled=True, audio_out_enabled=True, **params_kwargs)
    transport = SIPTransport(connection, params)
    # The transport pushes frames off the input processor; a mock avoids
    # standing up the full frame-processor machinery.
    transport._input = Mock(push_frame=AsyncMock())
    return transport, connection


def record_events(transport, names):
    """Collect (event, args) tuples in emission order."""
    recorded = []
    for name in names:

        def make_handler(event_name):
            async def handler(transport, *args):
                recorded.append((event_name, args))

            return handler

        transport.add_event_handler(name, make_handler(name))
    return recorded


async def settle():
    for _ in range(20):
        await asyncio.sleep(0)


def set_active_call(connection, handle=7, incoming=False):
    connection._call = Mock(handle=handle)
    connection._call_incoming = incoming


OUT_PAYLOAD = {
    "sessionId": "7",
    "sipCallId": "out-abc",
    "direction": "out",
    "origin": "sip:1001@example.com",
    "destination": "sip:9196@example.com",
}

IN_PAYLOAD = {
    "sessionId": "9",
    "sipCallId": "in-abc",
    "direction": "in",
    "sipFrom": "sip:2002@example.com",
    "sipTo": "sip:1001@example.com",
    "displayName": "Bob",
    "sipHeaders": {"X-Case": "42"},
}


def test_params_default_20ms_chunks():
    assert SIPParams().audio_out_10ms_chunks == 2
    assert SIPParams().auto_answer is True
    assert SIPParams().trunk is None


@pytest.mark.asyncio
async def test_start_dialout_returns_session_and_error_pair():
    transport, connection = make_transport()

    session_id, error = await transport.start_dialout({"sipUri": "sip:9196@example.com"})
    assert (session_id, error) == ("7", None)
    connection.dial.assert_awaited_once_with("sip:9196@example.com", headers=None, video=False)

    _, error = await transport.start_dialout({})
    assert error == "settings must include 'sipUri' or 'phoneNumber'"


@pytest.mark.asyncio
async def test_start_dialout_phone_number_uses_trunk():
    transport, connection = make_transport(trunk="trunk.example.com")

    session_id, error = await transport.start_dialout({"phoneNumber": "+15551230000"})
    assert error is None
    connection.dial.assert_awaited_once_with(
        "sip:+15551230000@trunk.example.com", headers=None, video=False
    )

    no_trunk_transport, _ = make_transport()
    _, error = await no_trunk_transport.start_dialout({"phoneNumber": "+15551230000"})
    assert error == "phoneNumber dial-out requires SIPParams.trunk"


@pytest.mark.asyncio
async def test_start_dialout_busy_and_failure_never_raise():
    transport, connection = make_transport()
    set_active_call(connection)
    _, error = await transport.start_dialout({"sipUri": "sip:9196@example.com"})
    assert error == "transport already has an active call"

    transport, connection = make_transport()
    connection.dial = AsyncMock(side_effect=RuntimeError("boom"))
    session_id, error = await transport.start_dialout({"sipUri": "sip:9196@example.com"})
    assert session_id == ""
    assert "boom" in error


@pytest.mark.asyncio
async def test_dialout_event_sequence_and_compound_emission():
    transport, connection = make_transport()
    recorded = record_events(
        transport,
        [
            "on_dialout_connected",
            "on_dialout_answered",
            "on_dialout_stopped",
            "on_first_participant_joined",
            "on_participant_joined",
            "on_client_connected",
            "on_participant_left",
            "on_client_disconnected",
        ],
    )

    await connection._call_event_handler("call_progress", OUT_PAYLOAD)
    await connection._call_event_handler("call_progress", OUT_PAYLOAD)
    await settle()
    assert [name for name, _ in recorded] == ["on_dialout_connected"]

    await connection._call_event_handler("call_established", OUT_PAYLOAD)
    await settle()
    assert [name for name, _ in recorded] == [
        "on_dialout_connected",
        "on_dialout_answered",
        "on_first_participant_joined",
        "on_participant_joined",
        "on_client_connected",
    ]
    participant = recorded[2][1][0]
    assert participant["id"] == "7"
    assert participant["info"]["userName"] == "sip:9196@example.com"
    transport._input.push_frame.assert_awaited()
    assert isinstance(transport._input.push_frame.await_args.args[0], ClientConnectedFrame)

    closed = dict(OUT_PAYLOAD, reason="hangup", established=True)
    await connection._call_event_handler("call_closed", closed)
    await settle()
    assert [name for name, _ in recorded][5:] == [
        "on_dialout_stopped",
        "on_participant_left",
        "on_client_disconnected",
    ]
    stopped = recorded[5][1][0]
    assert stopped == {
        "sessionId": "7",
        "sipCallId": "out-abc",
        "destination": "sip:9196@example.com",
        "reason": "hangup",
    }
    left_args = recorded[6][1]
    assert left_args[1] == "leftCall"


@pytest.mark.asyncio
async def test_dialin_flow_auto_answers_and_fires_connected():
    transport, connection = make_transport()
    recorded = record_events(
        transport, ["on_dialin_connected", "on_dialin_stopped", "on_client_connected"]
    )

    await connection._call_event_handler("incoming", IN_PAYLOAD)
    await settle()
    connection.answer.assert_awaited_once_with(video=False)

    await connection._call_event_handler("call_established", IN_PAYLOAD)
    await settle()
    assert recorded[0][0] == "on_dialin_connected"
    assert recorded[0][1][0] == IN_PAYLOAD
    participant = recorded[1][1][0]
    assert participant["info"]["userName"] == "Bob"

    closed = dict(IN_PAYLOAD, reason="hangup", established=True)
    await connection._call_event_handler("call_closed", closed)
    await settle()
    stopped = next(args[0] for name, args in recorded if name == "on_dialin_stopped")
    assert stopped["sessionId"] == "9"
    assert stopped["sipCallId"] == "in-abc"
    assert stopped["reason"] == "hangup"
    assert "direction" not in stopped and "established" not in stopped


@pytest.mark.asyncio
async def test_auto_answer_disabled_leaves_call_ringing():
    transport, connection = make_transport(auto_answer=False)

    await connection._call_event_handler("incoming", IN_PAYLOAD)
    await settle()

    connection.answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_video_params_answer_with_video():
    transport, connection = make_transport(video_in_enabled=True)

    await connection._call_event_handler("incoming", IN_PAYLOAD)
    await settle()

    connection.answer.assert_awaited_once_with(video=True)


@pytest.mark.asyncio
async def test_dtmf_event_and_frame_push():
    transport, connection = make_transport()
    set_active_call(connection, handle=9, incoming=True)
    recorded = record_events(transport, ["on_dtmf_event"])

    await connection._call_event_handler("dtmf", SimpleNamespace(digit="5", duration_ms=80))
    await settle()

    assert recorded[0][1][0] == {"sessionId": "9", "tone": "5"}
    frame = transport._input.push_frame.await_args.args[0]
    assert isinstance(frame, InputDTMFFrame)
    assert frame.button.value == "5"


@pytest.mark.asyncio
async def test_dtmf_a_to_d_filtered_but_reported():
    transport, connection = make_transport()
    set_active_call(connection, handle=9, incoming=True)
    recorded = record_events(transport, ["on_dtmf_event"])

    await connection._call_event_handler("dtmf", SimpleNamespace(digit="A", duration_ms=80))
    await settle()

    assert recorded[0][1][0]["tone"] == "A"
    transport._input.push_frame.assert_not_awaited()


@pytest.mark.asyncio
async def test_send_dtmf_session_fallback_and_validation():
    transport, connection = make_transport()

    assert await transport.send_dtmf({"tones": "1"}) == "Can't send DTMF if 'sessionId' is not set"

    set_active_call(connection, handle=7)
    transport._dial_out_session_id = "7"
    assert await transport.send_dtmf({}) == "Can't send DTMF if 'tones' is not set"
    assert await transport.send_dtmf({"tones": "12#"}) is None
    connection.send_dtmf.assert_awaited_once_with("12#")


@pytest.mark.asyncio
async def test_send_dtmf_method_validated_against_connection_mode():
    # Default connection mode is rtpevent: telephone-event passes,
    # sip-info is refused with the remedy named.
    transport, connection = make_transport()
    set_active_call(connection, handle=7)
    transport._dial_out_session_id = "7"
    assert await transport.send_dtmf({"tones": "1", "method": "telephone-event"}) is None
    error = await transport.send_dtmf({"tones": "1", "method": "sip-info"})
    assert "dtmf_mode" in error
    assert "unknown DTMF method" in await transport.send_dtmf({"tones": "1", "method": "inband"})

    info_transport, info_connection = make_transport(connection=make_connection(dtmf_mode="info"))
    set_active_call(info_connection, handle=7)
    info_transport._dial_out_session_id = "7"
    assert await info_transport.send_dtmf({"tones": "1", "method": "sip-info"}) is None
    assert "dtmf_mode" in await info_transport.send_dtmf(
        {"tones": "1", "method": "telephone-event"}
    )

    auto_transport, auto_connection = make_transport(connection=make_connection(dtmf_mode="auto"))
    set_active_call(auto_connection, handle=7)
    auto_transport._dial_out_session_id = "7"
    assert await auto_transport.send_dtmf({"tones": "1", "method": "telephone-event"}) is None
    assert await auto_transport.send_dtmf({"tones": "1", "method": "sip-info"}) is None


@pytest.mark.asyncio
async def test_sip_refer_sends_refer_and_call_transfer_is_rejected():
    transport, connection = make_transport()
    set_active_call(connection, handle=7)
    transport._dial_in_session_id = "7"

    assert await transport.sip_refer({}) == "Can't transfer SIP call if 'toEndPoint' is not set"
    assert await transport.sip_refer({"toEndPoint": "sip:9197@example.com"}) is None
    connection.transfer.assert_awaited_once_with("sip:9197@example.com")

    connection.transfer = AsyncMock(side_effect=RuntimeError("declined"))
    assert "declined" in await transport.sip_refer({"toEndPoint": "sip:9198@x.com"})

    error = await transport.sip_call_transfer({"toEndPoint": "sip:9196@example.com"})
    assert "not supported" in error and "sip_refer" in error
    assert connection.transfer.await_count == 1  # the rejection never touched the call


def tx_info(**kwargs):
    """An AudioInfo stand-in with a healthy, empty transmit buffer."""
    fields = dict(tx_sample_rate=8000, tx_channels=1, tx_buffered=0, tx_ready=True)
    fields.update(kwargs)
    return SimpleNamespace(**fields)


@pytest.mark.asyncio
async def test_write_audio_frame_resamples_to_call_rate():
    transport, connection = make_transport()
    output = SIPOutputTransport(transport, connection, transport._params)
    connection.audio_info = Mock(return_value=tx_info())
    connection.write_audio = Mock(side_effect=len)

    # Feed a run of 20 ms frames; the QQ resampler produces output from
    # the very first chunk (that near-zero priming is the point of QQ).
    for _ in range(10):
        frame = OutputAudioRawFrame(audio=b"\x01\x00" * 320, sample_rate=16000, num_channels=1)
        assert await output.write_audio_frame(frame) is True

    written = b"".join(c.args[0] for c in connection.write_audio.call_args_list)
    # 200 ms in at 16 kHz -> ~100 ms (3200 bytes) out at 8 kHz.
    assert 2800 <= len(written) <= 3400


@pytest.mark.asyncio
async def test_write_audio_frame_without_ready_call_returns_false():
    transport, connection = make_transport()
    output = SIPOutputTransport(transport, connection, transport._params)
    connection.audio_info = Mock(return_value=None)

    frame = OutputAudioRawFrame(audio=b"\x01\x00" * 320, sample_rate=16000, num_channels=1)
    assert await output.write_audio_frame(frame) is False


@pytest.mark.asyncio
async def test_write_audio_frame_retries_rejected_remainder():
    transport, connection = make_transport()
    output = SIPOutputTransport(transport, connection, transport._params)
    connection.audio_info = Mock(return_value=tx_info())
    # The transmit buffer takes half the first time, the rest on retry.
    connection.write_audio = Mock(side_effect=[320, 320])

    frame = OutputAudioRawFrame(audio=b"\x01\x00" * 320, sample_rate=8000, num_channels=1)
    assert await output.write_audio_frame(frame) is True

    first, second = connection.write_audio.call_args_list
    assert len(first.args[0]) == 640
    assert len(second.args[0]) == 320  # only the rejected remainder


@pytest.mark.asyncio
async def test_write_audio_frame_sleeps_off_excess_buffer():
    transport, connection = make_transport()
    output = SIPOutputTransport(transport, connection, transport._params)
    # 8 kHz mono: high water is 1280 bytes (80 ms); report 3200 buffered.
    connection.audio_info = Mock(return_value=tx_info(tx_buffered=3200))
    connection.write_audio = Mock(side_effect=len)

    frame = OutputAudioRawFrame(audio=b"\x01\x00" * 160, sample_rate=8000, num_channels=1)
    with patch("asyncio.sleep", new=AsyncMock()) as sleep:
        assert await output.write_audio_frame(frame) is True

    # (3200 - 1280) bytes of excess at 16000 bytes/s -> 0.12 s.
    sleep.assert_awaited_once()
    assert sleep.await_args.args[0] == pytest.approx(0.12)


@pytest.mark.asyncio
async def test_write_audio_frame_bails_when_call_dies_mid_retry():
    transport, connection = make_transport()
    output = SIPOutputTransport(transport, connection, transport._params)
    # Ready at first, gone when the rejected remainder re-probes.
    connection.audio_info = Mock(side_effect=[tx_info(), None])
    connection.write_audio = Mock(return_value=0)

    frame = OutputAudioRawFrame(audio=b"\x01\x00" * 160, sample_rate=8000, num_channels=1)
    assert await output.write_audio_frame(frame) is False


@pytest.mark.asyncio
async def test_audio_warning_routed_by_direction():
    transport, connection = make_transport()
    recorded = record_events(transport, ["on_dialin_warning", "on_dialout_warning"])

    set_active_call(connection, handle=9, incoming=True)
    await connection._call_event_handler("audio_warning", "rx underrun")
    await settle()
    assert recorded[0][0] == "on_dialin_warning"
    assert recorded[0][1][0] == {"sessionId": "9", "errorMsg": "rx underrun"}

    set_active_call(connection, handle=7, incoming=False)
    await connection._call_event_handler("audio_warning", "tx starved")
    await settle()
    assert recorded[1][0] == "on_dialout_warning"


@pytest.mark.asyncio
async def test_remote_hold_fires_participant_updated():
    transport, connection = make_transport()
    recorded = record_events(transport, ["on_participant_updated"])

    payload = dict(IN_PAYLOAD, on=True)
    await connection._call_event_handler("remote_hold", payload)
    await settle()

    participant = recorded[0][1][0]
    assert participant["media"] == {"onHold": True}


@pytest.mark.asyncio
async def test_dialout_failure_fires_dialout_error():
    transport, connection = make_transport()
    recorded = record_events(transport, ["on_dialout_error"])

    failure = {"sessionId": "7", "error": "CallBusy", "message": "busy here"}
    await connection._call_event_handler("call_failed", failure)
    await settle()

    assert recorded[0][1][0] == {
        "sessionId": "7",
        "errorMsg": "busy here",
        "error": "CallBusy",
    }


@pytest.mark.asyncio
async def test_receive_audio_surfaces_unexpected_errors():
    from pipecat.transports.sip.transport import SIPInputTransport

    transport, connection = make_transport()
    input_transport = SIPInputTransport(transport, connection, transport._params)
    input_transport._streaming = True
    input_transport.push_error = AsyncMock()
    connection.audio_info = Mock(return_value=SimpleNamespace(rx_sample_rate=8000))
    connection.read_audio = Mock(side_effect=RuntimeError("runtime is dead"))

    await input_transport._receive_audio()

    input_transport.push_error.assert_awaited_once()
    assert "runtime is dead" in input_transport.push_error.await_args.args[0]


@pytest.mark.asyncio
async def test_registered_maps_to_dialin_ready_and_before_leave_once():
    transport, connection = make_transport()
    recorded = record_events(transport, ["on_dialin_ready", "on_before_leave"])

    await connection._call_event_handler("registered", "sip:1001@example.com")
    await settle()
    assert recorded[0] == ("on_dialin_ready", ("sip:1001@example.com",))

    await transport._before_leave()
    await transport._before_leave()
    await settle()
    assert [name for name, _ in recorded].count("on_before_leave") == 1
