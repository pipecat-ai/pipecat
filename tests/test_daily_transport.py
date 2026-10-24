#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Daily transport."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import (
    BotConnectedFrame,
    InputDTMFFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    STTMetadataFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.stt_latency import DEEPGRAM_TTFS_P99
from pipecat.transports.daily.transport import (
    DailyOutputDTMFFrame,
    DailyParams,
    DailyTransport,
)
from pipecat.workers.runner import WorkerRunner


def _make_transport(**params_kwargs) -> DailyTransport:
    with (
        patch("pipecat.transports.daily.transport.Daily"),
        patch("pipecat.transports.daily.transport.CallClient"),
    ):
        return DailyTransport(
            "https://mock.daily.co/mock", None, "bot", params=DailyParams(**params_kwargs)
        )


def _make_client_session(
    *, status: int = 200, body: str = "", error: Exception | None = None
) -> MagicMock:
    response = MagicMock(status=status)
    response.text = AsyncMock(return_value=body)
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=None)

    session = MagicMock()
    if error:
        session.post.side_effect = error
    else:
        session.post.return_value = response

    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=None)
    return context


def _make_dialin_transport() -> DailyTransport:
    return _make_transport(
        api_key="test-api-key",
        dialin_settings={"call_id": "test-call", "call_domain": "test-domain"},
    )


@pytest.mark.asyncio
async def test_on_joined_pushes_stt_metadata_when_transcription_starts():
    transport = _make_transport(transcription_enabled=True)
    transport.start_transcription = AsyncMock(return_value=None)
    transport._input = AsyncMock()

    await transport._on_joined({})

    transport._input.push_stt_metadata_frame.assert_awaited_once()
    # BotConnectedFrame is pushed before the STT metadata frame.
    call_names = [name for (name, _, _) in transport._input.mock_calls]
    assert call_names.index("push_frame") < call_names.index("push_stt_metadata_frame")
    (frame,) = transport._input.push_frame.await_args.args
    assert isinstance(frame, BotConnectedFrame)


@pytest.mark.asyncio
async def test_on_joined_skips_stt_metadata_when_transcription_fails():
    transport = _make_transport(transcription_enabled=True)
    transport.start_transcription = AsyncMock(return_value="some error")
    transport._on_error = AsyncMock()
    transport._input = AsyncMock()

    await transport._on_joined({})

    transport._on_error.assert_awaited_once()
    transport._input.push_stt_metadata_frame.assert_not_awaited()
    transport._input.push_frame.assert_awaited_once()


@pytest.mark.asyncio
async def test_on_joined_skips_stt_metadata_when_transcription_disabled():
    transport = _make_transport()
    transport._input = AsyncMock()

    await transport._on_joined({})

    transport._input.push_stt_metadata_frame.assert_not_awaited()
    transport._input.push_frame.assert_awaited_once()


@pytest.mark.asyncio
async def test_push_stt_metadata_frame_contents():
    transport = _make_transport(transcription_enabled=True)
    input_transport = transport.input()
    input_transport.broadcast_frame = AsyncMock()

    await input_transport.push_stt_metadata_frame()

    input_transport.broadcast_frame.assert_awaited_once_with(
        STTMetadataFrame,
        service_name=input_transport.name,
        ttfs_p99_latency=DEEPGRAM_TTFS_P99,
    )


@pytest.mark.asyncio
async def test_on_dialin_ready_reports_http_error_and_ready_event():
    transport = _make_dialin_transport()
    transport._on_error = AsyncMock()
    transport._call_event_handler = AsyncMock()
    session = _make_client_session(status=503, body="service unavailable")

    with patch("pipecat.transports.daily.transport.aiohttp.ClientSession", return_value=session):
        await transport._on_dialin_ready("sip:test@example.com")

    transport._on_error.assert_awaited_once_with(
        "Unable to handle dialin-ready event (status: 503, error: service unavailable)"
    )
    transport._call_event_handler.assert_awaited_once_with(
        "on_dialin_ready", "sip:test@example.com"
    )


@pytest.mark.asyncio
async def test_on_dialin_ready_reports_timeout():
    transport = _make_dialin_transport()
    transport._on_error = AsyncMock()
    session = _make_client_session(error=TimeoutError())

    with patch("pipecat.transports.daily.transport.aiohttp.ClientSession", return_value=session):
        await transport._handle_dialin_ready("sip:test@example.com")

    transport._on_error.assert_awaited_once_with(
        "Timeout handling dialin-ready event (https://api.daily.co/v1/dialin/pinlessCallUpdate)"
    )


@pytest.mark.asyncio
async def test_on_dialin_ready_reports_unexpected_error():
    transport = _make_dialin_transport()
    transport._on_error = AsyncMock()
    session = _make_client_session(error=RuntimeError("connection failed"))

    with patch("pipecat.transports.daily.transport.aiohttp.ClientSession", return_value=session):
        await transport._handle_dialin_ready("sip:test@example.com")

    transport._on_error.assert_awaited_once_with(
        "Error handling dialin-ready event "
        "(https://api.daily.co/v1/dialin/pinlessCallUpdate): connection failed"
    )


@pytest.mark.asyncio
async def test_on_dialin_ready_does_not_report_success_as_error():
    transport = _make_dialin_transport()
    transport._on_error = AsyncMock()
    session = _make_client_session()

    with patch("pipecat.transports.daily.transport.aiohttp.ClientSession", return_value=session):
        await transport._handle_dialin_ready("sip:test@example.com")

    transport._on_error.assert_not_awaited()


def _make_session_client(*, dial_out: str = "", dial_in: str = ""):
    """Return a transport client with session ids set and a fake Daily call client."""
    transport = _make_transport()
    client = transport._client
    client._dial_out_session_id = dial_out
    client._dial_in_session_id = dial_in
    client._task_manager = MagicMock()
    client._task_manager.get_event_loop.return_value = asyncio.get_running_loop()

    def _complete(settings, completion):
        completion(None)

    client._client.send_dtmf = MagicMock(side_effect=_complete)
    client._client.sip_refer = MagicMock(side_effect=_complete)
    return client


@pytest.mark.asyncio
async def test_send_dtmf_falls_back_to_dial_in_session_id():
    client = _make_session_client(dial_in="dial-in-session")

    error = await client.send_dtmf({"tones": "1"})

    assert error is None
    settings = client._client.send_dtmf.call_args.args[0]
    assert settings["sessionId"] == "dial-in-session"


@pytest.mark.asyncio
async def test_send_dtmf_prefers_dial_out_session_id_over_dial_in():
    client = _make_session_client(dial_out="dial-out-session", dial_in="dial-in-session")

    error = await client.send_dtmf({"tones": "1"})

    assert error is None
    settings = client._client.send_dtmf.call_args.args[0]
    assert settings["sessionId"] == "dial-out-session"


@pytest.mark.asyncio
async def test_send_dtmf_errors_without_any_session_id():
    client = _make_session_client()

    error = await client.send_dtmf({"tones": "1"})

    assert error == "Can't send DTMF if 'sessionId' is not set"
    client._client.send_dtmf.assert_not_called()


@pytest.mark.asyncio
async def test_sip_refer_falls_back_to_dial_in_session_id():
    client = _make_session_client(dial_in="dial-in-session")

    error = await client.sip_refer({"toEndPoint": "sip:agent@example.com"})

    assert error is None
    settings = client._client.sip_refer.call_args.args[0]
    assert settings["sessionId"] == "dial-in-session"


@pytest.mark.asyncio
async def test_sip_refer_keeps_explicit_session_id():
    client = _make_session_client(dial_out="dial-out-session")

    error = await client.sip_refer(
        {"sessionId": "explicit-session", "toEndPoint": "sip:agent@example.com"}
    )

    assert error is None
    settings = client._client.sip_refer.call_args.args[0]
    assert settings["sessionId"] == "explicit-session"


@pytest.mark.asyncio
async def test_sip_refer_errors_without_any_session_id():
    client = _make_session_client()

    error = await client.sip_refer({"toEndPoint": "sip:agent@example.com"})

    assert error == "Can't perform SIP REFER if 'sessionId' is not set"
    client._client.sip_refer.assert_not_called()


@pytest.mark.asyncio
async def test_dtmf_event_pushes_input_dtmf_frame():
    transport = _make_transport()
    transport._input = AsyncMock()
    transport._call_event_handler = AsyncMock()

    await transport._on_dtmf_event({"sessionId": "s", "tone": "5"})

    transport._call_event_handler.assert_awaited_once_with(
        "on_dtmf_event", {"sessionId": "s", "tone": "5"}
    )
    (frame,) = transport._input.push_frame.await_args.args
    assert isinstance(frame, InputDTMFFrame)
    assert frame.button == KeypadEntry.FIVE


@pytest.mark.asyncio
@pytest.mark.parametrize("tone", ["A", "D"])
async def test_dtmf_event_ignores_tone_outside_keypad(tone):
    transport = _make_transport()
    transport._input = AsyncMock()
    transport._call_event_handler = AsyncMock()

    await transport._on_dtmf_event({"sessionId": "s", "tone": tone})

    transport._call_event_handler.assert_awaited_once()
    transport._input.push_frame.assert_not_awaited()


@pytest.mark.asyncio
async def test_tone_outside_keypad_keeps_daily_events_flowing():
    transport = _make_transport()
    transport._input = AsyncMock()
    client = transport._client
    client._joined_event.set()
    queue = asyncio.Queue()
    task = asyncio.create_task(client._callback_task_handler(queue))
    next_event = AsyncMock()

    await queue.put((transport._on_dtmf_event, {"sessionId": "s", "tone": "A"}))
    await queue.put((next_event, {"id": "caller"}))
    await asyncio.wait_for(queue.join(), timeout=1)

    next_event.assert_awaited_once_with({"id": "caller"})
    task.cancel()


@pytest.mark.asyncio
async def test_failing_callback_keeps_daily_events_flowing():
    client = _make_transport()._client
    client._joined_event.set()
    queue = asyncio.Queue()
    task = asyncio.create_task(client._callback_task_handler(queue))
    next_event = AsyncMock()

    async def failing_callback(data):
        raise RuntimeError("boom")

    await queue.put((failing_callback, {"id": "caller"}))
    await queue.put((next_event, {"id": "caller"}))
    await asyncio.wait_for(queue.join(), timeout=1)

    next_event.assert_awaited_once_with({"id": "caller"})
    assert not task.done()
    task.cancel()


@pytest.mark.asyncio
async def test_dtmf_frame_pushes_error_when_send_fails():
    transport = _make_transport()
    output = transport.output()
    output._client.send_dtmf = AsyncMock(return_value="Can't send DTMF if 'sessionId' is not set")
    output.push_error = AsyncMock()

    await output._write_dtmf_native(DailyOutputDTMFFrame(button=KeypadEntry.ONE))

    output.push_error.assert_awaited_once_with(
        "Unable to send DTMF: Can't send DTMF if 'sessionId' is not set"
    )


@pytest.mark.asyncio
async def test_dtmf_frame_pushes_no_error_when_send_succeeds():
    transport = _make_transport()
    output = transport.output()
    output._client.send_dtmf = AsyncMock(return_value=None)
    output.push_error = AsyncMock()

    await output._write_dtmf_native(DailyOutputDTMFFrame(button=KeypadEntry.ONE))

    output._client.send_dtmf.assert_awaited_once()
    output.push_error.assert_not_awaited()


@pytest.mark.parametrize("event_name", ["on_transcription_stopped", "on_transcription_error"])
def test_transcription_events_accept_handlers(event_name):
    transport = _make_transport()

    async def handler(*args):
        pass

    transport.add_event_handler(event_name, handler)

    assert transport._event_handlers[event_name].handlers == [handler]


async def _join_client_settings(transport: DailyTransport) -> dict:
    client = transport._client
    client._get_event_loop = MagicMock(return_value=asyncio.get_running_loop())
    client._client.join.side_effect = lambda *args, completion, **kwargs: completion({}, None)

    await client._join()

    return client._client.join.call_args.kwargs["client_settings"]


@pytest.mark.asyncio
async def test_join_sets_initial_receive_bitrate():
    transport = _make_transport(initial_receive_bitrate=2_500_000)

    client_settings = await _join_client_settings(transport)

    assert client_settings["connection"] == {"initialReceiveBitrate": 2_500_000}


@pytest.mark.asyncio
async def test_join_omits_connection_without_initial_receive_bitrate():
    transport = _make_transport()

    client_settings = await _join_client_settings(transport)

    assert "connection" not in client_settings


async def _make_joined_client():
    """Return a joined transport client with fake callbacks and Daily calls."""
    transport = _make_transport()
    client = transport._client
    client._task_manager = MagicMock()
    client._task_manager.get_event_loop.return_value = asyncio.get_running_loop()
    for name in (
        "on_joined",
        "on_left",
        "on_before_leave",
        "on_error",
        "on_call_state_updated",
        "on_network_connection_status_updated",
    ):
        setattr(client._callbacks, name, AsyncMock())

    def _complete(*args, completion, **kwargs):
        completion(None)

    client._client.send_app_message = MagicMock(side_effect=_complete)
    client._client.leave = MagicMock(side_effect=_complete)
    client._join = AsyncMock(return_value=({}, None))

    await client.join()
    return client


@pytest.mark.asyncio
async def test_send_message_sends_while_connected():
    client = await _make_joined_client()

    error = await client.send_message(OutputTransportMessageUrgentFrame(message={"n": 1}))

    assert error is None
    client._client.send_app_message.assert_called_once()


class _BlockedSetup(FrameProcessor):
    async def setup(self, setup):
        await super().setup(setup)
        await asyncio.Event().wait()


async def _block_forever(*args, **kwargs):
    await asyncio.Event().wait()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "frame_type", [OutputTransportMessageFrame, OutputTransportMessageUrgentFrame]
)
@pytest.mark.parametrize("join_blocks", [False, True], ids=["joined", "join_blocks"])
async def test_sends_after_setup_timeout_are_dropped(join_blocks, frame_type):
    transport = _make_transport(audio_in_enabled=False, audio_out_enabled=False)
    client = transport._client
    call_client = client._client
    if join_blocks:
        client._join = AsyncMock(side_effect=_block_forever)
        await client.send_message(frame_type(message={"n": 0}))
    else:
        client._join = AsyncMock(return_value=({}, None))
    client._leave = AsyncMock(return_value=None)
    for name in ("on_joined", "on_left", "on_before_leave", "on_error"):
        setattr(client._callbacks, name, AsyncMock())
    processors = [transport.input(), transport.output()]
    if not join_blocks:
        processors.append(_BlockedSetup())
    worker = PipelineWorker(Pipeline(processors), setup_timeout_secs=0.5, enable_rtvi=False)
    setup_timed_out = asyncio.Event()

    @worker.event_handler("on_setup_timeout")
    async def on_setup_timeout(worker):
        setup_timed_out.set()

    runner = WorkerRunner(handle_sigint=False, handle_sigterm=False)
    await runner.add_workers(worker)
    try:
        await asyncio.wait_for(runner.run(), timeout=10)
    finally:
        await runner.cancel()

    assert setup_timed_out.is_set()
    client._join.assert_awaited_once()
    # The pipeline never started, so the client was released without leave().
    client._leave.assert_not_awaited()
    call_client.release.assert_called_once()
    assert client._client is None

    error = await client.send_message(frame_type(message={"n": 1}))

    assert error is None
    assert not client._join_message_queue
    call_client.send_app_message.assert_not_called()

    error = await client.send_prebuilt_chat_message("hello")
    assert error == "Can't send message if not joined"
    call_client.send_prebuilt_chat_message.assert_not_called()


@pytest.mark.asyncio
async def test_call_ending_without_leave_reports_once_and_drops_messages():
    client = await _make_joined_client()

    await client._handle_call_state_updated("left")
    for n in range(3):
        error = await client.send_message(OutputTransportMessageUrgentFrame(message={"n": n}))
        assert error is None

    client._callbacks.on_error.assert_awaited_once()
    client._callbacks.on_call_state_updated.assert_awaited_once_with("left")
    client._client.send_app_message.assert_not_called()


@pytest.mark.asyncio
async def test_repeated_left_reports_once():
    client = await _make_joined_client()

    await client._handle_call_state_updated("left")
    await client._handle_call_state_updated("left")

    client._callbacks.on_error.assert_awaited_once()


@pytest.mark.asyncio
async def test_left_after_leave_is_not_an_error():
    client = await _make_joined_client()

    await client.leave()
    await client._handle_call_state_updated("left")

    client._client.leave.assert_called_once()
    client._callbacks.on_left.assert_awaited_once()
    client._callbacks.on_error.assert_not_awaited()
    client._callbacks.on_call_state_updated.assert_awaited_once_with("left")


@pytest.mark.asyncio
async def test_leave_after_call_ends_skips_daily_leave():
    client = await _make_joined_client()

    await client._handle_call_state_updated("left")
    await client.leave()

    client._client.leave.assert_not_called()
    client._callbacks.on_before_leave.assert_awaited_once()
    client._callbacks.on_left.assert_awaited_once()
    client._callbacks.on_error.assert_awaited_once()


@pytest.mark.asyncio
async def test_messages_after_leave_are_dropped():
    client = await _make_joined_client()

    await client.leave()
    error = await client.send_message(OutputTransportMessageUrgentFrame(message={"n": 1}))

    assert error is None
    assert not client._join_message_queue
    client._client.send_app_message.assert_not_called()


@pytest.mark.asyncio
async def test_other_call_states_keep_messages_flowing():
    client = await _make_joined_client()

    await client._handle_call_state_updated("leaving")
    error = await client.send_message(OutputTransportMessageUrgentFrame(message={"n": 1}))

    assert error is None
    client._callbacks.on_error.assert_not_awaited()
    client._client.send_app_message.assert_called_once()


def test_transcription_started_leaves_unmanaged_session_alone():
    transport = _make_transport(transcription_enabled=False)
    client = transport._client
    client._call_event_callback = MagicMock()

    client.on_transcription_started({})

    client._call_event_callback.assert_not_called()


@pytest.mark.asyncio
async def test_transcription_started_scopes_to_captured_participants():
    transport = _make_transport(transcription_enabled=True)
    client = transport._client
    await client.capture_participant_transcription("user-1")
    client._call_event_callback = MagicMock()

    client.on_transcription_started({})

    client._call_event_callback.assert_called_once_with(client.update_transcription, ["user-1"])


def _signalling(event: str) -> dict:
    return {"connection": "signalling", "event": event}


@pytest.mark.asyncio
async def test_signalling_interruption_warns_once_and_drops_messages():
    client = await _make_joined_client()

    with patch("pipecat.transports.daily.transport.logger") as log:
        await client._handle_network_connection_status_updated(_signalling("interrupted"))
        await client._handle_network_connection_status_updated(_signalling("interrupted"))
    for n in range(3):
        error = await client.send_message(OutputTransportMessageUrgentFrame(message={"n": n}))
        assert error is None

    log.warning.assert_called_once()
    client._callbacks.on_error.assert_not_awaited()
    assert client._callbacks.on_network_connection_status_updated.await_count == 2
    client._client.send_app_message.assert_not_called()


@pytest.mark.asyncio
async def test_signalling_reconnect_resumes_messages():
    client = await _make_joined_client()

    await client._handle_network_connection_status_updated(_signalling("interrupted"))
    await client._handle_network_connection_status_updated(_signalling("connected"))
    error = await client.send_message(OutputTransportMessageUrgentFrame(message={"n": 1}))

    assert error is None
    client._client.send_app_message.assert_called_once()


@pytest.mark.asyncio
async def test_signalling_flaps_follow_the_latest_event():
    client = await _make_joined_client()

    for event in ("interrupted", "connected", "interrupted"):
        await client._handle_network_connection_status_updated(_signalling(event))
    dropped = await client.send_message(OutputTransportMessageUrgentFrame(message={"n": 1}))
    await client._handle_network_connection_status_updated(_signalling("connected"))
    sent = await client.send_message(OutputTransportMessageUrgentFrame(message={"n": 2}))

    assert dropped is None and sent is None
    client._callbacks.on_error.assert_not_awaited()
    assert client._callbacks.on_network_connection_status_updated.await_count == 4
    client._client.send_app_message.assert_called_once()


@pytest.mark.asyncio
async def test_call_ending_during_signalling_interruption_is_an_error():
    client = await _make_joined_client()

    await client._handle_network_connection_status_updated(_signalling("interrupted"))
    await client._handle_call_state_updated("left")
    error = await client.send_message(OutputTransportMessageUrgentFrame(message={"n": 1}))

    assert error is None
    client._callbacks.on_error.assert_awaited_once()
    client._client.send_app_message.assert_not_called()


@pytest.mark.asyncio
async def test_signalling_connected_while_connected_is_not_reported():
    client = await _make_joined_client()

    await client._handle_network_connection_status_updated(_signalling("connected"))

    client._callbacks.on_error.assert_not_awaited()
    client._callbacks.on_network_connection_status_updated.assert_awaited_once_with(
        _signalling("connected")
    )


@pytest.mark.asyncio
async def test_media_transport_interruption_keeps_messages_flowing():
    client = await _make_joined_client()

    await client._handle_network_connection_status_updated(
        {"connection": "sendTransport", "event": "interrupted"}
    )
    error = await client.send_message(OutputTransportMessageUrgentFrame(message={"n": 1}))

    assert error is None
    client._callbacks.on_error.assert_not_awaited()
    client._client.send_app_message.assert_called_once()


@pytest.mark.asyncio
async def test_output_reports_failed_sends_once_until_one_succeeds():
    transport = _make_transport()
    output = transport.output()
    output.push_error = AsyncMock()
    transport._client.send_message = AsyncMock(
        side_effect=["Disconnected", "Disconnected", None, "Disconnected"]
    )

    for n in range(4):
        await output.send_message(OutputTransportMessageUrgentFrame(message={"n": n}))

    assert output.push_error.await_count == 2


@pytest.mark.asyncio
async def test_transcription_capture_continues_during_signalling_interruption():
    client = await _make_joined_client()
    client._params.transcription_enabled = True
    client._transcription_status = {"transcriptId": "t"}
    client.update_transcription = AsyncMock()

    await client._handle_network_connection_status_updated(_signalling("interrupted"))
    await client.capture_participant_transcription("participant")

    client.update_transcription.assert_awaited_once_with(["participant"])
