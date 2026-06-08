#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import parse_qs, urlparse

import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close
from websockets.protocol import State

from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.sarvam.stt_realtime import (
    REALTIME_MODEL,
    SARVAM_STT_REALTIME_URL,
    SarvamRealtimeSTTError,
    SarvamRealtimeSTTService,
    _build_realtime_ws_url,
)
from pipecat.transcriptions.language import Language


class _FakeWebsocket:
    def __init__(self, *, send_side_effect=None):
        self.sent = []
        self.state = State.OPEN
        self.close = AsyncMock()
        self.send = AsyncMock(side_effect=send_side_effect or self._send)

    async def _send(self, payload):
        self.sent.append(payload)


def _parse_ws_url(url: str) -> dict[str, str]:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    return {key: value[0] for key, value in qs.items()}


def _make_service(
    *,
    endpointing: str = "vad",
    reconnect_on_error: bool = False,
) -> SarvamRealtimeSTTService:
    service = SarvamRealtimeSTTService(
        api_key="sk_test",
        reconnect_on_error=reconnect_on_error,
        settings=SarvamRealtimeSTTService.Settings(
            language="hi-IN",
            endpointing=endpointing,
            stream_type="fast",
        ),
    )
    service._pushed_frames = []
    service._broadcast_frames = []
    service._interruptions = 0
    service._usage_events = []

    async def push_frame(frame, *args, **kwargs):
        service._pushed_frames.append(frame)

    async def broadcast_frame(frame_class, *args, **kwargs):
        service._broadcast_frames.append(frame_class())

    async def broadcast_interruption(*args, **kwargs):
        service._interruptions += 1

    async def call_event_handler(event_name, *args, **kwargs):
        if event_name == "on_usage":
            service._usage_events.append(args)

    service.push_frame = push_frame
    service.broadcast_frame = broadcast_frame
    service.broadcast_interruption = broadcast_interruption
    service._call_event_handler = call_event_handler
    service.start_processing_metrics = AsyncMock()
    service.stop_processing_metrics = AsyncMock()
    service.start_ttfb_metrics = AsyncMock()
    service.stop_ttfb_metrics = AsyncMock()
    return service


def _frame_types(service: SarvamRealtimeSTTService) -> list[type]:
    return [type(frame) for frame in service._broadcast_frames + service._pushed_frames]


def test_realtime_stt_does_not_retry_by_default() -> None:
    service = SarvamRealtimeSTTService(api_key="sk_test")

    assert service._reconnect_on_error is False


def test_realtime_stt_can_opt_into_pipecat_websocket_retries() -> None:
    service = SarvamRealtimeSTTService(api_key="sk_test", reconnect_on_error=True)

    assert service._reconnect_on_error is True


def test_realtime_ws_url_includes_core_and_vad_params() -> None:
    settings = SarvamRealtimeSTTService.Settings(
        language="hi-IN",
        stream_type="fast",
        endpointing="vad",
        vad_sot_threshold=0.4,
        vad_min_speech_ms=300,
        vad_min_silence_ms=800,
        vad_smoothing_alpha=0.5,
    )

    url = _build_realtime_ws_url(SARVAM_STT_REALTIME_URL, settings, 16000)

    assert url.startswith(SARVAM_STT_REALTIME_URL)
    assert _parse_ws_url(url) == {
        "language-code": "hi-IN",
        "stream-type": "fast",
        "endpointing": "vad",
        "encoding": "linear16",
        "sample_rate": "16000",
        "model": REALTIME_MODEL,
        "vad_sot_threshold": "0.4",
        "vad_min_speech_ms": "300",
        "vad_min_silence_ms": "800",
        "vad_smoothing_alpha": "0.5",
    }


@pytest.mark.asyncio
async def test_realtime_stt_logs_raw_server_message_at_debug() -> None:
    service = _make_service()
    payload = {
        "event": "future.server_event",
        "utterance_idx": 7,
        "text": "नमस्ते",
        "confidence": 0.91,
    }

    class _MessageWebsocket:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if getattr(self, "_sent", False):
                raise StopAsyncIteration
            self._sent = True
            return json.dumps(payload)

    service._websocket = _MessageWebsocket()
    with patch("pipecat.services.sarvam.stt_realtime.logger") as logger:
        logger.debug = Mock()
        await service._receive_messages()

    debug_messages = [call.args[0] for call in logger.debug.call_args_list]
    message = next(msg for msg in debug_messages if "raw server message" in msg)
    assert "Sarvam realtime STT raw server message" in message
    assert "raw_data=" in message
    assert "future.server_event" in message
    assert "नमस्ते" in message


def test_realtime_ws_url_omits_vad_params_for_manual_endpointing() -> None:
    settings = SarvamRealtimeSTTService.Settings(
        language="en-IN",
        endpointing="manual",
        vad_sot_threshold=0.4,
        vad_min_speech_ms=300,
    )

    params = _parse_ws_url(_build_realtime_ws_url(SARVAM_STT_REALTIME_URL, settings, 16000))

    assert params["endpointing"] == "manual"
    assert "vad_sot_threshold" not in params
    assert "vad_min_speech_ms" not in params


def test_realtime_settings_validate_contract() -> None:
    with pytest.raises(ValueError, match="language auto is only supported"):
        SarvamRealtimeSTTService.Settings(language="auto", stream_type="fast")

    with pytest.raises(ValueError, match="sample_rate must be one of"):
        SarvamRealtimeSTTService(api_key="sk_test", sample_rate=44100)

    with pytest.raises(ValueError, match="language od-IN is not supported"):
        SarvamRealtimeSTTService.Settings(language="od-IN")


def test_realtime_language_mapping_uses_realtime_odia_code() -> None:
    settings = SarvamRealtimeSTTService.Settings(language=Language.OR_IN)

    assert settings.language == "or-IN"


def test_simulated_streaming_allows_auto_language_and_mode() -> None:
    settings = SarvamRealtimeSTTService.Settings(
        language="auto",
        stream_type="simulated",
        mode="translate",
    )

    params = _parse_ws_url(_build_realtime_ws_url(SARVAM_STT_REALTIME_URL, settings, 16000))

    assert params["language-code"] == "auto"
    assert params["stream-type"] == "simulated"
    assert params["mode"] == "translate"


@pytest.mark.asyncio
async def test_event_mapping_defers_final_until_speech_end() -> None:
    service = _make_service()
    service._audio_position = 1.25

    await service._handle_message(
        {
            "event": "session.begin",
            "session_id": "sess_123",
            "request_id": "srv_custom-id_v9",
        }
    )
    await service._handle_message({"event": "vad.speech_start", "utterance_idx": 0})
    await service._handle_message(
        {
            "event": "transcript.partial",
            "utterance_idx": 0,
            "text": "नमस्ते",
            "confidence": 0.91,
        }
    )
    await service._handle_message(
        {
            "event": "transcript.final",
            "utterance_idx": 0,
            "text": "नमस्ते आप कैसे हैं",
            "language": "hi-IN",
            "language_confidence": 0.99,
        }
    )

    assert _frame_types(service) == [UserStartedSpeakingFrame, InterimTranscriptionFrame]

    service._audio_position = 1.75
    await service._handle_message({"event": "vad.speech_end", "utterance_idx": 0})

    assert _frame_types(service) == [
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
        InterimTranscriptionFrame,
        TranscriptionFrame,
    ]
    final_frame = service._pushed_frames[1]
    assert final_frame.text == "नमस्ते आप कैसे हैं"
    assert final_frame.language == Language.HI_IN
    assert final_frame.finalized is True
    assert final_frame.result["request_id"] == "srv_custom-id_v9"
    assert final_frame.result["session_id"] == "sess_123"
    assert final_frame.result["end_time"] == 1.75
    assert final_frame.result["confidence"] == 0.99


@pytest.mark.asyncio
async def test_speech_end_waits_for_final_transcript() -> None:
    service = _make_service()
    service._audio_position = 2.0

    with patch.object(service, "create_task", side_effect=lambda coro, *_, **__: coro.close()):
        await service._handle_message({"event": "vad.speech_start", "utterance_idx": 0})
        await service._handle_message({"event": "vad.speech_end", "utterance_idx": 0})

    assert _frame_types(service) == [UserStartedSpeakingFrame]

    await service._handle_message(
        {
            "event": "transcript.final",
            "utterance_idx": 0,
            "text": "नमस्ते आप कैसे हैं",
            "language": "hi-IN",
            "language_confidence": 0.99,
        }
    )

    final_frame = service._pushed_frames[0]
    assert final_frame.result["end_time"] == 2.0
    assert final_frame.result["speech_end_wall_time"] > 0
    assert _frame_types(service) == [
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
        TranscriptionFrame,
    ]


@pytest.mark.asyncio
async def test_eos_fallback_emits_stop_when_final_never_arrives() -> None:
    service = _make_service()

    await service._handle_message({"event": "vad.speech_start", "utterance_idx": 0})
    await service._handle_message({"event": "vad.speech_end", "utterance_idx": 0})
    await service._emit_pending_eos_after_timeout(0)

    assert _frame_types(service) == [UserStartedSpeakingFrame, UserStoppedSpeakingFrame]
    service.stop_processing_metrics.assert_awaited_once()


@pytest.mark.asyncio
async def test_manual_endpointing_emits_final_immediately() -> None:
    service = _make_service(endpointing="manual")
    service._audio_position = 1.25

    await service._handle_message(
        {
            "event": "transcript.final",
            "utterance_idx": 0,
            "text": "नमस्ते आप कैसे हैं",
            "language": "hi-IN",
            "language_confidence": 0.99,
        }
    )

    final_frame = service._pushed_frames[0]
    assert final_frame.result["end_time"] == 1.25


@pytest.mark.asyncio
async def test_request_id_is_raw_and_never_falls_back_to_session_id() -> None:
    service = _make_service()

    await service._handle_message({"event": "session.begin", "session_id": "sess_only"})
    assert service._session_id == "sess_only"
    assert service._request_id == ""

    await service._handle_message(
        {
            "event": "session.begin",
            "session_id": "sess_xyz",
            "metadata": {"request_id": "srv_custom-id_v9"},
        }
    )
    assert service._request_id == "srv_custom-id_v9"


@pytest.mark.asyncio
async def test_request_id_can_be_captured_from_nested_data() -> None:
    service = _make_service()

    await service._handle_message(
        {
            "event": "session.begin",
            "session_id": "sess_xyz",
            "data": {"request_id": "nested_req_123"},
        }
    )

    assert service._request_id == "nested_req_123"


@pytest.mark.asyncio
async def test_empty_transcripts_are_skipped() -> None:
    service = _make_service()

    await service._handle_message({"event": "transcript.partial", "text": ""})
    await service._handle_message({"event": "transcript.final", "text": ""})

    assert service._pushed_frames == []


@pytest.mark.asyncio
async def test_usage_events_emit_periodic_and_session_end_deltas() -> None:
    service = _make_service()
    service._request_id = "req_123"

    await service._on_audio_duration_report(1.5)
    await service._handle_message(
        {
            "event": "session.end",
            "session_id": "sess_123",
            "audio_duration_s": 2.25,
        }
    )

    assert [event[1] for event in service._usage_events] == [1.5, 0.75]
    assert service._session_ended is True


@pytest.mark.asyncio
async def test_stop_drains_session_end_before_disconnect() -> None:
    service = _make_service()
    service._websocket = _FakeWebsocket()
    order = []

    async def delayed_session_end() -> None:
        await asyncio.sleep(0.01)
        order.append("session_end")
        await service._handle_session_end(
            {
                "event": "session.end",
                "session_id": "sess_123",
                "audio_duration_s": 2.0,
            }
        )

    async def disconnect() -> None:
        order.append("disconnect")

    service._disconnect = AsyncMock(side_effect=disconnect)
    drain_task = asyncio.create_task(delayed_session_end())

    await service.stop(EndFrame())
    await drain_task

    assert order == ["session_end", "disconnect"]
    assert service._usage_events[-1][1] == 2.0
    assert service._websocket.sent[-1] == json.dumps({"event": "end"})


@pytest.mark.asyncio
async def test_error_handling_distinguishes_fatal_and_non_fatal() -> None:
    service = _make_service()

    await service._handle_message(
        {
            "event": "error",
            "code": "chunk_too_large",
            "message": "chunk too large",
            "is_fatal": False,
        }
    )

    with pytest.raises(RuntimeError, match="backend saturated"):
        await service._handle_message(
            {
                "event": "error",
                "code": "model_unavailable",
                "message": "backend saturated",
                "is_fatal": True,
                "status_code": 503,
            }
        )


def test_close_code_mapping_marks_backend_unavailable_retryable() -> None:
    service = _make_service()

    error = service._exception_from_close_code(1013, "busy")

    assert "backend temporarily unavailable" in str(error)
    assert error.retryable is True


def test_close_code_mapping_describes_session_rejection() -> None:
    service = _make_service()

    error = service._exception_from_close_code(4000, "beta access denied")

    assert "rejected the session" in str(error)
    assert error.retryable is False
    assert error.body["close_code"] == 4000


@pytest.mark.asyncio
async def test_receive_messages_maps_websocket_close_code() -> None:
    service = _make_service()

    class _ClosingWebsocket:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise ConnectionClosedError(Close(4000, "session rejected"), None)

    service._websocket = _ClosingWebsocket()

    with pytest.raises(SarvamRealtimeSTTError) as excinfo:
        await service._receive_messages()

    assert "rejected the session" in str(excinfo.value)
    assert excinfo.value.retryable is False
    assert excinfo.value.body["close_code"] == 4000


@pytest.mark.asyncio
async def test_non_retryable_sarvam_error_does_not_reconnect_even_when_enabled() -> None:
    service = _make_service(reconnect_on_error=True)
    errors = []

    async def report_error(error: ErrorFrame) -> None:
        errors.append(error)

    service._receive_messages = AsyncMock(
        side_effect=SarvamRealtimeSTTError("fatal auth", retryable=False)
    )
    service._try_reconnect = AsyncMock()

    await service._receive_task_handler(report_error)

    service._try_reconnect.assert_not_awaited()
    assert len(errors) == 1
    assert "fatal auth" in errors[0].error


@pytest.mark.asyncio
async def test_retryable_sarvam_error_reconnects_only_when_enabled() -> None:
    service = _make_service(reconnect_on_error=True)
    errors = []

    async def report_error(error: ErrorFrame) -> None:
        errors.append(error)

    service._receive_messages = AsyncMock(
        side_effect=[
            SarvamRealtimeSTTError("backend busy", retryable=True),
            asyncio.CancelledError(),
        ]
    )
    service._try_reconnect = AsyncMock(return_value=False)

    await service._receive_task_handler(report_error)

    service._try_reconnect.assert_awaited_once()
    assert len(errors) == 1
    assert "backend busy" in errors[0].error


def test_reset_connection_state_clears_ids_usage_and_utterance_state() -> None:
    service = _make_service()
    service._request_id = "req_old"
    service._session_id = "sess_old"
    service._session_ended = True
    service._manual_speech_started = True
    service._pending_eos = True
    service._pending_eos_time = 1.0
    service._pending_final_data = {"text": "hello"}
    service._utterance_start_audio_pos = 3.0
    service._utterance_speech_end_audio_pos = 4.0
    service._utterance_speech_end_wall = 5.0
    service._final_received_for_utterance = True
    service._eos_emitted_for_utterance = True
    service._audio_position = 20.0
    service._total_reported_audio_duration = 50.0

    service._reset_connection_state()

    assert service._request_id == ""
    assert service._session_id == ""
    assert service._session_ended is False
    assert service._manual_speech_started is False
    assert service._pending_eos is False
    assert service._pending_eos_time is None
    assert service._pending_final_data is None
    assert service._utterance_start_audio_pos == 0.0
    assert service._utterance_speech_end_audio_pos is None
    assert service._utterance_speech_end_wall is None
    assert service._final_received_for_utterance is False
    assert service._eos_emitted_for_utterance is False
    assert service._audio_position == 0.0
    assert service._total_reported_audio_duration == 0.0


@pytest.mark.asyncio
async def test_run_stt_buffers_audio_to_stream_type_chunk_size() -> None:
    service = _make_service()
    service._websocket = _FakeWebsocket()
    service._chunk_size_bytes = 16000

    chunks = [b"\x01" * 8000, b"\x02" * 8000, b"\x03" * 16000]
    for chunk in chunks:
        async for frame in service.run_stt(chunk):
            assert frame is None

    assert service._websocket.sent == [b"\x01" * 8000 + b"\x02" * 8000, b"\x03" * 16000]
    assert service._audio_position == 1.0


@pytest.mark.asyncio
async def test_run_stt_send_failure_yields_error_without_retry_by_default() -> None:
    service = _make_service()
    service._websocket = _FakeWebsocket(send_side_effect=RuntimeError("send failed"))
    service._chunk_size_bytes = 4

    frames = []
    async for frame in service.run_stt(b"1234"):
        frames.append(frame)

    assert len(frames) == 1
    assert isinstance(frames[0], ErrorFrame)
    assert "send failed" in frames[0].error
    assert service._audio_position == 0.0
    assert service._usage_events == []


@pytest.mark.asyncio
async def test_run_stt_send_failure_uses_pipecat_retry_when_enabled() -> None:
    service = _make_service(reconnect_on_error=True)
    service._websocket = _FakeWebsocket(send_side_effect=RuntimeError("send failed"))
    service._chunk_size_bytes = 4
    service.send_with_retry = AsyncMock()

    frames = []
    async for frame in service.run_stt(b"1234"):
        frames.append(frame)

    service.send_with_retry.assert_awaited_once_with(b"1234", service._report_error)
    assert frames == [None]
    assert service._audio_position == 0.000125


@pytest.mark.asyncio
async def test_manual_endpointing_sends_speech_boundaries_around_audio() -> None:
    service = _make_service(endpointing="manual")
    service._websocket = _FakeWebsocket()
    service._chunk_size_bytes = 4

    async for frame in service.run_stt(b"1234"):
        assert frame is None
    await service._send_manual_speech_end()

    assert service._websocket.sent == [
        json.dumps({"event": "speech_start"}),
        b"1234",
        json.dumps({"event": "speech_end"}),
    ]


@pytest.mark.asyncio
async def test_manual_endpointing_maps_pipecat_vad_frames_to_boundaries() -> None:
    service = _make_service(endpointing="manual")
    service._websocket = _FakeWebsocket()

    with patch.object(SarvamRealtimeSTTService.__mro__[1], "process_frame", new=AsyncMock()):
        await service.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
        await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    assert service._websocket.sent == [
        json.dumps({"event": "speech_start"}),
        json.dumps({"event": "speech_end"}),
    ]
