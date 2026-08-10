#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close
from websockets.protocol import State

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.frames.frames import (
    ErrorFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi.processor import RTVIProcessor
from pipecat.services.sarvam._sdk import sdk_headers
from pipecat.services.sarvam.realtime_stt import SarvamRealtimeSTTService
from pipecat.transcriptions.language import Language
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies


class _FakeWebsocket:
    def __init__(self, messages=None, *, state=State.OPEN):
        self._messages = messages or []
        self.state = state
        self.sent = []
        self.closed = False

    async def send(self, message):
        self.sent.append(message)

    async def close(self):
        self.closed = True
        self.state = State.CLOSED

    def __aiter__(self):
        return self._iter_messages()

    async def _iter_messages(self):
        for message in self._messages:
            yield message


class _CapturingLogger:
    def __init__(self):
        self.debug_messages = []
        self.info_messages = []
        self.warning_messages = []

    def debug(self, message):
        self.debug_messages.append(message)

    def info(self, message):
        self.info_messages.append(message)

    def warning(self, message):
        self.warning_messages.append(message)


def _query(service: SarvamRealtimeSTTService, *, sample_rate: int = 16000) -> dict[str, list[str]]:
    # The URL is only built after StartFrame resolves the rate, so mirror what
    # STTService.start() does here.
    service._sample_rate = service._init_sample_rate or sample_rate
    return parse_qs(urlparse(service._build_ws_url()).query)


def _seconds_to_bytes(seconds: float, *, sample_rate: int = 16000) -> int:
    """Byte count for `seconds` of 16-bit mono audio."""
    return int(seconds * sample_rate * 2)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stream_type", "balanced"),
        ("sample_rate", 8000),
        ("mode", "translate"),
    ],
)
def test_settings_fields_cannot_be_constructor_kwargs(field, value):
    with pytest.raises(TypeError, match="settings="):
        SarvamRealtimeSTTService(api_key="test-key", **{field: value})


def test_settings_values_are_applied_via_settings():
    service = SarvamRealtimeSTTService(
        api_key="test-key",
        settings=SarvamRealtimeSTTService.Settings(
            stream_type="balanced",
            sample_rate=8000,
            mode="translate",
        ),
    )

    query = _query(service)

    assert query["stream_type"] == ["balanced"]
    assert query["sample_rate"] == ["8000"]
    assert query["mode"] == ["translate"]


def test_default_url_uses_realtime_contract_params():
    service = SarvamRealtimeSTTService(api_key="test-key")

    query = _query(service)

    assert (
        urlparse(service._build_ws_url())
        .geturl()
        .startswith("wss://api.sarvam.ai/speech-to-text-realtime/ws?")
    )
    assert query["language_code"] == ["en-IN"]
    assert query["stream_type"] == ["balanced"]
    assert query["endpointing"] == ["vad"]
    assert query["encoding"] == ["linear16"]
    assert query["sample_rate"] == ["16000"]
    assert query["model"] == ["saaras:v3-realtime"]
    assert query["mode"] == ["transcribe"]
    assert query["return_timestamps"] == ["false"]


def test_validation_accepts_auto_language_and_modes():
    service = SarvamRealtimeSTTService(
        api_key="test-key",
        settings=SarvamRealtimeSTTService.Settings(
            language_code="auto",
            mode="translate",
            threshold=0.4,
            silence_duration_ms=700,
        ),
    )

    query = _query(service)

    assert query["language_code"] == ["auto"]
    assert query["mode"] == ["translate"]
    assert query["threshold"] == ["0.4"]
    assert query["silence_duration_ms"] == ["700"]


def test_string_language_setting_does_not_use_enum_converter(monkeypatch):
    converter_calls = []
    from pipecat.services.sarvam.realtime_stt import language_to_sarvam_realtime_language

    monkeypatch.setattr(
        "pipecat.services.sarvam.realtime_stt.language_to_sarvam_realtime_language",
        lambda language: (
            converter_calls.append(language),
            language_to_sarvam_realtime_language(language),
        )[1],
    )

    service = SarvamRealtimeSTTService(
        api_key="test-key",
        settings=SarvamRealtimeSTTService.Settings(language="hi-IN"),
    )

    assert _query(service)["language_code"] == ["hi-IN"]
    # Resolved through the enum path rather than forwarded as a raw string.
    assert Language.HI_IN in converter_calls


@pytest.mark.parametrize(
    "settings",
    [
        SarvamRealtimeSTTService.Settings(model="saarika:v2.5"),
        SarvamRealtimeSTTService.Settings(language_code="fr-FR"),
        SarvamRealtimeSTTService.Settings(stream_type="slow"),
        SarvamRealtimeSTTService.Settings(endpointing="server"),
        SarvamRealtimeSTTService.Settings(sample_rate=44100),
        SarvamRealtimeSTTService.Settings(threshold=1.1),
        SarvamRealtimeSTTService.Settings(silence_duration_ms=-1),
    ],
)
def test_invalid_realtime_settings_raise(settings):
    with pytest.raises(ValueError):
        SarvamRealtimeSTTService(api_key="test-key", settings=settings)


@pytest.mark.asyncio
async def test_connect_uses_subscription_key_and_user_agent(monkeypatch):
    captured = {}

    async def fake_websocket_connect(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _FakeWebsocket()

    monkeypatch.setattr(
        "pipecat.services.sarvam.realtime_stt.websocket_connect", fake_websocket_connect
    )

    service = SarvamRealtimeSTTService(api_key="test-key")
    await service._connect_websocket()

    assert captured["url"] == service._build_ws_url()
    assert captured["kwargs"]["additional_headers"] == {"API-SUBSCRIPTION-KEY": "test-key"}
    assert captured["kwargs"]["user_agent_header"] == sdk_headers()["User-Agent"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_type", ["fast", "balanced", "simulated"])
async def test_client_sends_50ms_chunks_regardless_of_stream_type(stream_type):
    service = SarvamRealtimeSTTService(
        api_key="test-key",
        settings=SarvamRealtimeSTTService.Settings(stream_type=stream_type),
    )
    service._websocket = _FakeWebsocket()
    service._sample_rate = 16000

    # 16 kHz linear16 => 1600 bytes per 50 ms.
    await _consume(service.run_stt(b"\x01" * 800))
    assert service._websocket.sent == []

    await _consume(service.run_stt(b"\x02" * 800))

    expected_audio = b"\x01" * 800 + b"\x02" * 800
    assert service._websocket.sent == [
        json.dumps({"event": "audio_input", "audio": base64.b64encode(expected_audio).decode()})
    ]


@pytest.mark.asyncio
async def test_manual_endpointing_sends_speech_boundaries():
    service = SarvamRealtimeSTTService(
        api_key="test-key",
        settings=SarvamRealtimeSTTService.Settings(endpointing="manual"),
    )
    service._websocket = _FakeWebsocket()

    await service.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    assert service._websocket.sent == [
        json.dumps({"event": "speech_start"}),
        json.dumps({"event": "speech_end"}),
    ]


@pytest.mark.asyncio
async def test_manual_endpointing_flushes_buffered_audio_before_speech_end():
    service = SarvamRealtimeSTTService(
        api_key="test-key",
        settings=SarvamRealtimeSTTService.Settings(endpointing="manual"),
    )
    service._websocket = _FakeWebsocket()
    service._sample_rate = 16000

    # Less than one 50 ms chunk, so it stays buffered until the turn ends.
    await _consume(service.run_stt(b"\x01" * 400))
    assert service._websocket.sent == []

    await service.process_frame(VADUserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    assert service._websocket.sent == [
        json.dumps({"event": "audio_input", "audio": base64.b64encode(b"\x01" * 400).decode()}),
        json.dumps({"event": "speech_end"}),
    ]
    assert service._audio_buffer == bytearray()


@pytest.mark.asyncio
async def test_partial_transcript_emits_interim_frame(monkeypatch):
    service = SarvamRealtimeSTTService(api_key="test-key")
    pushed = []
    monkeypatch.setattr(service, "push_frame", _capture(pushed))

    await service._handle_message(
        {"event": "transcript.partial", "utterance_idx": 7, "text": "हेलो", "language": "hi"}
    )
    await service._handle_message({"event": "transcript.partial", "text": ""})

    assert len(pushed) == 1
    assert isinstance(pushed[0], InterimTranscriptionFrame)
    assert pushed[0].text == "हेलो"
    assert pushed[0].language == Language.HI_IN
    assert pushed[0].result["utterance_idx"] == 7
    assert pushed[0].result["language"] == "hi"


@pytest.mark.asyncio
async def test_speech_end_emits_eos_before_delayed_final(monkeypatch):
    service = SarvamRealtimeSTTService(api_key="test-key")
    pushed = []
    broadcasted = []
    monkeypatch.setattr(service, "push_frame", _capture(pushed))
    monkeypatch.setattr(service, "broadcast_frame", _capture_class(broadcasted))
    monkeypatch.setattr(service, "broadcast_interruption", _noop)
    monkeypatch.setattr(service, "start_processing_metrics", _noop)
    monkeypatch.setattr(service, "start_ttfb_metrics", _noop)
    monkeypatch.setattr(service, "stop_ttfb_metrics", _noop)
    monkeypatch.setattr(service, "stop_processing_metrics", _noop)

    service._sample_rate = 16000
    service._audio_position_bytes = _seconds_to_bytes(1.25)
    await service._handle_message({"event": "vad.speech_start", "utterance_idx": 3})
    await service._handle_message({"event": "vad.speech_end", "utterance_idx": 3})
    await service._handle_message({"event": "transcript.final", "utterance_idx": 3, "text": "हेलो।"})

    assert broadcasted == [UserStartedSpeakingFrame, UserStoppedSpeakingFrame]
    assert len(pushed) == 1
    assert isinstance(pushed[0], TranscriptionFrame)
    assert pushed[0].text == "हेलो।"
    assert pushed[0].result["speech_end_audio_position_s"] == 1.25


@pytest.mark.asyncio
async def test_duplicate_speech_end_does_not_emit_duplicate_eos(monkeypatch):
    service = SarvamRealtimeSTTService(api_key="test-key")
    broadcasted = []
    monkeypatch.setattr(service, "broadcast_frame", _capture_class(broadcasted))
    monkeypatch.setattr(service, "broadcast_interruption", _noop)
    monkeypatch.setattr(service, "start_processing_metrics", _noop)
    monkeypatch.setattr(service, "start_ttfb_metrics", _noop)

    await service._handle_message({"event": "vad.speech_start", "utterance_idx": 1})
    await service._handle_message({"event": "vad.speech_end", "utterance_idx": 1})
    await service._handle_message({"event": "vad.speech_end", "utterance_idx": 1})

    assert broadcasted == [UserStartedSpeakingFrame, UserStoppedSpeakingFrame]


@pytest.mark.asyncio
async def test_post_eos_partial_is_interim_without_changing_eos_timing(monkeypatch):
    service = SarvamRealtimeSTTService(api_key="test-key")
    pushed = []
    broadcasted = []
    monkeypatch.setattr(service, "push_frame", _capture(pushed))
    monkeypatch.setattr(service, "broadcast_frame", _capture_class(broadcasted))
    monkeypatch.setattr(service, "broadcast_interruption", _noop)
    monkeypatch.setattr(service, "start_processing_metrics", _noop)
    monkeypatch.setattr(service, "start_ttfb_metrics", _noop)
    monkeypatch.setattr(service, "stop_ttfb_metrics", _noop)
    monkeypatch.setattr(service, "stop_processing_metrics", _noop)

    service._sample_rate = 16000
    service._audio_position_bytes = _seconds_to_bytes(2.0)
    await service._handle_message({"event": "vad.speech_start", "utterance_idx": 2})
    await service._handle_message({"event": "vad.speech_end", "utterance_idx": 2})
    await service._handle_message({"event": "transcript.partial", "utterance_idx": 2, "text": "हेल"})
    await service._handle_message({"event": "transcript.final", "utterance_idx": 2, "text": "हेलो।"})

    assert broadcasted == [UserStartedSpeakingFrame, UserStoppedSpeakingFrame]
    assert [type(frame) for frame in pushed] == [InterimTranscriptionFrame, TranscriptionFrame]
    assert pushed[-1].result["speech_end_audio_position_s"] == 2.0


@pytest.mark.asyncio
async def test_config_updated_and_pong_are_noops(monkeypatch):
    service = SarvamRealtimeSTTService(api_key="test-key")
    pushed = []
    monkeypatch.setattr(service, "push_frame", _capture(pushed))

    await service._handle_message({"event": "config.updated", "applied": ["language_code"]})
    await service._handle_message({"event": "pong"})

    assert pushed == []


@pytest.mark.asyncio
async def test_nonfatal_error_emits_raw_payload(monkeypatch):
    service = SarvamRealtimeSTTService(api_key="test-key")
    pushed_errors = []

    async def fake_push_error(error_msg, exception=None, fatal=False):
        pushed_errors.append((error_msg, exception, fatal))

    monkeypatch.setattr(service, "push_error", fake_push_error)

    await service._handle_message(
        {
            "event": "error",
            "code": "transient_warning",
            "message": "retrying",
            "is_fatal": False,
        }
    )

    assert pushed_errors[0][0] == (
        'Sarvam realtime STT error: {"event": "error", "code": "transient_warning", '
        '"message": "retrying", "is_fatal": false}'
    )
    assert pushed_errors[0][1] is None
    assert pushed_errors[0][2] is False


@pytest.mark.asyncio
async def test_sarvam_error_reaches_rtvi_client(monkeypatch):
    service = SarvamRealtimeSTTService(api_key="test-key")
    rtvi = RTVIProcessor()
    client_messages = []

    async def capture_transport_message(message, **_kwargs):
        client_messages.append(message)

    async def deliver_error(error_msg, exception=None, fatal=False):
        await rtvi._send_error_frame(
            ErrorFrame(error=error_msg, exception=exception, fatal=fatal, processor=service)
        )

    monkeypatch.setattr(rtvi, "push_transport_message", capture_transport_message)
    monkeypatch.setattr(service, "push_error", deliver_error)
    payload = {
        "event": "error",
        "code": "transient_warning",
        "message": "retrying",
        "is_fatal": False,
        "diagnostic": {"attempt": 2},
    }

    await service._handle_message(payload)

    assert client_messages == [
        RTVI.Error(
            data=RTVI.ErrorData(
                error=(
                    'Sarvam realtime STT error: {"event": "error", '
                    '"code": "transient_warning", "message": "retrying", '
                    '"is_fatal": false, "diagnostic": {"attempt": 2}}'
                ),
                fatal=False,
            )
        )
    ]


@pytest.mark.asyncio
async def test_session_begin_logs_request_id_at_info(monkeypatch):
    service = SarvamRealtimeSTTService(api_key="test-key")
    captured_logger = _CapturingLogger()
    monkeypatch.setattr("pipecat.services.sarvam.realtime_stt.logger", captured_logger)

    await service._handle_message({"event": "session.begin", "request_id": "request-123"})

    assert captured_logger.info_messages == [
        f"{service} Sarvam realtime session.begin request_id=request-123"
    ]
    assert captured_logger.debug_messages == []


@pytest.mark.asyncio
async def test_config_update_sends_without_reconnect_and_rejects_simulated_change():
    service = SarvamRealtimeSTTService(api_key="test-key")
    service._websocket = _FakeWebsocket()

    await service.update_config(language_code="auto", mode="translate", prompt="prefer glossary")

    assert service._websocket.sent == [
        json.dumps(
            {
                "event": "config.update",
                "language_code": "auto",
                "mode": "translate",
                "prompt": "prefer glossary",
            }
        )
    ]

    with pytest.raises(ValueError):
        await service.update_config(stream_type="simulated")


@pytest.mark.asyncio
async def test_endpointing_change_is_not_effective_until_acked():
    service = SarvamRealtimeSTTService(api_key="test-key")
    service._websocket = _FakeWebsocket()

    await service.update_config(endpointing="manual")
    service._websocket.sent.clear()

    # Server is still in vad mode until it acknowledges, so no manual boundaries.
    await service.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    assert service._websocket.sent == []

    await service._handle_message({"event": "config.updated", "applied": ["endpointing=manual"]})
    await service.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    assert service._websocket.sent == [json.dumps({"event": "speech_start"})]


@pytest.mark.asyncio
async def test_unrelated_config_updated_does_not_promote_endpointing():
    service = SarvamRealtimeSTTService(api_key="test-key")
    service._websocket = _FakeWebsocket()

    await service.update_config(endpointing="manual")
    await service._handle_message({"event": "config.updated", "applied": ["prompt=hello"]})
    service._websocket.sent.clear()

    await service.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    assert service._websocket.sent == []


@pytest.mark.asyncio
async def test_unusable_applied_payload_falls_back_to_promoting_endpointing():
    service = SarvamRealtimeSTTService(api_key="test-key")
    service._websocket = _FakeWebsocket()

    await service.update_config(endpointing="manual")
    await service._handle_message({"event": "config.updated"})
    service._websocket.sent.clear()

    await service.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    assert service._websocket.sent == [json.dumps({"event": "speech_start"})]


@pytest.mark.asyncio
async def test_endpointing_is_not_pending_when_socket_is_closed():
    service = SarvamRealtimeSTTService(api_key="test-key")

    await service.update_config(endpointing="manual")
    await service._handle_message({"event": "config.updated", "applied": ["endpointing=manual"]})

    assert service._effective_endpointing == "vad"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_rate", 8000),
        ("return_timestamps", True),
        ("prefix_padding_ms", 200),
        ("lid_gate_seconds", 1.5),
        ("lid_confidence_threshold", 0.5),
    ],
)
async def test_connection_only_fields_rejected_by_update_config(field, value):
    service = SarvamRealtimeSTTService(api_key="test-key")
    service._websocket = _FakeWebsocket()

    with pytest.raises(ValueError, match="connection"):
        await service.update_config(**{field: value})

    assert service._websocket.sent == []


@pytest.mark.asyncio
async def test_connection_only_setting_change_is_not_sent_as_config_update():
    service = SarvamRealtimeSTTService(api_key="test-key")
    service._websocket = _FakeWebsocket()

    await service._update_settings(SarvamRealtimeSTTService.Settings(lid_gate_seconds=2.0))

    assert service._websocket.sent == []


def test_sample_rate_defaults_to_the_pipeline_rate():
    service = SarvamRealtimeSTTService(api_key="test-key")

    assert service._init_sample_rate is None
    assert _query(service, sample_rate=8000)["sample_rate"] == ["8000"]


def test_explicit_sample_rate_setting_pins_the_rate():
    service = SarvamRealtimeSTTService(
        api_key="test-key",
        settings=SarvamRealtimeSTTService.Settings(sample_rate=8000),
    )

    assert service._init_sample_rate == 8000


def test_unsupported_resolved_sample_rate_raises():
    service = SarvamRealtimeSTTService(api_key="test-key")
    service._sample_rate = 44100

    with pytest.raises(ValueError, match="sample_rate"):
        service._validate_resolved_sample_rate()


def test_vad_params_are_omitted_for_manual_endpointing():
    service = SarvamRealtimeSTTService(
        api_key="test-key",
        settings=SarvamRealtimeSTTService.Settings(
            endpointing="manual",
            threshold=0.4,
            silence_duration_ms=700,
            min_speech_duration_ms=120,
            prefix_padding_ms=200,
        ),
    )

    query = _query(service)

    for param in (
        "threshold",
        "silence_duration_ms",
        "min_speech_duration_ms",
        "prefix_padding_ms",
    ):
        assert param not in query


@pytest.mark.asyncio
async def test_blank_final_still_stops_processing_metrics(monkeypatch):
    service = SarvamRealtimeSTTService(api_key="test-key")
    stopped = []
    monkeypatch.setattr(service, "push_frame", _noop)
    monkeypatch.setattr(service, "broadcast_frame", _noop)
    monkeypatch.setattr(service, "broadcast_interruption", _noop)
    monkeypatch.setattr(service, "start_processing_metrics", _noop)
    monkeypatch.setattr(service, "start_ttfb_metrics", _noop)

    async def fake_stop_processing_metrics():
        stopped.append(True)

    monkeypatch.setattr(service, "stop_processing_metrics", fake_stop_processing_metrics)

    await service._handle_message({"event": "vad.speech_start"})
    await service._handle_message({"event": "vad.speech_end"})
    await service._handle_message({"event": "transcript.final", "text": "   "})

    assert stopped == [True]


@pytest.mark.asyncio
async def test_session_end_mid_utterance_completes_the_turn(monkeypatch):
    service = SarvamRealtimeSTTService(api_key="test-key")
    broadcasted = []
    monkeypatch.setattr(service, "push_frame", _noop)
    monkeypatch.setattr(service, "broadcast_frame", _capture_class(broadcasted))
    monkeypatch.setattr(service, "broadcast_interruption", _noop)
    monkeypatch.setattr(service, "start_processing_metrics", _noop)
    monkeypatch.setattr(service, "start_ttfb_metrics", _noop)
    monkeypatch.setattr(service, "stop_processing_metrics", _noop)

    await service._handle_message({"event": "vad.speech_start"})
    await service._handle_message({"event": "session.end", "audio_duration_s": 1.0})

    assert broadcasted == [UserStartedSpeakingFrame, UserStoppedSpeakingFrame]


@pytest.mark.asyncio
async def test_disconnect_tolerates_socket_closing_during_flush(monkeypatch):
    service = SarvamRealtimeSTTService(api_key="test-key")
    service._sample_rate = 16000

    class _ClosingWebsocket(_FakeWebsocket):
        async def send(self, message):
            raise ConnectionResetError("socket went away")

    service._websocket = _ClosingWebsocket()
    service._audio_buffer.extend(b"\x01" * 400)
    monkeypatch.setattr(service, "push_error", _noop)

    await service._disconnect()

    assert service._websocket is None


@pytest.mark.asyncio
async def test_confidence_defaults_to_one_when_not_numeric(monkeypatch):
    service = SarvamRealtimeSTTService(api_key="test-key")
    pushed = []
    monkeypatch.setattr(service, "push_frame", _capture(pushed))
    monkeypatch.setattr(service, "stop_processing_metrics", _noop)

    await service._handle_message({"event": "transcript.partial", "text": "hi"})
    await service._handle_message({"event": "transcript.final", "text": "hi", "confidence": 0.42})

    assert pushed[0].result["confidence"] == 1.0
    assert pushed[1].result["confidence"] == 0.42


def test_explicit_language_code_is_not_overridden_by_language():
    service = SarvamRealtimeSTTService(
        api_key="test-key",
        settings=SarvamRealtimeSTTService.Settings(
            language=Language.EN_IN,
            language_code="hi-IN",
        ),
    )

    assert _query(service)["language_code"] == ["hi-IN"]


def test_service_metadata_recommends_external_turn_strategies_in_vad_mode():
    service = SarvamRealtimeSTTService(api_key="test-key")
    frame = service.service_metadata_frame()
    assert isinstance(frame.user_turn_strategies, ExternalUserTurnStrategies)


def test_service_metadata_leaves_turn_strategies_unset_in_manual_mode():
    service = SarvamRealtimeSTTService(
        api_key="test-key",
        settings=SarvamRealtimeSTTService.Settings(endpointing="manual"),
    )
    frame = service.service_metadata_frame()
    assert frame.user_turn_strategies is None


def test_reconnect_on_error_cannot_be_overridden():
    with pytest.raises(TypeError, match="reconnect_on_error"):
        SarvamRealtimeSTTService(api_key="test-key", reconnect_on_error=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "receive_error",
    [
        ConnectionClosedError(Close(1006, "Abnormal closure"), None),
        RuntimeError("unexpected receive failure"),
    ],
)
async def test_receive_errors_are_reported_without_reconnect(monkeypatch, receive_error):
    service = SarvamRealtimeSTTService(api_key="test-key")
    report_error = AsyncMock()
    try_reconnect = AsyncMock(return_value=False)
    monkeypatch.setattr(service, "_receive_messages", AsyncMock(side_effect=receive_error))
    monkeypatch.setattr(service, "_try_reconnect", try_reconnect)

    await service._receive_task_handler(report_error)

    try_reconnect.assert_not_awaited()
    report_error.assert_awaited_once()


@pytest.mark.asyncio
async def test_sarvam_error_is_reported_without_reconnect(monkeypatch):
    service = SarvamRealtimeSTTService(api_key="test-key")
    report_error = AsyncMock()
    try_reconnect = AsyncMock(return_value=False)
    pushed_errors = []

    async def fake_push_error(error_msg, exception=None, fatal=False):
        pushed_errors.append((error_msg, exception, fatal))

    monkeypatch.setattr(
        service,
        "_websocket",
        _FakeWebsocket(
            [
                json.dumps(
                    {
                        "event": "error",
                        "code": "invalid_subscription_key",
                        "message": "Invalid subscription key",
                        "is_fatal": True,
                    }
                )
            ]
        ),
    )
    monkeypatch.setattr(service, "_try_reconnect", try_reconnect)
    monkeypatch.setattr(service, "push_error", fake_push_error)

    await service._receive_task_handler(report_error)

    assert pushed_errors[0][2] is False
    try_reconnect.assert_not_awaited()
    report_error.assert_awaited_once()


@pytest.mark.asyncio
async def test_error_preserves_raw_payload(monkeypatch):
    service = SarvamRealtimeSTTService(api_key="test-key")
    pushed_errors = []

    async def fake_push_error(error_msg, exception=None, fatal=False):
        pushed_errors.append((error_msg, exception, fatal))

    monkeypatch.setattr(service, "push_error", fake_push_error)

    payload = {
        "event": "error",
        "code": "invalid_subscription_key",
        "message": "Invalid subscription key",
        "is_fatal": True,
        "status_code": 1003,
    }
    await service._handle_message(payload)

    assert pushed_errors
    assert "invalid_subscription_key" in pushed_errors[0][0]
    assert pushed_errors[0][1] is None
    assert pushed_errors[0][2] is False


async def _consume(generator):
    async for _ in generator:
        pass


async def _noop(*_args, **_kwargs):
    return None


def _capture(frames):
    async def inner(frame, *_args, **_kwargs):
        frames.append(frame)

    return inner


def _capture_class(frames):
    async def inner(frame_cls, *_args, **_kwargs):
        frames.append(frame_cls)

    return inner
