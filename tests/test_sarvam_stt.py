#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
from dataclasses import fields
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close
from websockets.protocol import State

pytest.importorskip("sarvamai")

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.frames.frames import (
    ErrorFrame,
    InterimTranscriptionFrame,
    MetricsFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import STTUsageMetricsData
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi.processor import RTVIProcessor
from pipecat.services.sarvam._sdk import sdk_headers
from pipecat.services.sarvam.stt import (
    MODEL_CONFIGS,
    SarvamRealtimeSTTService,
    SarvamRealtimeSTTSettings,
    SarvamSTTService,
)
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.errors import ErrorCategory
from tests.frame_processor_helpers import frame_processor_setup


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


def test_supported_models():
    """The sunset saarika:v2.5 and saaras:v2.5 models are no longer offered."""
    assert set(MODEL_CONFIGS) == {"saaras:v3", "saaras:v4"}


def test_default_model():
    """Constructing without a model picks up the latest one."""
    service = SarvamSTTService(api_key="test-key")
    assert service._settings.model == "saaras:v4"


def test_sunset_model_raises():
    """A model that was removed reports what it can be replaced with."""
    with pytest.raises(ValueError, match="saaras:v3, saaras:v4"):
        SarvamSTTService(
            api_key="test-key",
            settings=SarvamSTTService.Settings(model="saaras:v2.5"),
        )


def test_sarvam_vad_signals_recommend_external_strategies():
    """With ``vad_signals`` on, Sarvam's boundaries are what drive turns."""
    service = SarvamSTTService(
        api_key="test-key",
        settings=SarvamSTTService.Settings(model="saaras:v3", vad_signals=True),
    )
    strategies = service.service_metadata_frame().user_turn_strategies
    assert isinstance(strategies, ExternalUserTurnStrategies)


def test_sarvam_without_vad_signals_recommends_no_strategies():
    """Without them Sarvam proposes no turns, so the defaults stand."""
    service = SarvamSTTService(api_key="test-key")
    assert service.service_metadata_frame().user_turn_strategies is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stream_type", "balanced"),
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
            mode="translate",
        ),
    )

    query = _query(service)

    assert query["stream_type"] == ["balanced"]
    assert query["mode"] == ["translate"]


def test_connection_only_values_are_applied_via_constructor_arguments():
    service = SarvamRealtimeSTTService(
        api_key="test-key",
        sample_rate=8000,
        return_timestamps=True,
        prefix_padding_ms=200,
    )

    query = _query(service)

    assert query["sample_rate"] == ["8000"]
    assert query["return_timestamps"] == ["true"]
    assert query["prefix_padding_ms"] == ["200"]


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
    from pipecat.services.sarvam.stt import language_to_sarvam_realtime_language

    monkeypatch.setattr(
        "pipecat.services.sarvam.stt.language_to_sarvam_realtime_language",
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


def test_invalid_realtime_settings_raise():
    """Only the settings this integration itself depends on are checked here."""
    with pytest.raises(ValueError):
        SarvamRealtimeSTTService(
            api_key="test-key",
            settings=SarvamRealtimeSTTService.Settings(model="saaras:v3"),
        )


def test_unusable_sample_rate_raises():
    """The audio path has to produce this rate, so it can't wait for the wire."""
    with pytest.raises(ValueError, match="44100"):
        SarvamRealtimeSTTService(api_key="test-key", sample_rate=44100)


@pytest.mark.parametrize(
    "settings",
    [
        SarvamRealtimeSTTService.Settings(language_code="fr-FR"),
        SarvamRealtimeSTTService.Settings(stream_type="slow"),
        SarvamRealtimeSTTService.Settings(mode="sing"),
        SarvamRealtimeSTTService.Settings(threshold=1.1),
    ],
)
def test_sarvam_vocabulary_is_left_to_the_server(settings):
    """Sarvam rejects these on the wire, and the rejection reaches the app.

    Repeating its vocabulary here would block values Sarvam adds later, so
    construction has to accept anything it does not itself depend on.
    """
    SarvamRealtimeSTTService(api_key="test-key", settings=settings)


def test_endpointing_is_not_a_setting():
    """The mode picks the turn strategies, which are announced once at startup.

    Leaving it out of `Settings` is what makes a mid-session switch — which the
    user aggregator would never see — impossible to express.
    """
    with pytest.raises(TypeError):
        SarvamRealtimeSTTService.Settings(endpointing="manual")


@pytest.mark.asyncio
async def test_connect_uses_subscription_key_and_user_agent(monkeypatch):
    captured = {}

    async def fake_websocket_connect(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _FakeWebsocket()

    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect", fake_websocket_connect
    )

    service = SarvamRealtimeSTTService(api_key="test-key")
    await service._connect_websocket()

    assert captured["url"] == service._build_ws_url()
    assert captured["kwargs"]["additional_headers"] == {"API-SUBSCRIPTION-KEY": "test-key"}
    assert captured["kwargs"]["user_agent_header"] == sdk_headers()["User-Agent"]
    # Routed through the base helper, so teardown uses the service's close
    # timeout rather than the library's much longer default.
    assert captured["kwargs"]["close_timeout"] == service._ws_close_timeout


@pytest.mark.asyncio
async def test_failed_connect_leaves_the_service_usable(monkeypatch):
    """A socket that never opened can still be opened on a later attempt.

    `_try_reconnect` skips a service that has stopped being usable, so
    reporting the failure as permanent would bar the retry that could fix it.
    """
    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect",
        AsyncMock(side_effect=ConnectionError("no route to host")),
    )
    service = SarvamRealtimeSTTService(api_key="test-key")
    monkeypatch.setattr(service, "push_frame", AsyncMock())

    await service._connect_websocket()

    assert service._websocket is None
    assert service.is_usable is True


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
        endpointing="manual",
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
        endpointing="manual",
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
    monkeypatch.setattr(service, "start_ttfb_metrics", _noop)
    monkeypatch.setattr(service, "stop_ttfb_metrics", _noop)

    service._sample_rate = 16000
    service._audio_position_bytes = _seconds_to_bytes(1.25)
    await service._handle_message({"event": "vad.speech_start", "utterance_idx": 3})
    await service._handle_message({"event": "vad.speech_end", "utterance_idx": 3})
    await service._handle_message({"event": "transcript.final", "utterance_idx": 3, "text": "हेलो।"})

    assert broadcasted == [ProposedUserStartedSpeakingFrame, ProposedUserStoppedSpeakingFrame]
    assert len(pushed) == 1
    assert isinstance(pushed[0], TranscriptionFrame)
    assert pushed[0].text == "हेलो।"
    assert pushed[0].result["speech_end_audio_position_s"] == 1.25


@pytest.mark.asyncio
async def test_duplicate_speech_end_does_not_emit_duplicate_eos(monkeypatch):
    service = SarvamRealtimeSTTService(api_key="test-key")
    broadcasted = []
    monkeypatch.setattr(service, "broadcast_frame", _capture_class(broadcasted))
    monkeypatch.setattr(service, "start_ttfb_metrics", _noop)

    await service._handle_message({"event": "vad.speech_start", "utterance_idx": 1})
    await service._handle_message({"event": "vad.speech_end", "utterance_idx": 1})
    await service._handle_message({"event": "vad.speech_end", "utterance_idx": 1})

    assert broadcasted == [ProposedUserStartedSpeakingFrame, ProposedUserStoppedSpeakingFrame]


@pytest.mark.asyncio
async def test_post_eos_partial_is_interim_without_changing_eos_timing(monkeypatch):
    service = SarvamRealtimeSTTService(api_key="test-key")
    pushed = []
    broadcasted = []
    monkeypatch.setattr(service, "push_frame", _capture(pushed))
    monkeypatch.setattr(service, "broadcast_frame", _capture_class(broadcasted))
    monkeypatch.setattr(service, "start_ttfb_metrics", _noop)
    monkeypatch.setattr(service, "stop_ttfb_metrics", _noop)

    service._sample_rate = 16000
    service._audio_position_bytes = _seconds_to_bytes(2.0)
    await service._handle_message({"event": "vad.speech_start", "utterance_idx": 2})
    await service._handle_message({"event": "vad.speech_end", "utterance_idx": 2})
    await service._handle_message({"event": "transcript.partial", "utterance_idx": 2, "text": "हेल"})
    await service._handle_message({"event": "transcript.final", "utterance_idx": 2, "text": "हेलो।"})

    assert broadcasted == [ProposedUserStartedSpeakingFrame, ProposedUserStoppedSpeakingFrame]
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
    monkeypatch.setattr("pipecat.services.sarvam.stt.logger", captured_logger)

    await service._handle_message({"event": "session.begin", "request_id": "request-123"})

    assert captured_logger.info_messages == [
        f"{service} Sarvam realtime session.begin request_id=request-123"
    ]
    assert captured_logger.debug_messages == []


@pytest.mark.asyncio
async def test_config_update_sends_without_reconnect():
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


@pytest.mark.asyncio
async def test_update_config_rejects_a_field_sarvam_has_no_setting_for():
    service = SarvamRealtimeSTTService(api_key="test-key")
    service._websocket = _FakeWebsocket()

    with pytest.raises(ValueError, match="langauge_code"):
        await service.update_config(langauge_code="hi-IN")

    assert service._websocket.sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_rate", 8000),
        ("return_timestamps", True),
        ("prefix_padding_ms", 200),
        ("endpointing", "manual"),
    ],
)
async def test_connection_only_fields_rejected_by_update_config(field, value):
    """Connection-time values are constructor arguments, so `Settings` has none.

    That leaves nothing for a `config.update` to carry them in, which is what
    keeps them from reaching a stream that cannot apply them.
    """
    assert field not in {setting.name for setting in fields(SarvamRealtimeSTTSettings)}

    service = SarvamRealtimeSTTService(api_key="test-key")
    service._websocket = _FakeWebsocket()

    with pytest.raises(ValueError, match=field):
        await service.update_config(**{field: value})

    assert service._websocket.sent == []


@pytest.mark.asyncio
async def test_update_config_keeps_the_settings_store_in_step():
    service = SarvamRealtimeSTTService(api_key="test-key")
    service._websocket = _FakeWebsocket()

    await service.update_config(mode="translate")
    assert service._settings.mode == "translate"

    # Without the write-back this diffs against a stale "transcribe" and sends
    # nothing, stranding the server in translate mode.
    service._websocket.sent.clear()
    await service._update_settings(SarvamRealtimeSTTService.Settings(mode="transcribe"))

    assert service._websocket.sent == [json.dumps({"event": "config.update", "mode": "transcribe"})]
    assert service._settings.mode == "transcribe"


@pytest.mark.asyncio
async def test_stream_type_change_is_left_to_the_server():
    service = SarvamRealtimeSTTService(
        api_key="test-key",
        settings=SarvamRealtimeSTTService.Settings(stream_type="simulated"),
    )
    service._websocket = _FakeWebsocket()

    await service._update_settings(SarvamRealtimeSTTService.Settings(stream_type="fast"))

    assert service._websocket.sent == [
        json.dumps({"event": "config.update", "stream_type": "fast"})
    ]
    assert service._settings.stream_type == "fast"


@pytest.mark.asyncio
async def test_language_delta_is_sent_as_language_code():
    service = SarvamRealtimeSTTService(api_key="test-key")
    service._websocket = _FakeWebsocket()

    await service._update_settings(SarvamRealtimeSTTService.Settings(language=Language.HI_IN))

    assert service._websocket.sent == [
        json.dumps({"event": "config.update", "language_code": "hi-IN"})
    ]
    assert service._settings.language_code == "hi-IN"


@pytest.mark.asyncio
async def test_base_settings_delta_still_derives_a_language_code():
    """`STTUpdateSettingsFrame(delta=STTSettings(...))` carries no Sarvam fields.

    Widening the delta is what lets a caller who never imports the Sarvam
    settings change the language and have it reach the server.
    """
    service = SarvamRealtimeSTTService(api_key="test-key")
    service._websocket = _FakeWebsocket()

    await service._update_settings(STTSettings(language=Language.HI_IN))

    assert service._websocket.sent == [
        json.dumps({"event": "config.update", "language_code": "hi-IN"})
    ]
    assert service._settings.language_code == "hi-IN"


def test_sample_rate_defaults_to_the_pipeline_rate():
    service = SarvamRealtimeSTTService(api_key="test-key")

    assert service._init_sample_rate is None
    assert _query(service, sample_rate=8000)["sample_rate"] == ["8000"]


def test_explicit_sample_rate_pins_the_rate():
    service = SarvamRealtimeSTTService(api_key="test-key", sample_rate=8000)

    assert service._init_sample_rate == 8000


@pytest.mark.asyncio
async def test_unsupported_resolved_sample_rate_reports_and_skips_connect(monkeypatch):
    """An unusable pipeline rate has to surface as an error frame."""
    service = SarvamRealtimeSTTService(api_key="test-key")
    pushed_errors = []
    connects = []

    async def fake_push_error(error_msg, exception=None, fatal=False, category=None, **kwargs):
        pushed_errors.append((error_msg, fatal, category))

    async def fake_connect():
        connects.append(True)

    monkeypatch.setattr(service, "push_error", fake_push_error)
    monkeypatch.setattr(service, "_connect", fake_connect)
    monkeypatch.setattr(WebsocketSTTService, "setup", _noop)
    service._sample_rate = 44100

    await service.setup(frame_processor_setup())

    assert len(pushed_errors) == 1
    assert "sample_rate" in pushed_errors[0][0]
    # Non-fatal, so a ServiceSwitcher can fail over to another provider.
    assert pushed_errors[0][1] is False
    # Permanent, since the rate holds for the session: the service loses its
    # usability so the switcher stops handing it audio.
    assert pushed_errors[0][2] is ErrorCategory.INVALID_REQUEST
    assert pushed_errors[0][2].is_permanent
    assert connects == []


@pytest.mark.asyncio
async def test_unsupported_resolved_sample_rate_costs_the_service_its_usability(monkeypatch):
    """The verdict has to reach `is_usable`, which is what a switcher reads."""
    service = SarvamRealtimeSTTService(api_key="test-key")

    monkeypatch.setattr(service, "_connect", AsyncMock())
    monkeypatch.setattr(service, "push_frame", AsyncMock())
    monkeypatch.setattr(WebsocketSTTService, "setup", _noop)
    service._sample_rate = 44100

    await service.setup(frame_processor_setup())

    assert service.is_usable is False


@pytest.mark.asyncio
async def test_keepalive_uses_the_ping_event():
    """Sarvam's socket only accepts JSON events.

    The inherited keepalive writes raw PCM, which this endpoint rejects.
    """
    service = SarvamRealtimeSTTService(api_key="test-key")
    service._websocket = _FakeWebsocket()

    await service._send_keepalive(b"\x00" * 640)

    assert service._websocket.sent == [json.dumps({"event": "ping"})]


@pytest.mark.parametrize("final_text", ["hello", "   "])
@pytest.mark.asyncio
async def test_final_transcript_reports_usage(monkeypatch, final_text):
    """Usage is a per-utterance billing event.

    Leaving it to the teardown flush reports one lump sum, and a cancelled
    session reports nothing at all.
    """
    service = SarvamRealtimeSTTService(api_key="test-key")
    service._setup = frame_processor_setup(TaskManager(), enable_usage_metrics=True)
    service._stt_usage_pending_seconds = 2.5
    pushed = []
    monkeypatch.setattr(service, "push_frame", _capture(pushed))

    await service._handle_message({"event": "transcript.final", "text": final_text})

    usage = [
        data
        for frame in pushed
        if isinstance(frame, MetricsFrame)
        for data in frame.data
        if isinstance(data, STTUsageMetricsData)
    ]
    assert [data.value.audio_seconds for data in usage] == [2.5]
    assert service._stt_usage_pending_seconds == 0.0


@pytest.mark.asyncio
async def test_provider_vad_events_are_ignored_under_manual_endpointing(monkeypatch):
    """The pipeline owns turn boundaries in manual mode.

    Acting on server VAD telemetry too would give the aggregator two competing
    sets of boundaries for the same utterance.
    """
    service = SarvamRealtimeSTTService(
        api_key="test-key",
        endpointing="manual",
    )
    broadcasted = []
    monkeypatch.setattr(service, "push_frame", _noop)
    monkeypatch.setattr(service, "broadcast_frame", _capture_class(broadcasted))
    monkeypatch.setattr(service, "start_ttfb_metrics", _noop)

    await service._handle_message({"event": "vad.speech_start"})
    await service._handle_message({"event": "vad.speech_end"})

    assert broadcasted == []


def test_vad_params_are_omitted_for_manual_endpointing():
    service = SarvamRealtimeSTTService(
        api_key="test-key",
        endpointing="manual",
        prefix_padding_ms=200,
        settings=SarvamRealtimeSTTService.Settings(
            threshold=0.4,
            silence_duration_ms=700,
            min_speech_duration_ms=120,
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


@pytest.mark.parametrize("final_text", ["hello", "   "])
@pytest.mark.asyncio
async def test_speech_cycle_emits_no_processing_metrics(monkeypatch, final_text):
    """A processing window anchored to the speech boundary measures nothing useful.

    It would time how long the user talked, and the interruption raised on
    speech start closes it immediately anyway.
    """
    service = SarvamRealtimeSTTService(api_key="test-key")
    service._enable_metrics = True
    pushed = []
    monkeypatch.setattr(service, "push_frame", _capture(pushed))
    monkeypatch.setattr(service, "broadcast_frame", _noop)

    await service._handle_message({"event": "vad.speech_start"})
    await service._handle_message({"event": "vad.speech_end"})
    await service._handle_message({"event": "transcript.final", "text": final_text})

    assert not [frame for frame in pushed if isinstance(frame, MetricsFrame)]


@pytest.mark.asyncio
async def test_ttfb_is_anchored_to_the_vad_stop_frame(monkeypatch):
    """TTFB has to run from the real end of speech, like every other STT service.

    Sarvam's own `vad.speech_end` only arrives once the server's silence window
    has elapsed, so timing from it would report a shorter interval than the
    rest of the services do. The VAD frame carries the stop delay needed to
    place the actual boundary.
    """
    service = SarvamRealtimeSTTService(api_key="test-key")
    ttfb_starts = []
    monkeypatch.setattr(service, "push_frame", _noop)
    monkeypatch.setattr(service, "broadcast_frame", _noop)

    async def fake_start_ttfb_metrics(*, start_time=None):
        ttfb_starts.append(start_time)

    def fake_create_task(coro, name=None):
        # The base class arms a timeout task; this service has no task manager.
        coro.close()

    monkeypatch.setattr(service, "start_ttfb_metrics", fake_start_ttfb_metrics)
    monkeypatch.setattr(service, "create_task", fake_create_task)

    # The provider boundary alone must not start the measurement.
    await service._handle_message({"event": "vad.speech_start"})
    await service._handle_message({"event": "vad.speech_end"})
    assert ttfb_starts == []

    frame = VADUserStoppedSpeakingFrame(stop_secs=0.2, timestamp=1000.0)
    await service.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert ttfb_starts == [1000.0 - 0.2]


@pytest.mark.asyncio
async def test_session_end_mid_utterance_completes_the_turn(monkeypatch):
    service = SarvamRealtimeSTTService(api_key="test-key")
    broadcasted = []
    monkeypatch.setattr(service, "push_frame", _noop)
    monkeypatch.setattr(service, "broadcast_frame", _capture_class(broadcasted))
    monkeypatch.setattr(service, "start_ttfb_metrics", _noop)

    await service._handle_message({"event": "vad.speech_start"})
    await service._handle_message({"event": "session.end", "audio_duration_s": 1.0})

    assert broadcasted == [ProposedUserStartedSpeakingFrame, ProposedUserStoppedSpeakingFrame]


@pytest.mark.asyncio
async def test_socket_drop_mid_utterance_completes_the_turn(monkeypatch):
    """A dropped socket must still close the turn.

    Reconnection is disabled, so the boundary can never arrive on its own and
    external turn aggregation would wait on it forever.
    """
    service = SarvamRealtimeSTTService(api_key="test-key")
    broadcasted = []
    monkeypatch.setattr(service, "push_frame", _noop)
    monkeypatch.setattr(service, "broadcast_frame", _capture_class(broadcasted))
    monkeypatch.setattr(service, "start_ttfb_metrics", _noop)
    monkeypatch.setattr(service, "push_error", _noop)
    # The socket dies after speech starts, with no matching `vad.speech_end`.
    monkeypatch.setattr(
        service, "_websocket", _FakeWebsocket([json.dumps({"event": "vad.speech_start"})])
    )

    await service._receive_task_handler(AsyncMock())

    assert broadcasted == [ProposedUserStartedSpeakingFrame, ProposedUserStoppedSpeakingFrame]


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


@pytest.mark.parametrize("should_interrupt", [True, False])
def test_should_interrupt_reaches_the_turn_strategies(should_interrupt):
    """The strategies own the interruption, so the setting has to travel to them.

    Keeping it in the service would leave a pipeline that pins its own
    `ExternalUserTurnStrategies` interrupting regardless.
    """
    service = SarvamRealtimeSTTService(api_key="test-key", should_interrupt=should_interrupt)
    strategies = service.service_metadata_frame().user_turn_strategies
    assert strategies.enable_interruptions is should_interrupt


def test_service_metadata_leaves_turn_strategies_unset_in_manual_mode():
    service = SarvamRealtimeSTTService(
        api_key="test-key",
        endpointing="manual",
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
    # No reconnection path, so a dropped socket ends transcription for the
    # session and a switcher has to stop handing this service audio.
    assert service.is_usable is False


@pytest.mark.asyncio
async def test_intentional_disconnect_leaves_the_service_usable(monkeypatch):
    """Teardown ends the same loop, and must not be read as a failure."""
    service = SarvamRealtimeSTTService(api_key="test-key")
    drop = ConnectionClosedError(Close(1006, "Abnormal closure"), None)
    monkeypatch.setattr(service, "_receive_messages", AsyncMock(side_effect=drop))
    service._disconnecting = True

    await service._receive_task_handler(AsyncMock())

    assert service.is_usable is True


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
