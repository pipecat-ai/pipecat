#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Zoom Scribe Live and Fast STT services."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response
from websockets.protocol import State

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
)
from pipecat.services.zoom_ai import stt as zoom_ai_stt
from pipecat.services.zoom_ai.stt import ZoomAISTTService
from pipecat.transcriptions.language import Language
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.errors import ErrorCategory


class _FakeWebsocket:
    """A websocket that replays server messages and records what was sent."""

    def __init__(self, messages=None, *, state=State.OPEN):
        self._messages = messages or []
        self.state = state
        self.sent = []
        self.closed = False

    async def send(self, payload):
        self.sent.append(payload)

    async def close(self):
        self.closed = True
        self.state = State.CLOSED

    def __aiter__(self):
        return self._iter_messages()

    async def _iter_messages(self):
        for message in self._messages:
            yield message


def _patch_connect(monkeypatch, websocket: _FakeWebsocket) -> list[dict]:
    calls = []

    async def fake_websocket_connect(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return websocket

    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect", fake_websocket_connect
    )
    return calls


def _live(sample_rate: int = 16000, **kwargs) -> ZoomAISTTService:
    service = ZoomAISTTService(api_key="test-key", **kwargs)
    # sample_rate is normally set when the pipeline starts, which these tests skip.
    service._sample_rate = sample_rate
    return service


def _record_frames(monkeypatch, service) -> list:
    frames = []
    monkeypatch.setattr(service, "push_frame", AsyncMock(side_effect=lambda f: frames.append(f)))
    monkeypatch.setattr(
        service,
        "broadcast_frame",
        AsyncMock(side_effect=lambda frame_cls, **kw: frames.append(frame_cls)),
    )
    monkeypatch.setattr(service, "emit_stt_usage_metrics", AsyncMock())
    return frames


def _event(kind: str, **fields) -> str:
    return json.dumps({"type": kind, **fields})


# Scribe Live


@pytest.mark.asyncio
async def test_connect_sends_the_session_update(monkeypatch):
    websocket = _FakeWebsocket()
    calls = _patch_connect(monkeypatch, websocket)
    service = _live()

    await service._connect_websocket()

    assert calls[0]["args"][0] == "wss://api.zoom.us/v2/aiservices/scribe/live"
    assert calls[0]["kwargs"]["subprotocols"] == ["live-asr"]
    headers = calls[0]["kwargs"]["additional_headers"]
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["User-Agent"].startswith("pipecat/")
    assert headers["x-zm-trackingid"].startswith("zoom_ai_community_pipecat_")
    assert headers["x-zm-trackingid"] == service._tracking_id
    assert json.loads(websocket.sent[0]) == {
        "type": "session.update",
        "language": "en-US",
        "audio": {"format": "pcm16"},
        "turn_detection": {"silence_duration_ms": 350},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("silence_duration_ms", "expected"),
    [
        (500, {"silence_duration_ms": 500}),
        (None, None),  # Scribe's server default: exactly the documented session.update
    ],
)
async def test_silence_duration_setting(monkeypatch, silence_duration_ms, expected):
    websocket = _FakeWebsocket()
    _patch_connect(monkeypatch, websocket)
    service = _live(settings=ZoomAISTTService.Settings(silence_duration_ms=silence_duration_ms))

    await service._connect_websocket()

    assert json.loads(websocket.sent[0]).get("turn_detection") == expected


@pytest.mark.asyncio
async def test_language_setting_maps_to_a_scribe_locale(monkeypatch):
    websocket = _FakeWebsocket()
    _patch_connect(monkeypatch, websocket)
    service = _live(settings=ZoomAISTTService.Settings(language=Language.DE_DE))

    await service._connect_websocket()

    assert json.loads(websocket.sent[0])["language"] == "de-DE"


@pytest.mark.asyncio
@pytest.mark.parametrize("turn_open", [True, False])
async def test_input_gap_mid_turn_sends_silence_to_end_it(monkeypatch, turn_open):
    """Input stopping mid-turn (e.g. a mute) gets enough silence for Scribe to end the turn."""
    monkeypatch.setattr(zoom_ai_stt, "INPUT_GAP_SECS", 0.02)
    service = _live()
    websocket = _FakeWebsocket()
    service._websocket = websocket
    service._user_turn_open = turn_open
    service._last_input_time = 0  # input stopped long ago

    task = asyncio.create_task(service._input_gap_handler())
    await asyncio.sleep(0.1)
    task.cancel()

    sent = [m for m in websocket.sent if isinstance(m, bytes)]
    # silence_duration_ms (350) plus 100 ms, once per gap; nothing without an open turn
    assert sent == ([bytes(16000 * 2 * 450 // 1000)] if turn_open else [])
    assert service._stt_usage_pending_seconds == pytest.approx(0.45 if turn_open else 0)


@pytest.mark.asyncio
async def test_input_gap_silence_is_resampled_and_counted_at_the_pipeline_rate(monkeypatch):
    monkeypatch.setattr(zoom_ai_stt, "INPUT_GAP_SECS", 0.02)
    service = _live(sample_rate=8000)
    websocket = _FakeWebsocket()
    service._websocket = websocket
    service._user_turn_open = True
    service._last_input_time = 0

    task = asyncio.create_task(service._input_gap_handler())
    await asyncio.sleep(0.1)
    task.cancel()

    sent = b"".join(m for m in websocket.sent if isinstance(m, bytes))
    # Up to 450 ms at 16 kHz; the resampler holds back the rest.
    assert 0 < len(sent) <= 16000 * 2 * 450 // 1000
    assert service._stt_usage_pending_seconds == pytest.approx(0.45)


@pytest.mark.asyncio
async def test_keepalive_silence_is_resampled_and_counted(monkeypatch):
    service = _live(sample_rate=24000)
    websocket = _FakeWebsocket()
    service._websocket = websocket

    await service._send_keepalive(bytes(24000 * 2 // 10))  # 100 ms at 24 kHz

    sent = b"".join(m for m in websocket.sent if isinstance(m, bytes))
    assert 0 < len(sent) <= 16000 * 2 // 10
    assert service._stt_usage_pending_seconds == pytest.approx(0.1)


def test_plaintext_urls_warn_except_for_localhost(monkeypatch):
    warnings = []
    monkeypatch.setattr(zoom_ai_stt.logger, "warning", lambda msg, *a, **k: warnings.append(msg))
    ZoomAISTTService(api_key="test-key", url="ws://scribe.example.com/live")
    assert len(warnings) == 1 and "not encrypted" in warnings[0]
    warnings.clear()
    ZoomAISTTService(api_key="test-key", url="ws://localhost:8080/live")
    ZoomAISTTService(api_key="test-key")
    assert warnings == []


@pytest.mark.asyncio
async def test_audio_is_sent_as_it_arrives(monkeypatch):
    service = _live()
    service._websocket = _FakeWebsocket()

    async for _ in service.run_stt(b"\x01" * 640):
        pass
    async for _ in service.run_stt(b"\x02" * 640):
        pass

    assert service._websocket.sent == [b"\x01" * 640, b"\x02" * 640]


@pytest.mark.asyncio
async def test_off_rate_audio_is_resampled_to_16_khz(monkeypatch):
    service = _live(sample_rate=8000)
    service._websocket = _FakeWebsocket()

    for _ in range(10):
        async for _ in service.run_stt(b"\x00" * 1600):
            pass

    audio_sent = b"".join(service._websocket.sent)
    # 8 kHz in, 16 kHz out: twice the bytes, less what the resampler still holds.
    assert 0.9 < len(audio_sent) / (2 * 10 * 1600) <= 1.0


@pytest.mark.asyncio
async def test_scribe_turn_brackets_the_final_transcript(monkeypatch):
    service = _live()
    service._websocket = _FakeWebsocket(
        [
            _event("session.updated"),
            _event("input_audio_buffer.speech_started", item_id="item_1", audio_start_ms=100),
            _event("input_audio_buffer.speech_stopped", item_id="item_1", audio_end_ms=900),
            _event(
                "transcription.completed",
                item_id="item_1",
                transcript="[Speaker 1] What is the capital of France?",
                audio_start_ms=100,
                audio_end_ms=900,
            ),
        ]
    )
    frames = _record_frames(monkeypatch, service)

    await service._receive_messages()

    assert [f if isinstance(f, type) else type(f) for f in frames] == [
        ProposedUserStartedSpeakingFrame,
        TranscriptionFrame,
        ProposedUserStoppedSpeakingFrame,
    ]
    transcript = frames[1]
    assert transcript.text == "What is the capital of France?"
    assert transcript.finalized
    assert transcript.language is Language.EN_US


@pytest.mark.asyncio
async def test_repeated_speech_start_opens_one_turn(monkeypatch):
    service = _live()
    service._websocket = _FakeWebsocket(
        [
            _event("input_audio_buffer.speech_started"),
            _event("input_audio_buffer.speech_started"),
        ]
    )
    frames = _record_frames(monkeypatch, service)

    await service._receive_messages()

    assert frames == [ProposedUserStartedSpeakingFrame]


@pytest.mark.asyncio
async def test_usage_is_reported_before_the_transcription_frame(monkeypatch):
    service = _live()
    service._websocket = _FakeWebsocket(
        [
            _event("input_audio_buffer.speech_started", item_id="item_1"),
            _event("transcription.completed", item_id="item_1", transcript="Berlin."),
        ]
    )
    events = []
    monkeypatch.setattr(
        service, "push_frame", AsyncMock(side_effect=lambda f: events.append(type(f).__name__))
    )
    monkeypatch.setattr(service, "broadcast_frame", AsyncMock())
    monkeypatch.setattr(
        service, "emit_stt_usage_metrics", AsyncMock(side_effect=lambda: events.append("usage"))
    )

    await service._receive_messages()

    assert events == ["usage", "TranscriptionFrame"]


@pytest.mark.asyncio
async def test_disconnect_closes_an_open_turn(monkeypatch):
    service = _live()
    service._websocket = _FakeWebsocket([_event("input_audio_buffer.speech_started")])
    frames = _record_frames(monkeypatch, service)

    await service._receive_messages()
    await service._disconnect_websocket()

    assert frames == [ProposedUserStartedSpeakingFrame, ProposedUserStoppedSpeakingFrame]


@pytest.mark.asyncio
async def test_rejected_session_settings_are_a_permanent_error(monkeypatch):
    service = _live()
    service._websocket = _FakeWebsocket(
        [
            _event("session.created"),
            _event(
                "error",
                error={"code": "invalid_config", "message": "no polishing", "fatal": False},
            ),
            _event(
                "error",
                error={"code": "missing_config", "message": "send config", "fatal": True},
            ),
        ]
    )
    service._tracking_id = "track_1"
    push_error = AsyncMock()
    monkeypatch.setattr(service, "push_error", push_error)

    await service._receive_messages()

    [non_fatal, fatal] = push_error.await_args_list
    assert non_fatal.kwargs == {
        "error_msg": "Zoom Scribe Live error invalid_config: no polishing (tracking_id=track_1)"
    }
    assert fatal.kwargs == {
        "error_msg": "Zoom Scribe Live error missing_config: send config (tracking_id=track_1)",
        "category": ErrorCategory.INVALID_REQUEST,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("fatal", [True, False])
async def test_errors_after_the_session_starts_are_reported(monkeypatch, fatal):
    service = _live()
    service._websocket = _FakeWebsocket(
        [
            _event("session.updated"),
            _event("error", error={"code": "internal", "message": "bad things", "fatal": fatal}),
        ]
    )
    service._tracking_id = "track_1"
    push_error = AsyncMock()
    monkeypatch.setattr(service, "push_error", push_error)

    await service._receive_messages()

    push_error.assert_awaited_once_with(
        error_msg="Zoom Scribe Live error internal: bad things (tracking_id=track_1)"
    )


@pytest.mark.asyncio
async def test_settings_change_reconnects(monkeypatch):
    service = _live()
    reconnect = AsyncMock()
    monkeypatch.setattr(service, "_request_reconnect", reconnect)

    await service._update_settings(ZoomAISTTService.Settings(language=Language.JA_JP))

    assert service._settings.language == "ja-JP"
    reconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_live_diarization_is_requested_only_when_on(monkeypatch):
    websocket = _FakeWebsocket()
    _patch_connect(monkeypatch, websocket)
    service = _live(settings=ZoomAISTTService.Settings(diarization=True))

    await service._connect_websocket()

    assert json.loads(websocket.sent[0]) == {
        "type": "session.update",
        "language": "en-US",
        "audio": {"format": "pcm16"},
        "config": {"diarization": True},
        "turn_detection": {"silence_duration_ms": 350},
    }


@pytest.mark.asyncio
async def test_live_recognition_options_are_requested_only_when_set(monkeypatch):
    websocket = _FakeWebsocket()
    _patch_connect(monkeypatch, websocket)
    service = _live(
        settings=ZoomAISTTService.Settings(
            model="zoom-scribe-multilingual",
            language_detection="continuous",
            vocabulary=["Pipecat", "LiveKit"],
            text_polishing=True,
            extra={"config": {"profanity_filter": True}},
        )
    )

    await service._connect_websocket()

    update = json.loads(websocket.sent[0])
    assert update["config"] == {
        "model": "zoom-scribe-multilingual",
        "language_detection": {"mode": "continuous"},
        "profanity_filter": True,
        "vocabulary": {"phrases": ["Pipecat", "LiveKit"]},
    }
    assert update["outputs"] == {"text_polishing": {"enabled": True, "include_transcript": True}}


@pytest.mark.asyncio
async def test_live_extra_is_merged_into_the_session_update(monkeypatch):
    websocket = _FakeWebsocket()
    _patch_connect(monkeypatch, websocket)
    extra = {
        "config": {"vocabulary_id": "acct_vocab_1", "diarization": False},
        "turn_detection": {"max_segment_duration_ms": 15000},
        "outputs": {"future_output": {"enabled": True}},
        "audio": {"format": "opus"},  # the service sends PCM16, so this is kept
    }
    service = _live(settings=ZoomAISTTService.Settings(diarization=True, extra=extra))

    await service._connect_websocket()

    update = json.loads(websocket.sent[0])
    assert update["config"] == {"diarization": False, "vocabulary_id": "acct_vocab_1"}
    assert update["turn_detection"] == {
        "silence_duration_ms": 350,
        "max_segment_duration_ms": 15000,
    }
    assert update["outputs"] == {"future_output": {"enabled": True}}
    assert update["audio"] == {"format": "pcm16"}
    assert extra["audio"] == {"format": "opus"}  # the caller's dict isn't modified


@pytest.mark.asyncio
async def test_live_acts_on_settings_given_through_extra(monkeypatch):
    websocket = _FakeWebsocket()
    _patch_connect(monkeypatch, websocket)
    service = _live(
        settings=ZoomAISTTService.Settings(extra={"outputs": {"text_polishing": {"enabled": True}}})
    )
    await service._connect_websocket()
    update = json.loads(websocket.sent[0])
    # The transcript is the fallback when polishing fails, so it is always requested.
    assert update["outputs"] == {"text_polishing": {"enabled": True, "include_transcript": True}}

    service._websocket = _FakeWebsocket(
        [
            _event(
                "transcription.completed",
                item_id="item_1",
                transcript="Um, hello hello there.",
                outputs={"text_polishing": {"text": "Hello there."}},
            )
        ]
    )
    frames = _record_frames(monkeypatch, service)
    await service._receive_messages()

    [frame] = [f for f in frames if isinstance(f, TranscriptionFrame)]
    assert frame.text == "Hello there."


@pytest.mark.asyncio
async def test_live_frame_language_is_none_when_detection_reports_none(monkeypatch):
    service = _live(settings=ZoomAISTTService.Settings(language_detection="continuous"))
    service._websocket = _FakeWebsocket(
        [_event("transcription.completed", item_id="item_1", transcript="Hola.")]
    )
    frames = _record_frames(monkeypatch, service)

    await service._receive_messages()

    [frame] = [f for f in frames if isinstance(f, TranscriptionFrame)]
    assert frame.language is None


@pytest.mark.asyncio
async def test_live_frames_carry_the_detected_language(monkeypatch):
    service = _live(settings=ZoomAISTTService.Settings(language_detection="continuous"))
    service._websocket = _FakeWebsocket(
        [
            _event(
                "transcription.completed",
                item_id="item_1",
                transcript="Buenos días.",
                language="es",
            )
        ]
    )
    frames = _record_frames(monkeypatch, service)

    await service._receive_messages()

    [frame] = [f for f in frames if isinstance(f, TranscriptionFrame)]
    assert frame.language is Language.ES


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("polished", "expected"),
    [
        ("[polishing_failed: timeout]", "Um, hello hello there."),  # falls back to the transcript
        ("[Speaker 1] Hello there.", "Hello there."),  # speaker tags removed
    ],
)
async def test_live_polished_text_edge_cases(monkeypatch, polished, expected):
    service = _live(settings=ZoomAISTTService.Settings(text_polishing=True))
    service._websocket = _FakeWebsocket(
        [
            _event(
                "transcription.completed",
                item_id="item_1",
                transcript="Um, hello hello there.",
                outputs={"text_polishing": {"text": polished}},
            )
        ]
    )
    frames = _record_frames(monkeypatch, service)

    await service._receive_messages()

    [frame] = [f for f in frames if isinstance(f, TranscriptionFrame)]
    assert frame.text == expected


@pytest.mark.asyncio
async def test_live_text_polishing_pushes_the_polished_text(monkeypatch):
    service = _live(settings=ZoomAISTTService.Settings(text_polishing=True))
    service._websocket = _FakeWebsocket(
        [
            _event(
                "transcription.completed",
                item_id="item_1",
                transcript="Um, hello hello there.",
                outputs={"text_polishing": {"text": "Hello there."}},
            )
        ]
    )
    frames = _record_frames(monkeypatch, service)

    await service._receive_messages()

    [frame] = [f for f in frames if isinstance(f, TranscriptionFrame)]
    assert frame.text == "Hello there."
    assert frame.result["transcript"] == "Um, hello hello there."


@pytest.mark.asyncio
async def test_live_speaker_labels_stay_in_the_raw_result(monkeypatch):
    service = _live(settings=ZoomAISTTService.Settings(diarization=True))
    spans = [  # sent only when the speaker changed mid-utterance
        {"speaker": "Speaker 1", "transcript": "Hi", "audio_start_ms": 0, "audio_end_ms": 500},
        {"speaker": "Speaker 2", "transcript": "fine", "audio_start_ms": 500, "audio_end_ms": 900},
    ]
    service._websocket = _FakeWebsocket(
        [
            _event(
                "transcription.completed",
                item_id="item_1",
                transcript="[Speaker 1] Hello there.",
                speaker="Speaker 1",
            ),
            _event(
                "transcription.completed",
                item_id="item_2",
                transcript="Hi fine",
                speaker="Speaker 1",
                speaker_segments=spans,
            ),
        ]
    )
    frames = _record_frames(monkeypatch, service)

    await service._receive_messages()

    first, second = [f for f in frames if isinstance(f, TranscriptionFrame)]
    assert first.text == "Hello there."
    assert first.result["speaker"] == "Speaker 1"
    assert "speaker_segments" not in first.result
    assert second.result["speaker_segments"] == spans


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("diarization", "diarization_enabled", "warned"),
    [(True, False, True), (True, True, False), (True, None, False), (False, False, False)],
)
async def test_live_warns_when_scribe_cannot_diarize(
    monkeypatch, diarization, diarization_enabled, warned
):
    service = _live(settings=ZoomAISTTService.Settings(diarization=diarization))
    fields = {} if diarization_enabled is None else {"diarization_enabled": diarization_enabled}
    service._websocket = _FakeWebsocket([_event("session.updated", **fields)])
    warning = MagicMock()
    monkeypatch.setattr("pipecat.services.zoom_ai.stt.logger.warning", warning)

    await service._receive_messages()

    assert warning.called is warned


@pytest.mark.parametrize("should_interrupt", [True, False])
def test_metadata_recommends_external_turn_strategies(should_interrupt):
    service = _live(should_interrupt=should_interrupt)

    strategies = service.service_metadata_frame().user_turn_strategies

    assert isinstance(strategies, ExternalUserTurnStrategies)
    assert strategies.enable_interruptions is should_interrupt
    assert service.supports_ttfs is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("language", "locale"),
    [
        (Language.EN, "en-US"),
        (Language.PT, "pt-BR"),
        (Language.PT_PT, "pt-PT"),
        (Language.KO, "ko-KR"),
        (Language.NL_NL, "nl-NL"),
    ],
)
async def test_base_languages_map_to_a_scribe_locale(monkeypatch, language, locale):
    websocket = _FakeWebsocket()
    _patch_connect(monkeypatch, websocket)
    service = _live(settings=ZoomAISTTService.Settings(language=language))

    await service._connect_websocket()

    assert json.loads(websocket.sent[0])["language"] == locale


@pytest.mark.parametrize(
    ("language", "locale"),
    [(Language.EN_GB, "en-US"), (Language.ES_MX, "es-ES"), ("en-GB", "en-US"), (Language.XH, "xh")],
)
def test_unlisted_languages_resolve_through_their_base_language(language, locale):
    service = _live(settings=ZoomAISTTService.Settings(language=language))

    assert service._settings.language == locale


@pytest.mark.asyncio
async def test_stop_closes_the_session(monkeypatch):
    service = _live()
    websocket = _FakeWebsocket()
    service._websocket = websocket
    monkeypatch.setattr("pipecat.services.zoom_ai.stt.SESSION_CLOSE_TIMEOUT_SECS", 0.01)
    monkeypatch.setattr(
        "pipecat.services.stt_service.WebsocketSTTService.stop", AsyncMock(return_value=None)
    )

    await service.stop(EndFrame())

    assert service._disconnecting
    assert websocket.sent == [json.dumps({"type": "session.close"})]


@pytest.mark.asyncio
async def test_stop_waits_for_the_last_transcript(monkeypatch):
    service = _live()
    service._websocket = _FakeWebsocket()
    frames = _record_frames(monkeypatch, service)
    base_stop = AsyncMock(return_value=None)
    monkeypatch.setattr("pipecat.services.stt_service.WebsocketSTTService.stop", base_stop)
    sent = []

    async def scribe_answers(message):
        # Scribe sends the pending final, then session.closed, after session.close.
        sent.append(message)
        if message["type"] == "session.close":

            async def answer():
                await service._handle_event(
                    {"type": "transcription.completed", "transcript": "Last words."}
                )
                await service._handle_event({"type": "session.closed", "reason": "client"})

            asyncio.get_running_loop().create_task(answer())

    monkeypatch.setattr(service, "_send_json", scribe_answers)

    await service.stop(EndFrame())

    assert sent == [{"type": "session.close"}]
    assert [f.text for f in frames if isinstance(f, TranscriptionFrame)] == ["Last words."]
    base_stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_stop_gives_up_waiting_after_the_timeout(monkeypatch):
    service = _live()
    service._websocket = _FakeWebsocket()
    monkeypatch.setattr("pipecat.services.zoom_ai.stt.SESSION_CLOSE_TIMEOUT_SECS", 0.05)
    base_stop = AsyncMock(return_value=None)
    monkeypatch.setattr("pipecat.services.stt_service.WebsocketSTTService.stop", base_stop)

    await asyncio.wait_for(service.stop(EndFrame()), 1)

    base_stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_transcript_without_speech_started_opens_its_own_turn(monkeypatch):
    service = _live()
    service._websocket = _FakeWebsocket(
        [_event("transcription.completed", item_id="item_1", transcript="Hello.")]
    )
    frames = _record_frames(monkeypatch, service)

    await service._receive_messages()

    assert frames[0] is ProposedUserStartedSpeakingFrame
    assert isinstance(frames[1], TranscriptionFrame)
    assert frames[2] is ProposedUserStoppedSpeakingFrame


@pytest.mark.asyncio
async def test_empty_transcript_ends_the_turn_without_a_transcription(monkeypatch):
    service = _live()
    service._websocket = _FakeWebsocket(
        [
            _event("input_audio_buffer.speech_started", item_id="item_1"),
            _event("transcription.completed", item_id="item_1", transcript=""),
        ]
    )
    frames = _record_frames(monkeypatch, service)

    await service._receive_messages()

    assert frames == [ProposedUserStartedSpeakingFrame, ProposedUserStoppedSpeakingFrame]


@pytest.mark.asyncio
async def test_live_reports_no_processing_metrics(monkeypatch):
    service = _live()
    service._websocket = _FakeWebsocket(
        [
            _event("input_audio_buffer.speech_started", item_id="item_1"),
            _event("input_audio_buffer.speech_stopped", item_id="item_1"),
            _event("transcription.completed", item_id="item_1", transcript="Hi."),
        ]
    )
    _record_frames(monkeypatch, service)
    start, stop = AsyncMock(), AsyncMock()
    monkeypatch.setattr(service, "start_processing_metrics", start)
    monkeypatch.setattr(service, "stop_processing_metrics", stop)

    await service._receive_messages()

    start.assert_not_awaited()
    stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_settings_change_waits_for_the_scribe_turn_to_end(monkeypatch):
    service = _live()
    service._websocket = _FakeWebsocket(
        [_event("input_audio_buffer.speech_started", item_id="item_1")]
    )
    _record_frames(monkeypatch, service)
    reconnect = AsyncMock()
    monkeypatch.setattr(service, "_reconnect", reconnect)

    await service._receive_messages()  # Scribe heard speech; no local VAD in this pipeline
    await service._update_settings(ZoomAISTTService.Settings(language=Language.JA_JP))

    reconnect.assert_not_awaited()
    assert service._need_reconnect  # runs when the aggregator ends the turn


@pytest.mark.asyncio
async def test_held_settings_change_applies_when_the_scribe_turn_ends(monkeypatch):
    # No user aggregator, and the turn ends with an empty transcript: the change must
    # still be applied once Scribe's turn is over.
    service = _live()
    service._websocket = _FakeWebsocket(
        [_event("input_audio_buffer.speech_started", item_id="item_1")]
    )
    _record_frames(monkeypatch, service)
    reconnect = AsyncMock()
    monkeypatch.setattr(service, "_reconnect", reconnect)
    scheduled = []
    monkeypatch.setattr(service, "create_task", lambda coro, name=None: scheduled.append(coro))

    await service._receive_messages()
    await service._update_settings(ZoomAISTTService.Settings(silence_duration_ms=500))
    reconnect.assert_not_called()

    await service._handle_event({"type": "transcription.completed", "transcript": ""})

    assert len(scheduled) == 1
    await scheduled[0]
    reconnect.assert_awaited_once()
    assert service._can_reconnect


@pytest.mark.asyncio
async def test_settings_change_reconnects_with_the_new_config(monkeypatch):
    websocket = _FakeWebsocket()
    _patch_connect(monkeypatch, websocket)
    service = _live()

    await service._connect_websocket()
    await service._disconnect_websocket()
    service._settings.apply_update(
        ZoomAISTTService.Settings(silence_duration_ms=500, diarization=True)
    )
    websocket.state = State.OPEN
    await service._connect_websocket()

    update = json.loads(websocket.sent[-1])
    assert update["config"] == {"diarization": True}
    assert update["turn_detection"] == {"silence_duration_ms": 500}
    assert not service._session_closed.is_set()  # stop() waits again on the new session


@pytest.mark.asyncio
async def test_cancel_does_not_close_the_session_gracefully(monkeypatch):
    service = _live()
    websocket = _FakeWebsocket()
    service._websocket = websocket
    monkeypatch.setattr(
        "pipecat.services.stt_service.WebsocketSTTService.cancel", AsyncMock(return_value=None)
    )

    await service.cancel(CancelFrame())

    assert json.dumps({"type": "session.close"}) not in websocket.sent


@pytest.mark.asyncio
async def test_disconnect_reports_once(monkeypatch):
    service = _live()
    service._websocket = _FakeWebsocket()
    on_disconnected = AsyncMock()
    monkeypatch.setattr(service, "_call_event_handler", on_disconnected)

    await service._disconnect_websocket()
    await service._disconnect_websocket()

    on_disconnected.assert_awaited_once_with("on_disconnected")


@pytest.mark.asyncio
async def test_failed_first_connect_still_starts_the_receive_loop(monkeypatch):
    async def refuse(*args, **kwargs):
        raise ConnectionError("network down")

    monkeypatch.setattr("pipecat.services.websocket_service.websocket_connect", refuse)
    service = _live()
    monkeypatch.setattr(service, "push_error", AsyncMock())
    started = []
    monkeypatch.setattr(
        service, "create_task", lambda coro, name=None: started.append(name) or coro.close()
    )

    await service._connect()

    assert service._websocket is None
    # The receive loop owns reconnecting, so it must run even without a socket.
    assert "receive" in started


@pytest.mark.asyncio
async def test_connect_clears_a_previous_disconnect_before_connecting(monkeypatch):
    service = _live()
    service._disconnecting = True  # left set by the disconnect half of a reconnect
    disconnecting_while_connecting = []

    async def connect_websocket():
        disconnecting_while_connecting.append(service._disconnecting)

    monkeypatch.setattr(service, "_connect_websocket", connect_websocket)
    monkeypatch.setattr(service, "create_task", lambda coro, name=None: coro.close())

    await service._connect()

    assert disconnecting_while_connecting == [False]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            b'{"code":124, "message":"Invalid Access token"}',
            "Zoom refused the connection (HTTP 401): Invalid Access token",
        ),
        (
            b"",
            "Unable to connect to Zoom Scribe Live: server rejected WebSocket connection: HTTP 401",
        ),
    ],
)
async def test_refused_handshake_reports_zoom_reason(monkeypatch, body, expected):
    refusal = InvalidStatus(Response(401, "Unauthorized", Headers(), body))

    async def refuse(*args, **kwargs):
        raise refusal

    monkeypatch.setattr("pipecat.services.websocket_service.websocket_connect", refuse)
    service = _live()
    push_error = AsyncMock()
    monkeypatch.setattr(service, "push_error", push_error)
    connection_errors = []

    @service.event_handler("on_connection_error")
    async def on_connection_error(service, error):
        connection_errors.append(error)

    await service._connect_websocket()
    # Event handlers run as background tasks; let it finish.
    await asyncio.sleep(0)

    push_error.assert_awaited_once_with(error_msg=expected, exception=refusal)
    assert connection_errors == [expected.removeprefix("Unable to connect to Zoom Scribe Live: ")]
