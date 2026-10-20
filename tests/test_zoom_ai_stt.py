#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Zoom Scribe Live and Fast STT services."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from websockets.protocol import State

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
)
from pipecat.services.zoom_ai import stt as zoom_stt
from pipecat.services.zoom_ai.stt import (
    SEND_CHUNK_BYTES,
    ZoomScribeFastSTTService,
    ZoomScribeLiveSTTService,
)
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


def _live(sample_rate: int = 16000, **kwargs) -> ZoomScribeLiveSTTService:
    service = ZoomScribeLiveSTTService(api_key="test-key", **kwargs)
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
    assert headers["x-zm-trackingid"].startswith("zoom_ai_community_integration_pipecat_")
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
    service = _live(
        settings=ZoomScribeLiveSTTService.Settings(silence_duration_ms=silence_duration_ms)
    )

    await service._connect_websocket()

    assert json.loads(websocket.sent[0]).get("turn_detection") == expected


@pytest.mark.asyncio
async def test_language_setting_maps_to_a_scribe_locale(monkeypatch):
    websocket = _FakeWebsocket()
    _patch_connect(monkeypatch, websocket)
    service = _live(settings=ZoomScribeLiveSTTService.Settings(language=Language.DE_DE))

    await service._connect_websocket()

    assert json.loads(websocket.sent[0])["language"] == "de-DE"


@pytest.mark.asyncio
@pytest.mark.parametrize("turn_open", [True, False])
async def test_input_gap_mid_turn_sends_silence_to_end_it(monkeypatch, turn_open):
    """Input stopping mid-turn (e.g. a mute) gets enough silence for Scribe to end the turn."""
    monkeypatch.setattr(zoom_stt, "INPUT_GAP_SECS", 0.02)
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


def test_plaintext_urls_warn_except_for_localhost(monkeypatch):
    warnings = []
    monkeypatch.setattr(zoom_stt.logger, "warning", lambda msg, *a, **k: warnings.append(msg))
    ZoomScribeLiveSTTService(api_key="test-key", url="ws://scribe.example.com/live")
    ZoomScribeFastSTTService(api_key="test-key", url="http://scribe.example.com/transcribe")
    assert len(warnings) == 2 and all("not encrypted" in w for w in warnings)
    warnings.clear()
    ZoomScribeLiveSTTService(api_key="test-key", url="ws://localhost:8080/live")
    ZoomScribeLiveSTTService(api_key="test-key")
    assert warnings == []


@pytest.mark.asyncio
async def test_audio_is_sent_in_chunks(monkeypatch):
    service = _live()
    service._websocket = _FakeWebsocket()

    async for _ in service.run_stt(b"\x01" * (SEND_CHUNK_BYTES - 2)):
        pass
    assert service._websocket.sent == []

    async for _ in service.run_stt(b"\x01" * 2):
        pass
    assert service._websocket.sent == [b"\x01" * SEND_CHUNK_BYTES]


@pytest.mark.asyncio
async def test_off_rate_audio_is_resampled_to_16_khz(monkeypatch):
    service = _live(sample_rate=8000)
    service._websocket = _FakeWebsocket()

    for _ in range(10):
        async for _ in service.run_stt(b"\x00" * 1600):
            pass

    audio_sent = b"".join(service._websocket.sent) + bytes(service._send_buffer)
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
@pytest.mark.parametrize(
    "code, fatal, category",
    [
        ("invalid_config", True, ErrorCategory.INVALID_REQUEST),
        ("UNSUPPORTED_LANGUAGE", True, ErrorCategory.INVALID_REQUEST),
        ("internal", True, ErrorCategory.UNKNOWN),
        # Scribe sends this one as non-fatal, but it didn't apply the session.update.
        ("invalid_config", False, ErrorCategory.INVALID_REQUEST),
    ],
)
async def test_fatal_error_event_is_reported(monkeypatch, code, fatal, category):
    service = _live()
    service._websocket = _FakeWebsocket(
        [_event("error", error={"code": code, "message": "bad things", "fatal": fatal})]
    )
    push_error = AsyncMock()
    monkeypatch.setattr(service, "push_error", push_error)

    await service._receive_messages()

    assert "bad things" in push_error.await_args.kwargs["error_msg"]
    assert push_error.await_args.kwargs["category"] == category


@pytest.mark.asyncio
async def test_non_fatal_error_event_is_only_logged(monkeypatch):
    service = _live()
    service._websocket = _FakeWebsocket(
        [_event("error", error={"code": "audio_gap", "message": "late audio", "fatal": False})]
    )
    push_error = AsyncMock()
    monkeypatch.setattr(service, "push_error", push_error)

    await service._receive_messages()

    push_error.assert_not_awaited()


@pytest.mark.asyncio
async def test_settings_change_reconnects(monkeypatch):
    service = _live()
    reconnect = AsyncMock()
    monkeypatch.setattr(service, "_request_reconnect", reconnect)

    await service._update_settings(ZoomScribeLiveSTTService.Settings(language=Language.JA_JP))

    assert service._settings.language == "ja-JP"
    reconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_live_diarization_is_requested_only_when_on(monkeypatch):
    websocket = _FakeWebsocket()
    _patch_connect(monkeypatch, websocket)
    service = _live(settings=ZoomScribeLiveSTTService.Settings(diarization=True))

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
        settings=ZoomScribeLiveSTTService.Settings(
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
    service = _live(settings=ZoomScribeLiveSTTService.Settings(diarization=True, extra=extra))

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
        settings=ZoomScribeLiveSTTService.Settings(
            extra={"outputs": {"text_polishing": {"enabled": True}}}
        )
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
    service = _live(settings=ZoomScribeLiveSTTService.Settings(language_detection="continuous"))
    service._websocket = _FakeWebsocket(
        [_event("transcription.completed", item_id="item_1", transcript="Hola.")]
    )
    frames = _record_frames(monkeypatch, service)

    await service._receive_messages()

    [frame] = [f for f in frames if isinstance(f, TranscriptionFrame)]
    assert frame.language is None


@pytest.mark.asyncio
async def test_live_frames_carry_the_detected_language(monkeypatch):
    service = _live(settings=ZoomScribeLiveSTTService.Settings(language_detection="continuous"))
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
    service = _live(settings=ZoomScribeLiveSTTService.Settings(text_polishing=True))
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
    service = _live(settings=ZoomScribeLiveSTTService.Settings(text_polishing=True))
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
    service = _live(settings=ZoomScribeLiveSTTService.Settings(diarization=True))
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
    service = _live(settings=ZoomScribeLiveSTTService.Settings(diarization=diarization))
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
    [(Language.EN, "en-US"), (Language.PT, "pt-BR"), (Language.PT_PT, "pt-PT")],
)
async def test_base_languages_map_to_a_scribe_locale(monkeypatch, language, locale):
    websocket = _FakeWebsocket()
    _patch_connect(monkeypatch, websocket)
    service = _live(settings=ZoomScribeLiveSTTService.Settings(language=language))

    await service._connect_websocket()

    assert json.loads(websocket.sent[0])["language"] == locale


@pytest.mark.parametrize("language", [Language.EN_GB, Language.KO_KR, Language.XH])
def test_unsupported_language_points_to_the_docs(language):
    with pytest.raises(ValueError, match="supported-languages"):
        _live(settings=ZoomScribeLiveSTTService.Settings(language=language))
    with pytest.raises(ValueError, match="supported-languages"):
        ZoomScribeFastSTTService(
            api_key="test-key", settings=ZoomScribeFastSTTService.Settings(language=language)
        )


@pytest.mark.asyncio
async def test_stop_flushes_audio_and_closes_the_session(monkeypatch):
    service = _live()
    websocket = _FakeWebsocket()
    service._websocket = websocket
    service._send_buffer.extend(b"\x02" * 10)
    monkeypatch.setattr("pipecat.services.zoom_ai.stt.SESSION_CLOSE_TIMEOUT_SECS", 0.01)
    monkeypatch.setattr(
        "pipecat.services.stt_service.WebsocketSTTService.stop", AsyncMock(return_value=None)
    )

    await service.stop(EndFrame())

    assert service._disconnecting
    assert websocket.sent == [b"\x02" * 10, json.dumps({"type": "session.close"})]


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
async def test_processing_metrics_span_end_of_speech_to_final(monkeypatch):
    service = _live()
    service._websocket = _FakeWebsocket(
        [
            _event("input_audio_buffer.speech_started", item_id="item_1"),
            _event("input_audio_buffer.speech_stopped", item_id="item_1"),
            _event("transcription.completed", item_id="item_1", transcript="Hi."),
        ]
    )
    _record_frames(monkeypatch, service)
    calls = []
    monkeypatch.setattr(
        service, "start_processing_metrics", AsyncMock(side_effect=lambda: calls.append("start"))
    )
    monkeypatch.setattr(
        service, "stop_processing_metrics", AsyncMock(side_effect=lambda: calls.append("stop"))
    )

    await service._receive_messages()

    assert calls == ["start", "stop"]


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
    await service._update_settings(ZoomScribeLiveSTTService.Settings(language=Language.JA_JP))

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
    await service._update_settings(ZoomScribeLiveSTTService.Settings(silence_duration_ms=500))
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
        ZoomScribeLiveSTTService.Settings(silence_duration_ms=500, diarization=True)
    )
    websocket.state = State.OPEN
    await service._connect_websocket()

    update = json.loads(websocket.sent[-1])
    assert update["config"] == {"diarization": True}
    assert update["turn_detection"] == {"silence_duration_ms": 500}
    assert not service._session_closed.is_set()  # stop() waits again on the new session


@pytest.mark.asyncio
async def test_processing_metrics_only_stop_after_they_started(monkeypatch):
    service = _live()
    service._websocket = _FakeWebsocket(
        [_event("transcription.completed", item_id="item_1", transcript="Hi.")]
    )
    _record_frames(monkeypatch, service)
    stop = AsyncMock()
    monkeypatch.setattr(service, "stop_processing_metrics", stop)

    await service._receive_messages()  # no speech_stopped, e.g. lost in a reconnect

    stop.assert_not_awaited()


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


# Scribe Fast


class _ScribeFast:
    """A local HTTP server standing in for Scribe Fast."""

    def __init__(self, status: int = 200, body: dict | None = None):
        self.status = status
        self.body = body
        self.requests: list[dict] = []

    async def handle(self, request: web.Request) -> web.Response:
        form = await request.post()
        upload = form["file"]
        assert isinstance(upload, web.FileField)
        self.requests.append(
            {
                "headers": dict(request.headers),
                "config": json.loads(str(form["config"])),
                "outputs": json.loads(str(form["outputs"])) if "outputs" in form else None,
                "fields": {k: json.loads(str(v)) for k, v in form.items() if k != "file"},
                "filename": upload.filename,
                "content_type": upload.content_type,
                "audio": upload.file.read(),
            }
        )
        return web.json_response(self.body, status=self.status)


async def _run_fast(fake: _ScribeFast, **kwargs) -> list:
    app = web.Application()
    app.router.add_post("/transcribe", fake.handle)
    server = TestServer(app)
    await server.start_server()
    try:
        async with aiohttp.ClientSession() as session:
            service = ZoomScribeFastSTTService(
                api_key="test-key",
                url=str(server.make_url("/transcribe")),
                aiohttp_session=session,
                **kwargs,
            )
            service.start_processing_metrics = AsyncMock()
            service.stop_processing_metrics = AsyncMock()
            frames = [frame async for frame in service.run_stt(b"RIFF0000WAVE")]
            service.stop_processing_metrics.assert_awaited_once()
            return frames
    finally:
        await server.close()


_FAST_RESULT = {
    "request_id": "req_1",
    "result": {"text_display": "Hi there! What is the capital of France?", "segments": []},
}


@pytest.mark.asyncio
async def test_fast_uploads_the_utterance_and_config():
    fake = _ScribeFast(body=_FAST_RESULT)

    frames = await _run_fast(fake)

    [request] = fake.requests
    assert request["headers"]["Authorization"] == "Bearer test-key"
    assert request["headers"]["User-Agent"].startswith("pipecat/")
    assert request["headers"]["x-zm-trackingid"].startswith(
        "zoom_ai_community_integration_pipecat_"
    )
    assert (request["filename"], request["content_type"]) == ("utterance.wav", "audio/wav")
    assert request["audio"] == b"RIFF0000WAVE"
    assert request["config"] == {"language": "en-US"}
    assert request["outputs"] is None

    [frame] = frames
    assert isinstance(frame, TranscriptionFrame)
    assert frame.text == "Hi there! What is the capital of France?"
    assert frame.language is Language.EN_US


@pytest.mark.asyncio
async def test_fast_diarization_and_language_are_sent():
    fake = _ScribeFast(body=_FAST_RESULT)

    await _run_fast(
        fake, settings=ZoomScribeFastSTTService.Settings(language=Language.FR_FR, diarization=True)
    )

    assert fake.requests[0]["config"] == {"language": "fr-FR", "diarization": True}


@pytest.mark.asyncio
async def test_fast_speaker_labels_stay_in_the_raw_result():
    segments = [  # as production returns them
        {"start": 0.0, "end": 0.6, "speaker": "Speaker 1", "text": "Hi."},
        {"start": 0.6, "end": 4.8, "speaker": "Speaker 2", "text": "We built it."},
    ]
    body = {
        "request_id": "req_2",
        "result": {
            "text_display": "[Speaker 1] Hi. [Speaker 2] We built it.",
            "segments": segments,
        },
    }

    [frame] = await _run_fast(
        _ScribeFast(body=body), settings=ZoomScribeFastSTTService.Settings(diarization=True)
    )

    assert frame.text == "Hi. We built it."
    assert frame.result["result"]["segments"] == segments


@pytest.mark.asyncio
async def test_fast_recognition_options_and_text_polishing():
    body = {
        "request_id": "req_5",
        "result": {"text_display": "Um, hello hello there.", "segments": []},
        "outputs": {"text_polishing": {"text": "Hello there."}},
    }
    fake = _ScribeFast(body=body)

    [frame] = await _run_fast(
        fake,
        settings=ZoomScribeFastSTTService.Settings(
            model="zoom-scribe-en-pro",
            language_detection="once",
            vocabulary=["Pipecat"],
            text_polishing=True,
            extra={"config": {"profanity_filter": True}},
        ),
    )

    [request] = fake.requests
    assert request["config"] == {
        "language": "en-US",
        "model": "zoom-scribe-en-pro",
        "language_detection": {"mode": "once"},
        "profanity_filter": True,
        "vocabulary": {"phrases": ["Pipecat"]},
    }
    # Text polishing is its own multipart field, as `outputs` sits beside `config`.
    assert request["outputs"] == {"text_polishing": {"enabled": True}}
    assert frame.text == "Hello there."
    assert frame.result["result"]["text_display"] == "Um, hello hello there."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "detection", "expected"),
    [
        ({"language": "es-ES", "segments": []}, "once", Language.ES_ES),  # one-time detection
        ({"segments": [{"text": "Hola.", "language": "es"}]}, "continuous", Language.ES),
        ({"segments": [{"text": "Hola."}]}, "continuous", None),  # detecting, none reported
        ({"segments": [{"text": "Hola."}]}, None, Language.EN_US),  # the configured one
    ],
)
async def test_fast_frame_carries_the_detected_language(result, detection, expected):
    body = {"request_id": "req_6", "result": {"text_display": "Hola.", **result}}

    [frame] = await _run_fast(
        _ScribeFast(body=body),
        settings=ZoomScribeFastSTTService.Settings(language_detection=detection),
    )

    assert frame.language is expected


@pytest.mark.asyncio
async def test_fast_extra_becomes_request_fields():
    fake = _ScribeFast(body=_FAST_RESULT)

    await _run_fast(
        fake,
        settings=ZoomScribeFastSTTService.Settings(
            extra={
                "config": {"segmentation_mode": "none"},
                "outputs": {"future_output": {"enabled": True}},
                "reference_id": "req-42",
                "file": "ignored",  # the service uploads the audio
            }
        ),
    )

    [request] = fake.requests
    assert request["fields"] == {
        "config": {"language": "en-US", "segmentation_mode": "none"},
        "outputs": {"future_output": {"enabled": True}},
        "reference_id": "req-42",
    }
    assert request["audio"] == b"RIFF0000WAVE"


@pytest.mark.asyncio
async def test_fast_word_time_offsets_are_requested_only_when_on():
    off, on = _ScribeFast(body=_FAST_RESULT), _ScribeFast(body=_FAST_RESULT)

    await _run_fast(off)
    await _run_fast(on, settings=ZoomScribeFastSTTService.Settings(word_time_offsets=True))

    assert "word_time_offsets" not in off.requests[0]["config"]
    assert on.requests[0]["config"]["word_time_offsets"] is True


@pytest.mark.asyncio
async def test_fast_empty_transcript_yields_nothing():
    assert await _run_fast(_ScribeFast(body={"result": {"text_display": ""}})) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, category",
    [
        (401, ErrorCategory.AUTHENTICATION),
        (429, ErrorCategory.RATE_LIMIT),
        (503, ErrorCategory.SERVER),
    ],
)
async def test_fast_http_error_is_categorized(status, category):
    [frame] = await _run_fast(_ScribeFast(status=status, body={"code": 124, "message": "nope"}))

    assert isinstance(frame, ErrorFrame)
    assert str(status) in frame.error
    assert frame.category == category


@pytest.mark.asyncio
async def test_fast_non_json_response_is_an_error():
    async def garbage(request: web.Request) -> web.Response:
        await request.post()
        return web.Response(text="<html>gateway</html>", status=200)

    app = web.Application()
    app.router.add_post("/transcribe", garbage)
    server = TestServer(app)
    await server.start_server()
    try:
        async with aiohttp.ClientSession() as session:
            service = ZoomScribeFastSTTService(
                api_key="test-key", url=str(server.make_url("/transcribe")), aiohttp_session=session
            )
            [frame] = [f async for f in service.run_stt(b"RIFF0000WAVE")]
    finally:
        await server.close()

    assert isinstance(frame, ErrorFrame)


@pytest.mark.asyncio
async def test_fast_connection_error_is_an_error():
    async with aiohttp.ClientSession() as session:
        service = ZoomScribeFastSTTService(
            api_key="test-key", url="http://127.0.0.1:9/transcribe", aiohttp_session=session
        )
        [frame] = [f async for f in service.run_stt(b"RIFF0000WAVE")]

    assert isinstance(frame, ErrorFrame)
    assert "test-key" not in frame.error


@pytest.mark.asyncio
async def test_fast_closes_only_a_session_it_created():
    async with aiohttp.ClientSession() as injected:
        service = ZoomScribeFastSTTService(api_key="test-key", aiohttp_session=injected)
        await service.cleanup()
        assert not injected.closed

    service = ZoomScribeFastSTTService(api_key="test-key")
    service._session = aiohttp.ClientSession()
    created = service._session
    await service.cleanup()
    assert created.closed
