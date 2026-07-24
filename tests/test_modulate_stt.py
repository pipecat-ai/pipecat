#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import dataclasses
import json
from unittest.mock import AsyncMock

import pytest
from websockets.protocol import State

from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.services.modulate.stt import ModulateSTTService, language_to_modulate_language
from pipecat.services.settings import _NotGiven
from pipecat.transcriptions.language import Language


class _FakeWebsocket:
    def __init__(self, messages=(), *, state=State.OPEN, send_side_effect=None):
        self._messages = messages
        self.state = state
        self.send = AsyncMock(side_effect=send_side_effect)

    def __aiter__(self):
        return self._iter_messages()

    async def _iter_messages(self):
        for message in self._messages:
            yield message


def test_language_mapping_strips_region_subtags():
    assert language_to_modulate_language(Language.EN) == "en"
    assert language_to_modulate_language(Language.EN_US) == "en"
    assert language_to_modulate_language(Language.FR) == "fr"
    assert language_to_modulate_language(Language.PT_BR) == "pt"


def test_default_settings_are_complete():
    service = ModulateSTTService(api_key="test-key", sample_rate=16000)

    missing = [
        f.name
        for f in dataclasses.fields(service._settings)
        if isinstance(getattr(service._settings, f.name), _NotGiven)
    ]

    assert missing == []


def test_build_ws_url_contains_connection_params():
    service = ModulateSTTService(api_key="test-key", sample_rate=16000)
    service._sample_rate = 16000

    url = service._build_ws_url()

    assert url.startswith("wss://platform.modulate.ai/api/velma-2-stt-streaming?")
    assert "api_key=test-key" in url
    assert "audio_format=s16le" in url
    assert "sample_rate=16000" in url
    assert "num_channels=1" in url


def test_build_config_defaults():
    service = ModulateSTTService(api_key="test-key", sample_rate=16000)

    config = service._build_config()

    # Language is omitted so the server auto-detects per utterance.
    assert config == {"partial_results": True, "speaker_diarization": False}


def test_build_config_with_settings():
    service = ModulateSTTService(
        api_key="test-key",
        sample_rate=16000,
        settings=ModulateSTTService.Settings(
            language=Language.FR,
            speaker_diarization=True,
            emotion_signal=True,
            custom_terms=["Modulate", {"term": "Velma", "pronunciations": ["VEL-muh"]}],
        ),
    )

    config = service._build_config()

    assert config == {
        "partial_results": True,
        "speaker_diarization": True,
        "emotion_signal": True,
        "custom_terms": ["Modulate", {"term": "Velma", "pronunciations": ["VEL-muh"]}],
        "language": "fr",
    }


@pytest.mark.asyncio
async def test_connect_failure_clears_stale_websocket(monkeypatch):
    async def fake_websocket_connect(*args, **kwargs):
        raise RuntimeError("connection failed")

    monkeypatch.setattr("pipecat.services.modulate.stt.websocket_connect", fake_websocket_connect)

    service = ModulateSTTService(api_key="test-key", sample_rate=16000)
    service._sample_rate = 16000
    service._websocket = _FakeWebsocket(state=State.CLOSED)

    await service._connect_websocket()

    assert service._websocket is None


@pytest.mark.asyncio
async def test_connect_sends_config_frame(monkeypatch):
    websocket = _FakeWebsocket()

    async def fake_websocket_connect(*args, **kwargs):
        return websocket

    monkeypatch.setattr("pipecat.services.modulate.stt.websocket_connect", fake_websocket_connect)

    service = ModulateSTTService(api_key="test-key", sample_rate=16000)
    service._sample_rate = 16000

    await service._connect_websocket()

    websocket.send.assert_awaited_once()
    config = json.loads(websocket.send.await_args.args[0])
    assert config == {"partial_results": True, "speaker_diarization": False}


@pytest.mark.asyncio
async def test_run_stt_send_failure_does_not_clear_websocket():
    service = ModulateSTTService(api_key="test-key", sample_rate=16000)
    websocket = _FakeWebsocket(send_side_effect=RuntimeError("websocket closed"))
    service._websocket = websocket

    async for _ in service.run_stt(b"\x00" * 160):
        pass

    assert service._websocket is websocket


@pytest.mark.asyncio
async def test_receive_messages_pushes_interim_and_final_frames(monkeypatch):
    service = ModulateSTTService(api_key="test-key", sample_rate=16000)
    pushed_frames = []

    async def fake_push_frame(frame, direction=None):
        pushed_frames.append(frame)

    monkeypatch.setattr(service, "push_frame", fake_push_frame)
    monkeypatch.setattr(service, "stop_processing_metrics", AsyncMock())
    monkeypatch.setattr(service, "_trace_transcription", AsyncMock())

    service._websocket = _FakeWebsocket(
        [
            json.dumps(
                {
                    "type": "partial_utterance",
                    "partial_utterance": {"text": "Bonjour, comment", "start_ms": 120},
                }
            ),
            json.dumps(
                {
                    "type": "utterance",
                    "utterance": {
                        "utterance_uuid": "9e2f",
                        "text": "Bonjour, comment allez-vous?",
                        "start_ms": 120,
                        "duration_ms": 1500,
                        "speaker": 1,
                        "language": "fr",
                    },
                }
            ),
        ]
    )

    await service._receive_messages()

    assert len(pushed_frames) == 2
    interim = pushed_frames[0]
    assert isinstance(interim, InterimTranscriptionFrame)
    assert interim.text == "Bonjour, comment"
    assert interim.language is None
    final = pushed_frames[1]
    assert isinstance(final, TranscriptionFrame)
    assert final.text == "Bonjour, comment allez-vous?"
    assert final.finalized is True
    assert final.language == Language.FR


@pytest.mark.asyncio
async def test_receive_messages_skips_empty_transcripts_and_done(monkeypatch):
    service = ModulateSTTService(api_key="test-key", sample_rate=16000)
    pushed_frames = []

    async def fake_push_frame(frame, direction=None):
        pushed_frames.append(frame)

    monkeypatch.setattr(service, "push_frame", fake_push_frame)

    service._websocket = _FakeWebsocket(
        [
            json.dumps({"type": "partial_utterance", "partial_utterance": {"text": ""}}),
            json.dumps({"type": "utterance", "utterance": {"text": ""}}),
            json.dumps({"type": "done", "duration_ms": 1234}),
        ]
    )

    await service._receive_messages()

    assert pushed_frames == []
    assert service._disconnecting is False


@pytest.mark.asyncio
async def test_receive_messages_reports_server_error(monkeypatch):
    service = ModulateSTTService(api_key="test-key", sample_rate=16000)
    push_error = AsyncMock()
    monkeypatch.setattr(service, "push_error", push_error)

    service._websocket = _FakeWebsocket(
        [json.dumps({"type": "error", "error": "invalid audio format"})]
    )

    await service._receive_messages()

    push_error.assert_awaited_once()
    assert "invalid audio format" in push_error.await_args.kwargs["error_msg"]
    # No fabricated exception: an exception that was never raised has no
    # traceback, which push_error_frame's file/line reporting can't handle.
    assert "exception" not in push_error.await_args.kwargs


def test_language_from_code_prefers_message_language():
    service = ModulateSTTService(api_key="test-key", sample_rate=16000)

    assert service._language_from_code("fr") == Language.FR
    assert service._language_from_code("klingon") is None
    assert service._language_from_code(None) is None


def test_language_from_code_falls_back_to_configured_language():
    service = ModulateSTTService(
        api_key="test-key",
        sample_rate=16000,
        settings=ModulateSTTService.Settings(language=Language.DE),
    )

    assert service._language_from_code(None) == Language.DE
    assert service._language_from_code("fr") == Language.FR


@pytest.mark.asyncio
async def test_send_end_of_stream_sends_empty_text_frame():
    service = ModulateSTTService(api_key="test-key", sample_rate=16000)
    websocket = _FakeWebsocket()
    service._websocket = websocket

    await service._send_end_of_stream()

    websocket.send.assert_awaited_once_with("")


@pytest.mark.asyncio
async def test_send_end_of_stream_skips_closed_websocket():
    service = ModulateSTTService(api_key="test-key", sample_rate=16000)
    websocket = _FakeWebsocket(state=State.CLOSED)
    service._websocket = websocket

    await service._send_end_of_stream()

    websocket.send.assert_not_awaited()
