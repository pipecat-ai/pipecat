#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Gemini Live STT transcription handling and config building."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from google.genai.types import Modality

from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.services.google.gemini_live.stt import GeminiSTTService
from pipecat.transcriptions.language import Language


def make_service(**settings_kwargs):
    service = object.__new__(GeminiSTTService)
    service._name = "GeminiSTTService#0"
    service._user_id = "user"
    service._settings = GeminiSTTService.Settings(
        model="gemini-3.5-transcribe-live",
        language=None,
        languages=[],
        language_auto=None,
        adaptation_phrases=None,
    )
    service._settings.apply_update(GeminiSTTService.Settings(**settings_kwargs))

    service._session = None
    service._sample_rate = 16000

    frames = []
    transcriptions = []

    async def push_frame(frame):
        frames.append(frame)

    async def stop_processing_metrics():
        pass

    async def handle_transcription(transcript, is_final, language=None):
        transcriptions.append((transcript, is_final, language))

    service.push_frame = push_frame
    service.stop_processing_metrics = stop_processing_metrics
    service._handle_transcription = handle_transcription

    return service, frames, transcriptions


def interim_message(*, text: str, language_code: str | None = None):
    transcription = SimpleNamespace(text=text, language_code=language_code)
    return SimpleNamespace(
        server_content=SimpleNamespace(
            interim_input_transcription=transcription, input_transcription=None
        )
    )


def final_message(*, text: str, language_code: str | None = None):
    transcription = SimpleNamespace(text=text, language_code=language_code)
    return SimpleNamespace(
        server_content=SimpleNamespace(
            interim_input_transcription=None, input_transcription=transcription
        )
    )


@pytest.mark.asyncio
async def test_interim_transcriptions_emit_interim_frames():
    service, frames, transcriptions = make_service()

    for message in [
        interim_message(text="hello"),
        interim_message(text="hello there"),
    ]:
        await service._handle_server_message(message)

    assert all(isinstance(f, InterimTranscriptionFrame) for f in frames)
    assert [f.text for f in frames] == ["hello", "hello there"]
    assert transcriptions == []


@pytest.mark.asyncio
async def test_input_transcription_emits_finalized_frame():
    service, frames, transcriptions = make_service()

    for message in [
        interim_message(text="hello"),
        final_message(text="hello there"),
        final_message(text="again"),
    ]:
        await service._handle_server_message(message)

    assert isinstance(frames[0], InterimTranscriptionFrame)
    assert isinstance(frames[1], TranscriptionFrame)
    assert frames[1].text == "hello there"
    assert frames[1].finalized is True
    assert isinstance(frames[2], TranscriptionFrame)
    assert frames[2].text == "again"
    assert transcriptions == [("hello there", True, None), ("again", True, None)]


@pytest.mark.asyncio
async def test_empty_and_missing_transcriptions_are_ignored():
    service, frames, _ = make_service()

    await service._handle_server_message(interim_message(text=""))
    await service._handle_server_message(final_message(text=""))
    await service._handle_server_message(SimpleNamespace(server_content=None))

    assert frames == []


@pytest.mark.asyncio
async def test_language_code_mapping():
    service, frames, transcriptions = make_service()

    await service._handle_server_message(final_message(text="hola", language_code="es-ES"))
    await service._handle_server_message(final_message(text="??", language_code="not-a-language"))

    assert frames[0].language == Language.ES_ES
    assert frames[1].language is None
    assert transcriptions == [("hola", True, Language.ES_ES), ("??", True, None)]


def test_build_live_config_defaults_to_language_auto():
    service, _, _ = make_service()

    config = service._build_live_config()

    assert config.response_modalities == [Modality.TEXT]
    assert config.input_audio_transcription.language_auto is not None
    assert config.input_audio_transcription.language_hints is None
    assert config.input_audio_transcription.adaptation_phrases is None


def test_build_live_config_languages_become_hints():
    service, _, _ = make_service(languages=[Language.ES_ES, Language.EN_US])

    config = service._build_live_config()

    assert config.input_audio_transcription.language_auto is None
    assert config.input_audio_transcription.language_hints.language_codes == ["es-ES", "en-US"]


def test_build_live_config_single_language_becomes_hint():
    service, _, _ = make_service(language="fr-FR")

    config = service._build_live_config()

    assert config.input_audio_transcription.language_auto is None
    assert config.input_audio_transcription.language_hints.language_codes == ["fr-FR"]


def test_build_live_config_hints_win_over_explicit_auto():
    service, _, _ = make_service(languages=[Language.ES_ES], language_auto=True)

    config = service._build_live_config()

    assert config.input_audio_transcription.language_auto is None
    assert config.input_audio_transcription.language_hints.language_codes == ["es-ES"]


def test_build_live_config_auto_disabled_without_hints():
    service, _, _ = make_service(language_auto=False)

    config = service._build_live_config()

    assert config.input_audio_transcription.language_auto is None
    assert config.input_audio_transcription.language_hints is None


def test_build_live_config_adaptation_phrases():
    service, _, _ = make_service(adaptation_phrases=["oatmilk"])

    config = service._build_live_config()

    assert config.input_audio_transcription.adaptation_phrases == ["oatmilk"]


def test_build_live_config_keeps_automatic_activity_detection():
    service, _, _ = make_service()

    assert service._build_live_config().realtime_input_config is None


def make_session(service):
    """Attach a fake live session, returning the calls it records."""
    calls = []

    async def send_realtime_input(**kwargs):
        calls.append(kwargs)

    service._session = SimpleNamespace(send_realtime_input=send_realtime_input)
    return calls


async def drain(generator):
    async for _ in generator:
        pass


@pytest.mark.asyncio
async def test_streams_audio_continuously():
    service, _, _ = make_service()
    calls = make_session(service)

    await drain(service.run_stt(b"\x01\x02"))
    await drain(service.run_stt(b"\x03\x04"))

    assert [c["audio"].data for c in calls] == [b"\x01\x02", b"\x03\x04"]


@pytest.mark.asyncio
async def test_finalization_signal_sends_audio_stream_end():
    service, _, _ = make_service()
    calls = make_session(service)

    await drain(service.run_stt(b"speech"))
    await service._send_finalization_signal()
    await drain(service.run_stt(b"more speech"))

    assert calls[0]["audio"].data == b"speech"
    assert calls[1] == {"audio_stream_end": True}
    assert calls[2]["audio"].data == b"more speech"


@pytest.mark.asyncio
async def test_finalization_signal_without_session_is_a_noop():
    service, _, _ = make_service()

    await service._send_finalization_signal()


def make_failing_session(service):
    """Attach a live session whose sends fail, returning the attempts it records."""
    attempts = []

    async def send_realtime_input(**kwargs):
        attempts.append(kwargs)
        raise RuntimeError("gone")

    service._session = SimpleNamespace(send_realtime_input=send_realtime_input)
    return attempts


@pytest.mark.asyncio
async def test_finalization_signal_failure_reconnects():
    service, _, _ = make_service()
    service._request_reconnect = AsyncMock()
    make_failing_session(service)

    await service._send_finalization_signal()

    assert service._session is None
    service._request_reconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_audio_send_failure_reconnects():
    service, _, _ = make_service()
    service._request_reconnect = AsyncMock()
    make_failing_session(service)

    await drain(service.run_stt(b"speech"))

    assert service._session is None
    service._request_reconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_audio_after_a_send_failure_is_not_retried_on_the_dead_session():
    service, _, _ = make_service()
    service._request_reconnect = AsyncMock()
    attempts = make_failing_session(service)

    await drain(service.run_stt(b"one"))
    await drain(service.run_stt(b"two"))

    assert len(attempts) == 1
    service._request_reconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_settings_requests_reconnect_only_on_change():
    from pipecat.services.stt_service import STTService

    service, _, _ = make_service()
    service._request_reconnect = AsyncMock()

    with patch.object(STTService, "_update_settings", AsyncMock(return_value={"languages": []})):
        await service._update_settings(GeminiSTTService.Settings(languages=[Language.FR]))
    service._request_reconnect.assert_awaited_once()

    service._request_reconnect.reset_mock()
    with patch.object(STTService, "_update_settings", AsyncMock(return_value={})):
        await service._update_settings(GeminiSTTService.Settings())
    service._request_reconnect.assert_not_awaited()
