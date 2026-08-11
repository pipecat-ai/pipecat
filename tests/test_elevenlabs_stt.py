#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from urllib.parse import parse_qs, urlparse

import aiohttp
import pytest
from aiohttp import web

from pipecat.frames.frames import (
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.elevenlabs.stt import (
    CommitStrategy,
    ElevenLabsRealtimeSTTService,
    ElevenLabsSTTService,
    audio_format_from_sample_rate,
)
from pipecat.transcriptions.language import Language

COMMITTED_TEXT = "Hello. This is a test of the speech-to-text service."

PLAIN_COMMITTED_MESSAGE = {
    "message_type": "committed_transcript",
    "text": COMMITTED_TEXT,
}

TIMESTAMPED_COMMITTED_MESSAGE = {
    "message_type": "committed_transcript_with_timestamps",
    "text": COMMITTED_TEXT,
    "language_code": "en",
    "words": [{"text": "Hello.", "start": 0.0, "end": 0.5, "type": "word"}],
}

INTERIM_TEXT = "Hello. How's your day going?"

PARTIAL_MESSAGE = {
    "message_type": "partial_transcript",
    "text": INTERIM_TEXT,
}

COMMITTED_TRANSCRIPT_MESSAGE = {
    "message_type": "committed_transcript",
    "text": INTERIM_TEXT,
}

GROWN_PARTIAL_MESSAGE = {
    "message_type": "partial_transcript",
    "text": "Hello. How's your day going? Good so far.",
}

NEW_PARTIAL_MESSAGE = {
    "message_type": "partial_transcript",
    "text": "Good.",
}


def _capture_transcriptions(service: ElevenLabsRealtimeSTTService) -> list[TranscriptionFrame]:
    """Collect the TranscriptionFrames a service pushes."""
    captured: list[TranscriptionFrame] = []

    async def push_frame(frame, direction=None):
        if isinstance(frame, TranscriptionFrame):
            captured.append(frame)

    service.push_frame = push_frame
    return captured


def _capture_interims(service: ElevenLabsRealtimeSTTService) -> list[InterimTranscriptionFrame]:
    """Collect the InterimTranscriptionFrames a service pushes."""
    captured: list[InterimTranscriptionFrame] = []

    async def push_frame(frame, direction=None):
        if isinstance(frame, InterimTranscriptionFrame):
            captured.append(frame)

    service.push_frame = push_frame
    return captured


@pytest.mark.asyncio
async def test_elevenlabs_stt_sends_keyterms_multipart_fields(aiohttp_client):
    captured = {"headers": {}, "fields": []}

    async def handler(request):
        captured["headers"]["xi-api-key"] = request.headers.get("xi-api-key")
        reader = await request.multipart()

        async for part in reader:
            if part.name == "file":
                await part.read()
            else:
                captured["fields"].append((part.name, await part.text()))

        return web.json_response({"text": "hello", "language_code": "eng", "words": []})

    app = web.Application()
    app.router.add_post("/v1/speech-to-text", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("/")).rstrip("/")

    async with aiohttp.ClientSession() as session:
        service = ElevenLabsSTTService(
            api_key="test-key",
            aiohttp_session=session,
            base_url=base_url,
            settings=ElevenLabsSTTService.Settings(
                language=Language.EN,
                keyterms=["Pipecat", "Scribe V2"],
            ),
        )

        result = await service._transcribe_audio(b"RIFF")

    assert result["text"] == "hello"
    assert captured["headers"]["xi-api-key"] == "test-key"
    assert ("model_id", "scribe_v2") in captured["fields"]
    assert ("language_code", "eng") in captured["fields"]
    assert [value for name, value in captured["fields"] if name == "keyterms"] == [
        "Pipecat",
        "Scribe V2",
    ]


@pytest.mark.asyncio
async def test_elevenlabs_realtime_websocket_url_includes_keyterms(monkeypatch):
    captured = {}

    async def fake_websocket_connect(url, *, additional_headers, **kwargs):
        captured["url"] = url
        captured["headers"] = additional_headers
        return object()

    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect",
        fake_websocket_connect,
    )

    service = ElevenLabsRealtimeSTTService(
        api_key="test-key",
        base_url="example.test",
        commit_strategy=CommitStrategy.VAD,
        sample_rate=16000,
        include_timestamps=True,
        settings=ElevenLabsRealtimeSTTService.Settings(
            language=Language.EN,
            keyterms=["Pipecat", "Scribe V2"],
            vad_threshold=0.7,
        ),
    )
    service._audio_format = audio_format_from_sample_rate(16000)

    await service._connect_websocket()

    parsed = urlparse(captured["url"])
    query = parse_qs(parsed.query)
    assert parsed.scheme == "wss"
    assert parsed.netloc == "example.test"
    assert parsed.path == "/v1/speech-to-text/realtime"
    assert query["model_id"] == ["scribe_v2_realtime"]
    assert query["language_code"] == ["en"]
    assert query["audio_format"] == ["pcm_16000"]
    assert query["commit_strategy"] == ["vad"]
    assert query["include_timestamps"] == ["true"]
    assert query["vad_threshold"] == ["0.7"]
    assert query["keyterms"] == ["Pipecat", "Scribe V2"]
    assert captured["headers"] == {"xi-api-key": "test-key"}


@pytest.mark.asyncio
async def test_elevenlabs_realtime_websocket_url_includes_filter_background_audio(monkeypatch):
    captured = {}

    async def fake_websocket_connect(url, *, additional_headers, **kwargs):
        captured["url"] = url
        return object()

    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect",
        fake_websocket_connect,
    )

    # Background filtering applies under either commit strategy, unlike the VAD tuning params.
    service = ElevenLabsRealtimeSTTService(
        api_key="test-key",
        base_url="example.test",
        commit_strategy=CommitStrategy.MANUAL,
        sample_rate=16000,
        settings=ElevenLabsRealtimeSTTService.Settings(filter_background_audio=True),
    )
    service._audio_format = audio_format_from_sample_rate(16000)

    await service._connect_websocket()

    query = parse_qs(urlparse(captured["url"]).query)
    assert query["commit_strategy"] == ["manual"]
    assert query["filter_background_audio"] == ["true"]


@pytest.mark.asyncio
async def test_elevenlabs_realtime_websocket_url_omits_unset_filter_background_audio(monkeypatch):
    captured = {}

    async def fake_websocket_connect(url, *, additional_headers, **kwargs):
        captured["url"] = url
        return object()

    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect",
        fake_websocket_connect,
    )

    service = ElevenLabsRealtimeSTTService(
        api_key="test-key",
        base_url="example.test",
        sample_rate=16000,
    )
    service._audio_format = audio_format_from_sample_rate(16000)

    await service._connect_websocket()

    query = parse_qs(urlparse(captured["url"]).query)
    assert "filter_background_audio" not in query


@pytest.mark.asyncio
async def test_elevenlabs_realtime_language_detection_emits_single_final():
    """Language detection turns on the timestamped message, which alone carries language."""
    service = ElevenLabsRealtimeSTTService(
        api_key="test-key",
        sample_rate=16000,
        include_language_detection=True,
    )
    captured = _capture_transcriptions(service)

    # The server sends the timestamped message first in this configuration.
    await service._process_response(TIMESTAMPED_COMMITTED_MESSAGE)
    await service._process_response(PLAIN_COMMITTED_MESSAGE)

    assert len(captured) == 1
    assert captured[0].text == COMMITTED_TEXT
    assert captured[0].language == "en"


@pytest.mark.asyncio
async def test_elevenlabs_realtime_timestamps_emits_single_final():
    service = ElevenLabsRealtimeSTTService(
        api_key="test-key",
        sample_rate=16000,
        include_timestamps=True,
    )
    captured = _capture_transcriptions(service)

    await service._process_response(PLAIN_COMMITTED_MESSAGE)
    await service._process_response(TIMESTAMPED_COMMITTED_MESSAGE)

    assert len(captured) == 1
    assert captured[0].text == COMMITTED_TEXT


@pytest.mark.asyncio
async def test_elevenlabs_realtime_both_options_emit_single_final():
    service = ElevenLabsRealtimeSTTService(
        api_key="test-key",
        sample_rate=16000,
        include_timestamps=True,
        include_language_detection=True,
    )
    captured = _capture_transcriptions(service)

    await service._process_response(PLAIN_COMMITTED_MESSAGE)
    await service._process_response(TIMESTAMPED_COMMITTED_MESSAGE)

    assert len(captured) == 1
    assert captured[0].language == "en"


@pytest.mark.asyncio
async def test_elevenlabs_realtime_plain_committed_emitted_without_options():
    """Without either option the server sends only the plain message, so it must be emitted."""
    service = ElevenLabsRealtimeSTTService(
        api_key="test-key",
        sample_rate=16000,
    )
    captured = _capture_transcriptions(service)

    await service._process_response(PLAIN_COMMITTED_MESSAGE)

    assert len(captured) == 1
    assert captured[0].text == COMMITTED_TEXT
    assert captured[0].language is None


@pytest.mark.asyncio
async def test_elevenlabs_realtime_repeated_partial_is_suppressed():
    """An identical repeat of the last partial carries no new information."""
    service = ElevenLabsRealtimeSTTService(api_key="test-key", sample_rate=16000)
    captured = _capture_interims(service)

    await service._process_response(PARTIAL_MESSAGE)
    await service._process_response(PARTIAL_MESSAGE)

    assert len(captured) == 1


@pytest.mark.asyncio
async def test_elevenlabs_realtime_post_commit_echo_partial_is_suppressed():
    """The server echoes just-committed text as a stale partial shortly after commit."""
    service = ElevenLabsRealtimeSTTService(api_key="test-key", sample_rate=16000)
    captured = _capture_interims(service)

    await service._process_response(PARTIAL_MESSAGE)
    await service._process_response(COMMITTED_TRANSCRIPT_MESSAGE)
    await service._process_response(PARTIAL_MESSAGE)

    assert len(captured) == 1


@pytest.mark.asyncio
async def test_elevenlabs_realtime_new_partial_after_commit_is_emitted():
    """Genuinely new text after a commit is a new user turn and must pass through."""
    service = ElevenLabsRealtimeSTTService(api_key="test-key", sample_rate=16000)
    captured = _capture_interims(service)

    await service._process_response(PARTIAL_MESSAGE)
    await service._process_response(COMMITTED_TRANSCRIPT_MESSAGE)
    await service._process_response(NEW_PARTIAL_MESSAGE)

    assert [f.text for f in captured] == [INTERIM_TEXT, NEW_PARTIAL_MESSAGE["text"]]


@pytest.mark.asyncio
async def test_elevenlabs_realtime_changed_partial_text_is_emitted():
    """Partial progression within a turn (text grows/changes) must pass through."""
    service = ElevenLabsRealtimeSTTService(api_key="test-key", sample_rate=16000)
    captured = _capture_interims(service)

    await service._process_response(PARTIAL_MESSAGE)
    await service._process_response(GROWN_PARTIAL_MESSAGE)

    assert [f.text for f in captured] == [INTERIM_TEXT, GROWN_PARTIAL_MESSAGE["text"]]


@pytest.mark.asyncio
async def test_elevenlabs_realtime_partial_state_resets_on_reconnect(monkeypatch):
    """A reconnect must not carry stale partial/committed text into the new session."""

    async def fake_websocket_connect(url, *, additional_headers, **kwargs):
        return object()

    monkeypatch.setattr(
        "pipecat.services.websocket_service.websocket_connect",
        fake_websocket_connect,
    )

    service = ElevenLabsRealtimeSTTService(api_key="test-key", sample_rate=16000)
    service._audio_format = audio_format_from_sample_rate(16000)
    service.create_task = lambda coro, name=None: coro.close()

    captured = _capture_interims(service)

    await service._process_response(PARTIAL_MESSAGE)
    await service._process_response(COMMITTED_TRANSCRIPT_MESSAGE)
    assert len(captured) == 1

    await service._connect()

    # The same text arrives fresh in the new session and must not be
    # suppressed as a stale echo from the previous one.
    await service._process_response(PARTIAL_MESSAGE)

    assert len(captured) == 2


@pytest.mark.asyncio
async def test_elevenlabs_realtime_genuine_repeat_after_vad_start_is_emitted():
    """A VAD-detected turn start clears the markers, so a real repeat utterance fires."""
    service = ElevenLabsRealtimeSTTService(api_key="test-key", sample_rate=16000)
    captured = _capture_interims(service)

    # Turn 1: user says "Hello. How's your day going?"
    await service._process_response(PARTIAL_MESSAGE)
    await service._process_response(COMMITTED_TRANSCRIPT_MESSAGE)

    # Turn 2 begins: VAD detects real speech, not a phantom re-send.
    await service.process_frame(VADUserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)

    # The user genuinely says the same thing again.
    await service._process_response(PARTIAL_MESSAGE)

    assert [f.text for f in captured] == [INTERIM_TEXT, INTERIM_TEXT]


@pytest.mark.asyncio
async def test_elevenlabs_realtime_post_commit_echo_without_vad_start_still_suppressed():
    """Without a VAD-detected turn start, the post-commit echo is still a stale re-send."""
    service = ElevenLabsRealtimeSTTService(api_key="test-key", sample_rate=16000)
    captured = _capture_interims(service)

    await service._process_response(PARTIAL_MESSAGE)
    await service._process_response(COMMITTED_TRANSCRIPT_MESSAGE)

    # No VADUserStartedSpeakingFrame here -- this is the phantom post-commit echo.
    await service._process_response(PARTIAL_MESSAGE)

    assert len(captured) == 1
