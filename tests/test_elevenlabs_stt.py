#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from urllib.parse import parse_qs, urlparse

import aiohttp
import pytest
from aiohttp import web

from pipecat.frames.frames import TranscriptionFrame
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


def _capture_transcriptions(service: ElevenLabsRealtimeSTTService) -> list[TranscriptionFrame]:
    """Collect the TranscriptionFrames a service pushes."""
    captured: list[TranscriptionFrame] = []

    async def push_frame(frame, direction=None):
        if isinstance(frame, TranscriptionFrame):
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
