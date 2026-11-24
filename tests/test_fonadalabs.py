"""Tests for the FonadaLabs Pipecat services."""

import json

import pytest

from pipecat.frames.frames import ErrorFrame, TranscriptionFrame, TTSAudioRawFrame
from pipecat.services.fonadalabs import FonadaSTTService, FonadaTTSService


class FakeResponse:
    def __init__(self, status: int, body: str = "", chunks: list[bytes] | None = None):
        self.status = status
        self._body = body
        self.headers = {}
        self.content = self
        self._chunks = chunks or []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def text(self):
        return self._body

    async def iter_chunked(self, size):
        for chunk in self._chunks:
            yield chunk


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.responses)

    async def close(self):
        return None


@pytest.mark.asyncio
async def test_stt_emits_transcription_frame():
    session = FakeSession(
        [FakeResponse(200, json.dumps({"text": "नमस्ते", "transcript_chunks": []}))]
    )
    service = FonadaSTTService(api_key="secret", language="hi", aiohttp_session=session)

    frames = [frame async for frame in service.run_stt(b"wav")]

    assert len(frames) == 1
    assert isinstance(frames[0], TranscriptionFrame)
    assert frames[0].text == "नमस्ते"
    assert session.calls[0][0].endswith("/v3/asr/transcribe")
    await service.cleanup()


@pytest.mark.asyncio
async def test_stt_retries_transient_status():
    first = FakeResponse(502, "temporary")
    first.headers["Retry-After"] = "0"
    session = FakeSession([first, FakeResponse(200, json.dumps({"text": "hello"}))])
    service = FonadaSTTService(api_key="secret", aiohttp_session=session, max_retries=1)

    frames = [frame async for frame in service.run_stt(b"wav")]

    assert isinstance(frames[0], TranscriptionFrame)
    assert len(session.calls) == 2
    await service.cleanup()


@pytest.mark.asyncio
async def test_tts_emits_pcm_audio_frames():
    session = FakeSession([FakeResponse(200, chunks=[b"pcm-a", b"pcm-b"])])
    service = FonadaTTSService(
        api_key="secret", share_id="voice-1", language="en", aiohttp_session=session
    )

    frames = [frame async for frame in service.run_tts("hello", "context")]

    audio = [frame for frame in frames if isinstance(frame, TTSAudioRawFrame)]
    assert [frame.audio for frame in audio] == [b"pcm-a", b"pcm-b"]
    assert all(frame.sample_rate == 24000 and frame.num_channels == 1 for frame in audio)
    assert session.calls[0][1]["data"]["share_id"] == "voice-1"
    await service.cleanup()


@pytest.mark.asyncio
async def test_tts_reports_missing_voice():
    service = FonadaTTSService(api_key="secret", aiohttp_session=FakeSession([]))

    frames = [frame async for frame in service.run_tts("hello", "context")]

    assert isinstance(frames[0], ErrorFrame)
    assert "VOICE_SHARE_ID" in frames[0].error
    await service.cleanup()
