#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for HeyGenHttpTTSService."""

import asyncio

import aiohttp
import pytest
from aiohttp import web

from pipecat.frames.frames import (
    ErrorFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.heygen.tts import (
    HEYGEN_DEFAULT_VOICE_ID,
    HeyGenHttpTTSService,
    language_to_heygen_language,
)
from pipecat.tests.utils import run_test
from pipecat.transcriptions.language import Language

STREAM_PATH = "/v3/models/audio/tts/stream"


def _make_app(requests: list, audio: bytes = b"", status: int = 200, error: str = ""):
    async def handler(request):
        requests.append({"headers": dict(request.headers), "body": await request.json()})
        if status != 200:
            return web.json_response({"error": {"message": error}}, status=status)

        response = web.StreamResponse(
            status=200, headers={"Content-Type": "audio/pcm", "X-Request-Id": "req-123"}
        )
        await response.prepare(request)
        # Split mid-sample to check that emitted frames stay sample-aligned.
        await response.write(audio[:2047])
        await asyncio.sleep(0.01)
        await response.write(audio[2047:])
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post(STREAM_PATH, handler)
    return app


@pytest.mark.asyncio
async def test_heygen_tts_streams_pcm(aiohttp_client):
    """The service sends the documented request and emits sample-aligned PCM frames."""
    requests = []
    pcm_audio = b"\x00\x01\x02\x03" * 1024
    client = await aiohttp_client(_make_app(requests, pcm_audio))

    async with aiohttp.ClientSession() as session:
        tts = HeyGenHttpTTSService(
            api_key="test-key",
            aiohttp_session=session,
            base_url=str(client.make_url("/")),
            sample_rate=24000,
        )
        down_frames, _ = await run_test(
            tts, frames_to_send=[TTSSpeakFrame(text="Hello from HeyGen.")]
        )

    frame_types = [type(f) for f in down_frames]
    assert TTSStartedFrame in frame_types
    assert TTSStoppedFrame in frame_types

    audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
    assert audio_frames
    assert all(f.sample_rate == 24000 and f.num_channels == 1 for f in audio_frames)
    assert all(len(f.audio) % 2 == 0 for f in audio_frames)
    assert b"".join(f.audio for f in audio_frames) == pcm_audio

    assert len(requests) == 1
    assert requests[0]["headers"]["x-api-key"] == "test-key"
    assert requests[0]["body"] == {
        "text": "Hello from HeyGen.",
        "voice_id": HEYGEN_DEFAULT_VOICE_ID,
        "model": "heygen-voice-1-turbo",
        "output_format": {"encoding": "pcm", "sample_rate_hertz": 24000},
    }


@pytest.mark.asyncio
async def test_heygen_tts_settings_in_body(aiohttp_client):
    """Voice, model and language settings reach the request body."""
    requests = []
    client = await aiohttp_client(_make_app(requests, b"\x00\x00" * 2048))

    async with aiohttp.ClientSession() as session:
        tts = HeyGenHttpTTSService(
            api_key="test-key",
            aiohttp_session=session,
            base_url=str(client.make_url("/")),
            sample_rate=16000,
            settings=HeyGenHttpTTSService.Settings(
                voice="my-voice", model="heygen-voice-1", language=Language.EN_US
            ),
        )
        await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi.")])

    body = requests[0]["body"]
    assert body["voice_id"] == "my-voice"
    assert body["model"] == "heygen-voice-1"
    assert body["language"] == "en"
    assert body["output_format"]["sample_rate_hertz"] == 16000


@pytest.mark.asyncio
async def test_heygen_tts_unsupported_sample_rate_is_resampled(aiohttp_client):
    """A pipeline rate HeyGen doesn't offer requests 24 kHz and resamples to the pipeline rate."""
    requests = []
    client = await aiohttp_client(_make_app(requests, b"\x00\x00" * 24000))

    async with aiohttp.ClientSession() as session:
        tts = HeyGenHttpTTSService(
            api_key="test-key",
            aiohttp_session=session,
            base_url=str(client.make_url("/")),
            sample_rate=32000,
        )
        down_frames, _ = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi.")])

    assert requests[0]["body"]["output_format"]["sample_rate_hertz"] == 24000
    audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
    assert audio_frames
    assert all(f.sample_rate == 32000 for f in audio_frames)


@pytest.mark.asyncio
async def test_heygen_tts_http_error(aiohttp_client):
    """A non-200 response produces an ErrorFrame with the status and message."""
    requests = []
    client = await aiohttp_client(_make_app(requests, status=429, error="Request limit reached."))

    async with aiohttp.ClientSession() as session:
        tts = HeyGenHttpTTSService(
            api_key="test-key",
            aiohttp_session=session,
            base_url=str(client.make_url("/")),
            sample_rate=24000,
        )
        _, up_frames = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Hi.")])

    errors = [f for f in up_frames if isinstance(f, ErrorFrame)]
    assert errors
    assert "429" in errors[0].error
    assert "Request limit reached." in errors[0].error


def test_language_mapping():
    """Regional variants resolve to HeyGen's base ISO 639-1 codes."""
    assert language_to_heygen_language(Language.EN) == "en"
    assert language_to_heygen_language(Language.EN_GB) == "en"
    assert language_to_heygen_language(Language.ZH_HK) == "zh"
    assert language_to_heygen_language(Language.PT_BR) == "pt"
