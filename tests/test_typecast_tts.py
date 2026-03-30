#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for TypecastTTSService."""

import io
import struct
import unittest
import wave

import aiohttp
import pytest
from aiohttp import web

from pipecat.frames.frames import (
    AggregatedTextFrame,
    ErrorFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.typecast.tts import TypecastTTSService
from pipecat.tests.utils import run_test


def _make_wav_bytes(sample_rate: int = 44100, duration_frames: int = 4410) -> bytes:
    """Generate a minimal valid WAV file in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        # Silent PCM frames
        wf.writeframes(struct.pack(f"<{duration_frames}h", *([0] * duration_frames)))
    return buf.getvalue()


@pytest.mark.asyncio
async def test_typecast_tts_success(aiohttp_client):
    """Test successful TTS: mock server returns WAV → expect Started/Audio/Stopped frames."""

    async def handler(request):
        body = await request.json()
        assert body["voice_id"] is not None
        assert body["text"] is not None
        assert body["output"]["audio_format"] == "wav"
        return web.Response(
            status=200,
            body=_make_wav_bytes(),
            content_type="audio/wav",
        )

    app = web.Application()
    app.router.add_post("/v1/text-to-speech", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("")).rstrip("/")

    async with aiohttp.ClientSession() as session:
        tts = TypecastTTSService(
            api_key="test-key",
            base_url=base_url,
            aiohttp_session=session,
            settings=TypecastTTSService.Settings(
                voice="tc_60e5426de8b95f1d3000d7b5",
                model="ssfm-v30",
            ),
        )

        frames_received = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="Hello world.")],
        )
        down_frames = frames_received[0]
        frame_types = [type(f) for f in down_frames]

        assert TTSStartedFrame in frame_types
        assert TTSAudioRawFrame in frame_types
        assert TTSStoppedFrame in frame_types
        assert TTSTextFrame in frame_types
        assert AggregatedTextFrame in frame_types

        started_idx = frame_types.index(TTSStartedFrame)
        text_idx = frame_types.index(TTSTextFrame)
        stopped_idx = frame_types.index(TTSStoppedFrame)
        assert started_idx < text_idx < stopped_idx, (
            "Expected: TTSStartedFrame < TTSTextFrame < TTSStoppedFrame"
        )

        audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
        assert len(audio_frames) >= 1
        for af in audio_frames:
            assert af.sample_rate == 44100
            assert af.num_channels == 1
            assert len(af.audio) > 0


@pytest.mark.asyncio
async def test_typecast_tts_with_emotion(aiohttp_client):
    """Test that prompt/output objects are sent correctly when emotion settings are used."""

    received_bodies = []

    async def handler(request):
        received_bodies.append(await request.json())
        return web.Response(
            status=200,
            body=_make_wav_bytes(),
            content_type="audio/wav",
        )

    app = web.Application()
    app.router.add_post("/v1/text-to-speech", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("")).rstrip("/")

    async with aiohttp.ClientSession() as session:
        tts = TypecastTTSService(
            api_key="test-key",
            base_url=base_url,
            aiohttp_session=session,
            settings=TypecastTTSService.Settings(
                voice="tc_60e5426de8b95f1d3000d7b5",
                model="ssfm-v30",
                emotion_type="preset",
                emotion_preset="happy",
                emotion_intensity=1.5,
                audio_tempo=1.2,
                audio_pitch=2,
                volume=120,
                seed=42,
            ),
        )

        await run_test(tts, frames_to_send=[TTSSpeakFrame(text="This is a test.")])

    assert len(received_bodies) == 1
    body = received_bodies[0]

    assert body["model"] == "ssfm-v30"
    assert body["output"]["audio_format"] == "wav"
    assert body["output"]["audio_tempo"] == 1.2
    assert body["output"]["audio_pitch"] == 2
    assert body["output"]["volume"] == 120
    assert body["seed"] == 42

    prompt = body["prompt"]
    assert prompt["emotion_type"] == "preset"
    assert prompt["emotion_preset"] == "happy"
    assert prompt["emotion_intensity"] == 1.5


@pytest.mark.asyncio
async def test_typecast_tts_error(aiohttp_client):
    """Test that a non-200 response produces an ErrorFrame."""

    async def handler(_request):
        return web.Response(status=401, text='{"error": "Unauthorized"}')

    app = web.Application()
    app.router.add_post("/v1/text-to-speech", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("")).rstrip("/")

    async with aiohttp.ClientSession() as session:
        tts = TypecastTTSService(
            api_key="bad-key",
            base_url=base_url,
            aiohttp_session=session,
            settings=TypecastTTSService.Settings(
                voice="tc_60e5426de8b95f1d3000d7b5",
                model="ssfm-v30",
            ),
        )

        frames_received = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="Error test.", append_to_context=False)],
            expected_down_frames=[
                AggregatedTextFrame,
                TTSStartedFrame,
                TTSTextFrame,
                TTSStoppedFrame,
            ],
            expected_up_frames=[ErrorFrame],
        )
        up_frames = frames_received[1]
        assert isinstance(up_frames[0], ErrorFrame)
        assert "Typecast API error" in up_frames[0].error


@pytest.mark.asyncio
async def test_typecast_tts_no_voice(aiohttp_client):
    """Test that missing voice_id produces an ErrorFrame without hitting the API."""

    called = []

    async def handler(request):
        called.append(True)
        return web.Response(status=200, body=_make_wav_bytes(), content_type="audio/wav")

    app = web.Application()
    app.router.add_post("/v1/text-to-speech", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("")).rstrip("/")

    async with aiohttp.ClientSession() as session:
        tts = TypecastTTSService(
            api_key="test-key",
            base_url=base_url,
            aiohttp_session=session,
            # no voice configured
            settings=TypecastTTSService.Settings(model="ssfm-v30"),
        )

        frames_received = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="no voice.", append_to_context=False)],
            expected_up_frames=[ErrorFrame],
        )
        up_frames = frames_received[1]
        assert isinstance(up_frames[0], ErrorFrame)
        assert "voice_id" in up_frames[0].error
        assert not called, "API should not be called when voice_id is missing"


if __name__ == "__main__":
    unittest.main()
