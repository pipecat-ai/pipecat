#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Inworld speech-to-text service."""

import base64

import aiohttp
import pytest
from aiohttp import web

from pipecat.frames.frames import ErrorFrame, TranscriptionFrame
from pipecat.services.inworld.frames import InworldVoiceProfile, InworldVoiceProfileFrame
from pipecat.services.inworld.stt import (
    InworldSTTService,
    language_to_inworld_stt_language,
)
from pipecat.transcriptions.language import Language


@pytest.mark.asyncio
async def test_inworld_stt_sends_documented_request_and_emits_transcription(aiohttp_client):
    """The service should send WAV audio and expose the Inworld result on the frame."""
    captured = {}

    async def handler(request):
        captured["headers"] = request.headers
        captured["body"] = await request.json()
        return web.json_response(
            {
                "transcription": {
                    "transcript": "  Hello from Inworld.  ",
                    "isFinal": True,
                    "wordTimestamps": [],
                },
                "voiceProfile": {
                    "age": [{"label": "adult", "confidence": 0.88}],
                    "emotion": [{"label": "calm", "confidence": 0.91}],
                    "pitch": [{"label": "medium", "confidence": 0.82}],
                    "vocalStyle": [{"label": "normal", "confidence": 0.95}],
                    "accent": [{"label": "en-US", "confidence": 0.76}],
                },
                "usage": {
                    "transcribedAudioMs": 1200,
                    "modelId": "inworld/inworld-stt-1",
                },
            }
        )

    app = web.Application()
    app.router.add_post("/stt/v1/transcribe", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("/")).rstrip("/")
    audio = b"RIFF-test-wav"

    async with aiohttp.ClientSession() as session:
        service = InworldSTTService(
            api_key="encoded-key",
            aiohttp_session=session,
            base_url=base_url,
            settings=InworldSTTService.Settings(
                language=Language.EN_US,
                prompts=["Pipecat", "Inworld"],
                enable_voice_profile=True,
                voice_profile_top_n=3,
            ),
        )
        service._sample_rate = 16000
        service._user_id = "user-1"

        frames = [frame async for frame in service.run_stt(audio)]

    assert captured["headers"]["Authorization"] == "Basic encoded-key"
    assert captured["headers"]["X-User-Agent"].startswith("pipecat/")
    assert captured["headers"]["X-Request-Id"]
    assert captured["body"] == {
        "transcribeConfig": {
            "modelId": "inworld/inworld-stt-1",
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": 16000,
            "numberOfChannels": 1,
            "language": "en",
            "prompts": ["Pipecat", "Inworld"],
            "voiceProfileConfig": {
                "enableVoiceProfile": True,
                "topN": 3,
            },
        },
        "audioData": {"content": base64.b64encode(audio).decode("ascii")},
    }

    assert len(frames) == 2
    assert isinstance(frames[0], InworldVoiceProfileFrame)
    assert frames[0].user_id == "user-1"
    assert frames[0].voice_profile.emotion[0].label == "calm"
    assert frames[0].voice_profile.emotion[0].confidence == 0.91
    assert frames[0].voice_profile.vocal_style[0].label == "normal"

    assert isinstance(frames[1], TranscriptionFrame)
    assert frames[1].text == "Hello from Inworld."
    assert frames[1].user_id == "user-1"
    assert frames[1].language == Language.EN
    assert frames[1].result is not None
    assert frames[1].result["usage"]["transcribedAudioMs"] == 1200


@pytest.mark.asyncio
async def test_inworld_stt_omits_optional_settings_when_auto_detecting(aiohttp_client):
    """Auto-detection requests should omit language and empty prompts."""
    request_bodies = []

    async def handler(request):
        request_bodies.append(await request.json())
        return web.json_response({"transcription": {"transcript": ""}})

    app = web.Application()
    app.router.add_post("/stt/v1/transcribe", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("/")).rstrip("/")

    async with aiohttp.ClientSession() as session:
        service = InworldSTTService(
            api_key="encoded-key",
            aiohttp_session=session,
            base_url=base_url,
        )
        service._sample_rate = 8000

        frames = [frame async for frame in service.run_stt(b"RIFF")]

    assert frames == []
    config = request_bodies[0]["transcribeConfig"]
    assert config["sampleRateHertz"] == 8000
    assert "language" not in config
    assert "prompts" not in config


@pytest.mark.asyncio
async def test_inworld_stt_emits_error_frame_for_api_errors(aiohttp_client):
    """Provider errors should enter Pipecat's normal non-fatal error path."""

    async def handler(request):
        return web.json_response(
            {"code": 8, "message": "rate limit exceeded", "details": []},
            status=429,
        )

    app = web.Application()
    app.router.add_post("/stt/v1/transcribe", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("/")).rstrip("/")

    async with aiohttp.ClientSession() as session:
        service = InworldSTTService(
            api_key="encoded-key",
            aiohttp_session=session,
            base_url=base_url,
        )
        service._sample_rate = 16000

        frames = [frame async for frame in service.run_stt(b"RIFF")]

    assert len(frames) == 1
    assert isinstance(frames[0], ErrorFrame)
    assert frames[0].fatal is False
    assert "Inworld API error (429)" in frames[0].error
    assert "rate limit exceeded" in frames[0].error
    assert isinstance(frames[0].exception, RuntimeError)


def test_inworld_stt_language_mapping_uses_base_codes():
    """Regional variants should resolve to the base codes accepted by Inworld STT."""
    assert language_to_inworld_stt_language(Language.EN_GB) == "en"
    assert language_to_inworld_stt_language(Language.PT_BR) == "pt"
    assert language_to_inworld_stt_language(Language.ZH_TW) == "zh"
    assert language_to_inworld_stt_language(Language.FIL) == "fil"


def test_inworld_voice_profile_accepts_snake_case_fields():
    """Voice Profile models should accept both documented response naming styles."""
    profile = InworldVoiceProfile.model_validate(
        {
            "vocal_style": [{"label": "whispering", "confidence": 0.97}],
            "emotion": [{"label": "tender", "confidence": 0.84}],
        }
    )

    assert profile.vocal_style[0].label == "whispering"
    assert profile.emotion[0].confidence == 0.84
