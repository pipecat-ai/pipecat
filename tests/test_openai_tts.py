#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Request and audio handling for OpenAI-compatible TTS endpoints."""

import json

import pytest
from openai import DefaultAsyncHttpxClient

from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame, TTSSpeakFrame
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.tests.utils import run_test
from tests.openai_http_helpers import http


@pytest.mark.asyncio
@pytest.mark.parametrize("voice", ["alloy", "Magpie-Multilingual.EN-US.Aria"])
async def test_sends_builtin_and_custom_voices_to_configured_endpoint(voice):
    requests = []
    pcm = b"\x01\x00" * 32

    async def handle(request):
        requests.append(request)
        return http.Response(200, content=pcm, headers={"Content-Type": "audio/pcm"})

    async with DefaultAsyncHttpxClient(transport=http.MockTransport(handle)) as client:
        service = OpenAITTSService(
            api_key="test-key",
            base_url="http://tts.example/v1",
            http_client=client,
            sample_rate=24000,
            stop_frame_timeout_s=0.01,
            settings=OpenAITTSService.Settings(model="custom-tts", voice=voice),
        )
        down, up = await run_test(service, frames_to_send=[TTSSpeakFrame("Hello.")])

    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert str(requests[0].url) == "http://tts.example/v1/audio/speech"
    assert json.loads(requests[0].content) == {
        "input": "Hello.",
        "model": "custom-tts",
        "voice": voice,
        "response_format": "pcm",
    }
    audio = [frame for frame in down if isinstance(frame, TTSAudioRawFrame)]
    assert audio
    assert b"".join(frame.audio for frame in audio) == pcm
    assert all(frame.sample_rate == 24000 and frame.num_channels == 1 for frame in audio)
    assert not any(isinstance(frame, ErrorFrame) for frame in up)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "voice, expected_error, request_count",
    [
        (None, "OpenAI TTS voice must be specified", 0),
        ("unsupported-voice", "Provider rejected the voice", 1),
    ],
)
async def test_reports_missing_voice_and_provider_validation_errors(
    voice, expected_error, request_count
):
    requests = []

    async def handle(request):
        requests.append(request)
        return http.Response(
            400,
            json={
                "error": {"message": "Provider rejected the voice", "type": "invalid_request_error"}
            },
        )

    async with DefaultAsyncHttpxClient(transport=http.MockTransport(handle)) as client:
        service = OpenAITTSService(
            api_key="test-key",
            base_url="http://tts.example/v1",
            http_client=client,
            stop_frame_timeout_s=0.01,
            settings=OpenAITTSService.Settings(voice=voice),
        )
        down, up = await run_test(service, frames_to_send=[TTSSpeakFrame("Hello.")])

    assert len(requests) == request_count
    assert any(expected_error in frame.error for frame in up if isinstance(frame, ErrorFrame))
    assert not any(isinstance(frame, TTSAudioRawFrame) for frame in down)
