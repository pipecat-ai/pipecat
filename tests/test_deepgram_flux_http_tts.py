#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for DeepgramFluxHttpTTSService."""

import asyncio

import aiohttp
import pytest
from aiohttp import web

from pipecat.frames.frames import (
    ErrorFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.pipeline.task import PipelineParams
from pipecat.services.deepgram.flux.tts import DeepgramFluxHttpTTSService
from pipecat.tests.utils import run_test

PCM_AUDIO = b"\x00\x01\x02\x03" * 1024


async def _flux_http_client(aiohttp_client, captured: list[dict]):
    """Start a fake Flux TTS batch endpoint that records each request it gets."""

    async def handler(request):
        captured.append(
            {
                "query": request.query,
                "auth_header": request.headers.get("Authorization"),
                "body": await request.json(),
            }
        )
        response = web.StreamResponse(headers={"Content-Type": "audio/l16;rate=24000"})
        await response.prepare(request)
        # Split mid-sample, the way an HTTP stream may.
        await response.write(PCM_AUDIO[:2047])
        await asyncio.sleep(0.01)
        await response.write(PCM_AUDIO[2047:])
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v2/speak", handler)
    return await aiohttp_client(app)


@pytest.mark.asyncio
async def test_flux_http_tts_requests_raw_pcm_from_v2_speak(aiohttp_client):
    """The text goes to /v2/speak and the audio comes back as whole 16-bit samples."""
    captured: list[dict] = []
    client = await _flux_http_client(aiohttp_client, captured)

    async with aiohttp.ClientSession() as session:
        tts_service = DeepgramFluxHttpTTSService(
            api_key="test-key",
            aiohttp_session=session,
            base_url=str(client.make_url("")).rstrip("/"),
            sample_rate=24000,
            settings=DeepgramFluxHttpTTSService.Settings(voice="flux-anjali-hi"),
        )

        # Metrics on: the TTFA scan reads the audio as 16-bit samples.
        down_frames, up_frames = await run_test(
            tts_service,
            frames_to_send=[TTSSpeakFrame(text="Hello from Flux.")],
            pipeline_params=PipelineParams(enable_metrics=True),
        )

    assert not any(isinstance(frame, ErrorFrame) for frame in list(down_frames) + list(up_frames))

    assert len(captured) == 1
    request = captured[0]
    assert request["auth_header"] == "Token test-key"
    assert request["body"] == {"text": "Hello from Flux."}
    assert dict(request["query"]) == {
        "model": "flux-anjali-hi",
        "encoding": "linear16",
        "sample_rate": "24000",
        "container": "none",
    }

    audio_frames = [frame for frame in down_frames if isinstance(frame, TTSAudioRawFrame)]
    assert audio_frames
    assert all(frame.sample_rate == 24000 for frame in audio_frames)
    assert all(len(frame.audio) % 2 == 0 for frame in audio_frames)
    assert b"".join(frame.audio for frame in audio_frames) == PCM_AUDIO


@pytest.mark.asyncio
async def test_flux_http_tts_sends_settings_as_query_parameters(aiohttp_client):
    captured: list[dict] = []
    client = await _flux_http_client(aiohttp_client, captured)

    async with aiohttp.ClientSession() as session:
        tts_service = DeepgramFluxHttpTTSService(
            api_key="test-key",
            aiohttp_session=session,
            base_url=str(client.make_url("")).rstrip("/"),
            sample_rate=16000,
            mip_opt_out=True,
            tag=["evals", "caller voice"],
            settings=DeepgramFluxHttpTTSService.Settings(
                voice="flux-alexis-en",
                speed=1.1,
                expressivity=-1,
                normalize_volume=False,
            ),
        )

        await run_test(tts_service, frames_to_send=[TTSSpeakFrame(text="Hello.")])

    query = captured[0]["query"]
    assert query["model"] == "flux-alexis-en"
    assert query["sample_rate"] == "16000"
    assert query["speed"] == "1.1"
    assert query["expressivity"] == "-1"
    assert query["normalize_volume"] == "false"
    assert query["mip_opt_out"] == "true"
    assert query.getall("tag") == ["evals", "caller voice"]


@pytest.mark.asyncio
async def test_flux_http_tts_settings_update_applies_to_the_next_request(aiohttp_client):
    captured: list[dict] = []
    client = await _flux_http_client(aiohttp_client, captured)

    async with aiohttp.ClientSession() as session:
        tts_service = DeepgramFluxHttpTTSService(
            api_key="test-key",
            aiohttp_session=session,
            base_url=str(client.make_url("")).rstrip("/"),
            sample_rate=24000,
        )
        assert tts_service._settings.model == "flux-heather-en"

        await run_test(
            tts_service,
            frames_to_send=[
                TTSSpeakFrame(text="First."),
                TTSUpdateSettingsFrame(
                    delta=DeepgramFluxHttpTTSService.Settings(voice="flux-anjali-hi", speed=0.9)
                ),
                TTSSpeakFrame(text="Second."),
            ],
        )

    assert [request["query"]["model"] for request in captured] == [
        "flux-heather-en",
        "flux-anjali-hi",
    ]
    assert "speed" not in captured[0]["query"]
    assert captured[1]["query"]["speed"] == "0.9"
    # Deepgram's `model` parameter is the voice, so metrics report the new voice.
    assert tts_service._settings.model == "flux-anjali-hi"


@pytest.mark.asyncio
async def test_flux_http_tts_reports_a_refused_request(aiohttp_client):
    """Deepgram's reason for refusing a request reaches the error frame."""
    refusal = (
        '{"err_code":"EXPRESSIVITY_OUT_OF_RANGE",'
        '"err_msg":"\'expressivity\' must be between -2 and 2. Got: 5."}'
    )

    async def handler(request):
        return web.Response(status=400, text=refusal, content_type="application/json")

    app = web.Application()
    app.router.add_post("/v2/speak", handler)
    client = await aiohttp_client(app)

    async with aiohttp.ClientSession() as session:
        tts_service = DeepgramFluxHttpTTSService(
            api_key="test-key",
            aiohttp_session=session,
            base_url=str(client.make_url("")).rstrip("/"),
            sample_rate=24000,
        )

        down_frames, up_frames = await run_test(
            tts_service, frames_to_send=[TTSSpeakFrame(text="Hello.")]
        )

    errors = [f.error for f in list(down_frames) + list(up_frames) if isinstance(f, ErrorFrame)]
    assert f"Error getting audio: HTTP 400: {refusal}" in errors
    assert not any(isinstance(frame, TTSAudioRawFrame) for frame in down_frames)
