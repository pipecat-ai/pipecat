#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for PiperTTSService."""

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
    TTSTextFrame,
)
from pipecat.services.piper.tts import PiperTTSService
from pipecat.tests.utils import run_test


@pytest.mark.asyncio
async def test_run_piper_tts_success(aiohttp_client):
    """Test successful TTS generation with chunked audio data.

    Checks frames for TTSStartedFrame -> TTSAudioRawFrame -> TTSStoppedFrame.
    """

    async def handler(request):
        # The service expects a /?text= param
        # Here we're just returning dummy chunked bytes to simulate an audio response
        text_query = request.rel_url.query.get("text", "")
        print(f"Mock server received text param: {text_query}")

        # Prepare a StreamResponse with chunked data
        resp = web.StreamResponse(
            status=200,
            reason="OK",
            headers={"Content-Type": "audio/raw"},
        )
        await resp.prepare(request)

        # Write out some chunked byte data
        # In reality, you’d return WAV data or similar
        CHUNK_SIZE = 24000
        data_chunk_1 = b"\x00\x01\x02\x03" * CHUNK_SIZE  # 4xTTSAudioRawFrame
        data_chunk_2 = b"\x04\x05\x06\x07" * CHUNK_SIZE  # another chunk
        await resp.write(data_chunk_1)
        await asyncio.sleep(0.01)  # simulate async chunk delay
        await resp.write(data_chunk_2)
        await resp.write_eof()

        return resp

    # Create an aiohttp test server
    app = web.Application()
    app.router.add_post("/", handler)
    client = await aiohttp_client(app)

    # Remove trailing slash if present in the test URL
    base_url = str(client.make_url("")).rstrip("/")

    async with aiohttp.ClientSession() as session:
        # Instantiate PiperTTSService with our mock server
        tts_service = PiperTTSService(base_url=base_url, aiohttp_session=session, sample_rate=24000)

        frames_to_send = [
            TTSSpeakFrame(text="Hello world."),
        ]

        expected_returned_frames = [
            TTSStartedFrame,
            TTSAudioRawFrame,
            TTSAudioRawFrame,
            TTSAudioRawFrame,
            TTSAudioRawFrame,
            TTSAudioRawFrame,
            TTSAudioRawFrame,
            TTSAudioRawFrame,
            TTSAudioRawFrame,
            TTSStoppedFrame,
            TTSTextFrame,
        ]

        frames_received = await run_test(
            tts_service,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_returned_frames,
        )
        down_frames = frames_received[0]
        audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
        for a_frame in audio_frames:
            assert a_frame.sample_rate == 24000, "Sample rate should match the default (24000)"


@pytest.mark.asyncio
async def test_run_piper_tts_error(aiohttp_client):
    """Test how the service handles a non-200 response from the server.

    Expects an ErrorFrame to be returned.
    """

    async def handler(_request):
        # Return an error status for any request
        return web.Response(status=404, text="Not found")

    app = web.Application()
    app.router.add_post("/", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("")).rstrip("/")

    async with aiohttp.ClientSession() as session:
        tts_service = PiperTTSService(base_url=base_url, aiohttp_session=session, sample_rate=24000)

        frames_to_send = [
            TTSSpeakFrame(text="Error case."),
        ]

        expected_down_frames = [TTSStoppedFrame, TTSTextFrame]

        expected_up_frames = [ErrorFrame]

        frames_received = await run_test(
            tts_service,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )
        up_frames = frames_received[1]

        assert isinstance(up_frames[0], ErrorFrame), "Must receive an ErrorFrame for 404"
        assert "status: 404" in up_frames[0].error, (
            "ErrorFrame should contain details about the 404"
        )
