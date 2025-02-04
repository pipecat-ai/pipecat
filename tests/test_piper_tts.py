"""Tests for PiperTTSService."""

import asyncio

import pytest
from aiohttp import web

from pipecat.frames.frames import (
    ErrorFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.piper import PiperTTSService


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
        data_chunk_1 = b"\x00\x01\x02\x03" * 12000  # 48000 bytes
        data_chunk_2 = b"\x04\x05\x06\x07" * 6000  # another chunk
        await resp.write(data_chunk_1)
        await asyncio.sleep(0.01)  # simulate async chunk delay
        await resp.write(data_chunk_2)
        await resp.write_eof()

        return resp

    # Create an aiohttp test server
    app = web.Application()
    app.router.add_get("/", handler)
    client = await aiohttp_client(app)

    # Remove trailing slash if present in the test URL
    base_url = str(client.make_url("")).rstrip("/")

    # Instantiate PiperTTSService with our mock server
    tts_service = PiperTTSService(base_url=base_url)

    # Collect frames from the generator
    frames = []
    async for frame in tts_service.run_tts("Hello world."):
        frames.append(frame)

    # Ensure we received frames in the expected order/types
    assert len(frames) >= 3, "Expecting at least TTSStartedFrame, TTSAudioRawFrame, TTSStoppedFrame"
    assert isinstance(frames[0], TTSStartedFrame), "First frame must be TTSStartedFrame"
    assert isinstance(frames[-1], TTSStoppedFrame), "Last frame must be TTSStoppedFrame"

    # Check we have at least one TTSAudioRawFrame
    audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
    assert len(audio_frames) > 0, "Should have received at least one TTSAudioRawFrame"
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
    app.router.add_get("/", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("")).rstrip("/")

    tts_service = PiperTTSService(base_url=base_url)

    frames = []
    async for frame in tts_service.run_tts("Error case."):
        frames.append(frame)

    assert len(frames) == 1, "Should only receive a single ErrorFrame"
    assert isinstance(frames[0], ErrorFrame), "Must receive an ErrorFrame for 404"
    assert "status: 404" in frames[0].error, "ErrorFrame should contain details about the 404"
