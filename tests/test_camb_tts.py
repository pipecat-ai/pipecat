#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for CambTTSService.

These tests use mock servers to simulate the Camb.ai API responses.
"""

import asyncio

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
from pipecat.services.camb.tts import CambTTSService, language_to_camb_language
from pipecat.tests.utils import run_test
from pipecat.transcriptions.language import Language


@pytest.mark.asyncio
async def test_run_camb_tts_success(aiohttp_client):
    """Test successful TTS generation with chunked PCM audio.

    Verifies the frame sequence: TTSStartedFrame -> TTSAudioRawFrame* -> TTSStoppedFrame
    """

    async def handler(request):
        # Verify request headers
        assert request.headers.get("x-api-key") == "test-api-key"
        assert request.headers.get("Content-Type") == "application/json"

        # Parse and verify request body
        body = await request.json()
        assert "text" in body
        assert body["voice_id"] == 2681
        assert body["language"] == "en-us"
        assert body["speech_model"] == "mars-8-flash"
        assert body["output_configuration"]["format"] == "pcm_s16le"

        # Prepare a StreamResponse with chunked PCM data
        resp = web.StreamResponse(
            status=200,
            reason="OK",
            headers={"Content-Type": "audio/raw"},
        )
        await resp.prepare(request)

        # Write out chunked PCM byte data (16-bit samples)
        # Use smaller chunks for more predictable frame count
        data = b"\x00\x01" * 4800  # Small chunk of audio
        await resp.write(data)
        await resp.write_eof()

        return resp

    # Create an aiohttp test server
    app = web.Application()
    app.router.add_post("/tts-stream", handler)
    client = await aiohttp_client(app)

    base_url = str(client.make_url("")).rstrip("/")

    async with aiohttp.ClientSession() as session:
        tts_service = CambTTSService(
            api_key="test-api-key",
            aiohttp_session=session,
            base_url=base_url,
            voice_id=2681,
            model="mars-8-flash",
        )

        # Manually set sample rate (normally done by StartFrame)
        tts_service._sample_rate = 24000

        # Test run_tts directly to avoid frame count variability
        text = "Hello world, this is a test."
        frames = []
        async for frame in tts_service.run_tts(text):
            frames.append(frame)

        # Verify we got the expected frame types
        frame_types = [type(f).__name__ for f in frames]
        assert "TTSStartedFrame" in frame_types, "Should have TTSStartedFrame"
        assert "TTSAudioRawFrame" in frame_types, "Should have TTSAudioRawFrame"
        assert "TTSStoppedFrame" in frame_types, "Should have TTSStoppedFrame"

        audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
        assert len(audio_frames) > 0, "Should have at least one audio frame"

        # Verify sample rate matches Camb.ai's output (24kHz)
        for a_frame in audio_frames:
            assert a_frame.sample_rate == 24000, "Sample rate should be 24000 Hz"
            assert a_frame.num_channels == 1, "Should be mono audio"


@pytest.mark.asyncio
async def test_run_camb_tts_error_401(aiohttp_client):
    """Test handling of invalid API key (401 Unauthorized)."""

    async def handler(request):
        return web.Response(
            status=401,
            text="Unauthorized: Invalid API key",
        )

    app = web.Application()
    app.router.add_post("/tts-stream", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("")).rstrip("/")

    async with aiohttp.ClientSession() as session:
        tts_service = CambTTSService(
            api_key="invalid-key",
            aiohttp_session=session,
            base_url=base_url,
        )

        frames_to_send = [
            TTSSpeakFrame(text="This should fail."),
        ]

        expected_down_frames = [AggregatedTextFrame, TTSStoppedFrame, TTSTextFrame]
        expected_up_frames = [ErrorFrame]

        frames_received = await run_test(
            tts_service,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )
        up_frames = frames_received[1]

        assert isinstance(up_frames[0], ErrorFrame), "Must receive an ErrorFrame for 401"
        assert "Invalid Camb.ai API key" in up_frames[0].error, (
            "ErrorFrame should mention invalid API key"
        )


@pytest.mark.asyncio
async def test_run_camb_tts_error_404(aiohttp_client):
    """Test handling of invalid voice ID (404 Not Found)."""

    async def handler(request):
        return web.Response(
            status=404,
            text="Voice not found",
        )

    app = web.Application()
    app.router.add_post("/tts-stream", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("")).rstrip("/")

    async with aiohttp.ClientSession() as session:
        tts_service = CambTTSService(
            api_key="test-api-key",
            aiohttp_session=session,
            base_url=base_url,
            voice_id=99999,  # Invalid voice ID
        )

        frames_to_send = [
            TTSSpeakFrame(text="This should fail."),
        ]

        expected_down_frames = [AggregatedTextFrame, TTSStoppedFrame, TTSTextFrame]
        expected_up_frames = [ErrorFrame]

        frames_received = await run_test(
            tts_service,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )
        up_frames = frames_received[1]

        assert isinstance(up_frames[0], ErrorFrame), "Must receive an ErrorFrame for 404"
        assert "Invalid voice ID" in up_frames[0].error, (
            "ErrorFrame should mention invalid voice ID"
        )


@pytest.mark.asyncio
async def test_run_camb_tts_error_429(aiohttp_client):
    """Test handling of rate limit (429 Too Many Requests)."""

    async def handler(request):
        return web.Response(
            status=429,
            text="Rate limit exceeded",
        )

    app = web.Application()
    app.router.add_post("/tts-stream", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("")).rstrip("/")

    async with aiohttp.ClientSession() as session:
        tts_service = CambTTSService(
            api_key="test-api-key",
            aiohttp_session=session,
            base_url=base_url,
        )

        frames_to_send = [
            TTSSpeakFrame(text="This should fail due to rate limit."),
        ]

        expected_down_frames = [AggregatedTextFrame, TTSStoppedFrame, TTSTextFrame]
        expected_up_frames = [ErrorFrame]

        frames_received = await run_test(
            tts_service,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )
        up_frames = frames_received[1]

        assert isinstance(up_frames[0], ErrorFrame), "Must receive an ErrorFrame for 429"
        assert "rate limit" in up_frames[0].error.lower(), (
            "ErrorFrame should mention rate limit"
        )


@pytest.mark.asyncio
async def test_list_voices(aiohttp_client):
    """Test voice listing endpoint."""

    async def handler(request):
        # Verify API key header
        assert request.headers.get("x-api-key") == "test-api-key"

        # Return mock voice data (matching actual API response structure)
        voices = [
            {
                "id": 2681,
                "voice_name": "Attic",
                "gender": 1,
                "age": 25,
                "language": None,
                "transcript": None,
                "description": None,
                "is_published": None,
            },
            {
                "id": 2682,
                "voice_name": "Cellar",
                "gender": 2,
                "age": 30,
                "language": 1,
                "transcript": None,
                "description": None,
                "is_published": False,
            },
        ]
        return web.json_response(voices)

    app = web.Application()
    app.router.add_get("/list-voices", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("")).rstrip("/")

    async with aiohttp.ClientSession() as session:
        voices = await CambTTSService.list_voices(
            api_key="test-api-key",
            aiohttp_session=session,
            base_url=base_url,
        )

        # Should return all voices
        assert len(voices) == 2, "Should return all voices"

        # Verify voice data structure
        attic_voice = next(v for v in voices if v["id"] == 2681)
        assert attic_voice["name"] == "Attic"
        assert attic_voice["gender"] == "Male"
        assert attic_voice["age"] == 25


@pytest.mark.asyncio
async def test_text_length_validation_too_short(aiohttp_client):
    """Test that text shorter than 3 characters is handled gracefully."""

    async def handler(request):
        # This should not be called for short text
        pytest.fail("Handler should not be called for text < 3 chars")

    app = web.Application()
    app.router.add_post("/tts-stream", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("")).rstrip("/")

    async with aiohttp.ClientSession() as session:
        tts_service = CambTTSService(
            api_key="test-api-key",
            aiohttp_session=session,
            base_url=base_url,
        )

        frames_to_send = [
            TTSSpeakFrame(text="Hi"),  # Only 2 characters
        ]

        # For short text, we expect TTSStoppedFrame but no audio
        expected_down_frames = [AggregatedTextFrame, TTSStoppedFrame, TTSTextFrame]

        frames_received = await run_test(
            tts_service,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
        )
        down_frames = frames_received[0]

        # Verify no audio frames were generated
        audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
        assert len(audio_frames) == 0, "Should not generate audio for text < 3 chars"


@pytest.mark.asyncio
async def test_input_params():
    """Test InputParams model validation and defaults."""

    # Test defaults
    params = CambTTSService.InputParams()
    assert params.language == Language.EN
    assert params.speed == 1.0
    assert params.user_instructions is None

    # Test custom values
    params = CambTTSService.InputParams(
        language=Language.ES,
        speed=1.5,
        user_instructions="Speak slowly and clearly",
    )
    assert params.language == Language.ES
    assert params.speed == 1.5
    assert params.user_instructions == "Speak slowly and clearly"


@pytest.mark.asyncio
async def test_language_mapping():
    """Test language enum to Camb.ai language code conversion."""

    # Test common languages
    assert language_to_camb_language(Language.EN) == "en-us"
    assert language_to_camb_language(Language.EN_US) == "en-us"
    assert language_to_camb_language(Language.EN_GB) == "en-gb"
    assert language_to_camb_language(Language.ES) == "es-es"
    assert language_to_camb_language(Language.FR) == "fr-fr"
    assert language_to_camb_language(Language.DE) == "de-de"
    assert language_to_camb_language(Language.JA) == "ja-jp"
    assert language_to_camb_language(Language.ZH) == "zh-cn"


@pytest.mark.asyncio
async def test_mars8_instruct_model(aiohttp_client):
    """Test that user_instructions are included for mars-8-instruct model."""

    received_payload = {}

    async def handler(request):
        nonlocal received_payload
        received_payload = await request.json()

        # Return minimal successful response
        resp = web.StreamResponse(status=200, headers={"Content-Type": "audio/raw"})
        await resp.prepare(request)
        await resp.write(b"\x00" * 1000)
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_post("/tts-stream", handler)
    client = await aiohttp_client(app)
    base_url = str(client.make_url("")).rstrip("/")

    async with aiohttp.ClientSession() as session:
        tts_service = CambTTSService(
            api_key="test-api-key",
            aiohttp_session=session,
            base_url=base_url,
            model="mars-8-instruct",
            params=CambTTSService.InputParams(user_instructions="Speak with excitement"),
        )

        frames_to_send = [
            TTSSpeakFrame(text="This is exciting news!"),
        ]

        await run_test(
            tts_service,
            frames_to_send=frames_to_send,
            expected_down_frames=[
                AggregatedTextFrame,
                TTSStartedFrame,
                TTSAudioRawFrame,
                TTSStoppedFrame,
                TTSTextFrame,
            ],
        )

        # Verify user_instructions was included in the request
        assert received_payload.get("speech_model") == "mars-8-instruct"
        assert received_payload.get("user_instructions") == "Speak with excitement"


@pytest.mark.asyncio
async def test_base_url_trailing_slash():
    """Test that trailing slash in base URL is handled correctly."""
    async with aiohttp.ClientSession() as session:
        tts = CambTTSService(
            api_key="test-key",
            aiohttp_session=session,
            base_url="https://api.example.com/",  # With trailing slash
        )

        # Should have removed the trailing slash
        assert tts._base_url == "https://api.example.com"
