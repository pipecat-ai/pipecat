#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for CambTTSService.

These tests mock the Camb.ai SDK client to test the service behavior.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


async def mock_tts_stream(*args, **kwargs):
    """Mock TTS stream that yields audio chunks."""
    yield b"\x00\x01" * 4800  # Small chunk of PCM audio


async def mock_tts_stream_error(*args, **kwargs):
    """Mock TTS stream that raises an error."""
    raise Exception("API error: Invalid API key")
    yield  # Make this a generator


@pytest.mark.asyncio
async def test_run_camb_tts_success():
    """Test successful TTS generation with chunked PCM audio.

    Verifies the frame sequence: TTSStartedFrame -> TTSAudioRawFrame* -> TTSStoppedFrame
    """
    with patch("pipecat.services.camb.tts.AsyncCambAI") as MockAsyncCambAI:
        mock_client = MagicMock()
        mock_client.text_to_speech.tts = mock_tts_stream
        MockAsyncCambAI.return_value = mock_client

        tts_service = CambTTSService(
            api_key="test-api-key",
            voice_id=2681,
            model="mars-flash",
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
async def test_run_camb_tts_error():
    """Test handling of TTS API errors."""
    with patch("pipecat.services.camb.tts.AsyncCambAI") as MockAsyncCambAI:
        mock_client = MagicMock()
        mock_client.text_to_speech.tts = mock_tts_stream_error
        MockAsyncCambAI.return_value = mock_client

        tts_service = CambTTSService(
            api_key="invalid-key",
            voice_id=147320,
        )

        frames_to_send = [
            TTSSpeakFrame(text="This should fail."),
        ]

        # TTSStartedFrame is emitted before we attempt to iterate the stream
        expected_down_frames = [AggregatedTextFrame, TTSStartedFrame, TTSStoppedFrame, TTSTextFrame]
        expected_up_frames = [ErrorFrame]

        frames_received = await run_test(
            tts_service,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down_frames,
            expected_up_frames=expected_up_frames,
        )
        up_frames = frames_received[1]

        assert isinstance(up_frames[0], ErrorFrame), "Must receive an ErrorFrame"
        assert "error" in up_frames[0].error.lower(), "ErrorFrame should contain error message"


@pytest.mark.asyncio
async def test_list_voices():
    """Test voice listing endpoint."""

    async def mock_list_voices(*args, **kwargs):
        # Return mock Voice objects
        mock_voice1 = MagicMock()
        mock_voice1.id = 2681
        mock_voice1.voice_name = "Attic"
        mock_voice1.gender = 1
        mock_voice1.age = 25
        mock_voice1.language = None

        mock_voice2 = MagicMock()
        mock_voice2.id = 2682
        mock_voice2.voice_name = "Cellar"
        mock_voice2.gender = 2
        mock_voice2.age = 30
        mock_voice2.language = 1

        return [mock_voice1, mock_voice2]

    with patch("pipecat.services.camb.tts.AsyncCambAI") as MockAsyncCambAI:
        mock_client = MagicMock()
        mock_client.voice_cloning.list_voices = mock_list_voices
        MockAsyncCambAI.return_value = mock_client

        voices = await CambTTSService.list_voices(api_key="test-api-key")

        # Should return all voices
        assert len(voices) == 2, "Should return all voices"

        # Verify voice data structure
        attic_voice = next(v for v in voices if v["id"] == 2681)
        assert attic_voice["name"] == "Attic"
        assert attic_voice["gender"] == "Male"
        assert attic_voice["age"] == 25


@pytest.mark.asyncio
async def test_text_length_validation_too_short():
    """Test that text shorter than 3 characters is handled gracefully."""
    with patch("pipecat.services.camb.tts.AsyncCambAI") as MockAsyncCambAI:
        mock_client = MagicMock()
        # TTS should not be called for short text
        mock_client.text_to_speech.tts = AsyncMock(side_effect=AssertionError("TTS should not be called"))
        MockAsyncCambAI.return_value = mock_client

        tts_service = CambTTSService(
            api_key="test-api-key",
            voice_id=147320,
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
    assert params.user_instructions is None

    # Test custom values
    params = CambTTSService.InputParams(
        language=Language.ES,
        user_instructions="Speak slowly and clearly",
    )
    assert params.language == Language.ES
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
async def test_mars_instruct_model():
    """Test that user_instructions are included for mars-instruct model."""
    received_kwargs = {}

    async def mock_tts_with_capture(*args, **kwargs):
        nonlocal received_kwargs
        received_kwargs = kwargs
        yield b"\x00" * 1000

    with patch("pipecat.services.camb.tts.AsyncCambAI") as MockAsyncCambAI:
        mock_client = MagicMock()
        mock_client.text_to_speech.tts = mock_tts_with_capture
        MockAsyncCambAI.return_value = mock_client

        tts_service = CambTTSService(
            api_key="test-api-key",
            model="mars-instruct",
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
        assert received_kwargs.get("speech_model") == "mars-instruct"
        assert received_kwargs.get("user_instructions") == "Speak with excitement"


@pytest.mark.asyncio
async def test_client_initialization_with_api_key():
    """Test that client is created when api_key is provided."""
    with patch("pipecat.services.camb.tts.AsyncCambAI") as MockAsyncCambAI:
        mock_client = MagicMock()
        MockAsyncCambAI.return_value = mock_client

        tts = CambTTSService(
            api_key="test-key",
            voice_id=147320,
        )

        # Should have created a client
        MockAsyncCambAI.assert_called_once()
        assert tts._owns_client is True


@pytest.mark.asyncio
async def test_client_initialization_with_existing_client():
    """Test that existing client is used when provided."""
    mock_client = MagicMock()

    tts = CambTTSService(
        client=mock_client,
        voice_id=147320,
    )

    # Should use the provided client
    assert tts._client is mock_client
    assert tts._owns_client is False


@pytest.mark.asyncio
async def test_client_initialization_requires_api_key_or_client():
    """Test that ValueError is raised when neither api_key nor client is provided."""
    with pytest.raises(ValueError, match="Either 'api_key' or 'client' must be provided"):
        CambTTSService(voice_id=147320)
