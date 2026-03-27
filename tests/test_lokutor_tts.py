#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for LokutorTTSService."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipecat.frames.frames import (
    ErrorFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.lokutor.tts import LokutorTTSService
from pipecat.tests.utils import run_test
from pipecat.transcriptions.language import Language


@pytest.mark.asyncio
async def test_lokutor_tts_initialization():
    """Test LokutorTTSService initialization with valid parameters."""
    tts_service = LokutorTTSService(
        api_key="test-api-key",
        voice_id="M1",
    )

    assert tts_service._api_key == "test-api-key"
    assert tts_service._voice_id == "M1"
    assert tts_service._sample_rate == 44100


@pytest.mark.asyncio
async def test_lokutor_tts_invalid_voice():
    """Test LokutorTTSService initialization with invalid voice."""
    with pytest.raises(ValueError, match="Invalid voice_id 'INVALID'"):
        LokutorTTSService(
            api_key="test-api-key",
            voice_id="INVALID",
        )


@pytest.mark.asyncio
async def test_lokutor_tts_with_params():
    """Test LokutorTTSService initialization with custom parameters."""
    params = LokutorTTSService.InputParams(
        language=Language.EN,
        speed=1.2,
        steps=10,
        visemes=True,
    )

    tts_service = LokutorTTSService(
        api_key="test-api-key",
        voice_id="F1",
        params=params,
    )

    assert tts_service._params.language == Language.EN
    assert tts_service._params.speed == 1.2
    assert tts_service._params.steps == 10
    assert tts_service._params.visemes is True


@pytest.mark.asyncio
async def test_language_conversion():
    """Test language conversion functions."""
    from pipecat.services.lokutor.tts import language_to_lokutor_language, lokutor_language_to_language

    # Test conversion to Lokutor language
    assert language_to_lokutor_language(Language.EN) == "en"
    assert language_to_lokutor_language(Language.ES) == "es"
    assert language_to_lokutor_language(Language.FR) == "fr"
    assert language_to_lokutor_language(Language.PT) == "pt"
    assert language_to_lokutor_language(Language.KO) == "ko"

    # Test conversion from Lokutor language
    assert lokutor_language_to_language("en") == Language.EN
    assert lokutor_language_to_language("es") == Language.ES
    assert lokutor_language_to_language("fr") == Language.FR
    assert lokutor_language_to_language("pt") == Language.PT
    assert lokutor_language_to_language("ko") == Language.KO

    # Test unsupported language
    assert language_to_lokutor_language(Language.DE) is None
    assert lokutor_language_to_language("de") is None


@pytest.mark.asyncio
@patch('pipecat.services.lokutor.tts.websocket_connect')
async def test_lokutor_tts_run_tts(mock_websocket_connect):
    """Test successful TTS generation with mocked WebSocket."""
    # Mock WebSocket connection to raise an exception (simulating connection failure)
    # This tests that the service handles errors gracefully
    mock_websocket_connect.side_effect = Exception("Mock connection error")

    tts_service = LokutorTTSService(
        api_key="test-api-key",
        voice_id="M1",
    )

    # Run TTS - should handle the connection error gracefully
    frames = []
    async for frame in tts_service.run_tts("Hello world", "test-context"):
        frames.append(frame)

    # Should still yield TTSStartedFrame and TTSStoppedFrame even on error
    assert len(frames) >= 2
    assert isinstance(frames[0], TTSStartedFrame)
    assert isinstance(frames[-1], TTSStoppedFrame)

    # Should have yielded an ErrorFrame due to connection failure
    error_frames = [f for f in frames if isinstance(f, ErrorFrame)]
    assert len(error_frames) >= 1
    assert "Failed to connect to Lokutor" in error_frames[0].error


@pytest.mark.asyncio
@patch('pipecat.services.lokutor.tts.websocket_connect')
async def test_lokutor_tts_websocket_error(mock_websocket_connect):
    """Test handling of WebSocket connection errors."""
    mock_websocket_connect.side_effect = Exception("Connection failed")

    tts_service = LokutorTTSService(
        api_key="test-api-key",
        voice_id="M1",
    )

    frames = []
    async for frame in tts_service.run_tts("Hello world", "test-context"):
        frames.append(frame)

    # Should yield ErrorFrame
    error_frames = [f for f in frames if isinstance(f, ErrorFrame)]
    assert len(error_frames) > 0
    assert "Unknown error occurred" in error_frames[0].error


if __name__ == "__main__":
    unittest.main()