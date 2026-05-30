#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for GeminiGenAITTSService."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipecat.frames.frames import (
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.google.tts import GeminiGenAITTSService
from pipecat.tests.utils import run_test


def _make_mock_response(audio_bytes: bytes) -> MagicMock:
    """Build a mock generate_content response with a single audio part."""
    part = MagicMock()
    part.inline_data = MagicMock()
    part.inline_data.data = audio_bytes
    part.inline_data.mime_type = "audio/pcm;rate=24000"

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    response = MagicMock()
    response.candidates = [candidate]
    return response


@pytest.mark.asyncio
async def test_gemini_genai_tts_success():
    """GeminiGenAITTSService should call generate_content and emit PCM frames."""
    AUDIO_BYTES = b"\x10\x20\x30\x40" * 1024

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_client.aio = MagicMock()
        mock_client.aio.models = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=_make_mock_response(AUDIO_BYTES)
        )

        tts = GeminiGenAITTSService(api_key="test-key", sample_rate=24000)
        down_frames, _ = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="Hello from Gemini.")],
        )

    frame_types = [type(f) for f in down_frames]
    assert TTSStartedFrame in frame_types
    assert TTSStoppedFrame in frame_types

    audio_frames = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
    assert audio_frames, "Expected at least one TTSAudioRawFrame"
    assert all(f.sample_rate == 24000 for f in audio_frames)
    assert all(f.num_channels == 1 for f in audio_frames)
    assert b"".join(f.audio for f in audio_frames) == AUDIO_BYTES

    # Verify the generate_content call shape
    mock_client.aio.models.generate_content.assert_called_once()
    call_kwargs = mock_client.aio.models.generate_content.call_args
    assert call_kwargs.kwargs["model"] == "gemini-3.1-flash-tts-preview"
    assert call_kwargs.kwargs["contents"] == "Hello from Gemini."
    config = call_kwargs.kwargs["config"]
    assert "AUDIO" in config.response_modalities


@pytest.mark.asyncio
async def test_gemini_genai_tts_custom_settings():
    """GeminiGenAITTSService should respect custom model and voice settings."""
    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.aio = MagicMock()
        mock_client.aio.models = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=_make_mock_response(b"\x00\x01" * 512)
        )

        tts = GeminiGenAITTSService(
            api_key="key",
            settings=GeminiGenAITTSService.Settings(
                model="gemini-3.1-flash-tts-preview",
                voice="Charon",
            ),
        )
        await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Custom voice.")])

    call_kwargs = mock_client.aio.models.generate_content.call_args
    assert call_kwargs.kwargs["model"] == "gemini-3.1-flash-tts-preview"
    config = call_kwargs.kwargs["config"]
    assert config.speech_config.voice_config.prebuilt_voice_config.voice_name == "Charon"


@pytest.mark.asyncio
async def test_gemini_genai_tts_api_error():
    """GeminiGenAITTSService should emit an ErrorFrame when generate_content raises."""
    from pipecat.frames.frames import ErrorFrame

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.aio = MagicMock()
        mock_client.aio.models = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=Exception("API quota exceeded")
        )

        tts = GeminiGenAITTSService(api_key="bad-key")
        down_frames, up_frames = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(text="This should fail.")],
        )

    all_frames = down_frames + up_frames
    error_frames = [f for f in all_frames if isinstance(f, ErrorFrame)]
    assert error_frames, "Expected an ErrorFrame on API error"
    assert any("Gemini GenAI TTS error" in f.error for f in error_frames)


if __name__ == "__main__":
    unittest.main()
