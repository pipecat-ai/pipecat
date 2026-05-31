#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for OpenRouterTTSService."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipecat.frames.frames import (
    ErrorFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.openrouter.tts import (
    OPENROUTER_DEFAULT_TTS_MODEL,
    OPENROUTER_DEFAULT_TTS_VOICE,
    OPENROUTER_TTS_BASE_URL,
    OpenRouterTTSService,
    OpenRouterTTSSettings,
)
from pipecat.services.settings import NOT_GIVEN, is_given


# ---------------------------------------------------------------------------
# Settings contract (delta-mode and store-mode)
# ---------------------------------------------------------------------------


def test_settings_delta_defaults_are_not_given():
    """Empty OpenRouterTTSSettings must have all fields set to NOT_GIVEN."""
    from dataclasses import fields

    s = OpenRouterTTSSettings()
    for f in fields(s):
        if f.name == "extra":
            continue
        val = getattr(s, f.name)
        assert not is_given(val), (
            f"OpenRouterTTSSettings.{f.name} defaults to {val!r}, expected NOT_GIVEN"
        )


def test_service_settings_complete_after_init():
    """After construction, _settings must have no NOT_GIVEN values."""
    from dataclasses import fields

    svc = OpenRouterTTSService(api_key="test-key")
    for f in fields(svc._settings):
        if f.name == "extra":
            continue
        val = getattr(svc._settings, f.name)
        assert is_given(val), (
            f"OpenRouterTTSService._settings.{f.name} is NOT_GIVEN after construction"
        )


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_base_url():
    svc = OpenRouterTTSService(api_key="test-key")
    assert svc._client.base_url.host == "openrouter.ai"


def test_default_model():
    svc = OpenRouterTTSService(api_key="test-key")
    assert svc._settings.model == OPENROUTER_DEFAULT_TTS_MODEL


def test_default_voice():
    svc = OpenRouterTTSService(api_key="test-key")
    assert svc._settings.voice == OPENROUTER_DEFAULT_TTS_VOICE


def test_settings_override_model_and_voice():
    svc = OpenRouterTTSService(
        api_key="test-key",
        settings=OpenRouterTTSService.Settings(
            model="google/gemini-flash-tts",
            voice="en_paul_happy",
        ),
    )
    assert svc._settings.model == "google/gemini-flash-tts"
    assert svc._settings.voice == "en_paul_happy"


def test_non_openai_voices_accepted():
    """OpenRouter supports provider-namespaced voices — no allowlist rejection."""
    for voice in ["en_paul_happy", "af_bella", "alloy", "nova", "shimmer"]:
        svc = OpenRouterTTSService(
            api_key="test-key",
            settings=OpenRouterTTSService.Settings(voice=voice),
        )
        assert svc._settings.voice == voice


# ---------------------------------------------------------------------------
# run_tts — mocked OpenAI client
# ---------------------------------------------------------------------------


def _make_streaming_response(audio_chunks: list[bytes], status_code: int = 200):
    """Return an async context manager that yields bytes chunks, mimicking
    AsyncOpenAI's audio.speech.with_streaming_response.create(...)."""

    async def iter_bytes(chunk_size=None):
        for chunk in audio_chunks:
            yield chunk

    response = MagicMock()
    response.status_code = status_code
    response.iter_bytes = iter_bytes
    response.text = AsyncMock(return_value="error body")

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=response)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


@pytest.mark.asyncio
async def test_run_tts_emits_audio_frames():
    """run_tts should yield TTSAudioRawFrame chunks for a successful response."""
    audio_data = b"\x00\x01\x02\x03" * 512
    chunks = [audio_data[:512], audio_data[512:]]

    svc = OpenRouterTTSService(api_key="test-key")
    svc._sample_rate = 24000  # normally set by StartFrame in a live pipeline
    svc._client.audio.speech.with_streaming_response.create = MagicMock(
        return_value=_make_streaming_response(chunks)
    )

    frames = []
    async for frame in svc.run_tts("Hello, world!", context_id="ctx-1"):
        frames.append(frame)

    audio_frames = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
    assert len(audio_frames) == 2
    assert all(f.sample_rate == 24000 for f in audio_frames)
    assert all(f.num_channels == 1 for f in audio_frames)
    assert all(f.context_id == "ctx-1" for f in audio_frames)


@pytest.mark.asyncio
async def test_run_tts_request_params():
    """run_tts should pass model, voice, input, and response_format=pcm."""
    captured = {}

    async def iter_bytes(chunk_size=None):
        yield b"\x00" * 128

    response = MagicMock()
    response.status_code = 200
    response.iter_bytes = iter_bytes

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=response)
    cm.__aexit__ = AsyncMock(return_value=False)

    def create(**kwargs):
        captured.update(kwargs)
        return cm

    svc = OpenRouterTTSService(api_key="test-key")
    svc._client.audio.speech.with_streaming_response.create = create

    async for _ in svc.run_tts("Test text", context_id="ctx-2"):
        pass

    assert captured["input"] == "Test text"
    assert captured["model"] == OPENROUTER_DEFAULT_TTS_MODEL
    assert captured["voice"] == OPENROUTER_DEFAULT_TTS_VOICE
    assert captured["response_format"] == "pcm"


@pytest.mark.asyncio
async def test_run_tts_passes_instructions_and_speed():
    """run_tts should include optional instructions and speed when set."""
    captured = {}

    async def iter_bytes(chunk_size=None):
        yield b"\x00" * 64

    response = MagicMock()
    response.status_code = 200
    response.iter_bytes = iter_bytes

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=response)
    cm.__aexit__ = AsyncMock(return_value=False)

    def create(**kwargs):
        captured.update(kwargs)
        return cm

    svc = OpenRouterTTSService(
        api_key="test-key",
        settings=OpenRouterTTSService.Settings(instructions="Speak slowly.", speed=0.8),
    )
    svc._client.audio.speech.with_streaming_response.create = create

    async for _ in svc.run_tts("Test", context_id="ctx-3"):
        pass

    assert captured["instructions"] == "Speak slowly."
    assert captured["speed"] == 0.8


@pytest.mark.asyncio
async def test_run_tts_error_status_yields_error_frame():
    """A non-200 response should yield an ErrorFrame."""
    svc = OpenRouterTTSService(api_key="test-key")
    svc._client.audio.speech.with_streaming_response.create = MagicMock(
        return_value=_make_streaming_response([], status_code=400)
    )

    frames = []
    async for frame in svc.run_tts("Hello", context_id="ctx-4"):
        frames.append(frame)

    assert any(isinstance(f, ErrorFrame) for f in frames)
    assert not any(isinstance(f, TTSAudioRawFrame) for f in frames)


@pytest.mark.asyncio
async def test_run_tts_missing_voice_yields_error_frame():
    """run_tts should yield an ErrorFrame when voice is None."""
    svc = OpenRouterTTSService(api_key="test-key")
    svc._settings.voice = None

    frames = []
    async for frame in svc.run_tts("Hello", context_id="ctx-5"):
        frames.append(frame)

    assert any(isinstance(f, ErrorFrame) for f in frames)


if __name__ == "__main__":
    unittest.main()
