#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Inworld WebSocket TTS fallback text handling."""

from unittest.mock import AsyncMock, patch

import pytest

from pipecat.frames.frames import TTSTextFrame
from pipecat.services.inworld.tts import InworldTTSService
from pipecat.services.tts_service import TTSService


def _service(*, push_on_interruption: bool = True) -> InworldTTSService:
    service = InworldTTSService(
        api_key="test-key",
        push_full_text_on_interruption_without_timestamps=push_on_interruption,
    )
    service.push_frame = AsyncMock()
    return service


@pytest.mark.asyncio
async def test_normal_completion_pushes_fallback_when_interruption_fallback_is_disabled():
    """Normal completion should always preserve text when timestamps are unavailable."""
    service = _service(push_on_interruption=False)
    service._context_texts["context-1"] = "Hello world "

    await service._maybe_push_fallback_text("context-1")

    service.push_frame.assert_awaited_once()
    frame = service.push_frame.await_args.args[0]
    assert isinstance(frame, TTSTextFrame)
    assert frame.text == "Hello world"
    assert frame.context_id == "context-1"


@pytest.mark.asyncio
async def test_interruption_pushes_fallback_by_default():
    """The default should retain the existing full-text interruption behavior."""
    service = _service()
    service._context_texts["context-1"] = "Hello world"

    await service._maybe_push_fallback_text("context-1", is_interruption=True)

    service.push_frame.assert_awaited_once()


@pytest.mark.asyncio
async def test_interruption_can_suppress_fallback_and_consumes_context_state():
    """Suppressed interruption text must not be emitted by a late contextClosed event."""
    service = _service(push_on_interruption=False)
    service._context_texts["context-1"] = "Hello world"

    await service._maybe_push_fallback_text("context-1", is_interruption=True)
    await service._maybe_push_fallback_text("context-1")

    service.push_frame.assert_not_awaited()
    assert "context-1" not in service._context_texts


@pytest.mark.asyncio
async def test_timestamped_interruption_does_not_push_full_text():
    """Timestamp-confirmed text should remain authoritative during interruption."""
    service = _service()
    service._context_texts["context-1"] = "Hello world"
    service._contexts_with_timestamps.add("context-1")

    await service._maybe_push_fallback_text("context-1", is_interruption=True)

    service.push_frame.assert_not_awaited()
    assert "context-1" not in service._contexts_with_timestamps


@pytest.mark.asyncio
async def test_audio_interruption_marks_fallback_as_interrupted():
    """The interruption callback should apply the configurable fallback policy."""
    service = _service()
    service._maybe_push_fallback_text = AsyncMock()
    service._close_context = AsyncMock()

    with patch.object(TTSService, "on_audio_context_interrupted", new=AsyncMock()):
        await service.on_audio_context_interrupted("context-1")

    service._maybe_push_fallback_text.assert_awaited_once_with("context-1", is_interruption=True)
