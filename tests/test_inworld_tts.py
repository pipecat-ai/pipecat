#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Inworld TTS service lifecycle."""

from unittest.mock import AsyncMock, patch

import pytest

from pipecat.services.inworld.tts import InworldTTSService
from pipecat.services.tts_service import TTSService
from pipecat.utils.text.base_text_aggregator import AggregationType


@pytest.mark.asyncio
async def test_context_closes_once_across_completion_and_interruption():
    """Completion followed by interruption should close one provider context once."""
    service = InworldTTSService(api_key="test-key")
    service._websocket = AsyncMock()
    service._turn_context_id = "context-1"
    service._sent_context_ids.add("context-1")
    service._context_texts["context-1"] = "Hello there."
    service._send_close_context = AsyncMock()
    service.push_frame = AsyncMock()

    with patch.object(TTSService, "on_turn_context_completed", new=AsyncMock()):
        await service.on_turn_context_completed()
    await service.on_audio_context_interrupted("context-1")

    service._send_close_context.assert_awaited_once_with("context-1")
    service.push_frame.assert_awaited_once()
    fallback = service.push_frame.await_args.args[0]
    assert fallback.text == "Hello there."
    assert fallback.aggregated_by is AggregationType.SENTENCE
    assert fallback.context_id == "context-1"
