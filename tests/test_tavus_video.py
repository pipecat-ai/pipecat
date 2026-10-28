#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the Tavus video service."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from pipecat.expressive.expression import ExpressionDelivery
from pipecat.frames.frames import InterruptionFrame, StartFrame, TextType, TTSTextFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_service import AIService
from pipecat.services.tavus.video import TavusVideoService
from pipecat.utils.asyncio.task_manager import TaskManager
from pipecat.utils.time import seconds_to_nanoseconds
from tests.frame_processor_helpers import frame_processor_setup


async def _make_service() -> tuple[TavusVideoService, MagicMock]:
    service = TavusVideoService(api_key="test-key", replica_id="replica", session=MagicMock())
    # Skip TavusVideoService.setup, which creates a Tavus conversation.
    await AIService.setup(service, frame_processor_setup(TaskManager()))
    client = MagicMock()
    for method in (
        "start_send_task",
        "cancel_send_task",
        "send_interrupt_message",
        "update_emotion",
        "stop",
    ):
        setattr(client, method, AsyncMock())
    service._client = client
    service.push_frame = AsyncMock()
    await service.start(StartFrame())
    return service, client


def _word(text: str, delivery: list[str], delay_secs: float, service: TavusVideoService):
    frame = TTSTextFrame(text, text_type=TextType.WORD, delivery=delivery)
    frame.pts = service.get_clock().get_time() + seconds_to_nanoseconds(delay_secs)
    return frame


@pytest.mark.asyncio
async def test_the_emotion_changes_when_the_word_is_heard():
    service, client = await _make_service()

    await service.process_frame(
        _word("Wow", [ExpressionDelivery.SURPRISED], 0.2, service), FrameDirection.DOWNSTREAM
    )
    await asyncio.sleep(0.1)
    client.update_emotion.assert_not_awaited()

    await asyncio.sleep(0.2)
    client.update_emotion.assert_awaited_once_with([ExpressionDelivery.SURPRISED])
    await service._cancel_emotion_task()


@pytest.mark.asyncio
async def test_an_interruption_drops_the_emotion_of_words_not_yet_heard():
    service, client = await _make_service()

    await service.process_frame(
        _word("Wow", [ExpressionDelivery.SURPRISED], 0.2, service), FrameDirection.DOWNSTREAM
    )
    await service.process_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM)
    await asyncio.sleep(0.3)

    client.update_emotion.assert_not_awaited()
    await service._cancel_emotion_task()
