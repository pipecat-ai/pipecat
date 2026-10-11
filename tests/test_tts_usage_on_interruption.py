#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that an interrupted turn still reports the TTS characters it was billed for.

In ``TextAggregationMode.TOKEN`` the per-token ``start_tts_usage_metrics`` calls
short-circuit and the text is accumulated instead, then reported once when the turn
flushes. An interrupted turn never reaches that flush, so clearing the accumulator in
``_handle_interruption`` dropped the whole turn's character count even though the text
had already been sent to the service and billed for.

Barge-in is the normal case in a voice agent rather than an edge case, and
``TextAggregationMode.TOKEN`` is the default for Deepgram Flux, so this ran on every
interrupted turn and nothing raised.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipecat.frames.frames import InterruptionFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_service import AIService
from pipecat.services.tts_service import TextAggregationMode, TTSService


class _StubTTSService(TTSService):
    """Minimal concrete TTSService; the transport is irrelevant to usage accounting."""

    async def run_tts(self, text: str):  # pragma: no cover - never driven here
        return
        yield


def _service(mode: TextAggregationMode) -> _StubTTSService:
    service = _StubTTSService(text_aggregation_mode=mode)
    # _handle_interruption also cycles word timestamps and the audio context
    # task. Neither exists outside a running pipeline, and the audio context
    # task needs a TaskManager, so both are stubbed. Usage accounting is
    # independent of both.
    service.reset_word_timestamps = AsyncMock()
    service._stop_audio_context_task = AsyncMock()
    service._create_audio_context_task = MagicMock()
    service.push_frame = AsyncMock()
    return service


async def _interrupt(service) -> None:
    await service._handle_interruption(InterruptionFrame(), FrameDirection.DOWNSTREAM)


def _reported_text(recorder: AsyncMock) -> list:
    return [call.args[0] for call in recorder.await_args_list]


class TestInterruptedTurnReportsUsage:
    @pytest.mark.asyncio
    async def test_accumulated_characters_are_reported(self):
        """The service was sent this text and billed for it before the barge-in."""
        service = _service(TextAggregationMode.TOKEN)
        service._streamed_text = "The nearest office is on Market Street and it opens at"
        with patch.object(AIService, "start_tts_usage_metrics", new_callable=AsyncMock) as recorder:
            await _interrupt(service)
        assert _reported_text(recorder) == [
            "The nearest office is on Market Street and it opens at"
        ]

    @pytest.mark.asyncio
    async def test_the_accumulator_is_still_cleared(self):
        """Reporting must not leave the text behind to be counted twice next turn."""
        service = _service(TextAggregationMode.TOKEN)
        service._streamed_text = "some spoken text"
        with patch.object(AIService, "start_tts_usage_metrics", new_callable=AsyncMock):
            await _interrupt(service)
        assert service._streamed_text == ""

    @pytest.mark.asyncio
    async def test_a_turn_interrupted_twice_is_counted_once(self):
        """The second interruption has nothing left to report."""
        service = _service(TextAggregationMode.TOKEN)
        service._streamed_text = "some spoken text"
        with patch.object(AIService, "start_tts_usage_metrics", new_callable=AsyncMock) as recorder:
            await _interrupt(service)
            await _interrupt(service)
        assert _reported_text(recorder) == ["some spoken text"]

    @pytest.mark.asyncio
    async def test_nothing_is_reported_when_no_text_was_sent(self):
        """An interruption before any token must not emit a zero-character metric."""
        service = _service(TextAggregationMode.TOKEN)
        service._streamed_text = ""
        with patch.object(AIService, "start_tts_usage_metrics", new_callable=AsyncMock) as recorder:
            await _interrupt(service)
        recorder.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sentence_mode_is_unaffected(self):
        """Sentence aggregation reports per sentence as it goes and never accumulates."""
        service = _service(TextAggregationMode.SENTENCE)
        assert service._streamed_text == ""
        with patch.object(AIService, "start_tts_usage_metrics", new_callable=AsyncMock) as recorder:
            await _interrupt(service)
        recorder.assert_not_awaited()
