#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for Dograh-managed AI services."""

from unittest.mock import AsyncMock, patch

import pytest
from websockets.protocol import State

from pipecat.frames.frames import CancelFrame, EndFrame, TTSStoppedFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.dograh.flux.stt import DograhFluxSTTService
from pipecat.services.dograh.llm import DograhLLMService
from pipecat.services.dograh.stt import DograhSTTService
from pipecat.services.dograh.tts import DograhTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies


class _OpenWebsocket:
    state = State.OPEN

    def __init__(self):
        self.send = AsyncMock()


def test_stt_metadata_recommends_external_turn_strategies_with_vad_events():
    service = DograhSTTService(api_key="test-key", vad_events=True)

    frame = service.service_metadata_frame()

    assert isinstance(frame.user_turn_strategies, ExternalUserTurnStrategies)


def test_stt_metadata_leaves_turn_strategies_unset_without_vad_events():
    service = DograhSTTService(api_key="test-key", vad_events=False)

    frame = service.service_metadata_frame()

    assert frame.user_turn_strategies is None


@pytest.mark.asyncio
async def test_stt_cleanup_disconnects_without_a_shutdown_frame():
    service = DograhSTTService(api_key="test-key")
    service._disconnect = AsyncMock()

    await service.cleanup()

    service._disconnect.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "frame"),
    [("stop", EndFrame()), ("cancel", CancelFrame())],
)
async def test_tts_shutdown_disconnects_once(method_name, frame):
    service = DograhTTSService(api_key="test-key")
    service._disconnect = AsyncMock()

    await getattr(service, method_name)(frame)

    service._disconnect.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_tts_stopped_frame_does_not_add_a_reset_word_timestamp():
    service = DograhTTSService(api_key="test-key")
    service._check_started = lambda frame: True
    service.add_word_timestamps = AsyncMock()

    await service.push_frame(TTSStoppedFrame())

    service.add_word_timestamps.assert_not_awaited()


@pytest.mark.asyncio
async def test_tts_usage_is_not_billed_again_when_context_finishes():
    service = DograhTTSService(api_key="test-key")
    service._websocket = _OpenWebsocket()
    service.audio_context_available = lambda _context_id: True
    service._send_text = AsyncMock()
    service.start_tts_usage_metrics = AsyncMock()

    frames = [frame async for frame in service.run_tts("hello", "ctx")]
    await service._finish_context("ctx")

    assert frames == [None]
    service.start_tts_usage_metrics.assert_awaited_once_with("hello")


@pytest.mark.asyncio
async def test_tts_terminal_context_ids_do_not_grow_after_context_cleanup():
    service = DograhTTSService(api_key="test-key")
    service._reset_state = lambda: None

    service._finished_context_ids.add("finished")
    await service.on_audio_context_completed("finished")
    assert "finished" not in service._finished_context_ids

    service._cancel_context = AsyncMock()
    service._cancelled_context_ids.add("cancelled")
    await service.on_audio_context_interrupted("cancelled")
    assert "cancelled" not in service._cancelled_context_ids


@pytest.mark.asyncio
async def test_flux_disconnect_discards_configure_state_from_old_connection():
    service = DograhFluxSTTService(api_key="test-key")
    service._configure_in_flight = True
    service._configure_sent_at = 1.0
    service._configure_pending_fields = {"eot_threshold"}
    service.stop_all_metrics = AsyncMock()

    await service._disconnect_websocket()

    assert service._configure_in_flight is False
    assert service._configure_sent_at is None
    assert service._configure_pending_fields is None


@pytest.mark.asyncio
async def test_llm_quota_error_returns_an_empty_stream_after_permanent_error():
    with patch.object(DograhLLMService, "create_client"):
        service = DograhLLMService(api_key="test-key")
    service.push_error = AsyncMock()

    with patch.object(
        OpenAILLMService,
        "get_chat_completions",
        new_callable=AsyncMock,
        side_effect=Exception("403 quota_exceeded"),
    ):
        stream = await service.get_chat_completions(LLMContext())

    assert [chunk async for chunk in stream] == []
    service.push_error.assert_awaited_once()
    assert service.push_error.await_args.args[0] == "Dograh Service quota exceeded"
    assert service.push_error.await_args.kwargs["force_treat_as_permanent"] is True
