#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Unit tests for the centralized tool-call text suppression mechanism.

``LLMService`` owns the suppression flag and the drop logic in
``_push_llm_text``; providers only call ``_note_tool_call_detected``. These
tests exercise that contract directly, independent of any provider's
streaming loop.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pipecat.frames.frames import LLMFullResponseStartFrame, LLMTextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings, ToolCallTextPolicy
from tests.frame_processor_helpers import frame_processor_setup


class _LLMService(LLMService):
    """Minimal LLM service for testing the centralized text-suppression logic."""

    def __init__(self, *, policy):
        super().__init__(
            settings=LLMSettings(
                model="test-model",
                system_instruction=None,
                temperature=None,
                max_tokens=None,
                top_p=None,
                top_k=None,
                frequency_penalty=None,
                presence_penalty=None,
                seed=None,
                filter_incomplete_user_turns=False,
                user_turn_completion_config=None,
                tool_call_text_policy=policy,
            )
        )
        self._setup = frame_processor_setup(pipeline_worker=SimpleNamespace(app_resources=None))


async def _capture_push(service, coro_factory):
    pushed: list[str] = []

    async def fake_push(self_, frame, direction=FrameDirection.DOWNSTREAM):
        if isinstance(frame, LLMTextFrame):
            pushed.append(frame.text)

    with patch.object(FrameProcessor, "push_frame", fake_push):
        await coro_factory()
    return pushed


@pytest.mark.asyncio
async def test_text_is_pushed_when_policy_is_preserve_even_after_detection():
    service = _LLMService(policy=ToolCallTextPolicy.PRESERVE)
    service._note_tool_call_detected()

    pushed = await _capture_push(service, lambda: service._push_llm_text("after"))
    assert pushed == ["after"]


@pytest.mark.asyncio
async def test_text_is_pushed_when_suppress_but_no_tool_call_detected():
    service = _LLMService(policy=ToolCallTextPolicy.SUPPRESS_AFTER_TOOL_CALL)

    pushed = await _capture_push(service, lambda: service._push_llm_text("before"))
    assert pushed == ["before"]


@pytest.mark.asyncio
async def test_text_is_dropped_when_suppress_and_tool_call_detected():
    service = _LLMService(policy=ToolCallTextPolicy.SUPPRESS_AFTER_TOOL_CALL)
    service._note_tool_call_detected()

    pushed = await _capture_push(service, lambda: service._push_llm_text("after"))
    assert pushed == []


@pytest.mark.asyncio
async def test_note_tool_call_detected_sets_the_flag():
    service = _LLMService(policy=ToolCallTextPolicy.SUPPRESS_AFTER_TOOL_CALL)
    assert service._tool_call_detected is False
    service._note_tool_call_detected()
    assert service._tool_call_detected is True


@pytest.mark.asyncio
async def test_full_response_start_frame_resets_the_flag():
    service = _LLMService(policy=ToolCallTextPolicy.SUPPRESS_AFTER_TOOL_CALL)
    service._note_tool_call_detected()
    assert service._tool_call_detected is True

    async def fake_push(self_, frame, direction=FrameDirection.DOWNSTREAM):
        pass

    with patch.object(FrameProcessor, "push_frame", fake_push):
        await service.push_frame(LLMFullResponseStartFrame())
    assert service._tool_call_detected is False
