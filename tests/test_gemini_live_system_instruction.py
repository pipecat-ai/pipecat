#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the system instruction GeminiLiveLLMService configures sessions with.

The service connects in ``setup()``, before ``LLMService.start()`` composes the
system instruction, so the first context reconnects when the composed
instruction differs from the one the session was configured with. These tests
fake the Live API connection and record the config of every connect.
"""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import CancelFrame, StartFrame
from pipecat.processors.aggregators.async_tool_messages import ASYNC_TOOL_INSTRUCTIONS
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.utils.asyncio.task_manager import TaskManager


class _FakeSession:
    """Stands in for the Live API session: receives nothing until closed."""

    def __init__(self):
        self._closed = asyncio.Event()

    async def _messages(self):
        await self._closed.wait()
        return
        yield

    def receive(self):
        return self._messages()

    async def close(self):
        self._closed.set()


class _FakeConnection:
    async def __aenter__(self):
        return _FakeSession()

    async def __aexit__(self, *exc):
        return False


class _ConnectRecorder:
    """Records the system instruction of every connect."""

    def __init__(self):
        self.system_instructions: list = []

    def connect(self, model, config):
        self.system_instructions.append(config.system_instruction)
        return _FakeConnection()

    async def wait_for(self, count: int):
        for _ in range(200):
            if len(self.system_instructions) >= count:
                return
            await asyncio.sleep(0.01)
        raise AssertionError(f"expected {count} connects, got {len(self.system_instructions)}")


def _service(system_instruction: str) -> GeminiLiveLLMService:
    return GeminiLiveLLMService(
        api_key="test-key",
        settings=GeminiLiveLLMService.Settings(system_instruction=system_instruction),
    )


def _context() -> LLMContext:
    return LLMContext(messages=[{"role": "user", "content": "Hello"}])


@asynccontextmanager
async def _running(service: GeminiLiveLLMService, recorder: _ConnectRecorder):
    """Run the service through setup (the first connect) and start, then cancel it."""
    with (
        patch.object(service._client.aio.live, "connect", side_effect=recorder.connect),
        patch.object(service, "_create_initial_response", new=AsyncMock()),
    ):
        await service.setup(
            FrameProcessorSetup(
                clock=SystemClock(),
                task_manager=TaskManager(),
                pipeline_worker=SimpleNamespace(app_resources=None),  # type: ignore[arg-type]
            )
        )
        await recorder.wait_for(1)
        await service.start(StartFrame())
        try:
            yield
        finally:
            await service.cancel(CancelFrame())


@pytest.mark.asyncio
async def test_first_context_reconnects_with_composed_instruction():
    """Async-tool guidance composed at start reaches the model on the first context."""
    service = _service("base prompt")
    service.register_function("check_order", AsyncMock(), cancel_on_interruption=False)
    recorder = _ConnectRecorder()
    async with _running(service, recorder):
        await service._handle_context(_context())
        await recorder.wait_for(2)

    first, reconnect = recorder.system_instructions
    assert first == "base prompt"
    assert reconnect.startswith("base prompt")
    assert ASYNC_TOOL_INSTRUCTIONS in reconnect


@pytest.mark.asyncio
async def test_first_context_keeps_session_when_instruction_unchanged():
    """Without composed additions the first context does not reconnect."""
    service = _service("base prompt")
    recorder = _ConnectRecorder()
    async with _running(service, recorder):
        await service._handle_context(_context())
        await asyncio.sleep(0.05)

    assert recorder.system_instructions == ["base prompt"]


@pytest.mark.asyncio
async def test_reconnect_sends_current_instruction():
    """A reconnect after a runtime system_instruction update sends the new value."""
    service = _service("first")
    recorder = _ConnectRecorder()
    async with _running(service, recorder):
        await service._handle_context(_context())
        await service._update_settings(GeminiLiveLLMService.Settings(system_instruction="second"))
        await service._reconnect()
        await recorder.wait_for(2)

    assert recorder.system_instructions == ["first", "second"]
