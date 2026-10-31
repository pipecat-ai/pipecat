#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for composed system instruction handling in GeminiLiveLLMService."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import LLMUpdateSettingsFrame
from pipecat.processors.aggregators.async_tool_messages import ASYNC_TOOL_INSTRUCTIONS
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.utils.asyncio.task_manager import TaskManager


async def _setup_service(service: GeminiLiveLLMService) -> None:
    """Give the service a task manager so _connect can create its tasks."""
    await service.setup(
        FrameProcessorSetup(
            clock=__import__("pipecat.clocks.system_clock", fromlist=["SystemClock"]).SystemClock(),
            task_manager=TaskManager(),
            pipeline_worker=None,  # type: ignore[arg-type]
        )
    )


def _service(**settings) -> GeminiLiveLLMService:
    return GeminiLiveLLMService(
        api_key="test-key", settings=GeminiLiveLLMService.Settings(**settings)
    )


def _async_tool() -> FunctionSchema:
    return FunctionSchema(
        name="check_order",
        description="Check an order in the background.",
        properties={},
        required=[],
    )


def test_init_snapshot_equals_composed_before_start():
    """Before start(), the init snapshot and the composed settings agree."""
    service = _service(system_instruction="base prompt")
    assert service._system_instruction_from_init == "base prompt"
    # No addons yet: composition is a no-op rebuild of the base.
    assert service._settings.system_instruction == "base prompt"


def test_composed_instruction_diverges_from_init_snapshot():
    """Registering an async tool composes guidance the init snapshot lacks.

    This is the divergence that made the service send a stale instruction:
    _connect reads _system_instruction_from_init, so everything composed by
    LLMService after construction never reached the model.
    """
    service = _service(system_instruction="base prompt")
    service.register_function("check_order", AsyncMock(), cancel_on_interruption=False)
    service._sync_registered_tool_handlers(None)
    composed = service._settings.system_instruction
    assert composed is not None
    assert "base prompt" in composed
    assert composed != service._system_instruction_from_init
    assert service._system_instruction_from_init == "base prompt"


@pytest.mark.asyncio
async def test_connect_sends_composed_not_snapshot():
    """_connect must configure the session with the composed instruction."""
    service = _service(system_instruction="base prompt")
    service.register_function("check_order", AsyncMock(), cancel_on_interruption=False)
    service._sync_registered_tool_handlers(None)

    captured = {}
    _last_config = {}

    class FakeSession:
        pass

    class FakeConnect:
        async def __aenter__(self):
            captured["system_instruction"] = _last_config["config"].system_instruction
            return FakeSession()

        async def __aexit__(self, *exc):
            return False

    def fake_connect(model, config):
        _last_config["config"] = config
        return FakeConnect()

    with (
        patch.object(service._client.aio.live, "connect", side_effect=fake_connect),
        patch.object(service, "_handle_session_ready", new=AsyncMock()),
    ):
        await _setup_service(service)
        # the connection runs as a task; give it a beat to reach connect()
        for _ in range(100):
            if _last_config.get("config") is not None:
                break
            await asyncio.sleep(0.05)
        if service._connection_task:
            service._connection_task.cancel()
            try:
                await service._connection_task
            except asyncio.CancelledError:
                pass

    sent = captured["system_instruction"]
    assert sent is not None
    assert "base prompt" in sent
    # The async-tool guidance the base service composes must reach the wire.
    assert ASYNC_TOOL_INSTRUCTIONS in sent


@pytest.mark.asyncio
async def test_connect_without_addons_keeps_init_behavior():
    """No addons: composed == init value, nothing changes on the wire."""
    service = _service(system_instruction="just base")
    captured = {}
    _last_config = {}

    class FakeSession:
        pass

    class FakeConnect:
        async def __aenter__(self):
            captured["system_instruction"] = _last_config["config"].system_instruction
            return FakeSession()

        async def __aexit__(self, *exc):
            return False

    def fake_connect(model, config):
        _last_config["config"] = config
        return FakeConnect()

    with (
        patch.object(service._client.aio.live, "connect", side_effect=fake_connect),
        patch.object(service, "_handle_session_ready", new=AsyncMock()),
    ):
        await _setup_service(service)
        # the connection runs as a task; give it a beat to reach connect()
        for _ in range(100):
            if _last_config.get("config") is not None:
                break
            await asyncio.sleep(0.05)
        if service._connection_task:
            service._connection_task.cancel()
            try:
                await service._connection_task
            except asyncio.CancelledError:
                pass

    assert captured["system_instruction"] == "just base"


@pytest.mark.asyncio
async def test_runtime_settings_update_recomposes_and_reaches_connect():
    """A runtime system_instruction change flows into the next connect."""
    service = _service(system_instruction="first")
    captured = {}
    _last_config = {}

    class FakeSession:
        pass

    class FakeConnect:
        async def __aenter__(self):
            return FakeSession()

        async def __aexit__(self, *exc):
            return False

    connects = []

    def fake_connect(model, config):
        captured["system_instruction"] = config.system_instruction
        connects.append(config)
        return FakeConnect()

    with (
        patch.object(service._client.aio.live, "connect", side_effect=fake_connect),
        patch.object(service, "_handle_session_ready", new=AsyncMock()),
    ):
        await _setup_service(service)
        # the connection runs as a task; give it a beat to reach connect()
        for _ in range(100):
            if len(connects) >= 1:
                break
            await asyncio.sleep(0.05)
        if service._connection_task:
            service._connection_task.cancel()
            try:
                await service._connection_task
            except asyncio.CancelledError:
                pass
        assert captured["system_instruction"] == "first"
        await service._update_settings(GeminiLiveLLMService.Settings(system_instruction="second"))
        await service._reconnect()
        # reconnects land asynchronously; wait until the newest connect carries the
        # new instruction (internal retries may connect several times, last wins)
        for _ in range(100):
            if captured.get("system_instruction") == "second":
                break
            await asyncio.sleep(0.05)

    assert captured["system_instruction"] == "second"
