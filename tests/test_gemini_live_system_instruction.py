#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests that GeminiLiveLLMService sends the composed system instruction.

`LLMService._compose_system_instruction()` rebuilds `self._settings.system_instruction`
from the base prompt plus active addons (async-tool guidance, cancellation guidance,
appended instructions). Gemini Live must connect with that composed value, not the
instruction captured at construction time.
"""

from unittest.mock import MagicMock

import pytest

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.async_tool_messages import ASYNC_TOOL_INSTRUCTIONS
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService


def _service(**settings) -> GeminiLiveLLMService:
    return GeminiLiveLLMService(
        api_key="test-key", settings=GeminiLiveLLMService.Settings(**settings)
    )


def _capture_connect_config(service: GeminiLiveLLMService) -> dict:
    """Patch the connection task entry point to capture the connect config."""
    captured = {}

    def fake_connection_task_handler(*, config):
        captured["config"] = config

        async def _noop():
            return None

        return _noop()

    def fake_create_task(coro, *, name=None, **kwargs):
        coro.close()  # never awaited; avoid "coroutine was never awaited"
        return MagicMock()

    service._connection_task_handler = fake_connection_task_handler
    service.create_task = fake_create_task
    return captured


@pytest.mark.asyncio
async def test_connect_sends_init_system_instruction_without_addons():
    service = _service(system_instruction="You are a helpful assistant.")
    captured = _capture_connect_config(service)
    await service._connect()
    assert captured["config"].system_instruction == "You are a helpful assistant."


@pytest.mark.asyncio
async def test_connect_sends_composed_system_instruction_with_async_tool():
    service = _service(system_instruction="You are a helpful assistant.")

    @tool_options(cancel_on_interruption=False)
    async def handler(params):
        await params.result_callback({})

    tools = ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="slow_lookup",
                description="A slow lookup.",
                properties={},
                required=[],
                handler=handler,
            )
        ]
    )
    service._sync_registered_tool_handlers(tools)

    # Composition ran: the settings now carry the async-tool guidance.
    composed = service._settings.system_instruction
    assert composed is not None
    assert "You are a helpful assistant." in composed
    assert composed != "You are a helpful assistant."

    captured = _capture_connect_config(service)
    await service._connect()
    assert captured["config"].system_instruction == composed
    assert ASYNC_TOOL_INSTRUCTIONS in captured["config"].system_instruction


@pytest.mark.asyncio
async def test_connect_sends_appended_system_instruction():
    service = _service(system_instruction="You are a helpful assistant.")
    service.append_system_instruction("Always answer in haiku.")

    composed = service._settings.system_instruction
    captured = _capture_connect_config(service)
    await service._connect()
    assert captured["config"].system_instruction == composed
    assert "Always answer in haiku." in captured["config"].system_instruction
