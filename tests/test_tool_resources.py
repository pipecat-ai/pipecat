#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock

from pipecat.adapters.schemas.direct_function import DirectFunctionWrapper
from pipecat.clocks.system_clock import SystemClock
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.llm_service import (
    FunctionCallParams,
    FunctionCallRegistryItem,
    FunctionCallRunnerItem,
    LLMService,
)
from pipecat.services.settings import LLMSettings
from pipecat.utils.asyncio.task_manager import TaskManager, TaskManagerParams


@dataclass
class _Resources:
    user_name: str
    db: dict[str, Any] = field(default_factory=dict)


class _MockLLMService(LLMService):
    def __init__(self, **kwargs):
        super().__init__(settings=LLMSettings(), **kwargs)


class TestFunctionCallParamsToolResources(unittest.TestCase):
    def test_default_is_none(self):
        params = FunctionCallParams(
            function_name="f",
            tool_call_id="1",
            arguments={},
            llm=None,  # type: ignore[arg-type]
            context=LLMContext(),
            result_callback=AsyncMock(),
        )
        self.assertIsNone(params.tool_resources)

    def test_holds_reference(self):
        resources = _Resources(user_name="John")
        params = FunctionCallParams(
            function_name="f",
            tool_call_id="1",
            arguments={},
            llm=None,  # type: ignore[arg-type]
            context=LLMContext(),
            result_callback=AsyncMock(),
            tool_resources=resources,
        )
        self.assertIs(params.tool_resources, resources)


class TestLLMServiceCachesToolResources(unittest.IsolatedAsyncioTestCase):
    async def test_setup_caches_tool_resources(self):
        service = _MockLLMService()
        resources = _Resources(user_name="John")
        task_manager = TaskManager()
        task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))

        await service.setup(
            FrameProcessorSetup(
                clock=SystemClock(),
                task_manager=task_manager,
                tool_resources=resources,
            )
        )
        await asyncio.sleep(0)
        await service.cleanup()

        self.assertIs(service._tool_resources, resources)

    async def test_function_call_params_receives_tool_resources(self):
        service = _MockLLMService()
        resources = _Resources(user_name="John")
        service._tool_resources = resources

        captured: dict[str, Any] = {}

        async def handler(params: FunctionCallParams):
            captured["params"] = params
            params.tool_resources.db["hit"] = True
            await params.result_callback({"ok": True})

        service._functions["lookup"] = FunctionCallRegistryItem(
            function_name="lookup",
            handler=handler,
            cancel_on_interruption=True,
        )
        service.broadcast_frame = AsyncMock()  # type: ignore[method-assign]

        runner_item = FunctionCallRunnerItem(
            registry_item=service._functions["lookup"],
            function_name="lookup",
            tool_call_id="call-1",
            arguments={},
            context=LLMContext(),
        )
        await service._run_function_call(runner_item)

        self.assertIs(captured["params"].tool_resources, resources)
        self.assertTrue(resources.db["hit"])

    async def test_direct_function_params_receives_tool_resources(self):
        service = _MockLLMService()
        resources = _Resources(user_name="John")
        service._tool_resources = resources
        captured: dict[str, Any] = {}

        async def lookup(params: FunctionCallParams):
            captured["params"] = params

        wrapper = DirectFunctionWrapper(lookup)
        service._functions[wrapper.name] = FunctionCallRegistryItem(
            function_name=wrapper.name,
            handler=wrapper,
            cancel_on_interruption=True,
        )
        service.broadcast_frame = AsyncMock()  # type: ignore[method-assign]

        runner_item = FunctionCallRunnerItem(
            registry_item=service._functions[wrapper.name],
            function_name=wrapper.name,
            tool_call_id="call-1",
            arguments={},
            context=LLMContext(),
        )
        await service._run_function_call(runner_item)

        self.assertIs(captured["params"].tool_resources, resources)
