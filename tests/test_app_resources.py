#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from pipecat.adapters.schemas.direct_function import DirectFunctionWrapper
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import EndFrame, Frame, StartFrame
from pipecat.pipeline.base_task import PipelineTaskParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
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


def _complete_llm_settings() -> LLMSettings:
    """Return an LLMSettings with every field set so test_service_init's
    auto-discovered ``_MockLLMService`` doesn't fail its NOT_GIVEN check."""
    return LLMSettings(
        model=None,
        system_instruction=None,
        temperature=None,
        max_tokens=None,
        top_p=None,
        top_k=None,
        frequency_penalty=None,
        presence_penalty=None,
        seed=None,
        filter_incomplete_user_turns=None,
        user_turn_completion_config=None,
    )


class _MockLLMService(LLMService):
    def __init__(self, **kwargs):
        super().__init__(settings=_complete_llm_settings(), **kwargs)


class TestFunctionCallParamsAppResources(unittest.TestCase):
    def test_default_is_none(self):
        params = FunctionCallParams(
            function_name="f",
            tool_call_id="1",
            arguments={},
            llm=None,  # type: ignore[arg-type]
            context=LLMContext(),
            result_callback=AsyncMock(),
        )
        self.assertIsNone(params.app_resources)

    def test_holds_reference(self):
        resources = _Resources(user_name="John")
        params = FunctionCallParams(
            function_name="f",
            tool_call_id="1",
            arguments={},
            llm=None,  # type: ignore[arg-type]
            context=LLMContext(),
            result_callback=AsyncMock(),
            app_resources=resources,
        )
        self.assertIs(params.app_resources, resources)

    def test_tool_resources_property_warns_and_aliases_app_resources(self):
        resources = _Resources(user_name="John")
        params = FunctionCallParams(
            function_name="f",
            tool_call_id="1",
            arguments={},
            llm=None,  # type: ignore[arg-type]
            context=LLMContext(),
            result_callback=AsyncMock(),
            app_resources=resources,
        )
        with self.assertWarns(DeprecationWarning):
            value = params.tool_resources
        self.assertIs(value, resources)


class TestLLMServiceFunctionCallReadsAppResources(unittest.IsolatedAsyncioTestCase):
    async def test_function_call_params_receives_app_resources(self):
        service = _MockLLMService()
        resources = _Resources(user_name="John")
        # Stub the pipeline task with just the bit LLMService reads.
        service._pipeline_task = SimpleNamespace(app_resources=resources)  # type: ignore[assignment]

        captured: dict[str, Any] = {}

        async def handler(params: FunctionCallParams):
            captured["params"] = params
            params.app_resources.db["hit"] = True
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

        self.assertIs(captured["params"].app_resources, resources)
        self.assertTrue(resources.db["hit"])

    async def test_direct_function_params_receives_app_resources(self):
        service = _MockLLMService()
        resources = _Resources(user_name="John")
        service._pipeline_task = SimpleNamespace(app_resources=resources)  # type: ignore[assignment]
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

        self.assertIs(captured["params"].app_resources, resources)

    async def test_app_resources_none_when_pipeline_task_unset(self):
        service = _MockLLMService()
        captured: dict[str, Any] = {}

        async def handler(params: FunctionCallParams):
            captured["params"] = params
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

        self.assertIsNone(captured["params"].app_resources)

    async def test_frame_processor_setup_tool_resources_warns_on_read(self):
        # ``FrameProcessorSetup.tool_resources`` is retained for backwards
        # compatibility with custom FrameProcessors whose ``setup()`` overrides
        # still read it. The field is populated, but reading it warns.
        task_manager = TaskManager()
        task_manager.setup(TaskManagerParams(loop=asyncio.get_running_loop()))
        resources = _Resources(user_name="John")

        # Construction itself does not warn — only reads do.
        setup = FrameProcessorSetup(
            clock=SystemClock(),
            task_manager=task_manager,
            tool_resources=resources,
        )

        with self.assertWarns(DeprecationWarning):
            value = setup.tool_resources
        self.assertIs(value, resources)


class TestPipelineTaskAppResources(unittest.TestCase):
    def test_getter_returns_constructor_value(self):
        resources = _Resources(user_name="John")
        task = PipelineTask(Pipeline([]), app_resources=resources)
        self.assertIs(task.app_resources, resources)

    def test_default_app_resources_is_none(self):
        task = PipelineTask(Pipeline([]))
        self.assertIsNone(task.app_resources)

    def test_tool_resources_kwarg_warns_and_aliases_app_resources(self):
        resources = _Resources(user_name="John")
        with self.assertWarns(DeprecationWarning):
            task = PipelineTask(Pipeline([]), tool_resources=resources)
        self.assertIs(task.app_resources, resources)

    def test_app_resources_takes_precedence_over_tool_resources(self):
        new = _Resources(user_name="new")
        old = _Resources(user_name="old")
        with self.assertWarns(DeprecationWarning):
            task = PipelineTask(Pipeline([]), app_resources=new, tool_resources=old)
        self.assertIs(task.app_resources, new)


class _RecordingProcessor(FrameProcessor):
    """Records the pipeline_task it sees once StartFrame reaches it."""

    def __init__(self):
        super().__init__()
        self.observed_task: Any = None
        self.observed_app_resources: Any = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            # setup() runs before any frame reaches us, so pipeline_task is wired up.
            assert self.pipeline_task is not None
            self.observed_task = self.pipeline_task
            self.observed_app_resources = self.pipeline_task.app_resources
        await self.push_frame(frame, direction)


class _LegacyToolResourcesReader(FrameProcessor):
    """Custom processor that reads the deprecated ``setup.tool_resources``.

    Models a previously-written user FrameProcessor whose ``setup()``
    override hasn't been migrated yet. The field is populated by
    ``PipelineTask`` for backwards compatibility; reading it emits a
    DeprecationWarning.
    """

    def __init__(self):
        super().__init__()
        self.captured_tool_resources: Any = None

    async def setup(self, setup):
        await super().setup(setup)
        self.captured_tool_resources = setup.tool_resources

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Forward all frames so the EndFrame reaches the pipeline sink and
        # ``task.run()`` can return cleanly.
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class TestFrameProcessorSetupToolResourcesBackwardsCompat(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_processor_receives_value_via_app_resources(self):
        resources = _Resources(user_name="John")
        legacy = _LegacyToolResourcesReader()
        pipeline = Pipeline([legacy])
        task = PipelineTask(pipeline, app_resources=resources)

        await task.queue_frame(EndFrame())
        with self.assertWarns(DeprecationWarning):
            await task.run(PipelineTaskParams(loop=asyncio.get_event_loop()))

        self.assertIs(legacy.captured_tool_resources, resources)

    async def test_legacy_processor_receives_value_via_deprecated_tool_resources_kwarg(
        self,
    ):
        # If the user is still constructing PipelineTask with the deprecated
        # ``tool_resources`` kwarg (and hasn't migrated to ``app_resources``),
        # legacy processors must still see the value too.
        resources = _Resources(user_name="John")
        legacy = _LegacyToolResourcesReader()
        pipeline = Pipeline([legacy])
        with self.assertWarns(DeprecationWarning):
            task = PipelineTask(pipeline, tool_resources=resources)

        await task.queue_frame(EndFrame())
        with self.assertWarns(DeprecationWarning):
            await task.run(PipelineTaskParams(loop=asyncio.get_event_loop()))

        self.assertIs(legacy.captured_tool_resources, resources)


class TestFrameProcessorPipelineTaskAccess(unittest.IsolatedAsyncioTestCase):
    async def test_processor_can_reach_pipeline_task_and_app_resources(self):
        resources = _Resources(user_name="John")
        recorder = _RecordingProcessor()
        pipeline = Pipeline([recorder])
        task = PipelineTask(pipeline, app_resources=resources)

        await task.queue_frame(EndFrame())
        await task.run(PipelineTaskParams(loop=asyncio.get_event_loop()))

        self.assertIs(recorder.observed_task, task)
        self.assertIs(recorder.observed_app_resources, resources)

    def test_pipeline_task_returns_none_when_not_set_up(self):
        recorder = _RecordingProcessor()
        self.assertIsNone(recorder.pipeline_task)


if __name__ == "__main__":
    unittest.main()
