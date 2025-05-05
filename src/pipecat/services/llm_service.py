#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Optional, Protocol, Set, Tuple, Type

from loguru import logger

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter
from pipecat.frames.frames import (
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
    StartInterruptionFrame,
    UserImageRequestFrame,
)
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantAggregatorParams,
    LLMUserAggregatorParams,
)
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_service import AIService

# Type alias for a callable that handles LLM function calls.
FunctionCallHandler = Callable[["FunctionCallParams"], Awaitable[None]]


# Type alias for a callback function that handles the result of an LLM function call.
class FunctionCallResultCallback(Protocol):
    async def __call__(
        self, result: Any, *, properties: Optional[FunctionCallResultProperties] = None
    ) -> None: ...


@dataclass
class FunctionCallEntry:
    """Represents an internal entry for a function call.

    Attributes:
        function_name (Optional[str]): The name of the function.
        handler (FunctionCallHandler): The handler for processing function call parameters.
        cancel_on_interruption (bool): Flag indicating whether to cancel the call on interruption.

    """

    function_name: Optional[str]
    handler: FunctionCallHandler
    cancel_on_interruption: bool


@dataclass
class FunctionCallParams:
    """Parameters for a function call.

    Attributes:
        function_name (str): The name of the function being called.
        arguments (Mapping[str, Any]): The arguments for the function.
        tool_call_id (str): A unique identifier for the function call.
        llm (LLMService): The LLMService instance being used.
        context (OpenAILLMContext): The LLM context.
        result_callback (FunctionCallResultCallback): Callback to handle the result of the function call.

    """

    function_name: str
    tool_call_id: str
    arguments: Mapping[str, Any]
    llm: "LLMService"
    context: OpenAILLMContext
    result_callback: FunctionCallResultCallback


class LLMService(AIService):
    """This class is a no-op but serves as a base class for LLM services."""

    # OpenAILLMAdapter is used as the default adapter since it aligns with most LLM implementations.
    # However, subclasses should override this with a more specific adapter when necessary.
    adapter_class: Type[BaseLLMAdapter] = OpenAILLMAdapter

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._functions = {}
        self._start_callbacks = {}
        self._adapter = self.adapter_class()
        self._function_call_tasks: Set[Tuple[asyncio.Task, str, str]] = set()

        self._register_event_handler("on_completion_timeout")

    def get_llm_adapter(self) -> BaseLLMAdapter:
        return self._adapter

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> Any:
        pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            await self._handle_interruptions(frame)

    async def _handle_interruptions(self, frame: StartInterruptionFrame):
        for function_name, entry in self._functions.items():
            if entry.cancel_on_interruption:
                await self._cancel_function_call(function_name)

    def register_function(
        self,
        function_name: Optional[str],
        handler: Any,
        start_callback=None,
        *,
        cancel_on_interruption: bool = False,
    ):
        # Registering a function with the function_name set to None will run
        # that handler for all functions
        self._functions[function_name] = FunctionCallEntry(
            function_name=function_name,
            handler=handler,
            cancel_on_interruption=cancel_on_interruption,
        )

        # Start callbacks are now deprecated.
        if start_callback:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Parameter 'start_callback' is deprecated, just put your code on top of the actual function call instead.",
                    DeprecationWarning,
                )

            self._start_callbacks[function_name] = start_callback

    def unregister_function(self, function_name: Optional[str]):
        del self._functions[function_name]
        if self._start_callbacks[function_name]:
            del self._start_callbacks[function_name]

    def has_function(self, function_name: str):
        if None in self._functions.keys():
            return True
        return function_name in self._functions.keys()

    async def call_function(
        self,
        *,
        context: OpenAILLMContext,
        tool_call_id: str,
        function_name: str,
        arguments: Mapping[str, Any],
        run_llm: bool = True,
    ):
        if not function_name in self._functions.keys() and not None in self._functions.keys():
            return

        task = self.create_task(
            self._run_function_call(context, tool_call_id, function_name, arguments, run_llm)
        )

        self._function_call_tasks.add((task, tool_call_id, function_name))

        task.add_done_callback(self._function_call_task_finished)

    async def call_start_function(self, context: OpenAILLMContext, function_name: str):
        if function_name in self._start_callbacks.keys():
            await self._start_callbacks[function_name](function_name, self, context)
        elif None in self._start_callbacks.keys():
            return await self._start_callbacks[None](function_name, self, context)

    async def request_image_frame(
        self,
        user_id: str,
        *,
        function_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        text_content: Optional[str] = None,
    ):
        await self.push_frame(
            UserImageRequestFrame(
                user_id=user_id,
                function_name=function_name,
                tool_call_id=tool_call_id,
                context=text_content,
            ),
            FrameDirection.UPSTREAM,
        )

    async def _run_function_call(
        self,
        context: OpenAILLMContext,
        tool_call_id: str,
        function_name: str,
        arguments: Mapping[str, Any],
        run_llm: bool = True,
    ):
        if function_name in self._functions.keys():
            entry = self._functions[function_name]
        elif None in self._functions.keys():
            entry = self._functions[None]
        else:
            return

        logger.debug(
            f"{self} Calling function [{function_name}:{tool_call_id}] with arguments {arguments}"
        )

        # NOTE(aleix): This needs to be removed after we remove the deprecation.
        await self.call_start_function(context, function_name)

        # Push a SystemFrame downstream. This frame will let our assistant context aggregator
        # know that we are in the middle of a function call. Some contexts/aggregators may
        # not need this. But some definitely do (Anthropic, for example).
        # Also push a SystemFrame upstream for use by other processors, like STTMuteFilter.
        progress_frame_downstream = FunctionCallInProgressFrame(
            function_name=function_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
            cancel_on_interruption=entry.cancel_on_interruption,
        )
        progress_frame_upstream = FunctionCallInProgressFrame(
            function_name=function_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
            cancel_on_interruption=entry.cancel_on_interruption,
        )

        # Push frame both downstream and upstream
        await self.push_frame(progress_frame_downstream, FrameDirection.DOWNSTREAM)
        await self.push_frame(progress_frame_upstream, FrameDirection.UPSTREAM)

        # Define a callback function that pushes a FunctionCallResultFrame upstream & downstream.
        async def function_call_result_callback(
            result: Any, *, properties: Optional[FunctionCallResultProperties] = None
        ):
            result_frame_downstream = FunctionCallResultFrame(
                function_name=function_name,
                tool_call_id=tool_call_id,
                arguments=arguments,
                result=result,
                properties=properties,
            )
            result_frame_upstream = FunctionCallResultFrame(
                function_name=function_name,
                tool_call_id=tool_call_id,
                arguments=arguments,
                result=result,
                properties=properties,
            )

            await self.push_frame(result_frame_downstream, FrameDirection.DOWNSTREAM)
            await self.push_frame(result_frame_upstream, FrameDirection.UPSTREAM)

        signature = inspect.signature(entry.handler)
        if len(signature.parameters) > 1:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Function calls with parameters `(function_name, tool_call_id, arguments, llm, context, result_callback)` are deprecated, use a single `FunctionCallParams` parameter instead.",
                    DeprecationWarning,
                )

            await entry.handler(
                function_name, tool_call_id, arguments, self, context, function_call_result_callback
            )
        else:
            params = FunctionCallParams(
                function_name=function_name,
                tool_call_id=tool_call_id,
                arguments=arguments,
                llm=self,
                context=context,
                result_callback=function_call_result_callback,
            )
            await entry.handler(params)

    async def _cancel_function_call(self, function_name: str):
        cancelled_tasks = set()
        for task, tool_call_id, name in self._function_call_tasks:
            if name == function_name:
                # We remove the callback because we are going to cancel the task
                # now, otherwise we will be removing it from the set while we
                # are iterating.
                task.remove_done_callback(self._function_call_task_finished)

                logger.debug(f"{self} Cancelling function call [{name}:{tool_call_id}]...")

                await self.cancel_task(task)

                frame = FunctionCallCancelFrame(
                    function_name=function_name, tool_call_id=tool_call_id
                )
                await self.push_frame(frame)

                logger.debug(f"{self} Function call [{name}:{tool_call_id}] has been cancelled")

                cancelled_tasks.add(task)

        # Remove all cancelled tasks from our set.
        for task in cancelled_tasks:
            self._function_call_task_finished(task)

    def _function_call_task_finished(self, task: asyncio.Task):
        tuple_to_remove = next((t for t in self._function_call_tasks if t[0] == task), None)
        if tuple_to_remove:
            self._function_call_tasks.discard(tuple_to_remove)
            # The task is finished so this should exit immediately. We need to
            # do this because otherwise the task manager would report a dangling
            # task if we don't remove it.
            asyncio.run_coroutine_threadsafe(self.wait_for_task(task), self.get_event_loop())
