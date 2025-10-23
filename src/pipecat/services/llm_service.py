#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base classes for Large Language Model services with function calling support."""

import asyncio
import inspect
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Type,
)

from loguru import logger

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.direct_function import DirectFunction, DirectFunctionWrapper
from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMConfigureOutputFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    StartFrame,
    UserImageRequestFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage
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
    """Protocol for function call result callbacks.

    Handles the result of an LLM function call execution.
    """

    async def __call__(
        self, result: Any, *, properties: Optional[FunctionCallResultProperties] = None
    ) -> None:
        """Call the result callback.

        Args:
            result: The result of the function call.
            properties: Optional properties for the result.
        """
        ...


@dataclass
class FunctionCallParams:
    """Parameters for a function call.

    Parameters:
        function_name: The name of the function being called.
        tool_call_id: A unique identifier for the function call.
        arguments: The arguments for the function.
        llm: The LLMService instance being used.
        context: The LLM context.
        result_callback: Callback to handle the result of the function call.
    """

    function_name: str
    tool_call_id: str
    arguments: Mapping[str, Any]
    llm: "LLMService"
    context: OpenAILLMContext | LLMContext
    result_callback: FunctionCallResultCallback


@dataclass
class FunctionCallRegistryItem:
    """Represents an entry in the function call registry.

    This is what the user registers when calling register_function.

    Parameters:
        function_name: The name of the function (None for catch-all handler).
        handler: The handler for processing function call parameters.
        cancel_on_interruption: Whether to cancel the call on interruption.
    """

    function_name: Optional[str]
    handler: FunctionCallHandler | "DirectFunctionWrapper"
    cancel_on_interruption: bool
    handler_deprecated: bool


@dataclass
class FunctionCallRunnerItem:
    """Internal function call entry for the function call runner.

    The runner executes function calls in order.

    Parameters:
        registry_item: The registry item containing handler information.
        function_name: The name of the function.
        tool_call_id: A unique identifier for the function call.
        arguments: The arguments for the function.
        context: The LLM context.
        run_llm: Optional flag to control LLM execution after function call.
    """

    registry_item: FunctionCallRegistryItem
    function_name: str
    tool_call_id: str
    arguments: Mapping[str, Any]
    context: OpenAILLMContext | LLMContext
    run_llm: Optional[bool] = None


class LLMService(AIService):
    """Base class for all LLM services.

    Handles function calling registration and execution with support for both
    parallel and sequential execution modes. Provides event handlers for
    completion timeouts and function call lifecycle events.

    The service supports the following event handlers:

    - on_completion_timeout: Called when an LLM completion timeout occurs
    - on_function_calls_started: Called when function calls are received and
      execution is about to start

    Example::

        @task.event_handler("on_completion_timeout")
        async def on_completion_timeout(service):
            logger.warning("LLM completion timed out")

        @task.event_handler("on_function_calls_started")
        async def on_function_calls_started(service, function_calls):
            logger.info(f"Starting {len(function_calls)} function calls")
    """

    # OpenAILLMAdapter is used as the default adapter since it aligns with most LLM implementations.
    # However, subclasses should override this with a more specific adapter when necessary.
    adapter_class: Type[BaseLLMAdapter] = OpenAILLMAdapter

    def __init__(self, run_in_parallel: bool = True, **kwargs):
        """Initialize the LLM service.

        Args:
            run_in_parallel: Whether to run function calls in parallel or sequentially.
                Defaults to True.
            **kwargs: Additional arguments passed to the parent AIService.
        """
        super().__init__(**kwargs)
        self._run_in_parallel = run_in_parallel
        self._start_callbacks = {}
        self._adapter = self.adapter_class()
        self._functions: Dict[Optional[str], FunctionCallRegistryItem] = {}
        self._function_call_tasks: Dict[asyncio.Task, FunctionCallRunnerItem] = {}
        self._sequential_runner_task: Optional[asyncio.Task] = None
        self._tracing_enabled: bool = False
        self._skip_tts: bool = False

        self._register_event_handler("on_function_calls_started")
        self._register_event_handler("on_completion_timeout")

    def get_llm_adapter(self) -> BaseLLMAdapter:
        """Get the LLM adapter instance.

        Returns:
            The adapter instance used for LLM communication.
        """
        return self._adapter

    def create_llm_specific_message(self, message: Any) -> LLMSpecificMessage:
        """Create an LLM-specific message (as opposed to a standard message) for use in an LLMContext.

        Args:
            message: The message content.

        Returns:
            A LLMSpecificMessage instance.
        """
        return self.get_llm_adapter().create_llm_specific_message(message)

    async def run_inference(self, context: LLMContext | OpenAILLMContext) -> Optional[str]:
        """Run a one-shot, out-of-band (i.e. out-of-pipeline) inference with the given LLM context.

        Must be implemented by subclasses.

        Args:
            context: The LLM context containing conversation history.

        Returns:
            The LLM's response as a string, or None if no response is generated.
        """
        raise NotImplementedError(f"run_inference() not supported by {self.__class__.__name__}")

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> Any:
        """Create a context aggregator for managing LLM conversation context.

        Must be implemented by subclasses.

        Args:
            context: The LLM context to create an aggregator for.
            user_params: Parameters for user message aggregation.
            assistant_params: Parameters for assistant message aggregation.

        Returns:
            A context aggregator instance.
        """
        pass

    async def start(self, frame: StartFrame):
        """Start the LLM service.

        Args:
            frame: The start frame.
        """
        await super().start(frame)
        if not self._run_in_parallel:
            await self._create_sequential_runner_task()
        self._tracing_enabled = frame.enable_tracing

    async def stop(self, frame: EndFrame):
        """Stop the LLM service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        if not self._run_in_parallel:
            await self._cancel_sequential_runner_task()

    async def cancel(self, frame: CancelFrame):
        """Cancel the LLM service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        if not self._run_in_parallel:
            await self._cancel_sequential_runner_task()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            await self._handle_interruptions(frame)
        elif isinstance(frame, LLMConfigureOutputFrame):
            self._skip_tts = frame.skip_tts

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Pushes a frame.

        Args:
            frame: The frame to push.
            direction: The direction of frame pushing.
        """
        if isinstance(frame, (LLMTextFrame, LLMFullResponseStartFrame, LLMFullResponseEndFrame)):
            frame.skip_tts = self._skip_tts

        await super().push_frame(frame, direction)

    async def _handle_interruptions(self, _: InterruptionFrame):
        for function_name, entry in self._functions.items():
            if entry.cancel_on_interruption:
                await self._cancel_function_call(function_name)

    def register_function(
        self,
        function_name: Optional[str],
        handler: Any,
        start_callback=None,
        *,
        cancel_on_interruption: bool = True,
    ):
        """Register a function handler for LLM function calls.

        Args:
            function_name: The name of the function to handle. Use None to handle
                all function calls with a catch-all handler.
            handler: The function handler. Should accept a single FunctionCallParams
                parameter.
            start_callback: Legacy callback function (deprecated). Put initialization
                code at the top of your handler instead.

                .. deprecated:: 0.0.59
                    The `start_callback` parameter is deprecated and will be removed in a future version.

            cancel_on_interruption: Whether to cancel this function call when an
                interruption occurs. Defaults to True.
        """
        signature = inspect.signature(handler)
        handler_deprecated = len(signature.parameters) > 1
        if handler_deprecated:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Function calls with parameters `(function_name, tool_call_id, arguments, llm, context, result_callback)` are deprecated, use a single `FunctionCallParams` parameter instead.",
                    DeprecationWarning,
                )

        # Registering a function with the function_name set to None will run
        # that handler for all functions
        self._functions[function_name] = FunctionCallRegistryItem(
            function_name=function_name,
            handler=handler,
            cancel_on_interruption=cancel_on_interruption,
            handler_deprecated=handler_deprecated,
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

    def register_direct_function(
        self,
        handler: DirectFunction,
        *,
        cancel_on_interruption: bool = True,
    ):
        """Register a direct function handler for LLM function calls.

        Direct functions have their metadata automatically extracted from their
        signature and docstring, eliminating the need for accompanying
        configurations (as FunctionSchemas or in provider-specific formats).

        Args:
            handler: The direct function to register. Must follow DirectFunction protocol.
            cancel_on_interruption: Whether to cancel this function call when an
                interruption occurs. Defaults to True.
        """
        wrapper = DirectFunctionWrapper(handler)
        self._functions[wrapper.name] = FunctionCallRegistryItem(
            function_name=wrapper.name,
            handler=wrapper,
            cancel_on_interruption=cancel_on_interruption,
            handler_deprecated=False,
        )

    def unregister_function(self, function_name: Optional[str]):
        """Remove a registered function handler.

        Args:
            function_name: The name of the function handler to remove.
        """
        del self._functions[function_name]
        if self._start_callbacks[function_name]:
            del self._start_callbacks[function_name]

    def unregister_direct_function(self, handler: Any):
        """Remove a registered direct function handler.

        Args:
            handler: The direct function handler to remove.
        """
        wrapper = DirectFunctionWrapper(handler)
        del self._functions[wrapper.name]
        # Note: no need to remove start callback here, as direct functions don't support start callbacks.

    def has_function(self, function_name: str):
        """Check if a function handler is registered.

        Args:
            function_name: The name of the function to check.

        Returns:
            True if the function is registered or if a catch-all handler (None)
            is registered.
        """
        if None in self._functions.keys():
            return True
        return function_name in self._functions.keys()

    def needs_mcp_alternate_schema(self) -> bool:
        """Check if this LLM service requires alternate MCP schema.

        Some LLM services have stricter JSON schema validation and require
        certain properties to be removed or modified for compatibility.

        Returns:
            True if MCP schemas should be cleaned for this service, False otherwise.
        """
        return False

    async def run_function_calls(self, function_calls: Sequence[FunctionCallFromLLM]):
        """Execute a sequence of function calls from the LLM.

        Triggers the on_function_calls_started event and executes functions
        either in parallel or sequentially based on the run_in_parallel setting.

        Args:
            function_calls: The function calls to execute.
        """
        if len(function_calls) == 0:
            return

        await self._call_event_handler("on_function_calls_started", function_calls)

        # Push frame both downstream and upstream
        started_frame_downstream = FunctionCallsStartedFrame(function_calls=function_calls)
        started_frame_upstream = FunctionCallsStartedFrame(function_calls=function_calls)
        await self.push_frame(started_frame_downstream, FrameDirection.DOWNSTREAM)
        await self.push_frame(started_frame_upstream, FrameDirection.UPSTREAM)

        for function_call in function_calls:
            if function_call.function_name in self._functions.keys():
                item = self._functions[function_call.function_name]
            elif None in self._functions.keys():
                item = self._functions[None]
            else:
                logger.warning(
                    f"{self} is calling '{function_call.function_name}', but it's not registered."
                )
                continue

            runner_item = FunctionCallRunnerItem(
                registry_item=item,
                function_name=function_call.function_name,
                tool_call_id=function_call.tool_call_id,
                arguments=function_call.arguments,
                context=function_call.context,
            )

            if self._run_in_parallel:
                task = self.create_task(self._run_function_call(runner_item))
                self._function_call_tasks[task] = runner_item
                task.add_done_callback(self._function_call_task_finished)
            else:
                await self._sequential_runner_queue.put(runner_item)

    async def _call_start_function(
        self, context: OpenAILLMContext | LLMContext, function_name: str
    ):
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
        video_source: Optional[str] = None,
    ):
        """Request an image from a user.

        Pushes a UserImageRequestFrame upstream to request an image from the
        specified user.

        Args:
            user_id: The ID of the user to request an image from.
            function_name: Optional function name associated with the request.
            tool_call_id: Optional tool call ID associated with the request.
            text_content: Optional text content/context for the image request.
            video_source: Optional video source identifier.
        """
        await self.push_frame(
            UserImageRequestFrame(
                user_id=user_id,
                function_name=function_name,
                tool_call_id=tool_call_id,
                context=text_content,
                video_source=video_source,
            ),
            FrameDirection.UPSTREAM,
        )

    async def _create_sequential_runner_task(self):
        if not self._sequential_runner_task:
            self._sequential_runner_queue = asyncio.Queue()
            self._sequential_runner_task = self.create_task(self._sequential_runner_handler())

    async def _cancel_sequential_runner_task(self):
        if self._sequential_runner_task:
            await self.cancel_task(self._sequential_runner_task)
            self._sequential_runner_task = None

    async def _sequential_runner_handler(self):
        while True:
            runner_item = await self._sequential_runner_queue.get()
            task = self.create_task(self._run_function_call(runner_item))
            self._function_call_tasks[task] = runner_item
            # Since we run tasks sequentially we don't need to call
            # task.add_done_callback(self._function_call_task_finished).
            await task
            del self._function_call_tasks[task]

    async def _run_function_call(self, runner_item: FunctionCallRunnerItem):
        if runner_item.function_name in self._functions.keys():
            item = self._functions[runner_item.function_name]
        elif None in self._functions.keys():
            item = self._functions[None]
        else:
            return

        logger.debug(
            f"{self} Calling function [{runner_item.function_name}:{runner_item.tool_call_id}] with arguments {runner_item.arguments}"
        )

        # NOTE(aleix): This needs to be removed after we remove the deprecation.
        await self._call_start_function(runner_item.context, runner_item.function_name)

        # Push a function call in-progress downstream. This frame will let our
        # assistant context aggregator know that we are in the middle of a
        # function call. Some contexts/aggregators may not need this. But some
        # definitely do (Anthropic, for example).  Also push it upstream for use
        # by other processors, like STTMuteFilter.
        progress_frame_downstream = FunctionCallInProgressFrame(
            function_name=runner_item.function_name,
            tool_call_id=runner_item.tool_call_id,
            arguments=runner_item.arguments,
            cancel_on_interruption=item.cancel_on_interruption,
        )
        progress_frame_upstream = FunctionCallInProgressFrame(
            function_name=runner_item.function_name,
            tool_call_id=runner_item.tool_call_id,
            arguments=runner_item.arguments,
            cancel_on_interruption=item.cancel_on_interruption,
        )

        # Push frame both downstream and upstream
        await self.push_frame(progress_frame_downstream, FrameDirection.DOWNSTREAM)
        await self.push_frame(progress_frame_upstream, FrameDirection.UPSTREAM)

        # Define a callback function that pushes a FunctionCallResultFrame upstream & downstream.
        async def function_call_result_callback(
            result: Any, *, properties: Optional[FunctionCallResultProperties] = None
        ):
            result_frame_downstream = FunctionCallResultFrame(
                function_name=runner_item.function_name,
                tool_call_id=runner_item.tool_call_id,
                arguments=runner_item.arguments,
                result=result,
                run_llm=runner_item.run_llm,
                properties=properties,
            )
            result_frame_upstream = FunctionCallResultFrame(
                function_name=runner_item.function_name,
                tool_call_id=runner_item.tool_call_id,
                arguments=runner_item.arguments,
                result=result,
                run_llm=runner_item.run_llm,
                properties=properties,
            )

            await self.push_frame(result_frame_downstream, FrameDirection.DOWNSTREAM)
            await self.push_frame(result_frame_upstream, FrameDirection.UPSTREAM)

        if isinstance(item.handler, DirectFunctionWrapper):
            # Handler is a DirectFunctionWrapper
            await item.handler.invoke(
                args=runner_item.arguments,
                params=FunctionCallParams(
                    function_name=runner_item.function_name,
                    tool_call_id=runner_item.tool_call_id,
                    arguments=runner_item.arguments,
                    llm=self,
                    context=runner_item.context,
                    result_callback=function_call_result_callback,
                ),
            )
        else:
            # Handler is a FunctionCallHandler
            if item.handler_deprecated:
                await item.handler(
                    runner_item.function_name,
                    runner_item.tool_call_id,
                    runner_item.arguments,
                    self,
                    runner_item.context,
                    function_call_result_callback,
                )
            else:
                params = FunctionCallParams(
                    function_name=runner_item.function_name,
                    tool_call_id=runner_item.tool_call_id,
                    arguments=runner_item.arguments,
                    llm=self,
                    context=runner_item.context,
                    result_callback=function_call_result_callback,
                )
                await item.handler(params)

    async def _cancel_function_call(self, function_name: Optional[str]):
        cancelled_tasks = set()
        for task, runner_item in self._function_call_tasks.items():
            if runner_item.registry_item.function_name == function_name:
                name = runner_item.function_name
                tool_call_id = runner_item.tool_call_id

                # We remove the callback because we are going to cancel the task
                # now, otherwise we will be removing it from the set while we
                # are iterating.
                task.remove_done_callback(self._function_call_task_finished)

                logger.debug(f"{self} Cancelling function call [{name}:{tool_call_id}]...")

                await self.cancel_task(task)

                frame = FunctionCallCancelFrame(function_name=name, tool_call_id=tool_call_id)
                await self.push_frame(frame)

                cancelled_tasks.add(task)

                logger.debug(f"{self} Function call [{name}:{tool_call_id}] has been cancelled")

        # Remove all cancelled tasks from our set.
        for task in cancelled_tasks:
            self._function_call_task_finished(task)

    def _function_call_task_finished(self, task: asyncio.Task):
        if task in self._function_call_tasks:
            del self._function_call_tasks[task]
