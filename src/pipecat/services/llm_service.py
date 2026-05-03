#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base classes for Large Language Model services with function calling support."""

from __future__ import annotations

import asyncio
import json
import uuid
import warnings
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    Protocol,
    cast,
)

from loguru import logger
from typing_extensions import TypeVar
from websockets.exceptions import ConnectionClosed
from websockets.protocol import State

from pipecat.adapters.base_llm_adapter import BaseLLMAdapter
from pipecat.adapters.schemas.direct_function import DirectFunction, DirectFunctionWrapper
from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallFromLLM,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
    FunctionCallsStartedFrame,
    InterruptionFrame,
    LLMConfigureOutputFrame,
    LLMContextSummaryRequestFrame,
    LLMContextSummaryResultFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    StartFrame,
)
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
    LLMSpecificMessage,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_service import AIService
from pipecat.services.settings import LLMSettings, assert_given
from pipecat.services.websocket_service import WebsocketService
from pipecat.turns.user_turn_completion_mixin import UserTurnCompletionLLMServiceMixin
from pipecat.utils.async_tool_cancellation import (
    ASYNC_TOOL_CANCELLATION_INSTRUCTIONS,
    CANCEL_ASYNC_TOOL_NAME,
    CANCEL_ASYNC_TOOL_SCHEMA,
)
from pipecat.utils.context.llm_context_summarization import (
    DEFAULT_SUMMARIZATION_TIMEOUT,
    LLMContextSummarizationUtil,
)

# Type alias for a callable that handles LLM function calls.
FunctionCallHandler = Callable[["FunctionCallParams"], Awaitable[None]]


# Type alias for a callback function that handles the result of an LLM function call.
class FunctionCallResultCallback(Protocol):
    """Protocol for function call result callbacks.

    Used for both final results and intermediate updates. Pass
    ``properties=FunctionCallResultProperties(is_final=False)`` to send an
    intermediate update (only valid for async function calls registered with
    ``cancel_on_interruption=False``).
    """

    async def __call__(
        self, result: Any, *, properties: FunctionCallResultProperties | None = None
    ) -> None:
        """Call the result callback.

        Args:
            result: The result of the function call, or an intermediate update.
            properties: Optional properties. Set ``is_final=False`` to send an
                intermediate update instead of the final result.
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
        result_callback: Callback to deliver the result of the function call.
            For async function calls (``cancel_on_interruption=False``), call
            it with ``properties=FunctionCallResultProperties(is_final=False)``
            to push intermediate updates before the final result.
        app_resources: The application-defined resources passed to
            ``PipelineTask(..., app_resources=...)``. Same object — passed by
            reference, not a copy. Use it to share DB handles, clients, state,
            feature flags, etc. across all of a session's tool handlers.
    """

    function_name: str
    tool_call_id: str
    arguments: Mapping[str, Any]
    # `LLMService[Any]` so any concrete subclass (regardless of how — or
    # whether — it parameterizes the adapter type) can be assigned here.
    # Plain `LLMService` would invoke the TypeVar default and pyright would
    # treat it invariantly, rejecting `LLMService[XAdapter]` at the call
    # sites that build FunctionCallParams.
    llm: LLMService[Any]
    context: LLMContext
    result_callback: FunctionCallResultCallback
    app_resources: Any = None

    @property
    def tool_resources(self) -> Any:
        """Deprecated alias for :attr:`app_resources`.

        .. deprecated:: 1.2.0
            Use :attr:`app_resources` instead. ``tool_resources`` will be
            removed in a future version.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.warn(
                "`FunctionCallParams.tool_resources` is deprecated since 1.2.0, "
                "use `app_resources` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self.app_resources


@dataclass
class FunctionCallRegistryItem:
    """Represents an entry in the function call registry.

    This is what the user registers when calling register_function.

    Parameters:
        function_name: The name of the function (None for catch-all handler).
        handler: The handler for processing function call parameters.
        cancel_on_interruption: Whether to cancel the call on interruption.
            When ``False`` the call is treated as asynchronous: the LLM
            continues the conversation immediately without waiting for the
            result, and the result is injected later via a developer message.
        timeout_secs: Optional per-tool timeout in seconds. Overrides the global
            ``function_call_timeout_secs`` for this specific function.
    """

    function_name: str | None
    handler: FunctionCallHandler | DirectFunctionWrapper
    cancel_on_interruption: bool
    timeout_secs: float | None = None


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
        group_id: Shared identifier for all function calls from the same LLM
            response batch. Used to trigger the LLM exactly once when the last
            call in the group completes.
    """

    registry_item: FunctionCallRegistryItem
    function_name: str
    tool_call_id: str
    arguments: Mapping[str, Any]
    context: LLMContext
    run_llm: bool | None = None
    group_id: str | None = None


# `default=BaseLLMAdapter` (PEP 696) so that unparameterized subclasses
# (e.g. third-party `class MyService(LLMService):` with no bracket) get
# `TAdapter = BaseLLMAdapter` instead of `Unknown` at type-check time —
# matching the pre-generic behavior of `get_llm_adapter()`.
TAdapter = TypeVar("TAdapter", bound=BaseLLMAdapter, default=BaseLLMAdapter)


class LLMService(UserTurnCompletionLLMServiceMixin, AIService, Generic[TAdapter]):
    """Base class for all LLM services.

    Handles function calling registration and execution with support for both
    parallel and sequential execution modes. Provides event handlers for
    completion timeouts and function call lifecycle events.

    The service supports the following event handlers:

    - on_completion_timeout: Called when an LLM completion timeout occurs
    - on_function_calls_started: Called when function calls are received and
      execution is about to start. Built-in tools (e.g. ``cancel_async_tool_call``)
      are excluded from this event.
    - on_function_calls_cancelled: Called after one or more async tool calls are
      cancelled.

    Example::

        @task.event_handler("on_completion_timeout")
        async def on_completion_timeout(service):
            logger.warning("LLM completion timed out")

        @task.event_handler("on_function_calls_started")
        async def on_function_calls_started(service, function_calls: List[FunctionCallFromLLM]):
            logger.info(f"Starting {len(function_calls)} function calls")

        @task.event_handler("on_function_calls_cancelled")
        async def on_function_calls_cancelled(service, function_calls: List[FunctionCallFromLLM]):
            logger.info(f"Cancelled {len(function_calls)} function calls")
    """

    _settings: LLMSettings
    _adapter: TAdapter

    # OpenAILLMAdapter is used as the default adapter since it aligns with most LLM implementations.
    # However, subclasses should override this with a more specific adapter when necessary.
    adapter_class: type[BaseLLMAdapter] = OpenAILLMAdapter

    def __init__(
        self,
        run_in_parallel: bool = True,
        group_parallel_tools: bool = True,
        function_call_timeout_secs: float | None = None,
        enable_async_tool_cancellation: bool = False,
        settings: LLMSettings | None = None,
        **kwargs,
    ):
        """Initialize the LLM service.

        Args:
            run_in_parallel: Whether to run function calls in parallel or sequentially.
                Defaults to True.
            group_parallel_tools: Whether to group parallel function calls so the LLM
                is triggered exactly once after all calls in the batch complete. When
                False, each function call result triggers the LLM independently as it
                arrives. Defaults to True.
            function_call_timeout_secs: Optional timeout in seconds for deferred function
                calls.
            enable_async_tool_cancellation: When True and at least one async function
                (``cancel_on_interruption=False``) is registered, automatically injects
                the ``cancel_async_tool_call`` built-in tool and its system instructions
                so the LLM can cancel stale in-progress calls. Defaults to False.
            settings: The runtime-updatable settings for the LLM service.
            **kwargs: Additional arguments passed to the parent AIService.

        """
        super().__init__(
            settings=settings
            # Here in case subclass doesn't implement more specific settings
            # (which hopefully should be rare)
            or LLMSettings(),
            **kwargs,
        )
        self._run_in_parallel = run_in_parallel
        self._group_parallel_tools = group_parallel_tools
        self._function_call_timeout_secs = function_call_timeout_secs
        self._enable_async_tool_cancellation: bool = enable_async_tool_cancellation
        self._filter_incomplete_user_turns: bool = False
        self._async_tool_cancellation_enabled: bool = False
        self._base_system_instruction: str | None = None
        # `adapter_class` is typed as `type[BaseLLMAdapter]` so subclasses
        # don't need to spell out the generic parameter just to subclass
        # (backward compatibility for 3rd-party providers outside this repo).
        # Cast to TAdapter to keep `_adapter` and `get_llm_adapter()` precisely
        # typed for callers that opt into `LLMService[XAdapter]`.
        self._adapter = cast(TAdapter, self.adapter_class())
        self._functions: dict[str | None, FunctionCallRegistryItem] = {}
        self._function_call_tasks: dict[asyncio.Task | None, FunctionCallRunnerItem] = {}
        self._sequential_runner_task: asyncio.Task | None = None
        self._skip_tts: bool | None = None
        self._summary_task: asyncio.Task | None = None

        self._register_event_handler("on_function_calls_started")
        self._register_event_handler("on_function_calls_cancelled")
        self._register_event_handler("on_completion_timeout")

    def get_llm_adapter(self) -> TAdapter:
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

    async def run_inference(
        self,
        context: LLMContext,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
    ) -> str | None:
        """Run a one-shot, out-of-band (i.e. out-of-pipeline) inference with the given LLM context.

        Must be implemented by subclasses.

        Args:
            context: The LLM context containing conversation history.
            max_tokens: Optional maximum number of tokens to generate. If provided,
                overrides the service's default max_tokens/max_completion_tokens setting.
            system_instruction: Optional system instruction to use for this inference.
                If provided, overrides any system instruction in the context.

        Returns:
            The LLM's response as a string, or None if no response is generated.
        """
        raise NotImplementedError(f"run_inference() not supported by {self.__class__.__name__}")

    async def start(self, frame: StartFrame):
        """Start the LLM service.

        Args:
            frame: The start frame.
        """
        await super().start(frame)
        if not self._run_in_parallel:
            await self._create_sequential_runner_task()
        if self._enable_async_tool_cancellation and self._has_async_tools():
            self._setup_async_tool_cancellation()

    async def stop(self, frame: EndFrame):
        """Stop the LLM service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        if not self._run_in_parallel:
            await self._cancel_sequential_runner_task()
        await self._cancel_summary_task()

    async def cancel(self, frame: CancelFrame):
        """Cancel the LLM service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        if not self._run_in_parallel:
            await self._cancel_sequential_runner_task()
        await self._cancel_summary_task()

    def _compose_system_instruction(self):
        """Compose system_instruction from the base and all active addon instructions.

        Combines the base system instruction with turn completion instructions
        (when enabled) and async tool cancellation instructions (when enabled),
        writing the result to ``self._settings.system_instruction``.
        """
        base = self._base_system_instruction
        parts = [base] if base else []
        if self._filter_incomplete_user_turns:
            parts.append(self._user_turn_completion_config.completion_instructions)
        if self._async_tool_cancellation_enabled:
            parts.append(ASYNC_TOOL_CANCELLATION_INSTRUCTIONS)
        composed = "\n\n".join(p for p in parts if p)
        self._settings.system_instruction = composed or None
        logger.debug(f"{self}: System instruction composed: {self._settings.system_instruction}")

    async def _update_settings(self, delta: LLMSettings) -> dict[str, Any]:
        """Apply a settings delta, handling turn-completion fields.

        Args:
            delta: An LLM settings delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)

        if "filter_incomplete_user_turns" in changed:
            self._filter_incomplete_user_turns = (
                self._settings.filter_incomplete_user_turns or False
            )
            logger.info(
                f"{self}: Incomplete turn filtering "
                f"{'enabled' if self._filter_incomplete_user_turns else 'disabled'}"
            )
            if self._filter_incomplete_user_turns:
                # Save the current system_instruction before composing
                self._base_system_instruction = self._settings.system_instruction
                self._compose_system_instruction()
            else:
                # Restore original system_instruction
                self._settings.system_instruction = self._base_system_instruction
                self._base_system_instruction = None

        if "user_turn_completion_config" in changed and self._filter_incomplete_user_turns:
            self.set_user_turn_completion_config(
                assert_given(self._settings.user_turn_completion_config)
            )
            self._compose_system_instruction()

        if (
            "system_instruction" in changed
            and (self._filter_incomplete_user_turns or self._async_tool_cancellation_enabled)
            and "filter_incomplete_user_turns" not in changed
        ):
            # system_instruction changed while composition is active.
            # Treat the new value as the new base and recompose.
            self._base_system_instruction = self._settings.system_instruction
            self._compose_system_instruction()

        return changed

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
        elif isinstance(frame, LLMUpdateSettingsFrame):
            if frame.service is not None and frame.service is not self:
                await self.push_frame(frame, direction)
            elif frame.delta is not None:
                await self._update_settings(frame.delta)
            elif frame.settings:
                # Backward-compatible path: convert legacy dict to settings object.
                with warnings.catch_warnings():
                    warnings.simplefilter("always")
                    warnings.warn(
                        "Passing a dict via LLMUpdateSettingsFrame(settings={...}) is deprecated "
                        "since 0.0.104, use LLMUpdateSettingsFrame(delta=LLMSettings(...)) instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                delta = type(self._settings).from_mapping(frame.settings)
                await self._update_settings(delta)
        elif isinstance(frame, LLMContextSummaryRequestFrame):
            await self._handle_summary_request(frame)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Pushes a frame.

        Args:
            frame: The frame to push.
            direction: The direction of frame pushing.
        """
        if isinstance(frame, (LLMTextFrame, LLMFullResponseStartFrame, LLMFullResponseEndFrame)):
            if self._skip_tts is not None:
                frame.skip_tts = self._skip_tts

        await super().push_frame(frame, direction)

    async def _push_llm_text(self, text: str):
        """Push LLM text, using turn completion detection if enabled.

        This helper method simplifies text pushing in LLM implementations by
        handling the conditional logic for turn completion internally.

        Args:
            text: The text content from the LLM to push.
        """
        if self._filter_incomplete_user_turns:
            await self._push_turn_text(text)
        else:
            await self.push_frame(LLMTextFrame(text))

    async def _handle_interruptions(self, _: InterruptionFrame):
        for function_name, entry in self._functions.items():
            if entry.cancel_on_interruption:
                await self._cancel_function_call(function_name)

    async def _handle_summary_request(self, frame: LLMContextSummaryRequestFrame):
        """Handle context summarization request from aggregator.

        Processes a summarization request by generating a compressed summary
        of conversation history. Uses the adapter to format the summary
        according to the provider's requirements. Broadcasts the result back
        to the aggregator for context reconstruction.

        Args:
            frame: The summary request frame containing context and parameters.
        """
        logger.debug(f"{self}: Processing summarization request {frame.request_id}")

        # Create a background task to generate the summary without blocking
        self._summary_task = self.create_task(self._generate_summary_task(frame))

    async def _generate_summary_task(self, frame: LLMContextSummaryRequestFrame):
        """Background task to generate summary without blocking the pipeline.

        Args:
            frame: The summary request frame containing context and parameters.
        """
        summary = ""
        last_index = -1
        error = None

        timeout = frame.summarization_timeout or DEFAULT_SUMMARIZATION_TIMEOUT

        try:
            summary, last_index = await asyncio.wait_for(
                self._generate_summary(frame),
                timeout=timeout,
            )
        except TimeoutError:
            await self.push_error(error_msg=f"Context summarization timed out after {timeout}s")
        except Exception as e:
            error = f"Error generating context summary: {e}"
            await self.push_error(error, exception=e)

        await self.broadcast_frame(
            LLMContextSummaryResultFrame,
            request_id=frame.request_id,
            summary=summary,
            last_summarized_index=last_index,
            error=error,
        )

        self._summary_task = None

    async def _generate_summary(self, frame: LLMContextSummaryRequestFrame) -> tuple[str, int]:
        """Generate a compressed summary of conversation context.

        Uses the message selection logic to identify which messages
        to summarize, formats them as a transcript, and invokes the LLM to
        generate a concise summary. The summary is formatted according to the
        LLM provider's requirements using the adapter.

        Args:
            frame: The summary request frame containing context and configuration.

        Returns:
            Tuple of (formatted summary message, last_summarized_index).

        Raises:
            RuntimeError: If there are no messages to summarize, the service doesn't
                support run_inference(), or the LLM returns an empty summary.

        Note:
            Requires the service to implement run_inference() method for
            synchronous LLM calls.
        """
        # Get messages to summarize using utility method
        result = LLMContextSummarizationUtil.get_messages_to_summarize(
            frame.context, frame.min_messages_to_keep
        )

        if not result.messages:
            logger.debug(f"{self}: No messages to summarize")
            raise RuntimeError("No messages to summarize")

        logger.debug(
            f"{self}: Generating summary for {len(result.messages)} messages "
            f"(index 0 to {result.last_summarized_index}), "
            f"target_context_tokens={frame.target_context_tokens}"
        )

        # Create summary context
        transcript = LLMContextSummarizationUtil.format_messages_for_summary(result.messages)
        summary_context = LLMContext(
            messages=[{"role": "user", "content": f"Conversation history:\n{transcript}"}]
        )

        # Generate summary using run_inference
        # This will be overridden by each LLM service implementation
        try:
            summary_text = await self.run_inference(
                summary_context,
                max_tokens=frame.target_context_tokens,
                system_instruction=frame.summarization_prompt,
            )
        except NotImplementedError:
            raise RuntimeError(
                f"LLM service {self.__class__.__name__} does not implement run_inference"
            )

        if not summary_text:
            raise RuntimeError("LLM returned empty summary")

        summary_text = summary_text.strip()
        logger.info(
            f"{self}: Generated summary of {len(summary_text)} characters "
            f"for {len(result.messages)} messages"
        )

        return summary_text, result.last_summarized_index

    def register_function(
        self,
        function_name: str | None,
        handler: Any,
        *,
        cancel_on_interruption: bool = True,
        timeout_secs: float | None = None,
    ):
        """Register a function handler for LLM function calls.

        Args:
            function_name: The name of the function to handle. Use None to handle
                all function calls with a catch-all handler.
            handler: The function handler. Should accept a single FunctionCallParams
                parameter.
            cancel_on_interruption: Whether to cancel this function call when an
                interruption occurs. When ``False`` the call is treated as
                asynchronous: the LLM continues the conversation immediately
                without waiting for the result, and the result is injected later
                via a developer message. Defaults to True.
            timeout_secs: Optional per-tool timeout in seconds. Overrides the global
                ``function_call_timeout_secs`` for this specific function. Defaults to
                None, which uses the global timeout.
        """
        if function_name == CANCEL_ASYNC_TOOL_NAME:
            raise ValueError(
                f"'{CANCEL_ASYNC_TOOL_NAME}' is a reserved built-in tool name and cannot be "
                "registered by user code."
            )
        # Registering a function with the function_name set to None will run
        # that handler for all functions
        self._functions[function_name] = FunctionCallRegistryItem(
            function_name=function_name,
            handler=handler,
            cancel_on_interruption=cancel_on_interruption,
            timeout_secs=timeout_secs,
        )

    def register_direct_function(
        self,
        handler: DirectFunction,
        *,
        cancel_on_interruption: bool = True,
        timeout_secs: float | None = None,
    ):
        """Register a direct function handler for LLM function calls.

        Direct functions have their metadata automatically extracted from their
        signature and docstring, eliminating the need for accompanying
        configurations (as FunctionSchemas or in provider-specific formats).

        Args:
            handler: The direct function to register. Must follow DirectFunction protocol.
            cancel_on_interruption: Whether to cancel this function call when an
                interruption occurs. When ``False`` the call is treated as
                asynchronous: the LLM continues the conversation immediately
                without waiting for the result, and the result is injected later
                via a developer message. Defaults to True.
            timeout_secs: Optional per-tool timeout in seconds. Overrides the global
                ``function_call_timeout_secs`` for this specific function. Defaults to
                None, which uses the global timeout.
        """
        wrapper = DirectFunctionWrapper(handler)
        if wrapper.name == CANCEL_ASYNC_TOOL_NAME:
            raise ValueError(
                f"'{CANCEL_ASYNC_TOOL_NAME}' is a reserved built-in tool name and cannot be "
                "registered by user code."
            )
        self._functions[wrapper.name] = FunctionCallRegistryItem(
            function_name=wrapper.name,
            handler=wrapper,
            cancel_on_interruption=cancel_on_interruption,
            timeout_secs=timeout_secs,
        )

    def unregister_function(self, function_name: str | None):
        """Remove a registered function handler.

        Args:
            function_name: The name of the function handler to remove.
        """
        del self._functions[function_name]
        if self._async_tool_cancellation_enabled and not self._has_async_tools():
            self._teardown_async_tool_cancellation()

    def unregister_direct_function(self, handler: Any):
        """Remove a registered direct function handler.

        Args:
            handler: The direct function handler to remove.
        """
        wrapper = DirectFunctionWrapper(handler)
        del self._functions[wrapper.name]
        # Note: no need to remove start callback here, as direct functions don't support start callbacks.
        if self._async_tool_cancellation_enabled and not self._has_async_tools():
            self._teardown_async_tool_cancellation()

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

    async def run_function_calls(self, function_calls: Sequence[FunctionCallFromLLM]):
        """Execute a sequence of function calls from the LLM.

        Triggers the on_function_calls_started event and executes functions
        either in parallel or sequentially based on the run_in_parallel setting.

        Args:
            function_calls: The function calls to execute.
        """
        if len(function_calls) == 0:
            return

        # Exclude the built-in cancel tool — it's an internal mechanism and
        # should not be surfaced to user-facing event handlers or frames.
        user_visible_calls = [
            fc for fc in function_calls if fc.function_name != CANCEL_ASYNC_TOOL_NAME
        ]
        if user_visible_calls:
            await self._call_event_handler("on_function_calls_started", user_visible_calls)
            await self.broadcast_frame(FunctionCallsStartedFrame, function_calls=user_visible_calls)

        # When group_parallel_tools is True all calls share a group_id so the
        # aggregator triggers the LLM exactly once after the last one completes.
        # When False, group_id is None and each result triggers inference independently.
        group_id = str(uuid.uuid4()) if self._group_parallel_tools else None

        runner_items = []
        for function_call in function_calls:
            if function_call.function_name in self._functions.keys():
                item = self._functions[function_call.function_name]
            elif None in self._functions.keys():
                item = self._functions[None]
            else:
                logger.warning(
                    f"{self} is calling '{function_call.function_name}', but it's not registered."
                )
                item = self._build_missing_function_call_registry_item(function_call.function_name)

            runner_items.append(
                FunctionCallRunnerItem(
                    registry_item=item,
                    function_name=function_call.function_name,
                    tool_call_id=function_call.tool_call_id,
                    arguments=function_call.arguments,
                    context=function_call.context,
                    group_id=group_id,
                )
            )

        if self._run_in_parallel:
            await self._run_parallel_function_calls(runner_items)
        else:
            await self._run_sequential_function_calls(runner_items)

    async def _create_sequential_runner_task(self):
        if not self._sequential_runner_task:
            self._sequential_runner_queue = asyncio.Queue()
            self._sequential_runner_task = self.create_task(self._sequential_runner_handler())

    async def _cancel_sequential_runner_task(self):
        if self._sequential_runner_task:
            await self.cancel_task(self._sequential_runner_task)
            self._sequential_runner_task = None

    async def _cancel_summary_task(self):
        if self._summary_task:
            await self.cancel_task(self._summary_task)
            self._summary_task = None

    async def _sequential_runner_handler(self):
        while True:
            runner_item = await self._sequential_runner_queue.get()
            task = self.create_task(self._run_function_call(runner_item))
            self._function_call_tasks[task] = runner_item
            # Since we run tasks sequentially we don't need to call
            # task.add_done_callback(self._function_call_task_finished).
            await task
            del self._function_call_tasks[task]

    async def _run_parallel_function_calls(self, runner_items: Sequence[FunctionCallRunnerItem]):
        tasks = []
        for runner_item in runner_items:
            task = self.create_task(self._run_function_call(runner_item))
            tasks.append(task)
            self._function_call_tasks[task] = runner_item
            task.add_done_callback(self._function_call_task_finished)

        # Wait for all parallel function calls to complete before returning.
        # Using return_exceptions=True to prevent one failing task from canceling others.
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_sequential_function_calls(self, runner_items: Sequence[FunctionCallRunnerItem]):
        # Enqueue all function calls for background execution.
        for runner_item in runner_items:
            await self._sequential_runner_queue.put(runner_item)

    async def _run_function_call(self, runner_item: FunctionCallRunnerItem):
        # Re-resolve the registry item at execution time. The function may have
        # been unregistered between queuing and execution, in which case we
        # fall back to the missing-function handler so the call still terminates
        # with a normal tool result.
        if runner_item.function_name in self._functions.keys():
            item = self._functions[runner_item.function_name]
        elif None in self._functions.keys():
            item = self._functions[None]
        elif runner_item.registry_item.handler == self._missing_function_call_handler:
            item = runner_item.registry_item
        else:
            logger.warning(
                f"{self} is calling '{runner_item.function_name}', but it was just unregistered."
            )
            item = self._build_missing_function_call_registry_item(runner_item.function_name)

        logger.debug(
            f"{self} Calling function [{runner_item.function_name}:{runner_item.tool_call_id}] with arguments {runner_item.arguments}"
        )

        # Broadcast function call in-progress. This frame will let our assistant
        # context aggregator know that we are in the middle of a function
        # call. Some contexts/aggregators may not need this. But some definitely
        # do (Anthropic, for example).
        await self.broadcast_frame(
            FunctionCallInProgressFrame,
            function_name=runner_item.function_name,
            tool_call_id=runner_item.tool_call_id,
            arguments=runner_item.arguments,
            cancel_on_interruption=item.cancel_on_interruption,
            group_id=runner_item.group_id,
        )

        timeout_task: asyncio.Task | None = None

        # Single callback for both intermediate updates and final results.
        # Pass properties=FunctionCallResultProperties(is_final=False) for updates.
        async def function_call_result_callback(
            result: Any, *, properties: FunctionCallResultProperties | None = None
        ):
            is_final = properties.is_final if properties else True
            if not is_final and item.cancel_on_interruption:
                logger.warning(
                    f"{self} result_callback called with is_final=False on sync function call"
                    f" [{runner_item.function_name}:{runner_item.tool_call_id}]."
                    " Intermediate updates are only valid for async function calls"
                    " (cancel_on_interruption=False)."
                )
                return

            nonlocal timeout_task

            # Cancel timeout task if it exists
            if timeout_task and not timeout_task.done():
                await self.cancel_task(timeout_task)

            await self.broadcast_frame(
                FunctionCallResultFrame,
                function_name=runner_item.function_name,
                tool_call_id=runner_item.tool_call_id,
                arguments=runner_item.arguments,
                result=result,
                run_llm=runner_item.run_llm,
                properties=properties,
            )

        # Start a timeout task for deferred function calls
        async def timeout_handler():
            try:
                effective_timeout = item.timeout_secs or self._function_call_timeout_secs
                await asyncio.sleep(effective_timeout)
                logger.warning(
                    f"{self} Function call [{runner_item.function_name}:{runner_item.tool_call_id}] timed out after {effective_timeout} seconds."
                    f" You can increase this timeout by passing `timeout_secs` to `register_function()`,"
                    f" or set a global default via `function_call_timeout_secs` on the LLM constructor."
                )
                await function_call_result_callback(None)
            except asyncio.CancelledError:
                raise

        if item.timeout_secs or self._function_call_timeout_secs:
            timeout_task = self.create_task(timeout_handler())

        # Yield to the event loop so the timeout task coroutine gets entered
        # before it could be cancelled. Without this, cancelling the task before
        # it starts would leave the coroutine in a "never awaited" state.
        await asyncio.sleep(0)

        # _pipeline_task may be unset when the service is driven without a PipelineTask.
        app_resources = self._pipeline_task.app_resources if self._pipeline_task else None

        try:
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
                        app_resources=app_resources,
                    ),
                )
            else:
                # Handler is a FunctionCallHandler
                params = FunctionCallParams(
                    function_name=runner_item.function_name,
                    tool_call_id=runner_item.tool_call_id,
                    arguments=runner_item.arguments,
                    llm=self,
                    context=runner_item.context,
                    result_callback=function_call_result_callback,
                    app_resources=app_resources,
                )
                await item.handler(params)
        except Exception as e:
            error_message = f"Error executing function call [{runner_item.function_name}]: {e}"
            logger.error(f"{self} {error_message}")
            await self.push_error(error_msg=error_message, exception=e, fatal=False)
        finally:
            if timeout_task and not timeout_task.done():
                await self.cancel_task(timeout_task)

    def _build_missing_function_call_registry_item(
        self, function_name: str
    ) -> FunctionCallRegistryItem:
        """Build a registry item that routes to the missing-function handler."""
        return FunctionCallRegistryItem(
            function_name=function_name,
            handler=self._missing_function_call_handler,
            cancel_on_interruption=True,
        )

    async def _missing_function_call_handler(self, params: FunctionCallParams):
        """Return a terminal tool result when the LLM calls an unknown function."""
        await params.result_callback(f"Error: function '{params.function_name}' is not registered.")

    def _has_async_tools(self) -> bool:
        """Return True if at least one non-builtin async tool is registered."""
        return any(
            not item.cancel_on_interruption
            for name, item in self._functions.items()
            if name != CANCEL_ASYNC_TOOL_NAME
        )

    def _setup_async_tool_cancellation(self):
        """Enable async tool cancellation.

        Saves the base system instruction, recomposes to include cancellation
        instructions, registers the built-in ``cancel_async_tool_call`` handler,
        and injects its schema into the adapter's built-in tool dict.
        """
        logger.debug(f"{self}: Enabling async tool cancellation")

        self._async_tool_cancellation_enabled = True

        if self._base_system_instruction is None:
            self._base_system_instruction = self._settings.system_instruction

        self._compose_system_instruction()

        self._adapter.builtin_tools[CANCEL_ASYNC_TOOL_NAME] = CANCEL_ASYNC_TOOL_SCHEMA

        if CANCEL_ASYNC_TOOL_NAME not in self._functions:
            self._functions[CANCEL_ASYNC_TOOL_NAME] = FunctionCallRegistryItem(
                function_name=CANCEL_ASYNC_TOOL_NAME,
                handler=self._cancel_async_tool_call_handler,
                cancel_on_interruption=True,
            )

    def _teardown_async_tool_cancellation(self):
        """Disable async tool cancellation.

        Removes the built-in ``cancel_async_tool_call`` handler and its schema,
        recomposes the system instruction without cancellation instructions.
        """
        logger.debug(f"{self}: Disabling async tool cancellation")

        self._async_tool_cancellation_enabled = False
        self._adapter.builtin_tools.pop(CANCEL_ASYNC_TOOL_NAME, None)
        self._functions.pop(CANCEL_ASYNC_TOOL_NAME, None)
        self._compose_system_instruction()

    async def _cancel_async_tool_call_handler(self, params: FunctionCallParams):
        """Handle a ``cancel_async_tool_call`` invocation from the LLM.

        Args:
            params: Function call parameters containing ``tool_call_id`` to cancel.
        """
        logger.debug(f"{self}: cancel_async_tool_call invoked")

        tool_call_id: str | None = params.arguments.get("tool_call_id")
        if not tool_call_id:
            logger.warning(f"{self} cancel_async_tool_call called with no tool_call_id")
            await params.result_callback({"cancelled": None})
            return

        await self._cancel_function_calls_by_tool_call_id(tool_call_id)
        await params.result_callback(
            {"cancelled": tool_call_id},
            properties=FunctionCallResultProperties(run_llm=True),
        )

    async def _cancel_function_calls_by_tool_call_id(self, tool_call_id: str):
        """Cancel in-progress function call tasks by their tool_call_id.

        Args:
            tool_call_id: tool_call_id to cancel.
        """
        cancelled_tasks = set()
        cancelled_items = []
        for task, runner_item in self._function_call_tasks.items():
            if runner_item.tool_call_id == tool_call_id:
                name = runner_item.function_name
                tool_call_id = runner_item.tool_call_id

                logger.debug(
                    f"{self} Cancelling async function call [{name}:{tool_call_id}] "
                    "by LLM request..."
                )

                if task:
                    task.remove_done_callback(self._function_call_task_finished)
                    await self.cancel_task(task)
                    cancelled_tasks.add(task)

                await self.broadcast_frame(
                    FunctionCallCancelFrame, function_name=name, tool_call_id=tool_call_id
                )

                cancelled_items.append(
                    FunctionCallFromLLM(
                        function_name=runner_item.function_name,
                        tool_call_id=runner_item.tool_call_id,
                        arguments=runner_item.arguments,
                        context=runner_item.context,
                    )
                )
                logger.debug(f"{self} Async function call [{name}:{tool_call_id}] cancelled")

        for task in cancelled_tasks:
            self._function_call_task_finished(task)

        if cancelled_items:
            await self._call_event_handler("on_function_calls_cancelled", cancelled_items)

    async def _cancel_function_call(self, function_name: str | None):
        cancelled_tasks = set()
        cancelled_items = []
        for task, runner_item in self._function_call_tasks.items():
            if runner_item.registry_item.function_name == function_name:
                name = runner_item.function_name
                tool_call_id = runner_item.tool_call_id

                logger.debug(f"{self} Cancelling function call [{name}:{tool_call_id}]...")

                if task:
                    # We remove the callback because we are going to cancel the
                    # task next, otherwise we will be removing it from the set
                    # while we are iterating.
                    task.remove_done_callback(self._function_call_task_finished)
                    await self.cancel_task(task)
                    cancelled_tasks.add(task)

                await self.broadcast_frame(
                    FunctionCallCancelFrame, function_name=name, tool_call_id=tool_call_id
                )

                cancelled_items.append(
                    FunctionCallFromLLM(
                        function_name=runner_item.function_name,
                        tool_call_id=runner_item.tool_call_id,
                        arguments=runner_item.arguments,
                        context=runner_item.context,
                    )
                )
                logger.debug(f"{self} Function call [{name}:{tool_call_id}] has been cancelled")

        # Remove all cancelled tasks from our set.
        for task in cancelled_tasks:
            self._function_call_task_finished(task)

        if cancelled_items:
            await self._call_event_handler("on_function_calls_cancelled", cancelled_items)

    def _function_call_task_finished(self, task: asyncio.Task):
        if task in self._function_call_tasks:
            del self._function_call_tasks[task]


# ---------------------------------------------------------------------------
# WebSocket LLM service base
# ---------------------------------------------------------------------------


class WebsocketReconnectedError(Exception):
    """Raised by ``_ws_send``/``_ws_recv`` after a transparent reconnection.

    Signals that the WebSocket connection was lost and automatically
    re-established.  The current inference should be restarted — any
    connection-local state on the server (e.g. cached responses) is gone.
    """

    pass


class WebsocketLLMService(LLMService[TAdapter], WebsocketService, Generic[TAdapter]):
    """Base class for websocket-based LLM services.

    Each LLM inference is a discrete request/response exchange: send one
    request, receive events inline until a terminal event, then wait for
    the next frame to trigger an inference.  This contrasts with
    ``WebsocketTTSService`` / ``WebsocketSTTService`` which stream data
    continuously via a background receive loop
    (``_receive_task_handler``).  This class does **not** start a
    background receive loop.

    Provides connection lifecycle management (connect on start, disconnect
    on stop/cancel), automatic reconnection with exponential backoff, and
    three helpers for running each inference:

    1. ``_ensure_connected()`` — verify the websocket is alive, reconnect
       with exponential backoff if not.
    2. ``_ws_send(message)`` — send the inference request as JSON.
    3. ``_ws_recv()`` — receive and parse response events one at a time
       until the caller sees a terminal event.

    ``_ws_send`` and ``_ws_recv`` catch ``ConnectionClosed`` transparently,
    auto-reconnect via ``_try_reconnect``, and raise
    ``WebsocketReconnectedError`` so callers know the inference must be
    restarted.  If reconnection fails, the original ``ConnectionClosed``
    propagates.

    Subclasses must implement:
        ``_connect_websocket()``: Establish the websocket connection.
        ``_disconnect_websocket()``: Close the websocket and clean up.

    Event handlers:
        on_connection_error: Called when a websocket connection error occurs.

    Example::

        @llm.event_handler("on_connection_error")
        async def on_connection_error(llm: LLMService, error: str):
            logger.error(f"LLM connection error: {error}")
    """

    def __init__(self, *, reconnect_on_error: bool = True, **kwargs):
        """Initialize the Websocket LLM service.

        Args:
            reconnect_on_error: Whether to automatically reconnect on websocket errors.
            **kwargs: Additional arguments passed to parent classes.
        """
        # pyright stumbles here because the TypeVar default makes
        # `LLMService` resolve to `LLMService[BaseLLMAdapter]` invariantly,
        # while `self` is `WebsocketLLMService[TAdapter]` for an arbitrary
        # TAdapter. The runtime call is fine — generics are erased.
        LLMService.__init__(self, **kwargs)  # pyright: ignore[reportArgumentType]
        WebsocketService.__init__(self, reconnect_on_error=reconnect_on_error, **kwargs)
        self._register_event_handler("on_connection_error")

    # -- lifecycle ------------------------------------------------------------

    async def _connect(self):
        """Connect: reset flags and establish the websocket."""
        await super()._connect()
        await self._connect_websocket()

    async def _disconnect(self):
        """Disconnect: set flags and close the websocket."""
        await super()._disconnect()
        await self._disconnect_websocket()

    async def start(self, frame: StartFrame):
        """Start the service and establish WebSocket connection.

        Args:
            frame: The start frame triggering service initialization.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service and close WebSocket connection.

        Args:
            frame: The end frame triggering service shutdown.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and close WebSocket connection.

        Args:
            frame: The cancel frame triggering service cancellation.
        """
        await super().cancel(frame)
        await self._disconnect()

    # -- per-inference helpers ------------------------------------------------

    async def _ws_send(self, message: dict):
        """Send a JSON message over the websocket.

        Guards against sends during intentional disconnect.  If the send
        fails with ``ConnectionClosed``, attempts to reconnect and raises
        ``WebsocketReconnectedError`` on success so the caller can restart
        the inference.  If reconnection fails, the original
        ``ConnectionClosed`` propagates.

        Args:
            message: The message dict to serialize and send.
        """
        if self._disconnecting or not self._websocket:
            return
        try:
            await self._websocket.send(json.dumps(message))
        except ConnectionClosed:
            if self._disconnecting:
                return
            success = await self._try_reconnect(report_error=self._report_error)
            if success:
                raise WebsocketReconnectedError()
            raise

    async def _ws_recv(self) -> dict:
        """Receive and parse a JSON message from the websocket.

        If the receive fails with ``ConnectionClosed``, attempts to
        reconnect and raises ``WebsocketReconnectedError`` on success.
        If reconnection fails, the original ``ConnectionClosed``
        propagates.

        Returns:
            The parsed JSON message as a dict.
        """
        # Should never happen — `_ensure_connected` (which callers must invoke
        # first) raises ConnectionError if it can't establish a websocket.
        # Match that contract here.
        if self._websocket is None:
            raise ConnectionError(f"{self} _ws_recv called without a websocket")
        try:
            raw = await self._websocket.recv()
            return json.loads(raw)
        except ConnectionClosed:
            if self._disconnecting:
                raise
            success = await self._try_reconnect(report_error=self._report_error)
            if success:
                raise WebsocketReconnectedError()
            raise

    async def _ensure_connected(self):
        """Ensure the websocket is connected, reconnecting if needed.

        Uses ``_try_reconnect`` with exponential backoff.

        Raises:
            ConnectionError: If the connection could not be established.
        """
        if self._websocket and self._websocket.state is not State.CLOSED:
            return
        success = await self._try_reconnect(report_error=self._report_error)
        if not success:
            raise ConnectionError(f"{self} failed to establish WebSocket connection")

    # -- WebsocketService interface -------------------------------------------

    async def _receive_messages(self):
        """Not used — messages are received inline during each inference.

        This satisfies the ``WebsocketService`` abstract method but is never
        called because ``_receive_task_handler`` is never started.
        """
        raise NotImplementedError(
            "WebsocketLLMService receives messages inline during inference, "
            "not via a continuous background loop"
        )

    async def _report_error(self, error: ErrorFrame):
        await self._call_event_handler("on_connection_error", error.error)
        await self.push_error_frame(error)
