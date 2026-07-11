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
    TYPE_CHECKING,
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
from pipecat.adapters.schemas.tools_schema import ToolsSchema
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
    LLMContextFrame,
    LLMContextSummaryRequestFrame,
    LLMContextSummaryResultFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMServiceMetadataFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    StartFrame,
)
from pipecat.processors.aggregators.llm_context import (
    NOT_GIVEN,
    LLMContext,
    LLMSpecificMessage,
    is_given,
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
from pipecat.utils.deprecation import deprecated

if TYPE_CHECKING:
    from pipecat.pipeline.worker import PipelineWorker


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
            ``PipelineWorker(..., app_resources=...)``. Same object — passed by
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
    pipeline_worker: PipelineWorker
    context: LLMContext
    result_callback: FunctionCallResultCallback
    app_resources: Any = None

    @property
    @deprecated(
        "`FunctionCallParams.tool_resources` is deprecated since 1.2.0 and will be removed in "
        "2.0.0. Use `app_resources` instead."
    )
    def tool_resources(self) -> Any:
        """Deprecated alias for :attr:`app_resources`.

        .. deprecated:: 1.2.0
            Use :attr:`app_resources` instead. ``tool_resources``.
            Will be removed in 2.0.0.
        """
        return self.app_resources


@dataclass
class FunctionCallRegistryItem:
    """Internal record of a registered function-call handler.

    Created by the service when a function is registered — directly via
    ``register_function`` / ``register_direct_function``, or automatically from a
    direct function advertised in an ``LLMContext`` / ``LLMSetToolsFrame``.
    Application code doesn't construct these.

    Parameters:
        function_name: The name of the function (None for catch-all handler).
        handler: The handler for processing function call parameters.
        cancel_on_interruption: Whether to cancel the call on interruption.
            When ``False`` the call is treated as asynchronous: the LLM
            continues the conversation immediately without waiting for the
            result, and the result is injected later via a developer message.
        timeout_secs: Optional per-tool timeout in seconds. Overrides the global
            ``function_call_timeout_secs`` for this specific function.
        auto_registered: True only for a direct function that was auto-registered
            from an advertised tool set (listed in an ``LLMContext`` or
            ``LLMSetToolsFrame``). False for every explicitly registered handler —
            direct or non-direct — and for the catch-all and built-in handlers.
    """

    function_name: str | None
    handler: FunctionCallHandler | DirectFunctionWrapper
    cancel_on_interruption: bool
    timeout_secs: float | None = None
    auto_registered: bool = False


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

    # Returned to the LLM as the tool result when an unavailable function is
    # called. Deliberately neutral about future availability so the LLM can
    # pick the function up again if it returns (e.g. via the
    # ``add_tool_change_messages`` activation message, or silently on a
    # later inference). ``{function_name}`` is substituted at runtime.
    MISSING_FUNCTION_CALL_MESSAGE_TEMPLATE = (
        "The function `{function_name}` is not currently available."
    )

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
        # The user's base system instruction, without composed addons. Captured
        # here and refreshed when the user changes ``system_instruction`` at
        # runtime; ``_compose_system_instruction`` always rebuilds the effective
        # instruction from this plus any appended / addon instructions.
        base_si = self._settings.system_instruction
        self._base_system_instruction: str | None = base_si if isinstance(base_si, str) else None
        self._appended_system_instructions: list[str] = []
        # `adapter_class` is typed as `type[BaseLLMAdapter]` so subclasses
        # don't need to spell out the generic parameter just to subclass
        # (backward compatibility for 3rd-party providers outside this repo).
        # Cast to TAdapter to keep `_adapter` and `get_llm_adapter()` precisely
        # typed for callers that opt into `LLMService[XAdapter]`.
        self._adapter = cast(TAdapter, self.adapter_class())
        self._functions: dict[str | None, FunctionCallRegistryItem] = {}
        # Names we've already warned about for a redundant manual registration
        # (an explicit register_function call for a tool whose advertised
        # FunctionSchema already carries a handler), so the warning fires once
        # rather than on every context frame.
        self._redundant_registration_warned: set[str] = set()
        # Names explicitly unregistered (via unregister_function /
        # unregister_direct_function) that auto-registration must not re-register
        # while they're still advertised — otherwise a standalone unregister would
        # be undone by the next context frame. Cleared when the name is registered
        # again or stops being advertised (see _sync_registered_tool_handlers).
        self._explicitly_unregistered_function_names: set[str | None] = set()
        self._function_call_tasks: dict[asyncio.Task | None, FunctionCallRunnerItem] = {}
        self._sequential_runner_task: asyncio.Task | None = None
        self._skip_tts: bool | None = None
        self._summary_task: asyncio.Task | None = None
        # Whether the one-time realtime-service "no turn frames" warning has
        # fired (see _warn_if_realtime_service_emits_no_turn_frames).
        self._warned_realtime_service_no_turn_frames: bool = False

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

    def service_metadata_frame(self) -> LLMServiceMetadataFrame:
        """The metadata frame this LLM service broadcasts at start.

        The base returns a plain (non-realtime) frame; realtime
        (speech-to-speech) subclasses override this to set
        ``is_realtime_service=True`` and, when their turns are provider-driven,
        recommend ``ExternalUserTurnStrategies`` via ``user_turn_strategies``.

        Mostly here for conceptual consistency — today only realtime services
        need to override it — but it's a natural placeholder for future
        LLM-service metadata.
        """
        return LLMServiceMetadataFrame(service_name=self.name)

    def _warn_if_realtime_service_emits_no_turn_frames(self, emits_turn_frames: bool) -> None:
        """Warn (once) when a realtime service won't emit its own turn frames.

        Realtime services call this from ``service_metadata_frame()`` at the point
        they determine whether their turns are provider-driven. When they aren't,
        downstream processors that expect ``UserStarted/StoppedSpeakingFrame``
        (e.g. RTVI) may need local VAD/turn detection.
        """
        if emits_turn_frames or self._warned_realtime_service_no_turn_frames:
            return
        self._warned_realtime_service_no_turn_frames = True
        logger.warning(
            f"{self} is not emitting turn frames "
            "(UserStartedSpeakingFrame/UserStoppedSpeakingFrame) — this service "
            "either doesn't support them or isn't configured to emit them. A "
            "couple of things to keep in mind:\n"
            "  - Other processors in the pipeline (e.g. RTVI) may expect "
            "turn frames. You can enable local VAD/turn detection by "
            "setting a vad_analyzer in LLMUserAggregatorParams.\n"
            "  - Be aware that local turns may NOT perfectly align with "
            'the "ground truth" of server-decided turns, so they should '
            "be thought of as APPROXIMATE (unless, of course, you've "
            "also configured local turn detection to *drive* the "
            "realtime service's turns, e.g. by setting "
            "vad=GeminiVADParams(disabled=True))."
        )

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

    async def cleanup(self):
        """Release LLM service resources at teardown."""
        await super().cleanup()
        if not self._run_in_parallel:
            await self._cancel_sequential_runner_task()
        await self._cancel_summary_task()
        await self._cancel_all_function_call_tasks()

    def append_system_instruction(self, instruction: str) -> None:
        """Append durable text to the system instruction, preserving the user's prompt.

        The text is composed onto the end of the system instruction (joined
        with a blank line) and re-applied on every inference, so it survives
        context-message resets (e.g. ``LLMMessagesUpdateFrame(messages=[])``).
        Intended for framework components that own an LLM and need to add
        standard guidance to a user-provided prompt — for example, ``UIWorker``
        appends the UI wire-format guide. Appended instructions compose after
        the user's base prompt and alongside the turn-completion and
        async-tool-cancellation instructions.

        Args:
            instruction: The instruction text to append.
        """
        self._appended_system_instructions.append(instruction)
        self._compose_system_instruction()

    def _compose_system_instruction(self):
        """Rebuild ``system_instruction`` from the base prompt and all active addons.

        Joins the user's base system instruction (the single source of truth,
        captured at construction and refreshed on runtime ``system_instruction``
        updates) with any appended instructions (e.g. the ``UIWorker`` prompt
        guide), turn completion instructions (when enabled), and async tool
        cancellation instructions (when enabled). Safe to call repeatedly — it
        always rebuilds from the base, so it never compounds.
        """
        base = self._base_system_instruction
        parts = [base] if base else []
        parts.extend(self._appended_system_instructions)
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

        if "system_instruction" in changed:
            # The user replaced the base prompt; re-snapshot it so composition
            # rebuilds the effective instruction from the new value.
            base_si = self._settings.system_instruction
            self._base_system_instruction = base_si if isinstance(base_si, str) else None

        if "user_turn_completion_config" in changed and self._filter_incomplete_user_turns:
            self.set_user_turn_completion_config(
                assert_given(self._settings.user_turn_completion_config)
            )

        # Any of these fields changes the composed instruction; rebuild it.
        if changed.keys() & {
            "filter_incomplete_user_turns",
            "system_instruction",
            "user_turn_completion_config",
        }:
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

        if isinstance(frame, LLMContextFrame):
            # Sync the registered handlers with the tools advertised in the
            # context: register any newly advertised handler, drop the ones we
            # auto-registered that are no longer advertised. The context carries
            # the current tool set on every inference, so this is the single place
            # tool changes take effect for text LLMs.
            #
            # Realtime (speech-to-speech) services run continuously and don't get a
            # fresh context frame per turn, so they additionally call
            # _sync_registered_tool_handlers on their own LLMSetToolsFrame
            # handling.
            self._sync_registered_tool_handlers(frame.context.tools)

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
        cancel_on_interruption: bool | None = None,
        timeout_secs: float | None = None,
    ):
        """Register a function handler for LLM function calls.

        Call options resolve with the precedence **explicit argument >
        ``@tool_options`` decorator > default**. ``None`` (the default) means
        "not provided" — the option falls back to the ``@tool_options`` value on
        the handler, then to the documented default.

        Args:
            function_name: The name of the function to handle. Use None to handle
                all function calls with a catch-all handler.
            handler: The function handler. Should accept a single FunctionCallParams
                parameter.
            cancel_on_interruption: Whether to cancel this function call when an
                interruption occurs. When ``False`` the call is treated as
                asynchronous: the LLM continues the conversation immediately
                without waiting for the result, and the result is injected later
                via a developer message. Defaults to ``None`` (fall back to the
                ``@tool_options`` decorator value, then to True). Note: realtime
                LLM services deliver only the final result to the provider;
                intermediate streamed results (reported via
                ``FunctionCallResultProperties(is_final=False)``) are
                dropped and an error is raised. Use a non-realtime LLM
                service if your tool needs to stream intermediate results.
            timeout_secs: Optional per-tool timeout in seconds, overriding the
                global ``function_call_timeout_secs``. Defaults to ``None`` (fall
                back to the ``@tool_options`` decorator value, then to the global
                timeout).
        """
        if function_name == CANCEL_ASYNC_TOOL_NAME:
            raise ValueError(
                f"'{CANCEL_ASYNC_TOOL_NAME}' is a reserved built-in tool name and cannot be "
                "registered by user code."
            )
        # Explicitly registering a handler clears any standalone-unregister
        # suppression for its name.
        self._explicitly_unregistered_function_names.discard(function_name)
        # Registering a function with the function_name set to None will run
        # that handler for all functions
        self._functions[function_name] = FunctionCallRegistryItem(
            function_name=function_name,
            handler=handler,
            cancel_on_interruption=self._resolve_tool_option(
                function_name,
                cancel_on_interruption,
                handler,
                "_pipecat_cancel_on_interruption",
                default=True,
            ),
            timeout_secs=self._resolve_tool_option(
                function_name, timeout_secs, handler, "_pipecat_timeout_secs", default=None
            ),
        )

    @deprecated(
        "`LLMService.register_direct_function` is deprecated since 1.4.0 and will be removed in "
        "2.0.0. Use `LLMContext(tools=[...])` instead."
    )
    def register_direct_function(
        self,
        handler: DirectFunction,
        *,
        cancel_on_interruption: bool | None = None,
        timeout_secs: float | None = None,
    ):
        """Register a direct function handler for LLM function calls.

        .. deprecated:: 1.4.0
            Direct functions are now registered automatically. List them in
            ``LLMContext(tools=[...])`` for tools available at session start, or
            push an :class:`LLMSetToolsFrame` to change tools mid-session.
            Will be removed in 2.0.0.

        Direct functions have their metadata automatically extracted from their
        signature and docstring, eliminating the need for accompanying
        configurations (as FunctionSchemas or in provider-specific formats).

        Call options resolve with the precedence **explicit argument >
        ``@tool_options`` decorator > default**. ``None`` (the default) means
        "not provided" — the option falls back to the decorator value, then the
        documented default.

        Args:
            handler: The direct function to register. Must follow DirectFunction protocol.
            cancel_on_interruption: Whether to cancel this function call when an
                interruption occurs. When ``False`` the call is treated as
                asynchronous: the LLM continues the conversation immediately
                without waiting for the result, and the result is injected later
                via a developer message. Defaults to ``None`` (fall back to the
                ``@tool_options`` decorator value, then to True).
                Note: realtime LLM services deliver only the final result to the
                provider; intermediate streamed results (reported via
                ``FunctionCallResultProperties(is_final=False)``) are
                dropped and an error is raised. Use a non-realtime LLM
                service if your tool needs to stream intermediate results.
            timeout_secs: Optional per-tool timeout in seconds, overriding the
                global ``function_call_timeout_secs``. Defaults to ``None`` (fall
                back to the ``@tool_options`` decorator value, then to the global
                timeout).
        """
        self._register_direct_function(
            handler,
            cancel_on_interruption=cancel_on_interruption,
            timeout_secs=timeout_secs,
        )

    def _register_direct_function(
        self,
        handler: DirectFunction,
        *,
        cancel_on_interruption: bool | None = None,
        timeout_secs: float | None = None,
    ):
        """Register a direct function handler.

        Shared core behind automatic registration (from an ``LLMContext`` or
        ``LLMSetToolsFrame``). Call options resolve by precedence:
        explicit argument > ``@tool_options`` decorator > default.

        Args:
            handler: The direct function to register.
            cancel_on_interruption: Explicit override, or ``None`` to use the
                decorator value (then the True default).
            timeout_secs: Explicit override, or ``None`` to use the decorator
                value (then the None default, i.e. the global timeout).
        """
        wrapper = DirectFunctionWrapper(handler)
        if wrapper.name == CANCEL_ASYNC_TOOL_NAME:
            raise ValueError(
                f"'{CANCEL_ASYNC_TOOL_NAME}' is a reserved built-in tool name and cannot be "
                "registered by user code."
            )
        cancel_on_interruption = self._resolve_tool_option(
            wrapper.name,
            cancel_on_interruption,
            handler,
            "_pipecat_cancel_on_interruption",
            default=True,
        )
        timeout_secs = self._resolve_tool_option(
            wrapper.name,
            timeout_secs,
            handler,
            "_pipecat_timeout_secs",
            default=None,
        )
        # Explicitly registering a handler clears any standalone-unregister
        # suppression for its name. (Auto-registration skips suppressed names
        # before reaching this method, so this only fires for explicit calls.)
        self._explicitly_unregistered_function_names.discard(wrapper.name)
        # The new entry defaults to auto_registered=False, so a handler registered
        # through this method is treated as explicit and is never auto-pruned.
        self._functions[wrapper.name] = FunctionCallRegistryItem(
            function_name=wrapper.name,
            handler=wrapper,
            cancel_on_interruption=cancel_on_interruption,
            timeout_secs=timeout_secs,
        )

    def _resolve_tool_option(
        self, function_name: str | None, explicit: Any, handler: Any, attr: str, *, default: Any
    ) -> Any:
        """Resolve a tool call option by precedence: explicit > decorator > default.

        An explicit ``None`` is treated as "not provided" and falls back to the
        decorator value, then the default — so ``None`` can't be passed to force a
        default past a decorator value (an intentionally unsupported niche; set
        ``function_call_timeout_secs`` on the service for that).

        Args:
            function_name: The tool's name, for logging.
            explicit: The value passed to ``register_direct_function``, or ``None``
                if the caller omitted it.
            handler: The direct function, which may carry decorator-set options.
            attr: The decorator attribute to read (e.g. ``_pipecat_timeout_secs``).
            default: The value to use when neither an explicit argument nor a
                decorator value is present.

        Returns:
            The resolved option value.
        """
        decorated = getattr(handler, attr, None)
        if explicit is not None:
            if decorated is not None and decorated != explicit:
                logger.debug(
                    f"{self}: '{function_name}' registered with explicit "
                    f"{attr.removeprefix('_pipecat_')}={explicit!r}, overriding the "
                    f"decorator value {decorated!r}."
                )
            return explicit
        return decorated if decorated is not None else default

    def _register_advertised_tool_handlers(self, tools: Any) -> None:
        """Register handlers for any tools in the given set that carry one.

        A tool carries a handler when it's advertised as a direct function or as
        a ``FunctionSchema`` with a ``handler`` set; either way the handler is
        registered automatically, so no ``register_function`` call is needed. A
        ``FunctionSchema`` without a handler is advertise-only and registers
        nothing.

        Accepts whatever ``LLMContext`` accepts for tools — a ``ToolsSchema``, a
        plain list of direct functions / ``FunctionSchema`` objects, or
        ``NOT_GIVEN`` — normalizing as needed.

        Any tool whose name is already registered (explicitly, or from a previous
        context / tool set) is left untouched, so explicit registration always
        wins and repeated frames don't re-register.

        Args:
            tools: The tools to scan for handlers.
        """
        # A context's ``tools`` may be ``None`` (rather than ``NOT_GIVEN``) — e.g.
        # from realtime services or stand-in contexts in tests. is_given(None) is
        # True and the normalizer rejects None, so guard it explicitly.
        if tools is None:
            return
        tools = LLMContext._normalize_and_validate_tools(tools)
        if not is_given(tools):
            return

        # Register direct functions.
        for wrapper in tools.direct_functions:
            if wrapper.name in self._functions:
                continue
            if wrapper.name in self._explicitly_unregistered_function_names:
                # Explicitly unregistered while still advertised — leave it gone so
                # calls hit the missing-handler recovery path.
                continue
            self._register_direct_function(wrapper.function)
            # Mark the entry as advertised-tool-set-managed so it can be pruned on a
            # later sync that stops advertising it. Names already in _functions are
            # skipped above, so explicit registrations keep their default
            # auto_registered=False and are never pruned.
            self._functions[wrapper.name].auto_registered = True
            logger.debug(
                f"{self}: auto-registered handler for advertised direct function '{wrapper.name}'"
            )

        # Register the handlers that FunctionSchemas carry. A schema handler
        # registers like any classic handler — register_function reads its
        # @tool_options off the handler — only marked auto_registered so it's
        # pruned when no longer advertised.
        for schema in tools.standard_tools:
            if schema.handler is None:
                continue
            if schema.name in self._functions:
                self._warn_if_redundant_manual_registration(schema.name)
                continue
            if schema.name in self._explicitly_unregistered_function_names:
                continue
            self.register_function(schema.name, schema.handler)
            self._functions[schema.name].auto_registered = True
            logger.debug(
                f"{self}: auto-registered handler for advertised FunctionSchema '{schema.name}'"
            )

    def _warn_if_redundant_manual_registration(self, function_name: str) -> None:
        """Warn that a manual registration is unnecessary, once per function.

        Fires when a tool is registered explicitly via ``register_function`` yet
        its advertised ``FunctionSchema`` already carries a handler — the schema
        alone would register it. An auto-registered entry (e.g. from a same-named
        direct function) is not a manual registration and is left silent.

        Args:
            function_name: The name shared by the manual registration and the
                handler-carrying schema. Must already be registered.
        """
        if self._functions[function_name].auto_registered:
            return
        if function_name in self._redundant_registration_warned:
            return
        self._redundant_registration_warned.add(function_name)
        logger.warning(
            f"{self}: '{function_name}' is registered with register_function() but its "
            "advertised FunctionSchema already carries a handler. The manual registration "
            "step is unnecessary when the handler is on the FunctionSchema."
        )

    def _service_tools(self) -> ToolsSchema | list[Any] | None:
        """Return the service's own configured tools, if any.

        Used by the auto-registration mechanism. Tools normally reach a service
        through the ``LLMContext`` on each inference. Some services (the realtime
        services) also accept tools configured directly on the service (e.g. via a
        constructor argument) as an alternative; they override this to return those
        tools verbatim, in whatever form they hold them — a ``ToolsSchema`` or a
        provider-native tool list. When the context advertises no tools,
        :meth:`_sync_registered_tool_handlers` normalizes the result and registers
        handlers for any standard tools it contains; provider-native tools carry no
        handlers, so they contribute nothing to register.

        Note:
            A service that overrides this **must follow the auto-registration
            mechanism's preference rules: context tools, then service-configured
            tools**.

        Returns:
            The service's configured tools verbatim (e.g. a ``ToolsSchema`` or a
            provider-native tool list), or ``None`` when there are none (the
            default).
        """
        return None

    def _sync_registered_tool_handlers(self, tools: Any) -> None:
        """Sync the registered handlers with the handlers advertised in ``tools``.

        Registers handlers for any tools the set advertises with one (direct
        functions, or ``FunctionSchema`` objects carrying a ``handler``), then
        drops the handlers we auto-registered for tools it no longer advertises —
        so the registry matches what the LLM can see.

        This is the single path for keeping handlers in step with the advertised
        tools. The base service runs it on every ``LLMContextFrame`` (the context
        carries the current tool set on each inference). Realtime services that
        support runtime tool changes also call it from their own ``LLMSetToolsFrame``
        handling, since they run continuously and don't get a context frame per turn.
        When the context advertises no tools, handlers fall back to
        :meth:`_service_tools` (the service's own configured tools).

        Explicit registrations (``register_function`` / ``register_direct_function``),
        the catch-all handler, and built-in tools are never pruned.

        Args:
            tools: The advertised tool set (a ``ToolsSchema``, a plain list, or
                ``None`` / ``NOT_GIVEN`` for "no tools").
        """
        # None and an empty list both mean "no tools advertised" (the normalizer
        # collapses an empty set to NOT_GIVEN), in which case every auto-registered
        # handler is pruned.
        normalized = (
            LLMContext._normalize_and_validate_tools(tools) if tools is not None else NOT_GIVEN
        )
        # No context tools? Fall back to the service's own configured tools.
        # Those may be provider-native (already formatted, carrying no handlers);
        # only standard tools (gathered into a ToolsSchema) contribute handlers.
        if not is_given(normalized):
            service_tools = self._service_tools()
            if service_tools is not None:
                service_tools = LLMContext._normalize_and_validate_tools(
                    service_tools, allow_provider_tools=True
                )
                if isinstance(service_tools, ToolsSchema):
                    normalized = service_tools
        self._register_advertised_tool_handlers(normalized)
        advertised: set[str | None] = set()
        if is_given(normalized):
            advertised |= {wrapper.name for wrapper in normalized.direct_functions}
            advertised |= {s.name for s in normalized.standard_tools if s.handler is not None}
        self._unregister_unadvertised_tool_handlers(advertised)
        # A standalone unregister only suppresses re-registration while the tool is
        # still advertised. Once it leaves the advertised set, drop the suppression
        # so re-advertising it (a later tool-set change) registers it afresh.
        self._explicitly_unregistered_function_names &= advertised

    def _unregister_unadvertised_tool_handlers(self, advertised: set[str | None]) -> None:
        """Drop auto-registered handlers for tools no longer advertised.

        Only entries with ``auto_registered=True`` are eligible; explicit
        registrations, the catch-all handler, and built-in tools are untouched.

        Args:
            advertised: Names of handler-carrying tools in the new advertised set.
        """
        stale = [
            name
            for name, item in self._functions.items()
            if item.auto_registered and name not in advertised
        ]
        for name in stale:
            del self._functions[name]
        # If the last async tool was just pruned, tear down the cancellation tool.
        if stale and self._async_tool_cancellation_enabled and not self._has_async_tools():
            self._teardown_async_tool_cancellation()

    def unregister_function(self, function_name: str | None):
        """Remove a registered function handler.

        Note:
            This removes the handler but does not stop advertising the tool to
            the LLM. To remove a tool cleanly, prefer an ``LLMSetToolsFrame`` with
            the updated tool set — that both stops advertising it and avoids the
            LLM trying to call a tool that's no longer there.

        Args:
            function_name: The name of the function handler to remove.
        """
        del self._functions[function_name]
        # Remember the explicit removal so auto-registration doesn't bring the
        # handler back on the next context frame while the tool is still advertised.
        self._explicitly_unregistered_function_names.add(function_name)
        if self._async_tool_cancellation_enabled and not self._has_async_tools():
            self._teardown_async_tool_cancellation()

    @deprecated(
        "`LLMService.unregister_direct_function` is deprecated since 1.4.0 and will be removed in "
        "2.0.0. Use `LLMSetToolsFrame` instead."
    )
    def unregister_direct_function(self, handler: Any):
        """Remove a registered direct function handler.

        .. deprecated:: 1.4.0
            Direct-function handlers are now managed automatically. To stop
            advertising a tool, push an :class:`LLMSetToolsFrame` with the updated tool
            set — the service unregisters the handler for any direct function no
            longer listed. Will be removed in 2.0.0.

        Note:
            This removes the handler but does not stop advertising the tool to
            the LLM. To remove a tool cleanly, prefer an ``LLMSetToolsFrame`` with
            the updated tool set — that both stops advertising it and avoids the
            LLM trying to call a tool that's no longer there.

        Args:
            handler: The direct function handler to remove.
        """
        self._unregister_direct_function(handler)

    def _unregister_direct_function(self, handler: Any):
        """Remove a registered direct function handler (internal; no warning).

        Args:
            handler: The direct function handler to remove.
        """
        wrapper = DirectFunctionWrapper(handler)
        del self._functions[wrapper.name]
        # Remember the explicit removal so auto-registration doesn't bring the
        # handler back on the next context frame while the tool is still advertised.
        self._explicitly_unregistered_function_names.add(wrapper.name)
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

    def _function_is_async(self, function_name: str) -> bool:
        """Whether the named function was registered with cancel_on_interruption=False.

        Mirrors the registry-lookup pattern in :meth:`run_function_calls`:
        a name-specific entry takes precedence; if there isn't one, fall
        back to the ``None``-keyed catch-all entry. Returns ``False`` if
        no entry matches.
        """
        item = self._functions.get(function_name)
        if item is None:
            item = self._functions.get(None)
        return item is not None and not item.cancel_on_interruption

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
                self._log_missing_function_call(function_call.function_name, function_call.context)
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

    async def _cancel_all_function_call_tasks(self):
        # Snapshot first: cancel_task awaits, during which done callbacks may
        # mutate _function_call_tasks.
        for task in list(self._function_call_tasks.keys()):
            if task:
                task.remove_done_callback(self._function_call_task_finished)
                await self.cancel_task(task)
        self._function_call_tasks.clear()

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
            # Function was unregistered between queue and execution; the
            # registry-item-handler check above already covered the
            # missing-from-the-start case.
            logger.warning(
                f"{self}: '{runner_item.function_name}' was just unregistered "
                f"between queueing and execution."
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
                        pipeline_worker=self.pipeline_worker,
                        context=runner_item.context,
                        result_callback=function_call_result_callback,
                        app_resources=self.pipeline_worker.app_resources,
                    ),
                )
            else:
                # Handler is a FunctionCallHandler
                params = FunctionCallParams(
                    function_name=runner_item.function_name,
                    tool_call_id=runner_item.tool_call_id,
                    arguments=runner_item.arguments,
                    llm=self,
                    pipeline_worker=self.pipeline_worker,
                    context=runner_item.context,
                    result_callback=function_call_result_callback,
                    app_resources=self.pipeline_worker.app_resources,
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
        await params.result_callback(
            self.MISSING_FUNCTION_CALL_MESSAGE_TEMPLATE.format(function_name=params.function_name)
        )

    @staticmethod
    def _advertised_tool_names(context) -> set[str]:
        """Return the set of standard-tool names currently advertised to the LLM.

        Custom (LLM-specific) tools are not included, since they have no
        consistent name field across adapters.
        """
        tools = context.tools if context is not None else None
        if tools is None or not is_given(tools):
            return set()
        return {t.name for t in tools.standard_tools}

    def _log_missing_function_call(self, function_name: str, context) -> None:
        """Log an appropriate message when a tool is called with no handler.

        Distinguishes two cases:

        - **Developer error:** the tool is advertised to the LLM but no handler
          was wired up (a ``FunctionSchema`` must've been provided with neither
          ``handler`` nor accompanying ``register_function`` call). Logged at
          error level since this almost always indicates a bug.
        - **Hallucination:** the tool is not in the currently advertised tool
          set. Logged at warning level since this is model behavior the
          application can do little about beyond returning a terminal result.
        """
        if function_name in self._advertised_tool_names(context):
            logger.error(
                f"{self}: tool '{function_name}' is advertised to the LLM but has "
                f"no handler — set FunctionSchema.handler (recommended) or call "
                f"register_function()."
            )
        else:
            logger.warning(
                f"{self}: LLM called '{function_name}', which is not in the "
                f"currently advertised tool set."
            )

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
