#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Core conversation flow management system.

This module provides the FlowManager class which orchestrates
conversations across different LLM providers. It supports:

- Flows with runtime-determined transitions
- State management and transitions
- Function registration and execution
- Action handling
- Cross-provider compatibility

The flow manager coordinates all aspects of a conversation, including:

- LLM context management
- Function registration
- State transitions
- Action execution
- Error handling
"""

import asyncio
import inspect
import warnings
from collections.abc import Callable
from typing import Any, cast

from loguru import logger

from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.flows.actions import ActionError, ActionManager
from pipecat.flows.adapters import LLMAdapter
from pipecat.flows.exceptions import (
    FlowError,
    FlowInitializationError,
    FlowTransitionError,
    InvalidFunctionError,
)
from pipecat.flows.types import (
    NO_RESPONSE,
    ActionConfig,
    ConsolidatedFunctionResult,
    ContextStrategy,
    ContextStrategyConfig,
    FlowArgs,
    FlowFunctionHandler,
    FlowsDirectFunction,
    FlowsDirectFunctionWrapper,
    FlowsFunctionSchema,
    FunctionHandler,
    LegacyFunctionHandler,
    NodeConfig,
    ZeroArgFunctionHandler,
    get_or_generate_node_name,
)
from pipecat.frames.frames import (
    FunctionCallResultProperties,
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMRunFrame,
    LLMSetToolsFrame,
    LLMUpdateSettingsFrame,
)
from pipecat.pipeline.llm_switcher import LLMSwitcher
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import NOT_GIVEN, LLMContext, NotGiven
from pipecat.services.llm_service import FunctionCallParams, LLMService
from pipecat.services.settings import LLMSettings
from pipecat.transports.base_transport import BaseTransport
from pipecat.utils.deprecation import deprecated


class FlowManager:
    """Manages conversation flows.

    The FlowManager orchestrates conversation flows by managing state transitions,
    function registration, and message handling across different LLM providers,
    with comprehensive action handling and error management.

    The manager coordinates all aspects of a conversation including LLM context
    management, function registration, state transitions, and action execution.
    """

    def __init__(
        self,
        *,
        llm: LLMService | LLMSwitcher,
        context_aggregator: Any,
        worker: PipelineWorker | None = None,
        task: PipelineWorker | None = None,
        context_strategy: ContextStrategyConfig | None = None,
        transport: BaseTransport | None = None,
        global_functions: list[FlowsFunctionSchema | FlowsDirectFunction] | None = None,
    ):
        """Initialize the flow manager.

        Args:
            llm: LLM service or LLMSwitcher.
            context_aggregator: Context aggregator for updating user context.
            worker: PipelineWorker instance for queueing frames.
            task: PipelineWorker instance for queueing frames.

                .. deprecated:: 1.5.0
                    Use ``worker`` instead. Will be removed in 2.0.0.

            context_strategy: Context strategy configuration for managing conversation
                context during transitions.
            transport: Transport instance for communication.
            global_functions: Optional list of FlowsFunctionSchemas or FlowsDirectFunctions
                that will be available at every node. These functions are registered once
                during initialization and automatically included alongside node-specific
                functions.
        """
        if worker is not None and task is not None:
            raise ValueError("Pass either 'worker' or 'task' (deprecated), not both.")
        if task is not None:
            warnings.warn(
                "The 'task' parameter is deprecated since 1.5.0 and will be removed "
                "in 2.0.0. Use 'worker' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            worker = task
        if worker is None:
            raise ValueError("FlowManager requires a 'worker' (PipelineWorker).")

        self._worker = worker
        self._llm = llm
        self._action_manager = ActionManager(worker, flow_manager=self)
        self._adapter = LLMAdapter()
        self._initialized = False
        self._context_aggregator = context_aggregator
        self._pending_transition: dict[str, Any] | None = None
        self._context_strategy = context_strategy or ContextStrategyConfig(
            strategy=ContextStrategy.APPEND
        )
        self._transport = transport
        self._global_functions = global_functions or []

        self._state: dict[str, Any] = {}  # Internal state storage
        self._current_functions: set[str] = set()  # Track registered functions
        self._current_node: str | None = None

        self._showed_deprecation_warning_for_role_messages = False
        self._showed_deprecation_warning_for_reset_with_summary = False
        self._showed_deprecation_warning_for_zero_arg_handler = False
        self._showed_deprecation_warning_for_legacy_handler = False

    @property
    def state(self) -> dict[str, Any]:
        """Access the shared state dictionary across nodes.

        This property provides access to a persistent dictionary that maintains
        data across node transitions. It can be used to store and retrieve
        conversation state, user preferences, or any other data that needs
        to persist throughout the flow.

        Returns:
            Dict[str, Any]: The shared state dictionary that can be used for
                reading and writing state data.

        Examples:
            Setting state::

                flow_manager.state["user_name"] = "Alice"
                flow_manager.state["age"] = 25

            Getting state::

                name = flow_manager.state.get("user_name", "Unknown")
                age = flow_manager.state["age"]

            Checking for state::

                if "user_preferences" in flow_manager.state:
                    preferences = flow_manager.state["user_preferences"]
        """
        return self._state

    @property
    def transport(self) -> BaseTransport | None:
        """Access the transport instance used for communication.

        This property provides access to the transport instance that handles
        communication with the client (e.g., DailyTransport for Daily rooms).
        The transport can be used to interact with participants, manage
        audio/video settings, or access platform-specific features.

        Returns:
            Optional[BaseTransport]: The transport instance if provided during
                initialization, None otherwise.

        Examples:
            Accessing transport in action handlers::

                async def mute_participant(action: dict, flow_manager: FlowManager):
                    transport = flow_manager.transport
                    if transport and hasattr(transport, 'update_participant'):
                        await transport.update_participant(participant_id, {"canSnd": False})

            Working with Daily transport features::

                async def get_room_info(action: dict, flow_manager: FlowManager):
                    transport = flow_manager.transport
                    if isinstance(transport, DailyTransport):
                        participants = transport.participants()
                        return {"participant_count": len(participants)}
        """
        return self._transport

    @property
    def current_node(self) -> str | None:
        """Access the identifier of the currently active conversation node.

        This property provides access to the current node name/identifier in the
        conversation flow. It can be used to make decisions based on the current
        state of the conversation, implement conditional logic, or for debugging
        and logging purposes.

        Returns:
            Optional[str]: The identifier of the current node if a node is active,
                None if no node has been set or before initialization.

        Examples:
            Conditional logic based on current node::

                async def participant_joined(action: dict, flow_manager: FlowManager):
                    current = flow_manager.current_node
                    if current == "transferring_to_human_agent":
                        await start_human_agent_interaction(flow_manager)
                    elif current == "collecting_payment":
                        await setup_secure_session(flow_manager)

            Logging and debugging::

                async def log_conversation_state(action: dict, flow_manager: FlowManager):
                    node = flow_manager.current_node
                    logger.info(f"Current conversation node: {node}")
                    return {"current_node": node}
        """
        return self._current_node

    @property
    def worker(self) -> PipelineWorker:
        """Access the pipeline worker instance for frame queueing.

        This property provides access to the PipelineWorker instance used by the
        FlowManager. The worker can be used to queue custom frames directly into
        the pipeline, enabling advanced flow control and custom frame injection.

        Returns:
            PipelineWorker: The pipeline worker instance used for frame processing
                and queueing operations.

        Examples:
            Queueing frames in handlers::

                async def send_custom_notification(action: dict, flow_manager: FlowManager):
                    from pipecat.frames.frames import TTSUpdateSettingsFrame

                    # Queue a TTS settings update frame
                    await flow_manager.worker.queue_frame(
                        TTSUpdateSettingsFrame(settings={"voice": "your-new-voice-id"})
                    )
        """
        return self._worker

    @property
    @deprecated(
        "`FlowManager.task` is deprecated since 1.5.0 and will be removed in 2.0.0. "
        "Use `FlowManager.worker` instead."
    )
    def task(self) -> PipelineWorker:
        """Access the pipeline worker instance for frame queueing.

        .. deprecated:: 1.5.0
            Use :attr:`worker` instead. Will be removed in 2.0.0.

        Returns:
            PipelineWorker: The pipeline worker instance used for frame processing
                and queueing operations.
        """
        return self._worker

    async def initialize(self, initial_node: NodeConfig | None = None) -> None:
        """Initialize the flow manager.

        Args:
            initial_node: Optional initial node configuration. If provided,
                the flow will start at this node immediately.

        Raises:
            FlowInitializationError: If initialization fails.

        Examples:
            Initialize with an initial node::

                flow_manager = FlowManager(
                    ... # Initialization parameters
                )
                await flow_manager.initialize(create_initial_node())

            Initialize without an initial node (set later via set_node_from_config)::

                flow_manager = FlowManager(
                    ... # Initialization parameters
                )
                await flow_manager.initialize()
        """
        if self._initialized:
            logger.warning(f"{self.__class__.__name__} already initialized")
            return

        try:
            self._initialized = True
            logger.debug(f"Initialized {self.__class__.__name__}")

            # Set initial node if provided (otherwise initial node
            # will be set later via set_node_from_config())
            if initial_node:
                node_name = get_or_generate_node_name(initial_node)
                logger.debug(f"Setting initial node: {node_name}")
                await self._set_node(node_name, initial_node)

        except Exception as e:
            self._initialized = False
            raise FlowInitializationError(f"Failed to initialize flow: {str(e)}") from e

    def get_current_context(self) -> list[dict]:
        """Get the current conversation context.

        Returns:
            List of messages in the current context, including system messages,
            user messages, and assistant responses.

        Raises:
            FlowError: If context aggregator is not available.
        """
        if not self._context_aggregator:
            raise FlowError("No context aggregator available")

        context = self._context_aggregator.user()._context

        return context.get_messages()

    def register_action(self, action_type: str, handler: Callable) -> None:
        """Register a handler for a specific action type.

        Args:
            action_type: String identifier for the action (e.g., "tts_say").
            handler: Async or sync function that handles the action.

        Example::

            async def custom_notification(action: dict):
                text = action.get("text", "")
                await notify_user(text)

            flow_manager.register_action("notify", custom_notification)
        """
        self._action_manager._register_action(action_type, handler)

    def _register_action_from_config(self, action: ActionConfig) -> None:
        """Register an action handler from action configuration.

        Args:
            action: Action configuration dictionary containing type and optional handler.

        Raises:
            ActionError: If action type is not registered and no valid handler provided.
        """
        action_type = action.get("type")
        handler = action.get("handler")

        # Register action if not already registered
        if action_type and action_type not in self._action_manager._action_handlers:
            # Register handler if provided
            if handler and callable(handler):
                self.register_action(action_type, handler)
                logger.debug(f"Registered action handler from config: {action_type}")
            else:
                raise ActionError(
                    f"Action '{action_type}' not registered. "
                    "Provide handler in action config or register manually."
                )

    async def _call_handler(
        self, handler: FunctionHandler, args: FlowArgs
    ) -> Any | ConsolidatedFunctionResult:
        """Call handler with appropriate parameters based on its signature.

        Detects whether the handler can accept a flow_manager parameter and
        calls it accordingly to maintain backward compatibility with legacy handlers.

        Args:
            handler: The function handler to call (either legacy or modern format).
            args: Arguments dictionary from the function call.

        Returns:
            The result returned by the handler.
        """
        # Get the function signature
        sig = inspect.signature(handler)

        # Calculate effective parameter count
        effective_param_count = len(sig.parameters)

        # Handle different function signatures. inspect.signature has already
        # proven the shape, so each cast narrows the union to the branch we know
        # we're in.
        if effective_param_count == 0:
            if not self._showed_deprecation_warning_for_zero_arg_handler:
                self._showed_deprecation_warning_for_zero_arg_handler = True
                warnings.warn(
                    "Zero-argument function handlers are deprecated and will be "
                    "removed in 2.0.0. Update handlers to accept "
                    "(args: FlowArgs, flow_manager: FlowManager) instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return await cast(ZeroArgFunctionHandler, handler)()
        elif effective_param_count == 1:
            if not self._showed_deprecation_warning_for_legacy_handler:
                self._showed_deprecation_warning_for_legacy_handler = True
                warnings.warn(
                    "Single-argument (legacy) function handlers are deprecated "
                    "and will be removed in 2.0.0. Update handlers to accept "
                    "(args: FlowArgs, flow_manager: FlowManager) instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return await cast(LegacyFunctionHandler, handler)(args)
        else:
            return await cast(FlowFunctionHandler, handler)(args, self)

    async def _create_transition_func(
        self,
        name: str,
        handler: Callable | FlowsDirectFunctionWrapper,
    ) -> Callable:
        """Create a transition function for the given name and handler.

        Args:
            name: Name of the function being registered.
            handler: Function to process the call: a Flows function handler or a
                direct-function wrapper.

        Returns:
            Async function that handles the tool invocation.
        """

        async def transition_func(params: FunctionCallParams) -> None:
            """Inner function that handles the actual tool invocation."""
            try:
                logger.debug(f"Function called: {name}")

                is_transition_only_function = False
                acknowledged_result = {"status": "acknowledged"}

                # Invoke the handler with the provided arguments
                if isinstance(handler, FlowsDirectFunctionWrapper):
                    handler_response = await handler.invoke(params.arguments, self)
                else:
                    # Convert Pipecat's Mapping to a fresh dict so handlers may
                    # mutate without touching Pipecat's internal state. (In 2.0.0
                    # FlowArgs is planned to widen to Mapping; this conversion
                    # can go away then.)
                    handler_response = await self._call_handler(handler, dict(params.arguments))
                # Support both "consolidated" handlers that return (result, next_node) and handlers
                # that return just the result.
                if isinstance(handler_response, tuple):
                    result, next_node = handler_response
                    if result is None:
                        result = acknowledged_result
                        is_transition_only_function = True
                else:
                    result = handler_response
                    next_node = None
                    # FlowsDirectFunctions should always be "consolidated" functions that return a tuple
                    if isinstance(handler, FlowsDirectFunctionWrapper):
                        raise InvalidFunctionError(
                            f"Direct function {name} expected to return a tuple (result, next_node) but got {type(result)}"
                        )

                logger.debug(
                    f"{'Transition-only function called for' if is_transition_only_function else 'Function handler completed for'} {name}"
                )

                is_no_response = next_node is NO_RESPONSE
                if is_no_response or not next_node:
                    # Node function: stay on the current node.
                    properties = FunctionCallResultProperties(
                        run_llm=not is_no_response,
                        on_context_updated=None,
                    )
                else:
                    # Edge function: transition to the returned node.
                    self._pending_transition = {
                        "next_node": next_node,
                        "function_name": name,
                        "arguments": params.arguments,
                        "result": result,
                    }
                    properties = FunctionCallResultProperties(
                        run_llm=False,
                        on_context_updated=self._check_and_execute_transition,
                    )

                await params.result_callback(result, properties=properties)

            except Exception as e:
                logger.error(f"Error in transition function {name}: {str(e)}")
                error_result = {"status": "error", "error": str(e)}
                await params.result_callback(error_result)

        return transition_func

    async def _check_and_execute_transition(self) -> None:
        """Check if all functions are complete and execute transition if so."""
        if not self._pending_transition:
            return

        # Check if all function calls are complete using Pipecat's state
        assistant_aggregator = self._context_aggregator.assistant()
        if not assistant_aggregator.has_function_calls_in_progress:
            # All functions complete, execute transition
            transition_info = self._pending_transition
            self._pending_transition = None

            await self._execute_transition(transition_info)

    async def _execute_transition(self, transition_info: dict[str, Any]) -> None:
        """Execute the stored transition."""
        next_node = transition_info.get("next_node")

        try:
            if next_node:
                node_name = get_or_generate_node_name(next_node)
                logger.debug(f"Transition to function-returned node: {node_name}")
                await self._set_node(node_name, next_node)
        except Exception as e:
            logger.error(f"Error executing transition: {str(e)}")
            raise

    async def _create_function_schema(
        self, tool: FlowsFunctionSchema | FlowsDirectFunctionWrapper
    ) -> FunctionSchema:
        """Build a FunctionSchema that carries the handler the LLM service will run.

        Flows wraps each tool's handler in a "transition function" that runs the
        tool's work and coordinates any node transition.

        Args:
            tool: The node's function, as a ``FlowsFunctionSchema`` or a wrapped
                direct function.

        Returns:
            A ``FunctionSchema`` describing the tool and carrying its handler.
        """
        # For a direct function the wrapper itself is the handler; a
        # FlowsFunctionSchema carries its handler separately.
        handler = tool if isinstance(tool, FlowsDirectFunctionWrapper) else tool.handler
        # Stamp the resolved call options onto the handler so the LLM service
        # applies them when it registers the advertised tool. This is base
        # Pipecat's ``tool_options`` (not ``flows_tool_options``): the handler
        # rides on a base ``FunctionSchema``, and ``tool`` already carries Flows'
        # resolved option values.
        transition_func = tool_options(
            cancel_on_interruption=tool.cancel_on_interruption,
            timeout_secs=tool.timeout_secs,
        )(await self._create_transition_func(tool.name, handler))
        base = tool.to_function_schema()
        return FunctionSchema(
            name=base.name,
            description=base.description,
            properties=base.properties,
            required=base.required,
            handler=transition_func,
        )

    async def set_node_from_config(self, node_config: NodeConfig) -> None:
        """Set up a new conversation node and transition to it.

        Used to manually transition between nodes in a flow.

        Args:
            node_config: Configuration for the new node.

        Raises:
            FlowTransitionError: If manager not initialized.
            FlowError: If node setup fails.
        """
        await self._set_node(get_or_generate_node_name(node_config), node_config)

    async def _set_node(self, node_id: str, node_config: NodeConfig) -> None:
        """Set up a new conversation node and transition to it.

        Handles the complete node transition process in the following order:
        1. Execute pre-actions (if any)
        2. Set up messages (role and task)
        3. Register node functions
        4. Update LLM context with messages and tools
        5. Update state (current node and functions)
        6. Trigger LLM completion with new context
        7. Execute post-actions (if any)

        Args:
            node_id: Identifier for the new node.
            node_config: Complete configuration for the node.

        Raises:
            FlowTransitionError: If manager not initialized.
            FlowError: If node setup fails.
        """
        if not self._initialized:
            raise FlowTransitionError(f"{self.__class__.__name__} must be initialized first")

        try:
            # Clear any pending transition state when starting a new node
            # This ensures clean state regardless of how we arrived here:
            # - Normal transition flow (already cleared in _check_and_execute_transition)
            # - Direct calls to set_node/set_node_from_config
            self._pending_transition = None

            self._validate_node_config(node_id, node_config)
            logger.debug(f"Setting node: {node_id}")

            # Clear any deferred post-actions from previous node
            self._action_manager.clear_deferred_post_actions()

            # Register action handlers from config
            for action_list in [
                node_config.get("pre_actions", []),
                node_config.get("post_actions", []),
            ]:
                for action in action_list:
                    self._register_action_from_config(action)

            # Execute pre-actions if any
            if pre_actions := node_config.get("pre_actions"):
                await self._execute_actions(pre_actions=pre_actions)

            # Build the node's function schemas (carrying handlers)
            new_functions: set[str] = set()

            # Mix in global functions that should be available at every node
            functions_list = self._global_functions + node_config.get("functions", [])

            standard_functions: list[FunctionSchema] = []
            for func_config in functions_list:
                if callable(func_config):
                    tool = FlowsDirectFunctionWrapper(function=func_config)
                elif isinstance(func_config, FlowsFunctionSchema):
                    tool = func_config
                else:
                    raise InvalidFunctionError(
                        f"Invalid function format in node '{node_id}'. "
                        "Use FlowsFunctionSchema or direct functions."
                    )
                standard_functions.append(await self._create_function_schema(tool))
                new_functions.add(tool.name)

            formatted_tools = (
                ToolsSchema(standard_tools=standard_functions) if standard_functions else NOT_GIVEN
            )

            role_message = node_config.get("role_message")
            role_messages = node_config.get("role_messages")

            if role_message and role_messages:
                logger.warning(
                    "Both 'role_message' and 'role_messages' specified; using 'role_message'"
                )

            if role_messages and not role_message:
                if not self._showed_deprecation_warning_for_role_messages:
                    self._showed_deprecation_warning_for_role_messages = True
                    warnings.warn(
                        "'role_messages' is deprecated and will be removed in 2.0.0. "
                        "Use 'role_message' (singular, str) instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )

            # Update LLM context
            await self._update_llm_context(
                role_message=role_message,
                role_messages=role_messages if not role_message else None,
                task_messages=node_config["task_messages"],
                functions=formatted_tools,
                strategy=node_config.get("context_strategy"),
            )
            logger.debug("Updated LLM context")

            # Update state
            self._current_node = node_id
            self._current_functions = new_functions

            # Trigger completion with new context
            respond_immediately = node_config.get("respond_immediately", True)
            if respond_immediately:
                await self._worker.queue_frames([LLMRunFrame()])

            # Execute post-actions if any
            if post_actions := node_config.get("post_actions"):
                if respond_immediately:
                    await self._execute_actions(post_actions=post_actions)
                else:
                    # Schedule post-actions for execution after first LLM response in this node
                    self._schedule_deferred_post_actions(post_actions=post_actions)

            logger.debug(f"Successfully set node: {node_id}")

        except Exception as e:
            logger.error(f"Error setting node {node_id}: {str(e)}")
            raise FlowError(f"Failed to set node {node_id}: {str(e)}") from e

    def _schedule_deferred_post_actions(self, post_actions: list[ActionConfig]) -> None:
        self._action_manager.schedule_deferred_post_actions(post_actions=post_actions)

    async def _create_conversation_summary(
        self, summary_prompt: str, context: LLMContext
    ) -> str | None:
        """Generate a conversation summary from a given context."""
        return await self._adapter.generate_summary(self._llm, summary_prompt, context)

    async def _update_llm_context(
        self,
        role_message: str | None,
        role_messages: list[dict] | None,
        task_messages: list[dict],
        functions: ToolsSchema | NotGiven,
        strategy: ContextStrategyConfig | None = None,
    ) -> None:
        """Update LLM context with new messages and functions.

        If ``role_message`` is provided, it is sent as an
        ``LLMUpdateSettingsFrame`` (system instruction on the LLM itself).

        If ``role_messages`` (deprecated) is provided, the messages are
        prepended to the conversation context alongside ``task_messages``.

        Args:
            role_message: Optional role/personality string sent as the LLM
                system instruction via ``LLMUpdateSettingsFrame``.
            role_messages: Deprecated list-of-dicts prepended to context
                messages for backward compatibility.
            task_messages: Task messages to add to context.
            functions: New functions to make available.
            strategy: Optional context update configuration.

        Raises:
            FlowError: If context update fails.
        """
        try:
            frames = []

            # New path: role_message as LLM system instruction (persists until changed)
            if role_message:
                frames.append(
                    LLMUpdateSettingsFrame(delta=LLMSettings(system_instruction=role_message))
                )

            messages = []

            # Legacy path: role_messages prepended to context messages
            if role_messages:
                messages.extend(role_messages)

            update_config = strategy or self._context_strategy

            if update_config.strategy == ContextStrategy.RESET_WITH_SUMMARY:
                if not self._showed_deprecation_warning_for_reset_with_summary:
                    self._showed_deprecation_warning_for_reset_with_summary = True
                    warnings.warn(
                        "RESET_WITH_SUMMARY is deprecated and will be removed in 2.0.0. "
                        "Use Pipecat's native context summarization instead. To trigger "
                        "on-demand summarization during a node transition, push an "
                        "LLMSummarizeContextFrame in a pre-action. See "
                        "https://docs.pipecat.ai/guides/fundamentals/context-summarization",
                        DeprecationWarning,
                        stacklevel=2,
                    )

            if (
                update_config.strategy == ContextStrategy.RESET_WITH_SUMMARY
                and self._context_aggregator
                and self._context_aggregator.user()._context
            ):
                # We know summary_prompt exists because of __post_init__ validation in ContextStrategyConfig
                summary_prompt = cast(str, update_config.summary_prompt)
                try:
                    # Try to get summary with 5 second timeout
                    summary = await asyncio.wait_for(
                        self._create_conversation_summary(
                            summary_prompt,
                            self._context_aggregator.user()._context,
                        ),
                        timeout=5.0,
                    )

                    if summary:
                        summary_message = self._adapter.format_summary_message(summary)
                        messages.append(summary_message)
                        logger.debug(f"Added conversation summary to context: {summary_message}")
                    else:
                        # Fall back to APPEND strategy if summary fails
                        logger.warning(
                            "Failed to generate summary, falling back to APPEND strategy"
                        )
                        update_config.strategy = ContextStrategy.APPEND

                except TimeoutError:
                    logger.warning("Summary generation timed out, falling back to APPEND strategy")
                    update_config.strategy = ContextStrategy.APPEND

            # Add task messages
            messages.extend(task_messages)

            # Use an "update" (replace) frame for the RESET/RESET_WITH_SUMMARY
            # strategies; otherwise append. (Note that even the first node follows
            # the same rule: appending ensures any prior context contributions,
            # such as by tts_say pre-actions, is preserved rather than replaced).
            frame_type = (
                LLMMessagesUpdateFrame
                if update_config.strategy
                in [ContextStrategy.RESET, ContextStrategy.RESET_WITH_SUMMARY]
                else LLMMessagesAppendFrame
            )

            frames.append(frame_type(messages=messages))
            frames.append(LLMSetToolsFrame(tools=functions))

            await self._worker.queue_frames(frames)

            logger.debug(
                f"Updated LLM context using {frame_type.__name__} with strategy {update_config.strategy}"
            )

        except Exception as e:
            logger.error(f"Failed to update LLM context: {str(e)}")
            raise FlowError(f"Context update failed: {str(e)}") from e

    async def _execute_actions(
        self,
        pre_actions: list[ActionConfig] | None = None,
        post_actions: list[ActionConfig] | None = None,
    ) -> None:
        """Execute pre and post actions.

        Args:
            pre_actions: Actions to execute before context update.
            post_actions: Actions to execute after context update.
        """
        if pre_actions:
            await self._action_manager.execute_actions(pre_actions)
        if post_actions:
            await self._action_manager.execute_actions(post_actions)

    def _validate_node_config(self, node_id: str, config: NodeConfig) -> None:
        """Validate the configuration of a conversation node.

        This method ensures that:
        1. Required fields (task_messages) are present.
        2. Each function is either a ``FlowsFunctionSchema`` or a valid direct
           function.

        Args:
            node_id: Identifier for the node being validated.
            config: Complete node configuration to validate.

        Raises:
            FlowError: If required fields are missing.
            InvalidFunctionError: If function format is invalid.
        """
        # Check required fields
        if "task_messages" not in config:
            raise FlowError(f"Node '{node_id}' missing required 'task_messages' field")

        # Get functions list with default empty list if not provided
        functions_list = config.get("functions", [])

        # Validate each function configuration if there are any
        for func in functions_list:
            if callable(func):
                FlowsDirectFunctionWrapper.validate_function(func)
            elif not isinstance(func, FlowsFunctionSchema):
                raise InvalidFunctionError(
                    f"Invalid function format in node '{node_id}'. "
                    "Use FlowsFunctionSchema or direct functions."
                )
