#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Type definitions for the conversation flow system.

This module defines the core types used throughout the flow system:

- FlowResult: Function return type
- FlowArgs: Function argument type
- NodeConfig: Node configuration type
- FlowsFunctionSchema: A uniform schema for function calls in flows

These types provide structure and validation for flow configurations
and function interactions.
"""

import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Required,
    TypedDict,
)

from pipecat.adapters.schemas.direct_function import BaseDirectFunctionWrapper, tool_options
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.flows.exceptions import InvalidFunctionError
from pipecat.utils.deprecation import deprecated

if TYPE_CHECKING:
    from pipecat.flows.manager import FlowManager


@deprecated("`FlowResult` is deprecated since 1.5.0 and will be removed in 2.0.0. No replacement.")
class FlowResult(TypedDict, total=False):
    """Optional convention TypedDict for ``status``/``error`` results.

    .. deprecated:: 1.5.0
        No replacement. ``FlowResult`` is no longer required or referenced by
        any handler type, and Pipecat's upstream function-call-result contract
        is ``Any`` — define your own ``TypedDict`` or return any
        JSON-serializable value. Will be removed in 2.0.0.

    Parameters:
        status: Status of the function execution.
        error: Optional error message if execution failed.
    """

    status: str
    error: str


FlowArgs = dict[str, Any]
"""Type alias for function handler arguments.

Each invocation gets its own dict, so handlers may mutate it freely without
affecting Pipecat's internal state.

.. note::

    In 2.0.0 this alias is planned to widen to ``Mapping[str, Any]`` to align
    with Pipecat's typing of ``FunctionCallParams.arguments``. Handlers that
    only read args will be unaffected; handlers that mutate args will need to
    keep the annotation as ``dict[str, Any]`` explicitly.

Example::

    {
        "user_name": "John",
        "age": 25,
        "preferences": {"color": "blue"}
    }
"""


LegacyActionHandler = Callable[[dict[str, Any]], Awaitable[None]]
"""Legacy action handler type that only receives the action dictionary.

.. deprecated:: 1.5.0
    Use :data:`FlowActionHandler` (``(action, flow_manager)``) instead. Will be
    removed in 2.0.0.

Args:
    action: Dictionary containing action configuration and parameters.

Example::

    async def simple_handler(action: dict):
        await notify(action["text"])
"""

FlowActionHandler = Callable[[dict[str, Any], "FlowManager"], Awaitable[None]]
"""Modern action handler type that receives both action and flow_manager.

Args:
    action: Dictionary containing action configuration and parameters.
    flow_manager: Reference to the FlowManager instance.

Example::

    async def advanced_handler(action: dict, flow_manager: FlowManager):
        await flow_manager.transport.notify(action["text"])
"""


class ActionConfig(TypedDict, total=False):
    """Configuration for an action.

    Parameters:
        type: Action type identifier (e.g. "tts_say", "notify_slack").
        handler: Callable to handle the action.
        text: Text to speak for the "tts_say" action, or the optional goodbye
            message for the "end_conversation" action.
        append_text_to_context: For the built-in TTS actions ("tts_say" and
            "end_conversation"), whether the spoken ``text`` is appended to the
            LLM context. Defaults to True.

    Note:
        Additional fields are allowed and passed to the handler.
    """

    type: Required[str]
    handler: LegacyActionHandler | FlowActionHandler
    text: str
    append_text_to_context: bool


class ContextStrategy(Enum):
    """Strategy for managing context during node transitions.

    Parameters:
        APPEND: Append new messages to existing context (default).
        RESET: Reset context with new messages only.
        RESET_WITH_SUMMARY: Reset context but include an LLM-generated summary.

            .. deprecated:: 1.5.0
                Use :class:`LLMSummarizeContextFrame` instead — push it in a
                pre-action to trigger on-demand summarization during a node
                transition. See
                https://docs.pipecat.ai/guides/fundamentals/context-summarization.
                Will be removed in 2.0.0.
    """

    APPEND = "append"
    RESET = "reset"
    RESET_WITH_SUMMARY = "reset_with_summary"


@dataclass
class ContextStrategyConfig:
    """Configuration for context management.

    Parameters:
        strategy: Strategy to use for context management.
        summary_prompt: Required prompt text when using RESET_WITH_SUMMARY.

            .. deprecated:: 1.5.0
                Use ``LLMContextSummaryConfig.summarization_prompt`` instead.
                Deprecated together with ``RESET_WITH_SUMMARY``. Will be removed
                in 2.0.0.
    """

    strategy: ContextStrategy
    summary_prompt: str | None = None

    def __post_init__(self):
        """Validate configuration.

        Raises:
            ValueError: If summary_prompt is missing when using RESET_WITH_SUMMARY.
        """
        if self.strategy == ContextStrategy.RESET_WITH_SUMMARY and not self.summary_prompt:
            raise ValueError("summary_prompt is required when using RESET_WITH_SUMMARY strategy")


class NodeConfig(TypedDict, total=False):
    """Configuration for a single node in the flow.

    Parameters:
        task_messages: List of message dicts defining the current node's objectives.
        name: Name of the node, useful for debug logging when returning a next node
            from a "consolidated" function.
        role_message: The bot's role/personality as a plain string, sent as the
            LLM's system instruction via ``LLMUpdateSettingsFrame``. When
            provided, the system instruction persists across node transitions
            until a new node explicitly sets ``role_message`` again.
        role_messages: Deprecated list-of-dicts format for the bot's role/personality.

            .. deprecated:: 1.5.0
                Use ``role_message`` (str) instead. Will be removed in 2.0.0.

        functions: List of FlowsFunctionSchema definitions or direct functions
            whose definitions are automatically extracted from their signatures.
        pre_actions: Actions to execute before LLM inference.
        post_actions: Actions to execute after LLM inference.
        context_strategy: Strategy for updating context during transitions.
        respond_immediately: Whether to run LLM inference as soon as the node is
            set (default: True).

    Example::

        {
            "role_message": "You are a helpful assistant...",
            "task_messages": [
                {
                    "role": "developer",
                    "content": "Ask the user for their name..."
                }
            ],
            "functions": [...],
            "pre_actions": [...],
            "post_actions": [...],
            "context_strategy": ContextStrategyConfig(strategy=ContextStrategy.APPEND),
            "respond_immediately": true,
        }
    """

    task_messages: Required[list[dict]]
    name: str
    role_message: str
    role_messages: list[dict[str, Any]]
    # ``FlowsFunctionSchema`` and ``FlowsDirectFunction`` are defined below
    # (see the note above ``ConsolidatedFunctionResult``); string forward
    # references keep ``NodeConfig`` definable here without re-introducing the
    # cross-module forward reference that ``ConsolidatedFunctionResult`` used
    # to require.
    functions: "list[FlowsFunctionSchema | FlowsDirectFunction]"
    pre_actions: list[ActionConfig]
    post_actions: list[ActionConfig]
    context_strategy: ContextStrategyConfig
    respond_immediately: bool


# ``ConsolidatedFunctionResult`` is the public return-type alias for "direct"
# functions. It must be defined **after** ``NodeConfig`` and without a string
# forward reference: ``get_type_hints()`` on a user-defined direct function
# resolves names against the user's module globals, not this module's, so a
# ``"NodeConfig"`` forward reference here would fail unless the user happened
# to import ``NodeConfig`` themselves.
ConsolidatedFunctionResult = tuple[Any, NodeConfig | None]
"""Return type for "consolidated" functions.

Return type for "consolidated" functions that do either or both of:
- doing some work
- specifying the next node to transition to after the work is done

The first tuple element is the function-call result delivered to the LLM.
Any JSON-serializable value is accepted (matching Pipecat's upstream
``FunctionCallResultCallback`` contract). Pass ``None`` to signal a
transition-only handler; FlowManager substitutes an acknowledgement result.
"""


ZeroArgFunctionHandler = Callable[[], Awaitable[Any]]
"""Function handler that takes no arguments.

.. deprecated:: 1.5.0
    Use :data:`FlowFunctionHandler` (``(args, flow_manager)``) instead. Will be
    removed in 2.0.0.

Returns:
    Any JSON-serializable value, or a :data:`ConsolidatedFunctionResult`
    tuple to also specify the next node.
"""

LegacyFunctionHandler = Callable[[FlowArgs], Awaitable[Any]]
"""Legacy function handler that only receives arguments.

.. deprecated:: 1.5.0
    Use :data:`FlowFunctionHandler` (``(args, flow_manager)``) instead. Will be
    removed in 2.0.0.

Args:
    args: Dictionary of arguments from the function call.

Returns:
    Any JSON-serializable value, or a :data:`ConsolidatedFunctionResult`
    tuple to also specify the next node.
"""

FlowFunctionHandler = Callable[[FlowArgs, "FlowManager"], Awaitable[Any]]
"""Modern function handler that receives both arguments and flow_manager.

Args:
    args: Dictionary of arguments from the function call.
    flow_manager: Reference to the FlowManager instance.

Returns:
    Any JSON-serializable value, or a :data:`ConsolidatedFunctionResult`
    tuple to also specify the next node.
"""


FunctionHandler = ZeroArgFunctionHandler | LegacyFunctionHandler | FlowFunctionHandler
"""Union type for function handlers supporting 0-arg, legacy, and modern patterns."""


FlowsDirectFunction = Callable[..., Awaitable[ConsolidatedFunctionResult]]
"""Type alias for "direct" functions with automatic metadata extraction.

"Direct" functions have their definition automatically extracted from the
function signature and docstring. This can be used in :data:`NodeConfig`
directly, in lieu of a :class:`FlowsFunctionSchema` or function definition
dict.

Expected shape:

.. code-block:: python

    async def f(flow_manager: FlowManager, **params) -> ConsolidatedFunctionResult:
        ...

where ``**params`` are any named parameters described by the function's
docstring.

This is defined as ``Callable[..., ...]`` rather than a Protocol because
Python's Protocol system cannot express "any concrete named-parameter list"
against ``**kwargs: Any`` — a function with named params like ``llm: str``
is not structurally compatible with a ``**kwargs: Any`` protocol signature.
Runtime validation of the expected shape happens in
:meth:`FlowsDirectFunctionWrapper.validate_function`.
"""


@dataclass
class FlowsFunctionSchema:
    """Function schema with Flows-specific properties.

    This class extends a standard function schema with the Flows-specific
    ``handler`` that runs when the function is called, plus its call options.

    Parameters:
        name: Name of the function.
        description: Description of the function.
        properties: Dictionary defining parameter types and descriptions.
        required: List of required parameter names.
        handler: Function handler to process the function call.
        cancel_on_interruption: Whether to cancel this function call when an
            interruption occurs. Defaults to False.
        timeout_secs: Optional per-tool timeout in seconds, overriding the global
            ``function_call_timeout_secs``. Defaults to None (use global timeout).
    """

    name: str
    description: str
    properties: dict[str, Any]
    required: list[str]
    handler: FunctionHandler
    cancel_on_interruption: bool = False
    timeout_secs: float | None = None

    def to_function_schema(self) -> FunctionSchema:
        """Convert to a standard FunctionSchema for use with LLMs.

        Returns:
            FunctionSchema without flow-specific fields.
        """
        return FunctionSchema(
            name=self.name,
            description=self.description,
            properties=self.properties,
            required=self.required,
        )


def flows_tool_options(
    *, cancel_on_interruption: bool = False, timeout_secs: float | None = None
) -> Callable[[Callable], Callable]:
    """Configure a Flows direct function's call options.

    This decorator is optional; use it to override the defaults for a Flows
    direct function (an async function whose first parameter is ``flow_manager``).

    Args:
        cancel_on_interruption: Whether to cancel the function call when the user
            interrupts. Defaults to False.
        timeout_secs: Optional per-tool timeout in seconds, overriding the global
            ``function_call_timeout_secs``. Defaults to None (use global timeout).

    Returns:
        A decorator that attaches the metadata to the function.

    Example::

        @flows_tool_options(cancel_on_interruption=False, timeout_secs=30)
        async def long_running_task(flow_manager: FlowManager, query: str):
            '''Perform a long-running task that should not be cancelled on interruption.'''
            # ... implementation
            return {"status": "complete"}, None
    """
    return tool_options(cancel_on_interruption=cancel_on_interruption, timeout_secs=timeout_secs)


@deprecated(
    "`flows_direct_function` is deprecated since 1.5.0 and will be removed in 2.0.0. "
    "Use `flows_tool_options` instead."
)
def flows_direct_function(
    *, cancel_on_interruption: bool = False, timeout_secs: float | None = None
) -> Callable[[Callable], Callable]:
    """Configure a Flows direct function's call options.

    .. deprecated:: 1.5.0
        Renamed to :func:`flows_tool_options` to align with Pipecat's
        ``@tool_options`` and make clearer that it configures call options.
        Will be removed in 2.0.0.

    Args:
        cancel_on_interruption: Whether to cancel the function call when the user
            interrupts. Defaults to False.
        timeout_secs: Optional per-tool timeout in seconds, overriding the global
            ``function_call_timeout_secs``. Defaults to None (use global timeout).

    Returns:
        A decorator that attaches the metadata to the function.
    """
    return flows_tool_options(
        cancel_on_interruption=cancel_on_interruption, timeout_secs=timeout_secs
    )


class FlowsDirectFunctionWrapper(BaseDirectFunctionWrapper):
    """Wrapper around a FlowsDirectFunction for metadata extraction and invocation.

    The wrapper:

    - extracts metadata from the function signature and docstring
    - generates a corresponding FunctionSchema
    - helps with function invocation
    """

    @classmethod
    def special_first_param_name(cls) -> str:
        """Get the special first parameter name for Flows direct functions.

        Returns:
            The string "flow_manager" which is expected as the first parameter.
        """
        return "flow_manager"

    @classmethod
    def validate_function(cls, function: Callable) -> None:
        """Validate the function signature and docstring.

        Args:
            function: The function to validate.

        Raises:
            InvalidFunctionError: If the function does not meet the requirements.
        """
        try:
            super().validate_function(function)
        except Exception as e:
            raise InvalidFunctionError(str(e)) from e

    def _initialize_metadata(self):
        """Initialize metadata from function signature, docstring, and decorator."""
        super()._initialize_metadata()
        # Read the call options attached by @flows_tool_options (built on Pipecat's
        # @tool_options, which stores them under the _pipecat_* attributes). Fall
        # back to Flows' defaults when the function is undecorated.
        self.cancel_on_interruption = getattr(
            self.function, "_pipecat_cancel_on_interruption", False
        )
        self.timeout_secs = getattr(self.function, "_pipecat_timeout_secs", None)

    async def invoke(self, args: Mapping[str, Any], flow_manager: "FlowManager"):
        """Invoke the wrapped function with the provided arguments.

        Args:
            args: Arguments to pass to the function.
            flow_manager: FlowManager instance for function execution context.

        Returns:
            The result of the function call.
        """
        return await self.function(flow_manager=flow_manager, **args)


def get_or_generate_node_name(node_config: NodeConfig) -> str:
    """Get the node name from configuration or generate a UUID if not set.

    Args:
        node_config: Node configuration dictionary.

    Returns:
        Node name from config or generated UUID string.
    """
    return node_config.get("name", str(uuid.uuid4()))
