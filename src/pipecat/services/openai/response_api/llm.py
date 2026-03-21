"""OpenAI Responses API LLM service implementation."""

import asyncio
import base64
import json
from typing import Any, Dict, List, Mapping, Optional

from loguru import logger
from openai import (
    NOT_GIVEN,
    APITimeoutError,
    AsyncStream,
)
from openai.types.responses import (
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseCompletedEvent,
)
from pydantic import BaseModel, Field

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.frames.frames import (
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMContextFrame,
    LLMTextFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantAggregatorParams,
    LLMAssistantContextAggregator,
    LLMUserAggregatorParams,
    LLMUserContextAggregator,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.services.llm_service import FunctionCallFromLLM
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.utils.tracing.service_decorators import traced_llm
from dataclasses import dataclass


@dataclass
class OpenAIContextAggregatorPair:
    """Pair of OpenAI context aggregators for user and assistant messages.

    Parameters:
        _user: User context aggregator for processing user messages.
        _assistant: Assistant context aggregator for processing assistant messages.
    """

    _user: "OpenAIUserContextAggregator"
    _assistant: "OpenAIAssistantContextAggregator"

    def user(self) -> "OpenAIUserContextAggregator":
        """Get the user context aggregator.

        Returns:
            The user context aggregator instance.
        """
        return self._user

    def assistant(self) -> "OpenAIAssistantContextAggregator":
        """Get the assistant context aggregator.

        Returns:
            The assistant context aggregator instance.
        """
        return self._assistant


class OpenAIResponsesLLMService(BaseOpenAILLMService):
    """OpenAI Responses API LLM service implementation.

    This service uses the newer OpenAI Responses API (client.responses.create)
    instead of the Chat Completions API. The Responses API provides:
    - Stateful conversations with server-side storage (when store=True)
    - Event-based streaming with ResponseStreamEvent
    - Improved tool calling support
    - Better handling of multi-turn conversations

    This service consumes OpenAILLMContextFrame or LLMContextFrame frames,
    which contain a reference to an OpenAILLMContext or LLMContext object.
    """

    class InputParams(BaseModel):
        """Input parameters for OpenAI Responses model configuration.

        Parameters:
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens in response.
            store: Whether to store conversation server-side (enables stateful mode).
            metadata: Additional metadata for the response.
            reasoning_effort: Reasoning effort level for o-series models (minimal, low, medium, high).
            extra: Additional model-specific parameters.
        """

        temperature: Optional[float] = Field(default_factory=lambda: NOT_GIVEN, ge=0.0, le=2.0)
        max_tokens: Optional[int] = Field(default_factory=lambda: NOT_GIVEN, ge=1)
        store: Optional[bool] = Field(default=False)
        metadata: Optional[Dict[str, str]] = Field(default_factory=dict)
        reasoning_effort: Optional[str] = Field(default_factory=lambda: NOT_GIVEN)
        extra: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def __init__(
        self,
        *,
        model: str,
        api_key=None,
        base_url=None,
        organization=None,
        project=None,
        default_headers: Optional[Mapping[str, str]] = None,
        params: Optional[InputParams] = None,
        retry_timeout_secs: Optional[float] = 5.0,
        retry_on_timeout: Optional[bool] = False,
        **kwargs,
    ):
        """Initialize the OpenAI Responses API LLM service.

        Args:
            model: The OpenAI model name to use (e.g., "gpt-4o", "gpt-4.1").
            api_key: OpenAI API key. If None, uses environment variable.
            base_url: Custom base URL for OpenAI API. If None, uses default.
            organization: OpenAI organization ID.
            project: OpenAI project ID.
            default_headers: Additional HTTP headers to include in requests.
            params: Input parameters for model configuration and behavior.
            retry_timeout_secs: Request timeout in seconds. Defaults to 5.0 seconds.
            retry_on_timeout: Whether to retry the request once if it times out.
            **kwargs: Additional arguments passed to the parent BaseOpenAILLMService.
        """
        # Initialize parent with base parameters (will create the client)
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            project=project,
            default_headers=default_headers,
            params=BaseOpenAILLMService.InputParams(),  # Pass empty params to parent
            retry_timeout_secs=retry_timeout_secs,
            retry_on_timeout=retry_on_timeout,
            **kwargs,
        )

        # Override with Responses API specific settings
        params = params or OpenAIResponsesLLMService.InputParams()

        self._settings = {
            "temperature": params.temperature,
            "max_tokens": params.max_tokens,
            "store": params.store,
            "metadata": params.metadata if isinstance(params.metadata, dict) else {},
            "reasoning_effort": params.reasoning_effort,
            "extra": params.extra if isinstance(params.extra, dict) else {},
        }

        # Track response IDs for stateful conversations (when store=True)
        # Maps context id to last response_id
        self._response_ids: Dict[int, str] = {}
        # Track how many messages we've already sent for each context
        self._sent_message_counts: Dict[int, int] = {}
        # Track message content hash to detect RESET with same count
        self._message_content_hashes: Dict[int, str] = {}

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as OpenAI Responses service supports metrics generation.
        """
        return True

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> OpenAIContextAggregatorPair:
        """Create OpenAI-specific context aggregators.

        Creates a pair of context aggregators optimized for OpenAI's message format,
        including support for function calls, tool usage, and image handling.

        Args:
            context: The LLM context to create aggregators for.
            user_params: Parameters for user message aggregation.
            assistant_params: Parameters for assistant message aggregation.

        Returns:
            OpenAIContextAggregatorPair: A pair of context aggregators, one for
            the user and one for the assistant, encapsulated in an
            OpenAIContextAggregatorPair.
        """
        context.set_llm_adapter(self.get_llm_adapter())
        user = OpenAIUserContextAggregator(context, params=user_params)
        assistant = OpenAIAssistantContextAggregator(context, params=assistant_params)
        return OpenAIContextAggregatorPair(_user=user, _assistant=assistant)

    def _convert_messages_to_input(self, messages: List[Dict]) -> List[Dict]:
        """Convert chat completion messages format to Responses API input format.

        Args:
            messages: List of messages in chat completion format.

        Returns:
            List of input items in Responses API format.
        """
        input_items = []

        for idx, message in enumerate(messages):
            role = message.get("role")
            content = message.get("content", "")

            if role == "system":
                # System messages become input_text with system role
                input_items.append({
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": content}]
                })
            elif role == "user":
                # User messages
                if isinstance(content, str):
                    input_items.append({
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": content}]
                    })
                elif isinstance(content, list):
                    # Handle multimodal content (text + images)
                    formatted_content = []
                    for item in content:
                        if item.get("type") == "text":
                            formatted_content.append({"type": "input_text", "text": item.get("text", "")})
                        elif item.get("type") == "image_url":
                            # Extract base64 image data
                            image_url = item.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:image/jpeg;base64,"):
                                image_data = image_url.split(",", 1)[1]
                                formatted_content.append({
                                    "type": "input_image",
                                    "image": image_data
                                })
                    if formatted_content:
                        input_items.append({
                            "type": "message",
                            "role": "user",
                            "content": formatted_content
                        })
            elif role == "assistant":
                # Assistant messages use output_text
                if content:
                    input_items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content}]
                    })
                # Handle tool calls from assistant
                if "tool_calls" in message:
                    for tool_call in message["tool_calls"]:
                        if tool_call.get("type") == "function":
                            function_info = tool_call.get("function", {})
                            call_id = tool_call.get("id")
                            if call_id:
                                input_items.append({
                                    "type": "function_call",
                                    "call_id": call_id,  # Changed from "id" to "call_id"
                                    "name": function_info.get("name"),
                                    "arguments": function_info.get("arguments", "{}")
                                })
                            else:
                                logger.error(f"Tool call at message index {idx} missing id: {tool_call}")
            elif role == "tool":
                # Tool results - must include call_id
                tool_call_id = message.get("tool_call_id")
                if tool_call_id:
                    input_items.append({
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": content
                    })
                else:
                    logger.error(f"Tool message at index {idx} missing tool_call_id: {message}")

        return input_items

    def _convert_tools_to_responses_format(self, tools: List[Dict]) -> List[Dict]:
        """Convert Chat Completions tools format to Responses API format.

        Args:
            tools: List of tools in Chat Completions format.

        Returns:
            List of tools in Responses API format.
        """
        if not tools:
            return tools

        responses_tools = []
        for tool in tools:
            # Chat Completions format: {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
            # Responses API format: {"type": "function", "name": "...", "description": "...", "parameters": {...}}
            if tool.get("type") == "function":
                function_info = tool.get("function", {})
                responses_tool = {
                    "type": "function",
                    "name": function_info.get("name"),
                    "description": function_info.get("description", ""),
                    "parameters": function_info.get("parameters", {})
                }
                # Only add strict if it exists
                if "strict" in function_info:
                    responses_tool["strict"] = function_info["strict"]
                responses_tools.append(responses_tool)
            else:
                # If it's already in the correct format or unknown type, pass through
                responses_tools.append(tool)

        return responses_tools

    async def get_responses(
        self, params_from_context: OpenAILLMInvocationParams, context: OpenAILLMContext | LLMContext = None
    ) -> AsyncStream[ResponseStreamEvent]:
        """Get streaming responses from OpenAI Responses API with optional timeout and retry.

        Args:
            params_from_context: Parameters, derived from the LLM context, to
                use for the response. Contains messages, tools, and tool choice.
            context: Optional context for tracking conversation state.

        Returns:
            Async stream of response events.
        """
        params = self.build_response_params(params_from_context, context)

        if self._retry_on_timeout:
            try:
                events = await asyncio.wait_for(
                    self._client.responses.create(**params), timeout=self._retry_timeout_secs
                )
                return events
            except (APITimeoutError, asyncio.TimeoutError):
                # Retry, this time without a timeout so we get a response
                logger.debug(f"{self}: Retrying response creation due to timeout")
                events = await self._client.responses.create(**params)
                return events
        else:
            events = await self._client.responses.create(**params)
            return events

    def build_response_params(self, params_from_context: OpenAILLMInvocationParams, context: OpenAILLMContext | LLMContext = None) -> dict:
        """Build parameters for Responses API request.

        Args:
            params_from_context: Parameters, derived from the LLM context, to
                use for the response. Contains messages, tools, and tool choice.
            context: Optional context for tracking conversation state.

        Returns:
            Dictionary of parameters for the Responses API request.
        """
        messages = params_from_context.get("messages", [])
        tools = params_from_context.get("tools")
        tool_choice = params_from_context.get("tool_choice")

        # Get context ID for tracking (use id() for unique identification)
        context_id = id(context) if context else None

        # Check if we have a previous response_id for incremental updates
        previous_response_id = None
        messages_to_send = messages

        if context_id and self._settings["store"] and context_id in self._response_ids:
            # We have a previous response - check if context was reset
            previous_response_id = self._response_ids[context_id]
            sent_count = self._sent_message_counts.get(context_id, 0)

            # Calculate content hash for reset detection
            # Use first 3 messages (usually system + initial exchange) for faster hashing
            messages_sample = messages[:min(3, len(messages))]
            current_hash = hash(json.dumps(messages_sample, sort_keys=True))
            previous_hash = self._message_content_hashes.get(context_id)

            # Detect context reset through multiple signals:
            # 1. Message count decreased (definite reset)
            # 2. Message count same but content changed (RESET with same count edge case)
            if len(messages) < sent_count:
                # Definite reset - message count decreased
                logger.debug(f"Context reset detected (count): {len(messages)} messages < {sent_count} previously sent. Clearing tracking.")
                del self._response_ids[context_id]
                del self._sent_message_counts[context_id]
                if context_id in self._message_content_hashes:
                    del self._message_content_hashes[context_id]
                previous_response_id = None
                messages_to_send = messages
            elif len(messages) == sent_count and previous_hash is not None and current_hash != previous_hash:
                # Edge case: same count but content changed (RESET with same message count)
                logger.debug(f"Context reset detected (content): {len(messages)} messages same count but content changed. Clearing tracking.")
                del self._response_ids[context_id]
                del self._sent_message_counts[context_id]
                if context_id in self._message_content_hashes:
                    del self._message_content_hashes[context_id]
                previous_response_id = None
                messages_to_send = messages
            elif sent_count < len(messages):
                # Incremental mode: send only new messages
                messages_to_send = messages[sent_count:]
                logger.debug(f"Incremental mode: sending {len(messages_to_send)} new messages (out of {len(messages)} total)")
            else:
                # No new messages (same count, same content)
                messages_to_send = []
                logger.debug("Incremental mode: no new messages to send")

        # Convert messages to input format
        input_items = self._convert_messages_to_input(messages_to_send)

        # Log the input items for debugging
        logger.debug(f"Converting {len(messages_to_send)} messages to {len(input_items)} input items")
        for idx, item in enumerate(input_items):
            if item.get("type") == "function_call_output" and not item.get("call_id"):
                logger.error(f"Input item {idx} is function_call_output but missing call_id: {item}")

        params = {
            "model": self.model_name,
            "input": input_items,
            "stream": True,
        }

        # Add previous_response_id for incremental updates
        if previous_response_id:
            params["previous_response_id"] = previous_response_id
            logger.debug(f"Using previous_response_id: {previous_response_id}")

        # Add optional parameters
        if self._settings["temperature"] is not NOT_GIVEN:
            params["temperature"] = self._settings["temperature"]

        if self._settings["max_tokens"] is not NOT_GIVEN:
            params["max_output_tokens"] = self._settings["max_tokens"]

        if self._settings["store"]:
            params["store"] = True

        if self._settings["metadata"]:
            params["metadata"] = self._settings["metadata"]

        if self._settings["reasoning_effort"] is not NOT_GIVEN:
            params["reasoning_effort"] = self._settings["reasoning_effort"]

        # Add tools if present - convert to Responses API format
        if tools and tools is not NOT_GIVEN:
            params["tools"] = self._convert_tools_to_responses_format(tools)

        if tool_choice and tool_choice is not NOT_GIVEN:
            params["tool_choice"] = tool_choice

        # Add any extra parameters
        params.update(self._settings["extra"])

        return params

    async def run_inference(self, context: LLMContext | OpenAILLMContext) -> Optional[str]:
        """Run a one-shot, out-of-band (i.e. out-of-pipeline) inference with the given LLM context.

        Args:
            context: The LLM context containing conversation history.

        Returns:
            The LLM's response as a string, or None if no response is generated.
        """
        if isinstance(context, LLMContext):
            adapter = self.get_llm_adapter()
            params: OpenAILLMInvocationParams = adapter.get_llm_invocation_params(context)
            messages = params["messages"]
        else:
            messages = context.messages

        # Convert messages to input format
        input_items = self._convert_messages_to_input(messages)

        # Create non-streaming response
        response = await self._client.responses.create(
            model=self.model_name,
            input=input_items,
            stream=False,
        )

        # Extract text from output
        if response.output:
            for output_item in response.output:
                if output_item.type == "message":
                    for content_item in output_item.content:
                        if content_item.type == "text":
                            return content_item.text

        return None

    async def _stream_responses_specific_context(
        self, context: OpenAILLMContext
    ) -> AsyncStream[ResponseStreamEvent]:
        """Stream responses using OpenAI-specific context.

        Args:
            context: The OpenAI-specific LLM context.

        Returns:
            Async stream of response events.
        """
        logger.debug(
            f"{self}: Generating response from LLM-specific context {context.get_messages_for_logging()}"
        )

        messages: List[Dict] = context.get_messages()

        # base64 encode any images
        for message in messages:
            if message.get("mime_type") == "image/jpeg":
                # Avoid .getvalue() which makes a full copy of BytesIO
                raw_bytes = message["data"].read()
                encoded_image = base64.b64encode(raw_bytes).decode("utf-8")
                text = message.get("content", "")
                message["content"] = [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ]
                # Explicit cleanup
                del message["data"]
                del message["mime_type"]

        params = OpenAILLMInvocationParams(
            messages=messages, tools=context.tools, tool_choice=context.tool_choice
        )
        events = await self.get_responses(params, context)

        return events

    async def _stream_responses_universal_context(
        self, context: LLMContext
    ) -> AsyncStream[ResponseStreamEvent]:
        """Stream responses using universal LLM context.

        Args:
            context: The universal LLM context.

        Returns:
            Async stream of response events.
        """
        adapter = self.get_llm_adapter()
        logger.debug(
            f"{self}: Generating response from universal context {adapter.get_messages_for_logging(context)}"
        )

        params: OpenAILLMInvocationParams = adapter.get_llm_invocation_params(context)
        events = await self.get_responses(params, context)

        return events

    @traced_llm
    async def _process_context(self, context: OpenAILLMContext | LLMContext):
        """Process the LLM context and stream responses.

        Args:
            context: The LLM context to process.
        """
        functions_list = []
        arguments_list = []
        tool_id_list = []
        function_name = ""
        arguments = ""
        tool_call_id = ""

        await self.start_ttfb_metrics()

        # Generate responses using either OpenAILLMContext or universal LLMContext
        event_stream = await (
            self._stream_responses_specific_context(context)
            if isinstance(context, OpenAILLMContext)
            else self._stream_responses_universal_context(context)
        )

        async for event in event_stream:
            # Handle response completion event with usage metrics
            if isinstance(event, ResponseCompletedEvent):
                # Capture response_id for incremental updates (when store=True)
                if hasattr(event, 'response') and event.response:
                    response_id = getattr(event.response, 'id', None)
                    if response_id and self._settings["store"]:
                        context_id = id(context)
                        self._response_ids[context_id] = response_id

                        # Get current messages and update tracking
                        if isinstance(context, OpenAILLMContext):
                            messages = context.messages
                            self._sent_message_counts[context_id] = len(messages)
                        else:
                            # For LLMContext, get messages through adapter
                            adapter = self.get_llm_adapter()
                            params = adapter.get_llm_invocation_params(context)
                            messages = params.get("messages", [])
                            self._sent_message_counts[context_id] = len(messages)

                        # Store content hash for bulletproof reset detection
                        # Use first 3 messages (usually system + initial exchange) for faster hashing
                        messages_sample = messages[:min(3, len(messages))]
                        self._message_content_hashes[context_id] = hash(json.dumps(messages_sample, sort_keys=True))

                        logger.debug(f"Stored response_id {response_id} for context {context_id}, message count: {self._sent_message_counts[context_id]}")

                # Usage is on the response object, not the event itself
                if hasattr(event, 'response') and event.response and hasattr(event.response, 'usage') and event.response.usage:
                    usage = event.response.usage
                    cached_tokens = None
                    if hasattr(usage, 'input_token_details') and usage.input_token_details:
                        cached_tokens = usage.input_token_details.cached_tokens

                    tokens = LLMTokenUsage(
                        prompt_tokens=usage.input_tokens,
                        completion_tokens=usage.output_tokens,
                        total_tokens=usage.input_tokens + usage.output_tokens,
                        cache_read_input_tokens=cached_tokens,
                    )
                    await self.start_llm_usage_metrics(tokens)

            await self.stop_ttfb_metrics()

            # Handle text delta events
            if isinstance(event, ResponseTextDeltaEvent):
                if event.delta:
                    await self.push_frame(LLMTextFrame(event.delta))

            # Handle function call argument deltas
            elif isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
                if event.delta:
                    arguments += event.delta

            # Handle function call argument completion
            elif isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
                # Arguments are complete, but name comes from OutputItemAdded event
                # Just store the call_id for now
                if hasattr(event, 'call_id'):
                    tool_call_id = event.call_id

            # Handle output item added (new function call started)
            elif isinstance(event, ResponseOutputItemAddedEvent):
                if event.item.type == "function_call":
                    # Save previous function if exists
                    if function_name:
                        functions_list.append(function_name)
                        arguments_list.append(arguments)
                        tool_id_list.append(tool_call_id)

                    # Get function name from the item
                    function_name = event.item.name if hasattr(event.item, 'name') else ""
                    tool_call_id = event.item.call_id if hasattr(event.item, 'call_id') else ""
                    # Reset arguments for new function
                    arguments = ""

            # Handle output item completion
            elif isinstance(event, ResponseOutputItemDoneEvent):
                if event.item.type == "function_call":
                    # Function call is complete, save it
                    if function_name:
                        functions_list.append(function_name)
                        arguments_list.append(arguments)
                        tool_id_list.append(tool_call_id)

        # Process any collected function calls
        if function_name and arguments:
            # Add the last function if not already added
            if not functions_list or functions_list[-1] != function_name:
                functions_list.append(function_name)
                arguments_list.append(arguments)
                tool_id_list.append(tool_call_id)

        if functions_list:
            function_calls = []

            for func_name, args, tool_id in zip(functions_list, arguments_list, tool_id_list):
                try:
                    args_dict = json.loads(args) if isinstance(args, str) else args
                    function_calls.append(
                        FunctionCallFromLLM(
                            function_name=func_name,
                            tool_call_id=tool_id,
                            arguments=args_dict,
                            context=context,
                        )
                    )
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse function arguments: {e}")
                    logger.error(f"Arguments string: {args}")

            if function_calls:
                await self.run_function_calls(function_calls)

    async def _update_settings(self, settings: Dict[str, Any]):
        """Update service settings dynamically.

        Args:
            settings: Dictionary of settings to update.
        """
        for key, value in settings.items():
            if key in self._settings:
                self._settings[key] = value
                logger.debug(f"Updated setting {key} to {value}")


class OpenAIUserContextAggregator(LLMUserContextAggregator):
    """OpenAI-specific user context aggregator.

    Handles aggregation of user messages for OpenAI LLM services.
    Inherits all functionality from the base LLMUserContextAggregator.
    """

    pass


class OpenAIAssistantContextAggregator(LLMAssistantContextAggregator):
    """OpenAI-specific assistant context aggregator.

    Handles aggregation of assistant messages for OpenAI LLM services,
    with specialized support for OpenAI's function calling format,
    tool usage tracking, and image message handling.
    """

    async def handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame):
        """Handle a function call in progress.

        Adds the function call to the context with an IN_PROGRESS status
        to track ongoing function execution.

        Args:
            frame: Frame containing function call progress information.
        """
        self._context.add_message(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": frame.tool_call_id,
                        "function": {
                            "name": frame.function_name,
                            "arguments": json.dumps(frame.arguments),
                        },
                        "type": "function",
                    }
                ],
            }
        )
        self._context.add_message(
            {
                "role": "tool",
                "content": "IN_PROGRESS",
                "tool_call_id": frame.tool_call_id,
            }
        )

    async def handle_function_call_result(self, frame: FunctionCallResultFrame):
        """Handle the result of a function call.

        Updates the context with the function call result, replacing any
        previous IN_PROGRESS status.

        Args:
            frame: Frame containing the function call result.
        """
        if frame.result:
            result = json.dumps(frame.result)
            await self._update_function_call_result(frame.function_name, frame.tool_call_id, result)
        else:
            await self._update_function_call_result(
                frame.function_name, frame.tool_call_id, "COMPLETED"
            )

    async def handle_function_call_cancel(self, frame: FunctionCallCancelFrame):
        """Handle a cancelled function call.

        Updates the context to mark the function call as cancelled.

        Args:
            frame: Frame containing the function call cancellation information.
        """
        await self._update_function_call_result(
            frame.function_name, frame.tool_call_id, "CANCELLED"
        )

    async def _update_function_call_result(
        self, function_name: str, tool_call_id: str, result: Any
    ):
        """Update a function call result in the context.

        Args:
            function_name: Name of the function.
            tool_call_id: Unique identifier for the tool call.
            result: Result of the function call.
        """
        for message in self._context.messages:
            if (
                message["role"] == "tool"
                and message["tool_call_id"]
                and message["tool_call_id"] == tool_call_id
            ):
                message["content"] = result
