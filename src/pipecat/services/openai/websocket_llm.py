#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI WebSocket LLM service using the Responses API."""

import json
from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.adapters.services.open_ai_websocket_adapter import (
    OpenAIWebSocketLLMAdapter,
)
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    FunctionCallFromLLM,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMSetToolsFrame,
    StartFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantAggregatorParams,
    LLMUserAggregatorParams,
)
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings
from pipecat.services.websocket_service import WebsocketService

try:
    from websockets.asyncio.client import connect as websocket_connect
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use OpenAI WebSocket LLM, you need to `pip install pipecat-ai[openai]`."
    )
    raise Exception(f"Missing module: {e}")


class InputParams(BaseModel):
    """Configuration parameters for OpenAI WebSocket LLM requests.

    Parameters:
        temperature: Sampling temperature for response generation.
        max_tokens: Maximum number of tokens to generate.
        top_p: Nucleus sampling probability.
        frequency_penalty: Penalty for token frequency.
        presence_penalty: Penalty for token presence.
    """

    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None


@dataclass
class OpenAIWebSocketLLMSettings(LLMSettings):
    """Settings for OpenAI WebSocket LLM service.

    Parameters:
        store: Whether to store responses on the server.
        use_previous_response_id: Whether to use previous_response_id for context continuity.
    """

    store: bool = False
    use_previous_response_id: bool = True


class OpenAIWebSocketLLMService(LLMService, WebsocketService):
    """OpenAI WebSocket LLM service using the Responses API.

    Connects to ``wss://api.openai.com/v1/responses`` (or a custom base_url)
    and sends ``response.create`` events over WebSocket. Receives streaming
    response events including text deltas, function calls, and usage metrics.

    Supports automatic reconnection via ``WebsocketService`` and tracks
    ``previous_response_id`` for context continuity across requests.

    Event handlers available:

    - on_connection_error: Called when a WebSocket connection error occurs.

    Example::

        @service.event_handler("on_connection_error")
        async def on_connection_error(service, error):
            logger.error(f"Connection error: {error}")
    """

    _settings: OpenAIWebSocketLLMSettings

    adapter_class = OpenAIWebSocketLLMAdapter

    def __init__(
        self,
        *,
        model: str,
        api_key: str = "",
        base_url: str = "wss://api.openai.com/v1/responses",
        store: bool = False,
        use_previous_response_id: bool = True,
        params: InputParams = InputParams(),
        reconnect_on_error: bool = True,
        **kwargs,
    ):
        """Initialize the OpenAI WebSocket LLM service.

        Args:
            model: The model identifier to use for responses.
            api_key: OpenAI API key for authentication. Can be empty for local services.
            base_url: WebSocket URL for the Responses API.
                Defaults to "wss://api.openai.com/v1/responses".
            store: Whether to store responses on the server. Defaults to False.
            use_previous_response_id: Whether to chain responses via previous_response_id.
                Defaults to True.
            params: Input parameters for response generation.
            reconnect_on_error: Whether to automatically reconnect on errors.
            **kwargs: Additional arguments passed to parent classes.
        """
        LLMService.__init__(
            self,
            settings=OpenAIWebSocketLLMSettings(
                model=model,
                temperature=params.temperature,
                max_tokens=params.max_tokens,
                top_p=params.top_p,
                top_k=None,
                frequency_penalty=params.frequency_penalty,
                presence_penalty=params.presence_penalty,
                seed=None,
                filter_incomplete_user_turns=False,
                user_turn_completion_config=None,
                store=store,
                use_previous_response_id=use_previous_response_id,
            ),
            **kwargs,
        )
        WebsocketService.__init__(self, reconnect_on_error=reconnect_on_error)

        self._api_key = api_key
        self._base_url = base_url
        self._receive_task = None
        self._context: Optional[LLMContext] = None
        self._previous_response_id: Optional[str] = None

        # Function call tracking: keyed by call_id to handle parallel calls
        self._pending_function_calls: dict[str, dict] = {}

        self._register_event_handler("on_connection_error")

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate usage metrics.

        Returns:
            True, as this service supports metrics generation.
        """
        return True

    # -----------------------------------------------------------------------
    # WebsocketService abstract method implementations
    # -----------------------------------------------------------------------

    async def _connect_websocket(self):
        """Establish the WebSocket connection to the Responses API."""
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        self._websocket = await websocket_connect(
            uri=self._base_url,
            additional_headers=headers,
        )

    async def _disconnect_websocket(self):
        """Close the WebSocket connection."""
        if self._websocket:
            await self._websocket.close()
            self._websocket = None

    async def _receive_messages(self):
        """Receive and dispatch WebSocket messages."""
        async for message in self._websocket:
            await self._on_message(message)

    # -----------------------------------------------------------------------
    # Connection lifecycle
    # -----------------------------------------------------------------------

    async def _connect(self):
        """Connect to the WebSocket and start the receive task."""
        await super()._connect()
        try:
            await self._connect_websocket()
            self._receive_task = self.create_task(
                self._receive_task_handler(self._report_error),
                name="ws_llm_receive",
            )
        except Exception as e:
            await self.push_error(error_msg=f"Error connecting: {e}", exception=e)
            self._websocket = None

    async def _disconnect(self):
        """Disconnect from the WebSocket and cancel the receive task."""
        await super()._disconnect()
        try:
            await self.stop_all_metrics()
            await self._disconnect_websocket()
            if self._receive_task:
                await self.cancel_task(self._receive_task, timeout=1.0)
                self._receive_task = None
        except Exception as e:
            await self.push_error(error_msg=f"Error disconnecting: {e}", exception=e)

    async def _report_error(self, error):
        """Report a WebSocket connection error.

        Args:
            error: The error frame to report.
        """
        await self._call_event_handler("on_connection_error", error.error)
        await self.push_error(error_msg=str(error.error))

    # -----------------------------------------------------------------------
    # Frame processing
    # -----------------------------------------------------------------------

    async def start(self, frame: StartFrame):
        """Start the service and establish the WebSocket connection.

        Args:
            frame: The start frame triggering service initialization.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service and close the WebSocket connection.

        Args:
            frame: The end frame triggering service shutdown.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and close the WebSocket connection.

        Args:
            frame: The cancel frame triggering service cancellation.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames from the pipeline.

        Handles LLMContextFrame to trigger response generation,
        InterruptionFrame to cancel ongoing responses, and
        LLMSetToolsFrame to update available tools.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            self._context = frame.context
            await self._send_response_create(frame.context)
        elif isinstance(frame, InterruptionFrame):
            self._pending_function_calls.clear()
            await self.stop_all_metrics()
        elif isinstance(frame, LLMSetToolsFrame):
            # Tools will be included in the next response.create request
            pass

        await self.push_frame(frame, direction)

    # -----------------------------------------------------------------------
    # Sending requests
    # -----------------------------------------------------------------------

    async def _send_response_create(self, context: LLMContext):
        """Build and send a response.create event over WebSocket.

        Converts the LLM context via the adapter and sends a response.create
        message including model, input, tools, and optional previous_response_id.

        Args:
            context: The LLM context to use for generating the response.
        """
        adapter: OpenAIWebSocketLLMAdapter = self.get_llm_adapter()
        invocation_params = adapter.get_llm_invocation_params(context)

        logger.debug(
            f"{self} Sending response.create with messages: "
            f"{adapter.get_messages_for_logging(context)}"
        )

        await self.push_frame(LLMFullResponseStartFrame())
        await self.start_processing_metrics()
        await self.start_ttfb_metrics()

        request: dict[str, Any] = {
            "type": "response.create",
            "response": {
                "modalities": ["text"],
                "model": self._settings.model,
                "input": invocation_params["input"],
            },
        }

        # Add optional parameters
        response = request["response"]

        if invocation_params.get("system_instruction"):
            response["instructions"] = invocation_params["system_instruction"]

        if invocation_params.get("tools"):
            response["tools"] = invocation_params["tools"]

        if self._settings.store:
            response["store"] = True

        if self._settings.use_previous_response_id and self._previous_response_id:
            request["response"]["previous_response_id"] = self._previous_response_id

        # Add generation parameters
        if self._settings.temperature is not None:
            response["temperature"] = self._settings.temperature
        if self._settings.max_tokens is not None:
            response["max_output_tokens"] = self._settings.max_tokens
        if self._settings.top_p is not None:
            response["top_p"] = self._settings.top_p
        if self._settings.frequency_penalty is not None:
            response["frequency_penalty"] = self._settings.frequency_penalty
        if self._settings.presence_penalty is not None:
            response["presence_penalty"] = self._settings.presence_penalty

        message = json.dumps(request)
        try:
            await self.send_with_retry(message, self._report_error)
        except Exception as e:
            await self.push_error(error_msg=f"Error sending response.create: {e}", exception=e)

    # -----------------------------------------------------------------------
    # Message handling
    # -----------------------------------------------------------------------

    async def _on_message(self, message: str):
        """Dispatch an incoming WebSocket message by event type.

        Args:
            message: The raw JSON message string from the WebSocket.
        """
        try:
            data = json.loads(message)
        except json.JSONDecodeError as e:
            logger.error(f"{self} Failed to parse message: {e}")
            return

        event_type = data.get("type", "")

        if event_type == "response.created":
            await self._handle_response_created(data)
        elif event_type == "response.output_text.delta":
            await self._handle_text_delta(data)
        elif event_type == "response.output_item.added":
            await self._handle_output_item_added(data)
        elif event_type == "response.function_call_arguments.delta":
            await self._handle_function_call_arguments_delta(data)
        elif event_type == "response.function_call_arguments.done":
            await self._handle_function_call_arguments_done(data)
        elif event_type == "response.done":
            await self._handle_response_done(data)
        elif event_type == "error":
            await self._handle_error(data)

    async def _handle_response_created(self, data: dict):
        """Handle response.created event - stop TTFB metrics.

        Args:
            data: The parsed event data.
        """
        await self.stop_ttfb_metrics()

    async def _handle_text_delta(self, data: dict):
        """Handle response.output_text.delta event - push text downstream.

        Args:
            data: The parsed event data containing the text delta.
        """
        delta = data.get("delta", "")
        if delta:
            await self._push_llm_text(delta)

    async def _handle_output_item_added(self, data: dict):
        """Handle response.output_item.added event - track function calls.

        Args:
            data: The parsed event data containing the output item.
        """
        item = data.get("item", {})
        if item.get("type") == "function_call":
            call_id = item.get("call_id", "")
            if call_id:
                self._pending_function_calls[call_id] = {
                    "name": item.get("name", ""),
                    "arguments": "",
                }

    async def _handle_function_call_arguments_delta(self, data: dict):
        """Handle response.function_call_arguments.delta - accumulate arguments.

        Args:
            data: The parsed event data containing the argument delta.
        """
        call_id = data.get("call_id", "")
        delta = data.get("delta", "")
        if call_id in self._pending_function_calls:
            self._pending_function_calls[call_id]["arguments"] += delta

    async def _handle_function_call_arguments_done(self, data: dict):
        """Handle response.function_call_arguments.done - execute the function call.

        Args:
            data: The parsed event data with complete arguments.
        """
        call_id = data.get("call_id", "")
        fc = self._pending_function_calls.pop(call_id, None)
        if not fc:
            logger.warning(f"{self} No pending function call for call_id: {call_id}")
            return

        arguments_str = data.get("arguments", fc["arguments"])
        try:
            args = json.loads(arguments_str)
        except json.JSONDecodeError:
            args = {}
            logger.error(f"{self} Failed to parse function call arguments: {arguments_str}")

        function_calls = [
            FunctionCallFromLLM(
                context=self._context,
                tool_call_id=call_id,
                function_name=fc["name"],
                arguments=args,
            )
        ]

        await self.run_function_calls(function_calls)

    async def _handle_response_done(self, data: dict):
        """Handle response.done event - store response ID, push metrics and end frame.

        Args:
            data: The parsed event data containing the complete response.
        """
        response = data.get("response", {})

        # Store response ID for context continuity
        response_id = response.get("id")
        if response_id and self._settings.use_previous_response_id:
            self._previous_response_id = response_id

        # Push usage metrics
        usage = response.get("usage", {})
        if usage:
            tokens = LLMTokenUsage(
                prompt_tokens=usage.get("input_tokens", 0),
                completion_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            )
            await self.start_llm_usage_metrics(tokens)

        await self.stop_processing_metrics()
        await self.push_frame(LLMFullResponseEndFrame())

        # Handle errors in the response
        if response.get("status") == "failed":
            status_details = response.get("status_details", {})
            error_info = status_details.get("error", {})
            error_msg = error_info.get("message", "Unknown error")
            await self.push_error(error_msg=f"Response failed: {error_msg}")

    async def _handle_error(self, data: dict):
        """Handle error events from the WebSocket.

        Handles previous_response_not_found by clearing previous_response_id
        and retrying. Other errors are pushed upstream.

        Args:
            data: The parsed error event data.
        """
        error = data.get("error", {})
        error_code = error.get("code", "")
        error_message = error.get("message", "Unknown error")

        if error_code == "previous_response_not_found":
            logger.warning(f"{self} previous_response_id not found, clearing and retrying")
            self._previous_response_id = None
            # Retry with context if available
            if self._context:
                await self._send_response_create(self._context)
            return

        await self.push_error(error_msg=f"WebSocket error [{error_code}]: {error_message}")

    # -----------------------------------------------------------------------
    # Context aggregator (backward compatibility)
    # -----------------------------------------------------------------------

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> LLMContextAggregatorPair:
        """Create a context aggregator pair for backward compatibility.

        New code should use ``LLMContext`` and ``LLMContextAggregatorPair`` directly.

        Args:
            context: The OpenAI LLM context.
            user_params: User aggregator parameters.
            assistant_params: Assistant aggregator parameters.

        Returns:
            A pair of context aggregators for user and assistant.

        .. deprecated:: 0.0.99
            Use ``LLMContext`` and ``LLMContextAggregatorPair`` directly instead.
        """
        super().create_context_aggregator(context)
        context = LLMContext.from_openai_context(context)
        return LLMContextAggregatorPair(
            context, user_params=user_params, assistant_params=assistant_params
        )
