#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Responses API LLM service implementations (WebSocket and HTTP)."""

import hashlib
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import httpx
from loguru import logger
from openai import NOT_GIVEN, AsyncOpenAI, AsyncStream, DefaultAsyncHttpxClient
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)

from pipecat.adapters.services.open_ai_responses_adapter import (
    OpenAIResponsesLLMAdapter,
    OpenAIResponsesLLMInvocationParams,
)
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    StartFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.services.settings import NOT_GIVEN as _NOT_GIVEN
from pipecat.services.settings import LLMSettings, _NotGiven
from pipecat.utils.tracing.service_decorators import traced_llm

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.exceptions import ConnectionClosed
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use OpenAI, you need to `pip install pipecat-ai[openai]`.")
    raise Exception(f"Missing module: {e}")


# ---------------------------------------------------------------------------
# Private retry exception classes
# ---------------------------------------------------------------------------


class _RetryableError(Exception):
    """Base for errors that should trigger a retry in _process_context."""

    pass


class _PreviousResponseNotFoundError(_RetryableError):
    """Server could not find the previous response (connection-local cache miss)."""

    pass


class _ConnectionLimitReachedError(_RetryableError):
    """WebSocket connection hit the 60-minute server-side limit."""

    pass


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class OpenAIResponsesLLMSettings(LLMSettings):
    """Settings for OpenAI Responses API LLM services.

    Parameters:
        max_completion_tokens: Maximum completion tokens to generate.
    """

    max_completion_tokens: int | _NotGiven = field(default_factory=lambda: _NOT_GIVEN)


# ---------------------------------------------------------------------------
# Shared base class (private)
# ---------------------------------------------------------------------------


class _BaseOpenAIResponsesLLMService(LLMService):
    """Shared base for HTTP and WebSocket OpenAI Responses API services.

    Contains settings, adapter reference, HTTP client creation, parameter
    building, ``run_inference``, and metrics support. Subclasses implement
    ``process_frame`` and ``_process_context`` for their transport.
    """

    Settings = OpenAIResponsesLLMSettings
    _settings: Settings

    adapter_class = OpenAIResponsesLLMAdapter

    def __init__(
        self,
        *,
        api_key=None,
        base_url=None,
        organization=None,
        project=None,
        default_headers: Optional[Mapping[str, str]] = None,
        service_tier: Optional[str] = None,
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        """Initialize the OpenAI Responses API LLM service.

        Args:
            api_key: OpenAI API key. If None, uses environment variable.
            base_url: Custom base URL for OpenAI API. If None, uses default.
            organization: OpenAI organization ID.
            project: OpenAI project ID.
            default_headers: Additional HTTP headers to include in requests.
            service_tier: Service tier to use (e.g., "auto", "flex", "priority").
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to the parent LLMService.
        """
        default_settings = self.Settings(
            model="gpt-4.1",
            system_instruction=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            temperature=NOT_GIVEN,
            top_p=NOT_GIVEN,
            top_k=None,
            max_tokens=None,
            max_completion_tokens=NOT_GIVEN,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
            extra={},
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._service_tier = service_tier
        self._client = self._create_client(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            project=project,
            default_headers=default_headers,
        )

        if self._settings.system_instruction:
            logger.debug(f"{self}: Using system instruction: {self._settings.system_instruction}")

    def _create_client(
        self,
        api_key=None,
        base_url=None,
        organization=None,
        project=None,
        default_headers=None,
    ) -> AsyncOpenAI:
        """Create an AsyncOpenAI client instance.

        Args:
            api_key: OpenAI API key.
            base_url: Custom base URL for the API.
            organization: OpenAI organization ID.
            project: OpenAI project ID.
            default_headers: Additional HTTP headers.

        Returns:
            Configured AsyncOpenAI client instance.
        """
        return AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            project=project,
            http_client=DefaultAsyncHttpxClient(
                limits=httpx.Limits(
                    max_keepalive_connections=100, max_connections=1000, keepalive_expiry=None
                )
            ),
            default_headers=default_headers,
        )

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True

    def _build_response_params(self, invocation_params: OpenAIResponsesLLMInvocationParams) -> dict:
        """Build parameters for a Responses API call.

        Args:
            invocation_params: Parameters derived from the LLM context.

        Returns:
            Dictionary of parameters for the Responses API call.
        """
        params: Dict[str, Any] = {
            "model": self._settings.model,
            "stream": True,
            "store": False,
            "input": invocation_params["input"],
        }

        # instructions (set by the adapter when input is non-empty)
        if "instructions" in invocation_params:
            params["instructions"] = invocation_params["instructions"]

        # Optional parameters - only include if given
        if isinstance(self._settings.temperature, (int, float)):
            params["temperature"] = self._settings.temperature

        if isinstance(self._settings.top_p, (int, float)):
            params["top_p"] = self._settings.top_p

        if isinstance(self._settings.max_completion_tokens, int):
            params["max_output_tokens"] = self._settings.max_completion_tokens

        if self._service_tier is not None:
            params["service_tier"] = self._service_tier

        # Tools
        tools = invocation_params.get("tools")
        if tools is not None and not isinstance(tools, type(NOT_GIVEN)):
            params["tools"] = tools

        # Extra settings
        params.update(self._settings.extra)

        return params

    async def run_inference(
        self,
        context: LLMContext,
        max_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
    ) -> Optional[str]:
        """Run a one-shot, out-of-band inference with the given LLM context.

        Always uses the HTTP client regardless of transport variant.

        Args:
            context: The LLM context containing conversation history.
            max_tokens: Optional maximum number of tokens to generate.
            system_instruction: Optional system instruction for this inference.

        Returns:
            The LLM's response as a string, or None if no response is generated.
        """
        adapter: OpenAIResponsesLLMAdapter = self.get_llm_adapter()
        effective_instruction = system_instruction or self._settings.system_instruction
        invocation_params = adapter.get_llm_invocation_params(
            context, system_instruction=effective_instruction
        )

        params = self._build_response_params(invocation_params)

        # Override for non-streaming
        params["stream"] = False

        if max_tokens is not None:
            params["max_output_tokens"] = max_tokens

        response = await self._client.responses.create(**params)

        return response.output_text

    def _process_function_calls(
        self,
        context: LLMContext,
        function_calls: Dict[str, Dict[str, str]],
    ) -> List[FunctionCallFromLLM]:
        """Convert accumulated function call data into FunctionCallFromLLM list.

        Args:
            context: The LLM context for the current inference.
            function_calls: Map of item_id to {name, call_id, arguments}.

        Returns:
            List of parsed function call objects.
        """
        fc_list: List[FunctionCallFromLLM] = []
        for item_id, fc in function_calls.items():
            try:
                arguments = json.loads(fc["arguments"]) if fc["arguments"] else {}
            except json.JSONDecodeError:
                logger.warning(
                    f"{self}: Failed to parse function call arguments: {fc['arguments']}"
                )
                arguments = {}
            fc_list.append(
                FunctionCallFromLLM(
                    context=context,
                    tool_call_id=fc["call_id"],
                    function_name=fc["name"],
                    arguments=arguments,
                )
            )
        return fc_list


# ---------------------------------------------------------------------------
# WebSocket variant (default / recommended)
# ---------------------------------------------------------------------------


class OpenAIResponsesLLMService(_BaseOpenAIResponsesLLMService):
    """OpenAI Responses API LLM service using WebSocket transport.

    Maintains a persistent WebSocket connection to ``wss://api.openai.com/v1/responses``
    for lower-latency inference, especially beneficial for tool-call-heavy workflows.
    Automatically uses ``previous_response_id`` to send only incremental context when
    possible, and falls back to full context on reconnection or cache miss.

    This is the recommended variant for real-time / conversational use.

    Example::

        llm = OpenAIResponsesLLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            settings=OpenAIResponsesLLMService.Settings(
                model="gpt-4.1",
                system_instruction="You are a helpful assistant.",
            ),
        )
    """

    def __init__(
        self,
        *,
        ws_url: str = "wss://api.openai.com/v1/responses",
        **kwargs,
    ):
        """Initialize the WebSocket-based OpenAI Responses API LLM service.

        Args:
            ws_url: WebSocket endpoint URL.
                Defaults to ``wss://api.openai.com/v1/responses``.
            **kwargs: Additional arguments passed to the base class (api_key,
                base_url, organization, project, default_headers, service_tier,
                settings, etc.).
        """
        super().__init__(**kwargs)

        self._ws_url = ws_url
        self._websocket = None
        self._disconnecting = False

        # State for previous_response_id optimization
        self._previous_response_id: Optional[str] = None
        self._previous_input_hash: Optional[str] = None
        self._previous_input_length: Optional[int] = None
        self._previous_response_output: Optional[list] = None

    # -- lifecycle ------------------------------------------------------------

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

    # -- connection management ------------------------------------------------

    async def _connect(self):
        """Establish the WebSocket connection."""
        self._disconnecting = False
        try:
            if self._websocket:
                return
            self._websocket = await websocket_connect(
                uri=self._ws_url,
                additional_headers={
                    "Authorization": f"Bearer {self._api_key}",
                },
            )
        except Exception as e:
            await self.push_error(error_msg=f"Error connecting to WebSocket: {e}", exception=e)
            self._websocket = None

    async def _disconnect(self):
        """Close the WebSocket connection and clear state."""
        try:
            self._disconnecting = True
            await self.stop_all_metrics()
            if self._websocket:
                await self._websocket.close()
                self._websocket = None
            self._clear_previous_response_state()
            self._disconnecting = False
        except Exception as e:
            await self.push_error(error_msg=f"Error disconnecting from WebSocket: {e}", exception=e)

    async def _reconnect(self):
        """Reconnect to the WebSocket, clearing previous_response_id state."""
        await self._disconnect()
        await self._connect()

    async def _ensure_connected(self):
        """Ensure a WebSocket connection is available, reconnecting if needed.

        Raises:
            _RetryableError: If the connection could not be established.
        """
        if self._websocket is None:
            await self._connect()
        if self._websocket is None:
            raise _RetryableError("Failed to establish WebSocket connection")

    async def _ws_send(self, message: dict):
        """Send a JSON message over the WebSocket.

        Args:
            message: The message dict to serialize and send.
        """
        if self._disconnecting or not self._websocket:
            return
        await self._websocket.send(json.dumps(message))

    # -- previous_response_id optimization ------------------------------------

    @staticmethod
    def _hash_input_items(items: list) -> str:
        """Compute a deterministic hash of input items for comparison.

        Args:
            items: List of Responses API input items.

        Returns:
            Hex digest of the SHA-256 hash.
        """
        return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()

    def _apply_previous_response_optimization(self, params: dict, full_input: list) -> dict:
        """Try to use previous_response_id to send only new input items.

        If the prefix of ``full_input`` matches the stored hash from the
        previous inference call, only new items are sent along with
        ``previous_response_id``. Otherwise the full input is sent.

        Args:
            params: The response params dict (modified in place).
            full_input: The complete input items list from the adapter.

        Returns:
            The (possibly modified) params dict.
        """
        if (
            self._previous_response_id is not None
            and self._previous_input_length is not None
            and self._previous_input_hash is not None
            and len(full_input) > self._previous_input_length
        ):
            prefix = full_input[: self._previous_input_length]
            prefix_hash = self._hash_input_items(prefix)
            if prefix_hash == self._previous_input_hash:
                items_after_prefix = full_input[self._previous_input_length :]
                response_output = self._previous_response_output or []

                if self._starts_with_response_output(items_after_prefix, response_output):
                    # The server already knows its own output — skip those items
                    items_to_send = items_after_prefix[len(response_output) :]
                    cached = self._previous_input_length + len(response_output)
                    params["input"] = items_to_send
                    params["previous_response_id"] = self._previous_response_id
                    logger.debug(
                        f"{self}: Sending incremental context via previous_response_id "
                        f"({len(items_to_send)} new items, {cached} cached)"
                    )
                    return params

        logger.debug(f"{self}: Sending full context ({len(full_input)} items)")
        return params

    @staticmethod
    def _starts_with_response_output(items: list, response_output: list) -> bool:
        """Check whether ``items`` begins with entries that match ``response_output``.

        When using ``previous_response_id``, the server already knows its own
        output.  After confirming that the input prefix matches what we
        previously sent, this method checks whether the items immediately
        following that prefix correspond to the server's response output.
        If they do, those items can be skipped so we send only the truly
        new items (user messages, tool results, etc.).

        For messages, the comparison checks role and text content (extracting
        text from the output's ``output_text`` content parts and comparing
        against the input's content).  For function calls, it matches by
        ``call_id``.  This avoids requiring exact format equality while
        still confirming the items represent the same data.  If the match
        fails for any reason, the caller falls back to sending the full
        context.

        Args:
            items: The input items following the matched prefix.
            response_output: Raw ``output`` array from the previous
                ``response.completed`` event.

        Returns:
            True if the leading items correspond to the response output.
        """
        if len(items) < len(response_output):
            return False

        for output_item, input_item in zip(response_output, items):
            output_type = output_item.get("type")
            if output_type == "message":
                if input_item.get("role") != output_item.get("role", "assistant"):
                    return False
                # Extract text from the output's content array and compare
                # against the input's content (which the adapter stores as
                # a plain string for simple text responses).
                output_content = output_item.get("content", [])
                if isinstance(output_content, list):
                    output_text = "".join(
                        p.get("text", "") for p in output_content if p.get("type") == "output_text"
                    )
                else:
                    output_text = str(output_content)
                input_content = input_item.get("content", "")
                if isinstance(input_content, list):
                    # Adapter may produce multimodal content parts
                    input_text = "".join(
                        p.get("text", "") for p in input_content if p.get("type") == "input_text"
                    )
                else:
                    input_text = str(input_content)
                if output_text != input_text:
                    return False
            elif output_type == "function_call":
                if input_item.get("type") != "function_call" or input_item.get(
                    "call_id"
                ) != output_item.get("call_id"):
                    return False
            else:
                # Unknown output type — can't confirm match
                return False

        return True

    def _store_previous_response_state(
        self, response_id: str, full_input: list, response_output: list
    ):
        """Store state for the next call's previous_response_id optimization.

        Args:
            response_id: The response ID returned by the server.
            full_input: The complete input items list that was sent.
            response_output: Raw ``output`` array from the ``response.completed``
                event, stored for loose comparison on the next call.
        """
        self._previous_response_id = response_id
        self._previous_input_length = len(full_input)
        self._previous_input_hash = self._hash_input_items(full_input)
        self._previous_response_output = response_output

    def _clear_previous_response_state(self):
        """Clear stored previous_response_id state."""
        self._previous_response_id = None
        self._previous_input_length = None
        self._previous_input_hash = None
        self._previous_response_output = None

    # -- frame processing -----------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for LLM completion requests.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        context = None
        if isinstance(frame, LLMContextFrame):
            context = frame.context
        else:
            await self.push_frame(frame, direction)

        if context:
            try:
                await self.push_frame(LLMFullResponseStartFrame())
                await self.start_processing_metrics()
                await self._process_context(context)
            except Exception as e:
                await self.push_error(error_msg=f"Error during completion: {e}", exception=e)
            finally:
                await self.stop_processing_metrics()
                await self.push_frame(LLMFullResponseEndFrame())

    # -- core inference -------------------------------------------------------

    @traced_llm
    async def _process_context(self, context: LLMContext):
        """Run inference over WebSocket with retry and previous_response_id.

        Args:
            context: The LLM context containing conversation history.
        """
        adapter: OpenAIResponsesLLMAdapter = self.get_llm_adapter()
        logger.debug(
            f"{self}: Generating response from universal context "
            f"{adapter.get_messages_for_logging(context)}"
        )

        invocation_params = adapter.get_llm_invocation_params(
            context, system_instruction=self._settings.system_instruction
        )

        full_input = invocation_params["input"]

        max_attempts = 2
        for attempt in range(max_attempts):
            params = self._build_response_params(invocation_params)
            # WebSocket mode does not use the "stream" parameter
            params.pop("stream", None)

            # Apply previous_response_id optimization (skipped after a retry)
            if attempt == 0:
                params = self._apply_previous_response_optimization(params, full_input)

            try:
                await self._ensure_connected()
                await self.start_ttfb_metrics()
                await self._ws_send({"type": "response.create", **params})
                await self._receive_response_events(context, full_input)
                return  # Success
            except _PreviousResponseNotFoundError:
                logger.warning(
                    f"{self}: previous_response_not_found — "
                    f"retrying with full context ({len(full_input)} items)"
                )
                self._clear_previous_response_state()
                await self.stop_ttfb_metrics()
                if attempt >= max_attempts - 1:
                    await self.push_error(
                        error_msg="previous_response_not_found: retry also failed"
                    )
                    return
            except _ConnectionLimitReachedError:
                logger.warning(
                    f"{self}: WebSocket connection limit reached — "
                    f"reconnecting and retrying with full context ({len(full_input)} items)"
                )
                self._clear_previous_response_state()
                await self.stop_ttfb_metrics()
                await self._reconnect()
                if attempt >= max_attempts - 1:
                    await self.push_error(error_msg="WebSocket connection limit: retry also failed")
                    return
            except ConnectionClosed as e:
                logger.warning(
                    f"{self}: WebSocket connection closed during inference: {e} — "
                    f"reconnecting and retrying with full context ({len(full_input)} items)"
                )
                self._clear_previous_response_state()
                self._websocket = None
                await self.stop_ttfb_metrics()
                await self._reconnect()
                if attempt >= max_attempts - 1:
                    await self.push_error(
                        error_msg=f"WebSocket connection closed: retry also failed: {e}",
                        exception=e,
                    )
                    return

    async def _receive_response_events(self, context: LLMContext, full_input: list):
        """Receive and process WebSocket events until the response completes.

        Args:
            context: The LLM context for the current inference.
            full_input: The complete input items list (for storing state on success).

        Raises:
            _PreviousResponseNotFoundError: Server couldn't find previous response.
            _ConnectionLimitReachedError: 60-minute connection limit reached.
            ConnectionClosed: WebSocket connection was closed unexpectedly.
        """
        function_calls: Dict[str, Dict[str, str]] = {}
        current_arguments: Dict[str, str] = {}

        while True:
            raw = await self._websocket.recv()
            event = json.loads(raw)
            event_type = event.get("type")

            if event_type == "response.output_text.delta":
                await self.stop_ttfb_metrics()
                await self._push_llm_text(event.get("delta", ""))

            elif event_type == "response.output_item.added":
                await self.stop_ttfb_metrics()
                item = event.get("item", {})
                if item.get("type") == "function_call":
                    item_id = item.get("id", "")
                    function_calls[item_id] = {
                        "name": item.get("name", ""),
                        "call_id": item.get("call_id", ""),
                        "arguments": "",
                    }
                    current_arguments[item_id] = ""

            elif event_type == "response.function_call_arguments.delta":
                item_id = event.get("item_id", "")
                if item_id in current_arguments:
                    current_arguments[item_id] += event.get("delta", "")

            elif event_type == "response.function_call_arguments.done":
                item_id = event.get("item_id", "")
                if item_id in function_calls:
                    function_calls[item_id]["arguments"] = event.get("arguments", "")

            elif event_type == "response.output_item.done":
                item = event.get("item", {})
                if item.get("type") == "function_call":
                    item_id = item.get("id", "")
                    if item_id in function_calls:
                        function_calls[item_id]["name"] = item.get("name", "")
                        function_calls[item_id]["call_id"] = item.get("call_id", "")
                        function_calls[item_id]["arguments"] = item.get("arguments", "")

            elif event_type == "response.completed":
                response = event.get("response", {})
                usage = response.get("usage")
                if usage:
                    input_details = usage.get("input_tokens_details") or {}
                    output_details = usage.get("output_tokens_details") or {}
                    tokens = LLMTokenUsage(
                        prompt_tokens=usage.get("input_tokens", 0),
                        completion_tokens=usage.get("output_tokens", 0),
                        total_tokens=usage.get("total_tokens", 0),
                        cache_read_input_tokens=input_details.get("cached_tokens", 0),
                        reasoning_tokens=output_details.get("reasoning_tokens", 0),
                    )
                    await self.start_llm_usage_metrics(tokens)

                self._full_model_name = response.get("model")

                # Store state for next call's previous_response_id optimization.
                # Include the response output so the hash covers the assistant's
                # reply — the server already knows it, so we won't resend it.
                response_id = response.get("id")
                if response_id:
                    response_output = response.get("output") or []
                    self._store_previous_response_state(response_id, full_input, response_output)

                break  # Response complete

            elif event_type in ("response.failed", "response.incomplete"):
                response = event.get("response", {})
                status_details = response.get("status_details") or {}
                error_info = status_details.get("error") or {}
                error_msg = error_info.get("message", f"Response {event_type.split('.')[-1]}")
                await self.push_error(error_msg=f"LLM response error: {error_msg}")
                break

            elif event_type == "error":
                error = event.get("error", {})
                code = error.get("code", "")
                message = error.get("message", "Unknown error")

                if code == "previous_response_not_found":
                    raise _PreviousResponseNotFoundError(message)
                elif code == "websocket_connection_limit_reached":
                    raise _ConnectionLimitReachedError(message)
                else:
                    await self.push_error(error_msg=f"WebSocket API error: {message}")
                    break

        # Process any function calls
        if function_calls:
            fc_list = self._process_function_calls(context, function_calls)
            await self.run_function_calls(fc_list)


# ---------------------------------------------------------------------------
# HTTP variant
# ---------------------------------------------------------------------------


class OpenAIResponsesHttpLLMService(_BaseOpenAIResponsesLLMService):
    """OpenAI Responses API LLM service using HTTP streaming transport.

    Uses server-sent events (SSE) via the OpenAI Python SDK for streaming
    inference. Each ``_process_context`` call opens a new HTTP connection.

    Example::

        llm = OpenAIResponsesHttpLLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            settings=OpenAIResponsesHttpLLMService.Settings(
                model="gpt-4.1",
                system_instruction="You are a helpful assistant.",
            ),
        )
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for LLM completion requests.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        context = None
        if isinstance(frame, LLMContextFrame):
            context = frame.context
        else:
            await self.push_frame(frame, direction)

        if context:
            try:
                await self.push_frame(LLMFullResponseStartFrame())
                await self.start_processing_metrics()
                await self._process_context(context)
            except httpx.TimeoutException as e:
                await self._call_event_handler("on_completion_timeout")
                await self.push_error(error_msg="LLM completion timeout", exception=e)
            except Exception as e:
                await self.push_error(error_msg=f"Error during completion: {e}", exception=e)
            finally:
                await self.stop_processing_metrics()
                await self.push_frame(LLMFullResponseEndFrame())

    @traced_llm
    async def _process_context(self, context: LLMContext):
        adapter: OpenAIResponsesLLMAdapter = self.get_llm_adapter()
        logger.debug(
            f"{self}: Generating response from universal context "
            f"{adapter.get_messages_for_logging(context)}"
        )

        invocation_params = adapter.get_llm_invocation_params(
            context, system_instruction=self._settings.system_instruction
        )

        params = self._build_response_params(invocation_params)

        await self.start_ttfb_metrics()

        stream: AsyncStream[ResponseStreamEvent] = await self._client.responses.create(**params)

        # Track function calls across stream events
        function_calls: Dict[str, Dict[str, str]] = {}  # item_id -> {name, call_id, arguments}
        current_arguments: Dict[str, str] = {}  # item_id -> accumulated arguments

        # Ensure stream and its async iterator are closed on cancellation/exception
        # to prevent socket leaks and uvloop crashes. Closing the iterator first
        # cascades cleanup through nested async generators (httpx/httpcore internals),
        # preventing uvloop's broken asyncgen finalizer from firing on Python 3.12+
        # (MagicStack/uvloop#699).
        @asynccontextmanager
        async def _closing(stream):
            chunk_iter = stream.__aiter__()
            try:
                yield chunk_iter
            finally:
                # Close the iterator first to cascade cleanup through
                # nested async generators (httpx/httpcore internals).
                if hasattr(chunk_iter, "aclose"):
                    await chunk_iter.aclose()
                # Then close the stream to release HTTP resources.
                if hasattr(stream, "close"):
                    await stream.close()
                elif hasattr(stream, "aclose"):
                    await stream.aclose()

        async with _closing(stream) as event_iter:
            async for event in event_iter:
                if isinstance(event, ResponseTextDeltaEvent):
                    await self.stop_ttfb_metrics()
                    await self._push_llm_text(event.delta)

                elif isinstance(event, ResponseOutputItemAddedEvent):
                    await self.stop_ttfb_metrics()
                    item = event.item
                    if isinstance(item, ResponseFunctionToolCall):
                        item_id = item.id or ""
                        function_calls[item_id] = {
                            "name": item.name,
                            "call_id": item.call_id,
                            "arguments": "",
                        }
                        current_arguments[item_id] = ""

                elif isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
                    item_id = event.item_id
                    if item_id in current_arguments:
                        current_arguments[item_id] += event.delta

                elif isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
                    item_id = event.item_id
                    if item_id in function_calls:
                        function_calls[item_id]["arguments"] = event.arguments

                elif isinstance(event, ResponseOutputItemDoneEvent):
                    item = event.item
                    if isinstance(item, ResponseFunctionToolCall):
                        item_id = item.id or ""
                        if item_id in function_calls:
                            function_calls[item_id]["name"] = item.name
                            function_calls[item_id]["call_id"] = item.call_id
                            function_calls[item_id]["arguments"] = item.arguments

                elif isinstance(event, ResponseCompletedEvent):
                    response = event.response
                    if response.usage:
                        tokens = LLMTokenUsage(
                            prompt_tokens=response.usage.input_tokens,
                            completion_tokens=response.usage.output_tokens,
                            total_tokens=response.usage.total_tokens,
                            cache_read_input_tokens=response.usage.input_tokens_details.cached_tokens,
                            reasoning_tokens=response.usage.output_tokens_details.reasoning_tokens,
                        )
                        await self.start_llm_usage_metrics(tokens)

                    # This field is used by @traced_llm for more detailed
                    # model name in tracing spans
                    self._full_model_name = response.model

        # Process any function calls
        if function_calls:
            fc_list = self._process_function_calls(context, function_calls)
            await self.run_function_calls(fc_list)


__all__ = [
    "OpenAIResponsesLLMService",
    "OpenAIResponsesHttpLLMService",
    "OpenAIResponsesLLMSettings",
]
