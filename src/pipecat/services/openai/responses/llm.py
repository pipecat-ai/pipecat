#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Responses API LLM service implementations (WebSocket and HTTP)."""

import asyncio
import hashlib
import json
import os
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
from loguru import logger
from openai import NOT_GIVEN, AsyncOpenAI, AsyncStream, DefaultAsyncHttpxClient
from openai._types import NotGiven as OpenAINotGiven
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
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import (
    FunctionCallFromLLM,
    LLMService,
    WebsocketLLMService,
    WebsocketReconnectedError,
)
from pipecat.services.settings import NOT_GIVEN as _NOT_GIVEN
from pipecat.services.settings import LLMSettings, _NotGiven, assert_given
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

    # Override inherited LLMSettings fields to also accept openai's NotGiven
    # sentinel. The service stores openai's NOT_GIVEN in these fields so they
    # can be passed through unchanged to the AsyncOpenAI client.
    temperature: float | None | _NotGiven | OpenAINotGiven = field(
        default_factory=lambda: _NOT_GIVEN
    )
    top_p: float | None | _NotGiven | OpenAINotGiven = field(default_factory=lambda: _NOT_GIVEN)
    max_completion_tokens: int | _NotGiven | OpenAINotGiven = field(
        default_factory=lambda: _NOT_GIVEN
    )


# ---------------------------------------------------------------------------
# Shared base class (private)
# ---------------------------------------------------------------------------


class _BaseOpenAIResponsesLLMService(LLMService[OpenAIResponsesLLMAdapter]):
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
        default_headers: Mapping[str, str] | None = None,
        service_tier: str | None = None,
        settings: Settings | None = None,
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

        # Resolve the API key from the environment if not provided. The
        # AsyncOpenAI HTTP client does this automatically, but the WebSocket
        # variant connects via raw websockets and needs the key explicitly.
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
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
        params: dict[str, Any] = {
            "model": self._settings.model,
            "stream": True,
            # store=False avoids OpenAI-side 30-day conversation storage.
            # The WebSocket variant's previous_response_id optimization
            # still works with store=False because it uses a connection-local
            # in-memory cache. See the class docstrings for details.
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
        max_tokens: int | None = None,
        system_instruction: str | None = None,
    ) -> str | None:
        """Run a one-shot, out-of-band inference with the given LLM context.

        Always uses the HTTP client regardless of transport variant.

        Args:
            context: The LLM context containing conversation history.
            max_tokens: Optional maximum number of tokens to generate.
            system_instruction: Optional system instruction for this inference.

        Returns:
            The LLM's response as a string, or None if no response is generated.
        """
        adapter = self.get_llm_adapter()
        effective_instruction = system_instruction or assert_given(
            self._settings.system_instruction
        )
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
        function_calls: dict[str, dict[str, str]],
    ) -> list[FunctionCallFromLLM]:
        """Convert accumulated function call data into FunctionCallFromLLM list.

        Args:
            context: The LLM context for the current inference.
            function_calls: Map of item_id to {name, call_id, arguments}.

        Returns:
            List of parsed function call objects.
        """
        fc_list: list[FunctionCallFromLLM] = []
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


class OpenAIResponsesLLMService(
    _BaseOpenAIResponsesLLMService, WebsocketLLMService[OpenAIResponsesLLMAdapter]
):
    """OpenAI Responses API LLM service using WebSocket transport.

    Maintains a persistent WebSocket connection to ``wss://api.openai.com/v1/responses``
    for lower-latency inference, especially beneficial for tool-call-heavy workflows.
    Automatically uses ``previous_response_id`` to send only incremental context when
    possible, and falls back to full context on reconnection or cache miss.

    The ``previous_response_id`` optimization works with ``store=False`` (the default)
    because WebSocket mode uses a connection-local in-memory cache — no conversations
    are stored on OpenAI's servers.  This is why the HTTP variant
    (``OpenAIResponsesHttpLLMService``) does not offer this optimization by default
    (or at all, yet): over HTTP, ``previous_response_id`` requires ``store=True``,
    which enables OpenAI-side 30-day conversation storage.

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

        # State for previous_response_id optimization
        self._previous_response_id: str | None = None
        self._previous_input_hash: str | None = None
        self._previous_input_length: int | None = None
        self._previous_response_output: list | None = None

        # Response cancellation state
        self._current_response_id: str | None = None  # ID of current non-cancelled response
        self._cancel_pending_response: bool = False
        self._needs_drain: bool = False

    # -- WebsocketLLMService interface ----------------------------------------

    async def _connect_websocket(self):
        """Establish the WebSocket connection."""
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
            self._websocket = None
            await self.push_error(error_msg=f"Error connecting to WebSocket: {e}", exception=e)

    async def _disconnect_websocket(self):
        """Close the WebSocket connection and clear state."""
        try:
            await self.stop_all_metrics()
            if self._websocket:
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error disconnecting from WebSocket: {e}", exception=e)
        finally:
            self._websocket = None
            self._clear_previous_response_state()
            self._clear_cancellation_state()

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
        if self._previous_response_id is None:
            logger.debug(f"{self}: Sending full context ({len(full_input)} items)")
            logger.trace(f"{self}: Reason: no previous response")
            return params

        if (
            self._previous_input_length is None
            or self._previous_input_hash is None
            or len(full_input) <= self._previous_input_length
        ):
            logger.debug(f"{self}: Sending full context ({len(full_input)} items)")
            logger.trace(
                f"{self}: Reason: input not longer than previous ({self._previous_input_length})"
            )
            return params

        prefix = full_input[: self._previous_input_length]
        prefix_hash = self._hash_input_items(prefix)
        if prefix_hash != self._previous_input_hash:
            logger.debug(f"{self}: Sending full context ({len(full_input)} items)")
            logger.trace(
                f"{self}: Reason: input prefix hash mismatch "
                f"(previous input: {json.dumps(prefix, indent=2, default=str)}, "
                f"expected hash: {self._previous_input_hash}, "
                f"actual hash: {prefix_hash})"
            )
            return params

        items_after_prefix = full_input[self._previous_input_length :]
        response_output = self._previous_response_output or []

        if not self._starts_with_response_output(items_after_prefix, response_output):
            logger.debug(f"{self}: Sending full context ({len(full_input)} items)")
            logger.trace(
                f"{self}: Reason: response output mismatch after prefix "
                f"(previous response output: {json.dumps(response_output, indent=2, default=str)}, "
                f"items after prefix: {json.dumps(items_after_prefix, indent=2, default=str)})"
            )
            return params

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

    # -- response cancellation ------------------------------------------------

    def _clear_cancellation_state(self):
        """Clear response cancellation tracking state."""
        self._current_response_id = None
        self._cancel_pending_response = False
        self._needs_drain = False

    async def _drain_cancelled_response(self):
        """Drain events from a cancelled response before starting a new one.

        After a cancellation, the WebSocket may still have in-flight events
        from the cancelled response.  We must drain them before sending a
        new ``response.create`` — we can't simply filter them inline because
        the API doesn't provide a reliable way to correlate events to a
        specific response (e.g. delta events carry neither a
        ``response_id`` nor any intermediary identifier that could be
        traced back to one).

        This method reads and discards events until a terminal event
        (``response.completed``, ``response.failed``, or
        ``response.incomplete``) arrives, ensuring the connection is clean.
        If draining times out or the connection drops, clears cancellation
        state and returns — ``_ensure_connected`` will handle reconnection
        before the next inference.
        """
        if not self._websocket:
            self._clear_cancellation_state()
            return

        logger.debug(f"{self}: Draining cancelled response events")
        try:
            while True:
                raw = await asyncio.wait_for(self._websocket.recv(), timeout=5.0)
                event = json.loads(raw)
                event_type = event.get("type")

                # If we were cancelled before response.created, the first
                # event here will be response.created for the cancelled
                # request — send cancel now that we have the id.
                if event_type == "response.created" and self._cancel_pending_response:
                    response_id = event.get("response", {}).get("id")
                    logger.debug(
                        f"{self}: Received response.created for pending-cancel "
                        f"response {response_id} — sending response.cancel"
                    )
                    self._cancel_pending_response = False
                    if response_id:
                        try:
                            await self._ws_send(
                                {"type": "response.cancel", "response_id": response_id}
                            )
                        except Exception:
                            pass
                    continue

                if event_type in ("response.completed", "response.failed", "response.incomplete"):
                    logger.debug(
                        f"{self}: Cancelled response terminated with {event_type} — "
                        f"connection is clean"
                    )
                    self._clear_cancellation_state()
                    return
        except (TimeoutError, WebsocketReconnectedError, ConnectionClosed) as e:
            logger.warning(f"{self}: Error draining cancelled response: {e}")
            self._clear_cancellation_state()

    # -- frame processing -----------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for LLM completion requests.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            try:
                await self.push_frame(LLMFullResponseStartFrame())
                await self.start_processing_metrics()
                await self._process_context(frame.context)
            except asyncio.CancelledError:
                # The pipeline cancelled us (e.g. due to an interruption).
                # Ask the server to stop generating and flag that we need
                # to drain stale events before the next inference.  We
                # can't just send a new response.create and filter stale
                # events inline — the API doesn't provide a reliable way
                # to correlate events to a specific response.
                if self._current_response_id:
                    logger.debug(
                        f"{self}: Cancelled during response {self._current_response_id} "
                        f"— sending response.cancel"
                    )
                    try:
                        await self._ws_send(
                            {"type": "response.cancel", "response_id": self._current_response_id}
                        )
                    except Exception:
                        pass
                else:
                    logger.debug(
                        f"{self}: Cancelled before response.created "
                        f"— will cancel on next response.created"
                    )
                    self._cancel_pending_response = True
                self._current_response_id = None
                self._needs_drain = True
                raise
            except Exception as e:
                await self.push_error(error_msg=f"Error during inference: {e}", exception=e)
            finally:
                await self.stop_processing_metrics()
                await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)

    # -- core inference -------------------------------------------------------

    @traced_llm
    async def _process_context(self, context: LLMContext):
        """Run inference over WebSocket with retry and previous_response_id.

        Tries once with the ``previous_response_id`` optimization.  On a
        retriable error (cache miss, connection limit, connection drop),
        clears state and retries once with the full context.  Transport-level
        ``ConnectionClosed`` errors are handled transparently by
        ``_ws_send``/``_ws_recv`` (auto-reconnect → ``WebsocketReconnectedError``).

        Args:
            context: The LLM context containing conversation history.
        """
        # If a previous response was cancelled, drain its remaining events
        # before starting a new one.
        if self._needs_drain:
            await self._drain_cancelled_response()

        adapter = self.get_llm_adapter()
        logger.debug(
            f"{self}: Generating response from universal context "
            f"{adapter.get_messages_for_logging(context)}"
        )

        invocation_params = adapter.get_llm_invocation_params(
            context, system_instruction=assert_given(self._settings.system_instruction)
        )

        full_input = invocation_params["input"]

        def build_params(*, apply_optimization: bool) -> dict:
            params = self._build_response_params(invocation_params)
            # WebSocket mode does not use the "stream" parameter.
            params.pop("stream", None)
            if apply_optimization:
                params = self._apply_previous_response_optimization(params, full_input)
            return params

        async def send_and_receive(params: dict):
            await self._ensure_connected()
            await self.start_ttfb_metrics()
            await self._ws_send({"type": "response.create", **params})
            await self._receive_response_events(context, full_input)

        async def cleanup():
            self._clear_previous_response_state()
            await self.stop_ttfb_metrics()

        # -- first attempt (with previous_response_id optimization) -----------

        try:
            await send_and_receive(build_params(apply_optimization=True))
            return  # Success
        except _PreviousResponseNotFoundError:
            logger.warning(
                f"{self}: previous_response_not_found — "
                f"retrying with full context ({len(full_input)} items)"
            )
            await cleanup()
        except _ConnectionLimitReachedError:
            logger.warning(
                f"{self}: WebSocket connection limit reached — "
                f"reconnecting and retrying with full context ({len(full_input)} items)"
            )
            await cleanup()
            await self._try_reconnect(report_error=self._report_error)
        except WebsocketReconnectedError:
            # ConnectionClosed was handled by the base class — connection is
            # fresh, so any connection-local server state is gone.
            logger.warning(
                f"{self}: Connection lost and recovered — "
                f"retrying with full context ({len(full_input)} items)"
            )
            await cleanup()
        except Exception:
            await cleanup()
            raise

        # -- retry with full context (no optimization) ------------------------

        try:
            await send_and_receive(build_params(apply_optimization=False))
        except Exception:
            await cleanup()
            raise

    async def _receive_response_events(self, context: LLMContext, full_input: list):
        """Receive and process WebSocket events until the response completes.

        Args:
            context: The LLM context for the current inference.
            full_input: The complete input items list (for storing state on success).

        Raises:
            _PreviousResponseNotFoundError: Server couldn't find previous response.
            _ConnectionLimitReachedError: 60-minute connection limit reached.
            WebsocketReconnectedError: Connection was lost and auto-recovered.
            ConnectionClosed: Connection was lost and could not be recovered.
        """
        function_calls: dict[str, dict[str, str]] = {}
        current_arguments: dict[str, str] = {}

        while True:
            event = await self._ws_recv()
            event_type = event.get("type")

            if event_type == "response.created":
                self._current_response_id = event.get("response", {}).get("id")
                logger.debug(f"{self}: Response started: {self._current_response_id}")
                continue

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

    Unlike the WebSocket variant, this service does not use
    ``previous_response_id`` for incremental context delivery by default
    (or at all, yet).  Over HTTP, ``previous_response_id`` requires
    ``store=True``, which enables OpenAI-side 30-day conversation storage
    — a privacy/compliance tradeoff that many users won't want.  The
    WebSocket variant avoids this because its ``previous_response_id``
    uses a connection-local in-memory cache that works with
    ``store=False`` (nothing is stored long-term).

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

        if isinstance(frame, LLMContextFrame):
            try:
                await self.push_frame(LLMFullResponseStartFrame())
                await self.start_processing_metrics()
                await self._process_context(frame.context)
            except httpx.TimeoutException as e:
                await self._call_event_handler("on_completion_timeout")
                await self.push_error(error_msg="LLM completion timeout", exception=e)
            except Exception as e:
                await self.push_error(error_msg=f"Error during inference: {e}", exception=e)
            finally:
                await self.stop_processing_metrics()
                await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)

    @traced_llm
    async def _process_context(self, context: LLMContext):
        adapter = self.get_llm_adapter()
        logger.debug(
            f"{self}: Generating response from universal context "
            f"{adapter.get_messages_for_logging(context)}"
        )

        invocation_params = adapter.get_llm_invocation_params(
            context, system_instruction=assert_given(self._settings.system_instruction)
        )

        params = self._build_response_params(invocation_params)

        await self.start_ttfb_metrics()

        stream: AsyncStream[ResponseStreamEvent] = await self._client.responses.create(**params)

        # Track function calls across stream events
        function_calls: dict[str, dict[str, str]] = {}  # item_id -> {name, call_id, arguments}
        current_arguments: dict[str, str] = {}  # item_id -> accumulated arguments

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
