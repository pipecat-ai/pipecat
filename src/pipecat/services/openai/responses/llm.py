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
import re
import time
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger
from openai import NOT_GIVEN as OPENAI_NOT_GIVEN
from openai import APITimeoutError, AsyncOpenAI, AsyncStream, DefaultAsyncHttpxClient
from openai._types import NotGiven as OpenAINotGiven
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseIncompleteEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseReasoningItem,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)
from pydantic import BaseModel
from websockets.exceptions import ConnectionClosed

from pipecat.adapters.services.open_ai_responses_adapter import (
    OpenAIResponsesLLMAdapter,
    OpenAIResponsesLLMInvocationParams,
)
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMThoughtEndFrame,
    LLMThoughtStartFrame,
    LLMThoughtTextFrame,
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
from pipecat.services.openai.base_llm import OPENAI_MODEL_WITHOUT_RESPONSE_SCHEMA
from pipecat.services.settings import LLMSettings
from pipecat.utils.http import TIMEOUT_EXCEPTIONS, connection_limits
from pipecat.utils.tracing.service_decorators import traced_llm
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given

DEFAULT_WS_URL = "wss://api.openai.com/v1/responses"

# How long the fallback drain, used when the server does not tag events with
# their lane, waits for a cancelled response's remaining events before
# replacing the connection instead. The server keeps generating a cancelled
# response, so the wait would otherwise last as long as the rest of that reply
# takes, while a new connection costs about a second.
CANCELLED_RESPONSE_DRAIN_SECS = 1.0

# How long a request sent while abandoned responses are still in flight may go
# without an event of its own before it is presumed queued behind them and the
# connection is replaced. A response is normally acknowledged well within it.
LANE_ACK_SECS = 1.5

# Per-connection limits of WebSocket mode, from OpenAI's documentation. Past
# the in-flight cap the server queues requests behind the active responses;
# past the lane cap it rejects the request with websocket_stream_limit_reached.
MAX_IN_FLIGHT_RESPONSES = 16
MAX_NAMED_LANES = 32

_TERMINAL_EVENT_TYPES = frozenset({"response.completed", "response.failed", "response.incomplete"})
_CONNECTION_ERROR_CODES = frozenset(
    {"websocket_connection_limit_reached", "websocket_stream_limit_reached"}
)

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


class _StreamLimitReachedError(_RetryableError):
    """WebSocket connection has used up its named lanes."""

    pass


class _LaneStalledError(_RetryableError):
    """The request is queued behind, or unacknowledged among, abandoned responses."""

    pass


class _LaneRejectedError(_RetryableError):
    """The server does not accept ``stream_id``; the request is re-sent without it."""

    pass


class _ResponseTimeoutError(_RetryableError):
    """Response did not begin producing output within the retry timeout."""

    pass


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class OpenAIResponsesReasoningConfig(BaseModel):
    """Reasoning configuration for reasoning-capable OpenAI Responses models.

    Only reasoning-capable models use this — the gpt-5.x series and the o-series.
    The service's default model, ``gpt-4.1``, does not reason, so this config has
    no effect there; select a reasoning-capable model to use it. See OpenAI's
    reasoning guide (https://platform.openai.com/docs/guides/reasoning) and model
    list (https://platform.openai.com/docs/models) to choose one and to check
    which effort levels it accepts.

    Reasoning models use internal reasoning tokens before producing a response.
    This controls how much reasoning they do and whether a human-readable summary
    of it is returned.

    Parameters:
        effort: How much reasoning effort the model applies. ``None`` (the
            default) leaves the field unset, so the model's own default applies;
            ``"none"`` disables reasoning for latency-sensitive use and
            ``"max"`` applies the model's deepest reasoning. At lower efforts
            (e.g. ``"low"``) the model reasons only when a turn calls for it, so
            simple prompts may produce no summary at all.
        summary: Verbosity of the reasoning summary to return. ``None`` (the
            default) requests no summary. Any summary is surfaced via thought
            frames (the ``on_assistant_thought`` event); the encrypted reasoning
            itself is preserved across turns regardless of this setting.
        mode: Reasoning mode for models that offer one, such as the gpt-5.6
            series: ``"standard"`` or the slower, more thorough ``"pro"``.
            ``None`` (the default) leaves the field unset, so the model's own
            default applies. ``effort`` selects the reasoning intensity within
            the chosen mode.
    """

    # ``| str`` for forward compatibility: if OpenAI adds new levels, users can
    # pass the new string without waiting for a Pipecat release.
    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | str | None = None
    summary: Literal["auto", "concise", "detailed"] | str | None = None
    mode: Literal["standard", "pro"] | str | None = None


@dataclass
class OpenAIResponsesLLMSettings(LLMSettings):
    """Settings for OpenAI Responses API LLM services.

    Parameters:
        max_completion_tokens: Maximum completion tokens to generate.
        reasoning: Reasoning configuration for reasoning-capable models. ``None``
            (the default) leaves reasoning unconfigured — the service then disables
            reasoning by default for whatever models possible, to keep latency low
            for real-time voice. Note that the default model, ``gpt-4.1``, does
            not reason.
    """

    # Override inherited LLMSettings fields to also accept the OpenAI SDK's
    # sentinel, which the service stores here so these fields can be passed
    # through unchanged to the AsyncOpenAI client.
    temperature: float | None | NotGiven | OpenAINotGiven = field(default_factory=lambda: NOT_GIVEN)
    top_p: float | None | NotGiven | OpenAINotGiven = field(default_factory=lambda: NOT_GIVEN)
    max_completion_tokens: int | NotGiven | OpenAINotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )
    reasoning: OpenAIResponsesReasoningConfig | None | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )


# ---------------------------------------------------------------------------
# Model classification
# ---------------------------------------------------------------------------


def _is_o_series(model: str) -> bool:
    """Whether the model is an o-series reasoning model (o1, o3, o4-mini, ...)."""
    return bool(re.match(r"o\d", model.lower()))


def _rejects_effort_none(model: str) -> bool:
    """Whether a reasoning model rejects ``effort="none"`` with an API error.

    The reasoning-first o-series and ``gpt-6-astra`` accept only a positive
    effort level, so reasoning cannot be switched off for them.
    """
    model = model.lower()
    return _is_o_series(model) or model.startswith("gpt-6-astra")


def _model_supports_reasoning(model: str) -> bool | None:
    """Classify whether an OpenAI model supports reasoning.

    Assumes that future (beyond gpt-5.x) mainline gpt series models will also
    support reasoning. This can be revisited if OpenAI changes their model lineup
    or reasoning support in the future.

    Args:
        model: The model name (e.g. ``"gpt-5.4"``, ``"o3"``, ``"gpt-4.1"``).

    Returns:
        ``True`` for reasoning-capable models — the o-series and the mainline gpt
        series from gpt-5 onward. ``False`` for models known *not* to reason:
        gpt-4.x and earlier, and the ``gpt-5-chat`` non-reasoning variant.
        ``None`` when the model is unrecognized and we can't tell either way.
    """
    model = model.lower()
    if _is_o_series(model):
        return True
    # Mainline gpt series: reasons from gpt-5 onward. ``gpt-5-chat`` is the
    # non-reasoning variant of the series.
    match = re.match(r"gpt-(\d+)", model)
    if match:
        return int(match.group(1)) >= 5 and "chat" not in model
    return None


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

    ReasoningConfig = OpenAIResponsesReasoningConfig

    adapter_class = OpenAIResponsesLLMAdapter

    supports_response_schema: bool = True

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
        retry_timeout_secs: float | None = 5.0,
        retry_on_timeout: bool | None = False,
        **kwargs,
    ):
        """Initialize the OpenAI Responses API LLM service.

        Args:
            api_key: OpenAI API key. If None, uses environment variable.
            base_url: Custom base URL for OpenAI API. If None, uses default.
            organization: OpenAI organization ID.
            project: OpenAI project ID.
            default_headers: Additional HTTP headers to include in requests.
            service_tier: Service tier to use: "auto", "default", "flex",
                "scale", "fast" or "priority". "fast" is OpenAI's low-latency
                tier, the name that replaced "priority"; both values are
                accepted.
            settings: Runtime-updatable settings.
            retry_timeout_secs: How long an inference may go without producing
                output before it is abandoned and re-issued, when
                ``retry_on_timeout`` is set. Defaults to 5.0 seconds.
            retry_on_timeout: Whether to re-issue the request once if the first
                attempt produces no output within ``retry_timeout_secs``. The
                retry is unbounded.
            **kwargs: Additional arguments passed to the parent LLMService.
        """
        default_settings = self.Settings(
            model="gpt-4.1",
            system_instruction=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            temperature=OPENAI_NOT_GIVEN,
            top_p=OPENAI_NOT_GIVEN,
            top_k=None,
            max_tokens=None,
            max_completion_tokens=OPENAI_NOT_GIVEN,
            reasoning=None,
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
        self._retry_timeout_secs = retry_timeout_secs
        self._retry_on_timeout = retry_on_timeout
        # Tracks the model we've already warned about (reasoning configured on a
        # non-reasoning model) so we log it once per model rather than per turn.
        self._reasoning_unsupported_warned_for: str | None = None
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
                limits=connection_limits(
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
        if tools is not None and not isinstance(tools, type(OPENAI_NOT_GIVEN)):
            params["tools"] = tools

        # Reasoning
        reasoning = assert_given(self._settings.reasoning)
        reasoning_params = reasoning.model_dump(exclude_none=True) if reasoning else {}
        if reasoning_params:
            params["reasoning"] = reasoning_params
            # Ask for the encrypted reasoning so it can be sent back on later
            # turns, preserving reasoning context across the conversation.
            params["include"] = ["reasoning.encrypted_content"]
            self._warn_if_reasoning_unsupported()
        else:
            # No reasoning configured: disable it by default on the gpt-5.x series
            # for real-time latency (see the helper).
            self._maybe_disable_reasoning(params)

        # Extra settings
        params.update(self._settings.extra)

        return params

    @staticmethod
    def model_supports_response_schema(model: str) -> bool:
        """Whether a model can enforce a response schema.

        OpenAI models before gpt-4o-mini and gpt-4o-2024-08-06 cannot.

        Args:
            model: The model name.
        """
        return not OPENAI_MODEL_WITHOUT_RESPONSE_SCHEMA.match(model)

    async def run_inference(
        self,
        context: LLMContext,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> str | None:
        """Run a one-shot, out-of-band inference with the given LLM context.

        Always uses the HTTP client regardless of transport variant.

        Args:
            context: The LLM context containing conversation history.
            max_tokens: Optional maximum number of tokens to generate.
            system_instruction: Optional system instruction for this inference.
            response_schema: Optional JSON schema the reply must follow. The
                service asks the provider to enforce it, so the reply is JSON
                text matching the schema.

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

        response_schema = self._check_response_schema(response_schema)
        if response_schema is not None:
            params["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "response",
                    "schema": response_schema,
                    "strict": True,
                }
            }

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

    # -- reasoning ------------------------------------------------------------

    def _maybe_disable_reasoning(self, params: dict):
        """Disable reasoning by default on the mainline gpt series for real-time voice.

        When the caller hasn't configured ``reasoning``, request ``effort="none"``
        for whatever models possible. Note that this is a no-op for models like
        ``gpt-5.4`` that already default to ``none``. Some models are left at the
        provider default: those that reject ``effort="none"`` outright (see
        :func:`_rejects_effort_none`), and gpt-4.x and earlier, which don't reason
        at all. Mirrors Gemini's ``_maybe_unset_thinking_budget``, which disables
        or minimizes thinking on its latency-sensitive models.

        Args:
            params: The response params dict (modified in place).
        """
        model = assert_given(self._settings.model)
        # Lower reasoning only for models that reason *and* accept effort="none".
        if model and _model_supports_reasoning(model) and not _rejects_effort_none(model):
            params["reasoning"] = {"effort": "none"}

    def _warn_if_reasoning_unsupported(self):
        """Log a clear error when reasoning is configured on a model that can't use it.

        The raw API error in this case ("Encrypted content is not supported with
        this model") is cryptic, so surface an actionable one instead. Only fires
        for models *known* not to reason; unrecognized models are left alone.
        Logs once per model rather than on every turn.
        """
        model = assert_given(self._settings.model)
        if not model or _model_supports_reasoning(model) is not False:
            # No model, reasoning-capable, or unrecognized — say nothing.
            return
        if model == self._reasoning_unsupported_warned_for:
            return
        self._reasoning_unsupported_warned_for = model
        logger.error(
            f"{self}: `reasoning` is configured but model '{model}' does not support "
            "reasoning, so requests will fail. Reasoning is supported only by "
            "reasoning-capable models — the gpt-5.x series and the o-series; see "
            "OpenAI's reasoning guide (https://platform.openai.com/docs/guides/reasoning). "
            "Remove the `reasoning` setting or select a reasoning-capable model."
        )

    async def _append_reasoning_message(
        self, item_id: str | None, summary: list[dict], encrypted_content: str | None
    ):
        """Persist a reasoning item so it round-trips on the next request.

        The encrypted reasoning is sent back on later requests to preserve
        reasoning context. Store it as an LLM-specific message the adapter
        re-emits as a Responses reasoning input item — positioned, by append
        order, before the assistant message or function call it produced.

        Args:
            item_id: The reasoning item id.
            summary: The reasoning summary parts (``summary_text`` dicts).
            encrypted_content: The encrypted reasoning payload.
        """
        if not encrypted_content:
            # Nothing to round-trip (e.g. reasoning disabled / effort="none").
            return
        message = {
            "type": "reasoning",
            "id": item_id,
            "summary": summary,
            "encrypted_content": encrypted_content,
        }
        await self.push_frame(
            LLMMessagesAppendFrame([self.get_llm_adapter().create_llm_specific_message(message)])
        )


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

    Each request goes out on a named ``stream_id`` lane. When the pipeline
    interrupts a response, the response is left to finish on its lane while the
    next request starts at once on a fresh one: the server runs lanes
    concurrently and tags every event with its lane, so the abandoned
    response's events are dropped as they arrive, by a reader that keeps the
    socket drained between turns. The abandoned reply is still generated in
    full and billed; its usage is reported like any other. A server that does
    not tag its events falls back to draining them before the next request,
    and one that rejects ``stream_id`` is sent plain requests from then on.

    This is the recommended variant for real-time / conversational use.

    Example::

        llm = OpenAIResponsesLLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            settings=OpenAIResponsesLLMService.Settings(
                system_instruction="You are a helpful assistant.",
            ),
        )
    """

    def __init__(
        self,
        *,
        ws_url: str = DEFAULT_WS_URL,
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
        self._needs_drain: bool = False

        # Lane state. The first two describe the server and survive
        # reconnects: whether it accepts ``stream_id`` at all, and whether it
        # tags its events with it (unknown until a response.created is seen).
        # The rest is per connection.
        self._lanes_supported: bool = True
        self._lane_tagging: bool | None = None
        self._lane_id: str | None = None
        self._lane_counter: int = 0
        # A request is out on the lane and has not reached a terminal event.
        # Set before the request is written, so a cancel that lands inside the
        # write still counts it as sent.
        self._lane_busy: bool = False
        # Lanes whose abandoned response may still be in flight, and lanes
        # whose abandoned response has ended and can carry a request again.
        self._abandoned_lanes: set[str] = set()
        self._free_lanes: list[str] = []
        # Untagged events the idle reader kept for the next response's loop.
        self._pending_events: list[dict] = []
        self._idle_reader_task: asyncio.Task | None = None

    # -- WebsocketLLMService interface ----------------------------------------

    async def _connect_websocket(self):
        """Establish the WebSocket connection."""
        try:
            if self._websocket:
                return
            self._websocket = await self._websocket_connect(
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
        await self._stop_idle_reader()
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
            self._clear_lane_state()

    async def cleanup(self):
        """Release resources at teardown."""
        await super().cleanup()
        await self._disconnect()

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
            elif output_type == "reasoning":
                # Reasoning items round-trip via an LLMSpecificMessage; match by
                # the server-assigned id so the optimization can skip them too.
                if input_item.get("type") != "reasoning" or input_item.get("id") != output_item.get(
                    "id"
                ):
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

    # -- response cancellation and lanes ---------------------------------------

    def _clear_cancellation_state(self):
        """Clear response cancellation tracking state."""
        self._current_response_id = None
        self._needs_drain = False

    def _clear_lane_state(self):
        """Forget this connection's lanes. What is known about the server is kept."""
        self._lane_id = None
        self._lane_counter = 0
        self._lane_busy = False
        self._abandoned_lanes.clear()
        self._free_lanes.clear()
        self._pending_events.clear()

    def _abandon_response(self):
        """Give up on the response in flight: the pipeline cancelled this inference.

        Runs inside the cancellation, so it must not await.
        """
        self._current_response_id = None

        if not self._lane_busy:
            logger.debug(f"{self}: Cancelled before the request was sent")
            return

        if self._lane_id is not None and self._lane_tagging:
            # The abandoned response keeps streaming on its lane and its tagged
            # events are dropped on arrival. The next request takes a fresh
            # lane, which holds no cached response to chain from.
            logger.debug(
                f"{self}: Leaving the response on {self._lane_id} — "
                f"the next request takes a new lane"
            )
            self._abandoned_lanes.add(self._lane_id)
            self._lane_id = None
            self._lane_busy = False
            self._clear_previous_response_state()
            return

        # Without lane tags the abandoned response's events cannot be told
        # from the next response's: drain them first.
        logger.debug(f"{self}: Cancelled mid-response — draining its events before the next")
        self._needs_drain = True

    async def _prepare_lane(self):
        """Give the next request a lane, on a connection with room for it.

        A lane freed by an abandoned response that has ended is reused before
        a new one is named. The connection is replaced when the next request
        would be queued behind abandoned responses still in flight, or when
        it needs a lane and the connection has named all it can.
        """
        if not self._lanes_supported:
            self._lane_id = None
            return

        if self._websocket and (
            len(self._abandoned_lanes) >= MAX_IN_FLIGHT_RESPONSES
            or (
                self._lane_id is None
                and not self._free_lanes
                and self._lane_counter >= MAX_NAMED_LANES
            )
        ):
            logger.debug(f"{self}: Connection is at its lane limits — replacing it")
            await self._replace_connection()

        if self._lane_id is None:
            if self._free_lanes:
                self._lane_id = self._free_lanes.pop()
            else:
                self._lane_counter += 1
                self._lane_id = f"lane-{self._lane_counter}"
            self._lane_busy = False

    def _is_stale(self, event: dict) -> bool:
        """Whether an event belongs to a lane other than the current one."""
        lane = event.get("stream_id")
        return lane is not None and lane != self._lane_id

    async def _note_stale_event(self, event: dict):
        """Account for an event from a lane whose response was abandoned."""
        event_type = event.get("type")
        lane = event.get("stream_id")
        if event_type in _TERMINAL_EVENT_TYPES or event_type == "error":
            if lane in self._abandoned_lanes:
                self._abandoned_lanes.discard(lane)
                self._free_lanes.append(lane)
                logger.debug(f"{self}: Abandoned response on {lane} ended with {event_type}")
        if event_type == "response.completed":
            # The abandoned reply was generated in full and billed.
            tokens = self._token_usage(event.get("response", {}))
            if tokens:
                await self.start_llm_usage_metrics(tokens)

    def _adopt_lane_tagging(self, event: dict) -> bool:
        """Turn a pending drain into a lane abandonment once an event carries a tag.

        Returns:
            Whether the event carried a tag and the drain was called off.
        """
        if not self._needs_drain or event.get("stream_id") is None:
            return False
        logger.debug(f"{self}: Events carry lane tags — leaving the response on its lane")
        self._lane_tagging = True
        self._clear_cancellation_state()
        if self._lane_id is not None:
            self._abandoned_lanes.add(self._lane_id)
        self._lane_id = None
        self._lane_busy = False
        return True

    async def _recv_event(self) -> dict:
        """The next event: one the idle reader kept, else the next from the socket."""
        if self._pending_events:
            return self._pending_events.pop(0)
        return await self._ws_recv()

    @staticmethod
    def _token_usage(response: dict) -> LLMTokenUsage | None:
        usage = response.get("usage")
        if not usage:
            return None
        input_details = usage.get("input_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or {}
        return LLMTokenUsage(
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            cache_read_input_tokens=input_details.get("cached_tokens", 0),
            cache_creation_input_tokens=input_details.get("cache_write_tokens", 0),
            reasoning_tokens=output_details.get("reasoning_tokens", 0),
        )

    # -- idle reader ----------------------------------------------------------

    def _start_idle_reader(self):
        """Keep reading the socket between inferences while responses we gave up on stream.

        Nothing else reads the socket then. Left unread, the client library
        stops reading the transport after a handful of frames, the server's
        pings go unanswered, and the connection is closed within its
        keepalive timeout.
        """
        if self._idle_reader_task or not self._websocket:
            return
        if not (self._abandoned_lanes or self._needs_drain):
            return
        self._idle_reader_task = self.create_task(
            self._idle_reader(self._websocket), name="idle_reader"
        )

    async def _stop_idle_reader(self):
        task = self._idle_reader_task
        if task:
            await self.cancel_task(task)
            if self._idle_reader_task is task:
                self._idle_reader_task = None

    async def _idle_reader(self, websocket):
        try:
            while self._abandoned_lanes or self._needs_drain:
                event = json.loads(await websocket.recv())
                await self._handle_idle_event(event)
        except ConnectionClosed:
            # _ensure_connected reconnects before the next request.
            pass
        finally:
            if self._idle_reader_task is asyncio.current_task():
                self._idle_reader_task = None

    async def _handle_idle_event(self, event: dict):
        if self._adopt_lane_tagging(event) or self._is_stale(event):
            await self._note_stale_event(event)
            return
        event_type = event.get("type")
        if self._needs_drain:
            # The cancelled response's own events, untagged: only its end matters.
            if event_type in _TERMINAL_EVENT_TYPES:
                logger.debug(f"{self}: Cancelled response terminated with {event_type}")
                self._clear_cancellation_state()
                self._lane_busy = False
            return
        if event.get("stream_id") is not None:
            logger.debug(f"{self}: Ignoring {event_type} on the idle lane {self._lane_id}")
            return
        # Not about any lane: a connection-scoped error, for the next loop.
        self._pending_events.append(event)

    # -- connection replacement -----------------------------------------------

    async def _replace_connection(self):
        """Open a new connection, dropping the old one without a closing handshake.

        The old connection may still carry responses we abandoned, and a server
        busy streaming them can take the whole close timeout to answer a close
        frame. Dropping the transport also ends those responses.
        """
        await self._stop_idle_reader()
        websocket = self._websocket
        self._websocket = None
        self._clear_previous_response_state()
        self._clear_cancellation_state()
        self._clear_lane_state()
        if websocket:
            self._abort_connection(websocket)
        await self._try_reconnect(report_error=self._report_error)

    @staticmethod
    def _abort_connection(websocket):
        transport = getattr(websocket, "transport", None)
        if transport is not None:
            transport.abort()

    async def _drain_cancelled_response(self):
        """Drain a cancelled response's events before starting the next one.

        This is the fallback for a server that does not tag events with their
        lane. The cancelled response's in-flight events then cannot be told
        from the next response's, and delta events carry neither a
        ``response_id`` nor any intermediary identifier that could be traced
        back to one, so they are read and discarded until a terminal event
        (``response.completed``, ``response.failed`` or ``response.incomplete``)
        arrives or ``CANCELLED_RESPONSE_DRAIN_SECS`` run out. The server keeps
        generating a cancelled response, so past that budget the connection is
        replaced and the next inference starts on one that carries no events
        from the abandoned response. A dropped connection is left to
        ``_ensure_connected``. An event that does carry a lane tag calls the
        drain off: the response is left on its lane instead.

        The cancelled response is now the connection's latest, so the response
        the next request would otherwise chain from can no longer be continued;
        the ``previous_response_id`` state is cleared and the next request
        sends the full context.
        """
        if not self._websocket:
            self._clear_cancellation_state()
            return

        self._clear_previous_response_state()

        logger.debug(f"{self}: Draining cancelled response events")
        deadline = time.monotonic() + CANCELLED_RESPONSE_DRAIN_SECS
        try:
            while True:
                event = await asyncio.wait_for(
                    self._recv_event(), timeout=max(deadline - time.monotonic(), 0)
                )
                event_type = event.get("type")

                if self._adopt_lane_tagging(event):
                    await self._note_stale_event(event)
                    return

                if event_type in _TERMINAL_EVENT_TYPES:
                    logger.debug(
                        f"{self}: Cancelled response terminated with {event_type} — "
                        f"connection is clean"
                    )
                    self._clear_cancellation_state()
                    self._lane_busy = False
                    return
        except TimeoutError:
            logger.warning(
                f"{self}: Cancelled response still streaming after "
                f"{CANCELLED_RESPONSE_DRAIN_SECS}s — replacing the connection"
            )
            await self._replace_connection()
        except (WebsocketReconnectedError, ConnectionClosed) as e:
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
                self._abandon_response()
                raise
            except Exception as e:
                await self.push_error(error_msg=f"Error during inference: {e}", exception=e)
            finally:
                await self.stop_processing_metrics()
                await self.push_frame(LLMFullResponseEndFrame())
                self._start_idle_reader()
        else:
            await self.push_frame(frame, direction)

    # -- core inference -------------------------------------------------------

    @traced_llm
    async def _process_context(self, context: LLMContext):
        """Run inference over WebSocket with retry and previous_response_id.

        Tries once with the ``previous_response_id`` optimization.  On a
        retriable error (cache miss, connection or lane limit, a request stuck
        behind abandoned responses, a rejected ``stream_id``, connection drop,
        or — when ``retry_on_timeout`` is set — a response that produces no
        output in time), clears state and retries once with the full context
        and no timeout.  Transport-level
        ``ConnectionClosed`` errors are handled transparently by
        ``_ws_send``/``_ws_recv`` (auto-reconnect → ``WebsocketReconnectedError``).

        Args:
            context: The LLM context containing conversation history.
        """
        # This loop owns the socket from here on.
        await self._stop_idle_reader()

        # If a previous response was cancelled on a server that does not tag
        # events, drain its remaining events before starting a new one.
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

        async def send_and_receive(
            *, apply_optimization: bool, output_timeout_secs: float | None = None
        ):
            await self._ensure_connected()
            await self._prepare_lane()
            # Built once the lane is ready: replacing the connection clears the
            # previous_response_id state the optimization reads.
            params = build_params(apply_optimization=apply_optimization)
            if self._lane_id is not None:
                params["stream_id"] = self._lane_id
            await self.start_ttfb_metrics()
            self._lane_busy = True
            await self._ws_send({"type": "response.create", **params})
            await self._receive_response_events(context, full_input, output_timeout_secs)

        async def cleanup():
            self._clear_previous_response_state()
            await self.stop_ttfb_metrics()

        # -- first attempt (with previous_response_id optimization) -----------

        try:
            await send_and_receive(
                apply_optimization=True,
                output_timeout_secs=self._retry_timeout_secs if self._retry_on_timeout else None,
            )
            return  # Success
        except _ResponseTimeoutError:
            # A new connection discards the abandoned response's events, so
            # the retry starts clean with no draining needed.
            logger.warning(
                f"{self}: No output within {self._retry_timeout_secs}s — reconnecting "
                f"and retrying with full context ({len(full_input)} items)"
            )
            await cleanup()
            await self._replace_connection()
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
            await self._replace_connection()
        except _StreamLimitReachedError:
            logger.warning(
                f"{self}: WebSocket lane limit reached — "
                f"reconnecting and retrying with full context ({len(full_input)} items)"
            )
            await cleanup()
            await self._replace_connection()
        except _LaneStalledError as e:
            logger.warning(
                f"{self}: {e} — reconnecting and retrying with full context "
                f"({len(full_input)} items)"
            )
            await cleanup()
            await self._replace_connection()
        except _LaneRejectedError:
            logger.warning(
                f"{self}: Server rejected stream_id — "
                f"retrying without lanes ({len(full_input)} items)"
            )
            await cleanup()
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
            await send_and_receive(apply_optimization=False)
        except Exception:
            await cleanup()
            raise

    async def _receive_response_events(
        self, context: LLMContext, full_input: list, output_timeout_secs: float | None = None
    ):
        """Receive and process WebSocket events until the response completes.

        Events tagged with another lane come from a response that was abandoned
        and are only accounted for. Once the server is known to tag its events,
        an untagged event is not about this request either, unless it is a
        connection-scoped error.

        Args:
            context: The LLM context for the current inference.
            full_input: The complete input items list (for storing state on success).
            output_timeout_secs: How long to wait for the response to start
                producing output before raising ``_ResponseTimeoutError``. Once
                the first output event arrives the wait becomes unbounded, since
                by then content is already on its way downstream and re-issuing
                would duplicate it. None waits indefinitely throughout.

        Raises:
            _PreviousResponseNotFoundError: Server couldn't find previous response.
            _ConnectionLimitReachedError: 60-minute connection limit reached.
            _StreamLimitReachedError: The connection's named lanes are used up.
            _LaneStalledError: The request is queued behind, or unacknowledged
                among, abandoned responses.
            _LaneRejectedError: The server does not accept ``stream_id``.
            _ResponseTimeoutError: Response produced no output in time.
            WebsocketReconnectedError: Connection was lost and auto-recovered.
            ConnectionClosed: Connection was lost and could not be recovered.
        """
        function_calls: dict[str, dict[str, str]] = {}
        current_arguments: dict[str, str] = {}
        reasoning_summary_open = False

        deadline = time.monotonic() + output_timeout_secs if output_timeout_secs else None
        # A request sent while abandoned responses are still in flight has to
        # be acknowledged on its own lane in time, or it is queued behind them.
        ack_deadline = (
            time.monotonic() + LANE_ACK_SECS if self._lane_id and self._abandoned_lanes else None
        )

        while True:
            remaining = [d - time.monotonic() for d in (deadline, ack_deadline) if d is not None]
            if not remaining:
                event = await self._recv_event()
            else:
                try:
                    event = await asyncio.wait_for(
                        self._recv_event(), timeout=max(min(remaining), 0)
                    )
                except TimeoutError:
                    if ack_deadline is not None and time.monotonic() >= ack_deadline:
                        raise _LaneStalledError(
                            f"No event on {self._lane_id} within {LANE_ACK_SECS}s"
                        ) from None
                    raise _ResponseTimeoutError(
                        f"No output within {output_timeout_secs}s"
                    ) from None

            event_type = event.get("type")

            lane = event.get("stream_id")
            if lane is not None:
                self._lane_tagging = True
                if lane != self._lane_id:
                    await self._note_stale_event(event)
                    continue
                ack_deadline = None
            elif event_type == "response.created" and self._lane_id is not None:
                # The request named a lane and the server did not echo it.
                self._lane_tagging = False
            elif (
                self._lane_tagging
                and self._lane_id is not None
                and not (
                    event_type == "error"
                    and event.get("error", {}).get("code") in _CONNECTION_ERROR_CODES
                )
            ):
                logger.debug(f"{self}: Ignoring untagged {event_type}")
                continue

            if event_type == "response.created":
                self._current_response_id = event.get("response", {}).get("id")
                logger.debug(f"{self}: Response started: {self._current_response_id}")
                continue

            if event_type == "response.queued":
                # Past the in-flight cap the server holds the request until an
                # abandoned response finishes; a new connection is quicker.
                raise _LaneStalledError("Request queued behind in-flight responses")

            # Anything past response.created means the response is under way, so
            # the window for abandoning and re-issuing it has closed.
            deadline = None

            if event_type == "response.output_text.delta":
                await self.stop_ttfb_metrics()
                await self._push_llm_text(event.get("delta", ""))

            elif event_type == "response.reasoning_summary_text.delta":
                await self.stop_ttfb_metrics()
                if not reasoning_summary_open:
                    await self.push_frame(LLMThoughtStartFrame())
                    reasoning_summary_open = True
                await self.push_frame(LLMThoughtTextFrame(text=event.get("delta", "")))

            elif event_type == "response.output_item.added":
                await self.stop_ttfb_metrics()
                item = event.get("item", {})
                if item.get("type") == "function_call":
                    # A turn that only calls tools produces no answer text, so the
                    # call itself is what the caller gets and TTFAT ends here
                    # rather than going unmeasured.
                    await self.stop_ttfat_metrics()
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
                elif item.get("type") == "reasoning":
                    if reasoning_summary_open:
                        await self.push_frame(LLMThoughtEndFrame())
                        reasoning_summary_open = False
                    await self._append_reasoning_message(
                        item.get("id"),
                        [
                            {"type": "summary_text", "text": s.get("text", "")}
                            for s in (item.get("summary") or [])
                        ],
                        item.get("encrypted_content"),
                    )

            elif event_type == "response.completed":
                response = event.get("response", {})
                tokens = self._token_usage(response)
                if tokens:
                    await self.start_llm_usage_metrics(tokens)

                self._full_model_name = response.get("model")

                # Store state for next call's previous_response_id optimization.
                # Include the response output so the hash covers the assistant's
                # reply — the server already knows it, so we won't resend it.
                response_id = response.get("id")
                if response_id:
                    response_output = response.get("output") or []
                    self._store_previous_response_state(response_id, full_input, response_output)

                self._lane_busy = False
                break  # Response complete

            elif event_type in ("response.failed", "response.incomplete"):
                response = event.get("response", {})
                status_details = response.get("status_details") or {}
                error_info = status_details.get("error") or {}
                error_msg = error_info.get("message", f"Response {event_type.split('.')[-1]}")
                await self.push_error(error_msg=f"LLM response error: {error_msg}")
                self._lane_busy = False
                break

            elif event_type == "error":
                error = event.get("error", {})
                code = error.get("code", "")
                message = error.get("message", "Unknown error")

                # Whatever the error, the request it answers is over.
                self._lane_busy = False
                if code == "previous_response_not_found":
                    raise _PreviousResponseNotFoundError(message)
                elif code == "websocket_connection_limit_reached":
                    raise _ConnectionLimitReachedError(message)
                elif code == "websocket_stream_limit_reached":
                    raise _StreamLimitReachedError(message)
                elif self._lanes_supported and (
                    error.get("param") == "stream_id" or "stream_id" in message
                ):
                    # A server that does not implement lanes rejects the field.
                    self._lanes_supported = False
                    self._lane_tagging = False
                    self._lane_id = None
                    raise _LaneRejectedError(message)
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
            except TIMEOUT_EXCEPTIONS as e:
                await self._call_event_handler("on_completion_timeout")
                await self.push_error(error_msg="LLM completion timeout", exception=e)
            except Exception as e:
                await self.push_error(error_msg=f"Error during inference: {e}", exception=e)
            finally:
                await self.stop_processing_metrics()
                await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)

    async def _create_stream(self, params: dict) -> AsyncStream[ResponseStreamEvent]:
        """Open a streaming Responses API request, retrying once on timeout.

        Args:
            params: Parameters for the Responses API call.

        Returns:
            Async stream of response events.
        """
        if not self._retry_on_timeout:
            return await self._client.responses.create(**params)

        try:
            return await asyncio.wait_for(
                self._client.responses.create(**params), timeout=self._retry_timeout_secs
            )
        except (TimeoutError, APITimeoutError):
            # Retry, this time without a timeout so we get a response
            logger.debug(f"{self}: Retrying response creation due to timeout")
            return await self._client.responses.create(**params)

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

        stream: AsyncStream[ResponseStreamEvent] = await self._create_stream(params)

        # Track function calls across stream events
        function_calls: dict[str, dict[str, str]] = {}  # item_id -> {name, call_id, arguments}
        current_arguments: dict[str, str] = {}  # item_id -> accumulated arguments
        reasoning_summary_open = False
        stream_errored = False

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

                elif isinstance(event, ResponseReasoningSummaryTextDeltaEvent):
                    await self.stop_ttfb_metrics()
                    if not reasoning_summary_open:
                        await self.push_frame(LLMThoughtStartFrame())
                        reasoning_summary_open = True
                    await self.push_frame(LLMThoughtTextFrame(text=event.delta))

                elif isinstance(event, ResponseOutputItemAddedEvent):
                    await self.stop_ttfb_metrics()
                    item = event.item
                    if isinstance(item, ResponseFunctionToolCall):
                        # A turn that only calls tools produces no answer text, so
                        # the call itself is what the caller gets and TTFAT ends
                        # here rather than going unmeasured.
                        await self.stop_ttfat_metrics()
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
                    elif isinstance(item, ResponseReasoningItem):
                        if reasoning_summary_open:
                            await self.push_frame(LLMThoughtEndFrame())
                            reasoning_summary_open = False
                        await self._append_reasoning_message(
                            item.id,
                            [
                                {"type": "summary_text", "text": s.text}
                                for s in (item.summary or [])
                            ],
                            item.encrypted_content,
                        )

                elif isinstance(event, ResponseCompletedEvent):
                    response = event.response
                    if response.usage:
                        usage = response.usage
                        # Third-party Responses API servers may omit fields the
                        # OpenAI SDK treats as required. The SDK's lenient
                        # streaming decoder leaves anything omitted as None — at
                        # the top level (token counts), as a missing detail
                        # sub-object, or as a missing field inside one. Coalesce
                        # each to 0 so a partial usage payload can't raise or
                        # leak None into metrics.
                        input_details = usage.input_tokens_details
                        output_details = usage.output_tokens_details
                        tokens = LLMTokenUsage(
                            prompt_tokens=usage.input_tokens or 0,
                            completion_tokens=usage.output_tokens or 0,
                            total_tokens=usage.total_tokens or 0,
                            cache_read_input_tokens=(input_details.cached_tokens or 0)
                            if input_details
                            else 0,
                            cache_creation_input_tokens=(
                                getattr(input_details, "cache_write_tokens", None) or 0
                            )
                            if input_details
                            else 0,
                            reasoning_tokens=(output_details.reasoning_tokens or 0)
                            if output_details
                            else 0,
                        )
                        await self.start_llm_usage_metrics(tokens)

                    # This field is used by @traced_llm for more detailed
                    # model name in tracing spans
                    self._full_model_name = response.model

                elif isinstance(event, ResponseFailedEvent):
                    # As with usage above, the detail objects and their fields
                    # are only as reliable as the server; coalesce to a generic
                    # message so a sparse payload still reports something.
                    error = event.response.error
                    message = error.message if error else None
                    await self.push_error(
                        error_msg=f"LLM response error: {message or 'Response failed'}"
                    )
                    stream_errored = True
                    break

                elif isinstance(event, ResponseIncompleteEvent):
                    details = event.response.incomplete_details
                    reason = details.reason if details else None
                    await self.push_error(
                        error_msg=f"LLM response error: {reason or 'Response incomplete'}"
                    )
                    stream_errored = True
                    break

                elif isinstance(event, ResponseErrorEvent):
                    await self.push_error(error_msg=f"Responses API error: {event.message}")
                    stream_errored = True
                    break

        # A stream that ended in a terminal error may have announced a function
        # call whose arguments never finished streaming — drop those rather than
        # run them with fabricated empty arguments. `arguments` is only written
        # by the Done events, so a non-empty string means the call completed
        # (e.g. parallel tool calls finished before a later item was truncated)
        # and it still runs.
        if stream_errored:
            function_calls = {
                item_id: call for item_id, call in function_calls.items() if call["arguments"]
            }

        # Process any function calls
        if function_calls:
            fc_list = self._process_function_calls(context, function_calls)
            await self.run_function_calls(fc_list)


__all__ = [
    "OpenAIResponsesLLMService",
    "OpenAIResponsesHttpLLMService",
    "OpenAIResponsesLLMSettings",
    "OpenAIResponsesReasoningConfig",
]
