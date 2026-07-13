#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base LLM service implementation for services that use the AsyncOpenAI client."""

import ast
import asyncio
import json
import re
from collections.abc import Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
from loguru import logger
from openai import (
    NOT_GIVEN,
    APITimeoutError,
    AsyncOpenAI,
    AsyncStream,
    DefaultAsyncHttpxClient,
)
from openai._types import NotGiven as OpenAINotGiven
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel, Field

from pipecat.adapters.services.open_ai_adapter import OpenAILLMAdapter, OpenAILLMInvocationParams
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    Frame,
    FunctionCallsFromLLMInfoFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.services.settings import NOT_GIVEN as _NOT_GIVEN
from pipecat.services.settings import LLMSettings, _NotGiven, assert_given
from pipecat.utils.deprecation import deprecated
from pipecat.utils.tracing.service_decorators import traced_llm


# Longest wait for the next streamed completion chunk before the generation
# is abandoned as wedged. Generous vs. healthy inter-chunk gaps (tens of ms;
# first chunk = prefill, seconds) but bounded, so a server that accepts the
# request and then never streams cannot silently freeze the turn.
STREAM_CHUNK_TIMEOUT_SECS = 15

# Longest wait for closing a completion stream before abandoning the
# connection. Cleanup over a dead socket can otherwise hang un-cancellably.
STREAM_CLEANUP_TIMEOUT_SECS = 2

# How many times to re-run a generation that came back empty (no text, no tool
# call) before giving up to idle handling. One retry was too few for a model
# that empties transiently (Gemma-4 stalled a live turn after a single failed
# retry); each retry is a fresh attempt and only fires on an otherwise-dead turn.
MAX_EMPTY_COMPLETION_RETRIES = 3


def _repair_truncated_json(text) -> Optional[dict]:
    """Best-effort repair of a JSON object cut off mid-generation.

    Closes an unterminated string and any unclosed braces, then re-parses.
    Returns the parsed dict, or None when the text is not a repairable
    object (wrong shape, or still invalid after closing).
    """
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s.startswith("{"):
        return None
    # Drop a dangling escape at the very end ('\' with nothing after it).
    if s.endswith("\\"):
        s = s[:-1]
    # Close an unterminated string: count unescaped quotes.
    quotes = len(re.findall(r'(?<!\\)"', s))
    if quotes % 2 == 1:
        s += '"'
    # Trim a trailing comma or colon left hanging by the cut.
    s = re.sub(r"[,:]\s*$", "", s)
    # A key without a value ("key" at object level) can't be salvaged —
    # drop it back to the previous comma/brace.
    s += "}" * max(0, s.count("{") - s.count("}"))
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        # Last resort: drop the final (possibly half-written) member.
        s2 = re.sub(r',\s*"[^"]*"?\s*:?\s*("[^"]*"?)?\s*}\s*$', "}", s)
        try:
            parsed = json.loads(s2)
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _recover_leaked_tool_call(content: str, known_names):
    """Recover a tool call the model emitted as plain text.

    Some models (notably Gemma-3 served by vLLM's ``pythonic`` tool parser)
    cannot emit prose and a structured ``tool_calls`` delta in the same
    generation, so they fall back to writing the call into the assistant
    *content* as ``[func(arg=...)]`` or bare ``func(arg=...)``. When that
    happens no structured tool call is produced and the call is never executed.

    This scans ``content`` for a call to one of the registered ``known_names``,
    balance-matching parentheses (so nested lists/dicts are handled) and parsing
    the keyword arguments with :func:`ast.literal_eval`. It is deliberately
    strict — it only matches a *known* tool name immediately followed by a
    balanced ``(...)`` whose kwargs are pure literals — so ordinary text (even
    Hebrew prose containing parentheses) will not false-positive.

    Args:
        content: The accumulated assistant content from the stream.
        known_names: Iterable of registered tool names to look for.

    Returns:
        ``(name, args_dict, start, end)`` for the matched span, where ``start``
        and ``end`` bound the call text (including any wrapping ``[`` / ``]``),
        or ``None`` if no recognizable call is found.
    """
    if not content or not known_names:
        return None
    for name in known_names:
        idx = 0
        while True:
            pos = content.find(name, idx)
            if pos == -1:
                break
            idx = pos + len(name)
            # Require a standalone identifier (not a substring of a larger word).
            if pos > 0 and (content[pos - 1].isalnum() or content[pos - 1] == "_"):
                continue
            # The next non-space character must open the argument list.
            j = idx
            while j < len(content) and content[j] == " ":
                j += 1
            if j >= len(content) or content[j] != "(":
                continue
            # Balance parentheses, respecting string literals.
            depth = 0
            k = j
            in_str = None
            while k < len(content):
                c = content[k]
                if in_str:
                    if c == in_str and content[k - 1] != "\\":
                        in_str = None
                elif c in "\"'":
                    in_str = c
                elif c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            if k >= len(content) or depth != 0:
                continue
            call_src = content[pos : k + 1]
            try:
                node = ast.parse(call_src, mode="eval").body
            except SyntaxError:
                continue
            if not isinstance(node, ast.Call):
                continue
            try:
                args = {
                    kw.arg: ast.literal_eval(kw.value)
                    for kw in node.keywords
                    if kw.arg is not None
                }
            except (ValueError, SyntaxError):
                continue
            start, end = pos, k + 1
            # Absorb a quote wrapper (`['func(...)']`) so the leftover `['`/`']`
            # never masks the real tail of any co-spoken prose.
            if (
                start > 0
                and content[start - 1] in "\"'"
                and end < len(content)
                and content[end] == content[start - 1]
            ):
                start -= 1
                end += 1
            # Absorb the pythonic-list wrapper `[ ... ]` if present.
            if start > 0 and content[start - 1] == "[":
                start -= 1
            if end < len(content) and content[end] == "]":
                end += 1
            return name, args, start, end
    # Degenerate list forms with no parsable call: the model names the tool as
    # a (possibly quoted) list item — `[func]`, `['func']`, `["func()"]`. No
    # arguments survive that form, but the intent is unambiguous for a
    # registered name, and leaving it as text means it is stripped from speech
    # and the call silently dropped.
    for name in known_names:
        m = re.search(
            r"\[\s*(['\"]?)" + re.escape(name) + r"(?:\s*\(\s*\))?\1\s*\]",
            content,
        )
        if m:
            return name, {}, m.start(), m.end()
    return None


@dataclass
class OpenAILLMSettings(LLMSettings):
    """Settings for BaseOpenAILLMService.

    Parameters:
        max_completion_tokens: Maximum completion tokens to generate.
    """

    # Override inherited LLMSettings fields to also accept openai's NotGiven
    # sentinel. The service stores openai's NOT_GIVEN in these fields so they
    # can be passed through unchanged to the AsyncOpenAI client.
    frequency_penalty: float | None | _NotGiven | OpenAINotGiven = field(
        default_factory=lambda: _NOT_GIVEN
    )
    presence_penalty: float | None | _NotGiven | OpenAINotGiven = field(
        default_factory=lambda: _NOT_GIVEN
    )
    seed: int | None | _NotGiven | OpenAINotGiven = field(default_factory=lambda: _NOT_GIVEN)
    temperature: float | None | _NotGiven | OpenAINotGiven = field(
        default_factory=lambda: _NOT_GIVEN
    )
    top_p: float | None | _NotGiven | OpenAINotGiven = field(default_factory=lambda: _NOT_GIVEN)
    max_tokens: int | None | _NotGiven | OpenAINotGiven = field(default_factory=lambda: _NOT_GIVEN)
    max_completion_tokens: int | None | _NotGiven | OpenAINotGiven = field(
        default_factory=lambda: _NOT_GIVEN
    )


class BaseOpenAILLMService(LLMService[OpenAILLMAdapter]):
    """Base class for all services that use the AsyncOpenAI client.

    This service consumes LLMContextFrame frames, which contain a reference to
    an LLMContext object. The context defines what is sent to the LLM for
    completion, including user, assistant, and system messages, as well as tool
    choices and function call configurations.
    """

    Settings = OpenAILLMSettings
    _settings: Settings

    supports_developer_role: bool = True
    """Whether this service's API supports the "developer" message role.

    OpenAI's native API supports it, but some OpenAI-compatible services
    (e.g. Cerebras) do not. Subclasses that don't support it should set
    this to ``False``, which causes the adapter to convert "developer"
    messages to "user" messages before sending them to the API.
    """

    @deprecated(
        "`BaseOpenAILLMService.InputParams` is deprecated since 0.0.105 and will be removed in "
        "2.0.0. Use `BaseOpenAILLMService.Settings` instead."
    )
    class InputParams(BaseModel):
        """Input parameters for OpenAI model configuration.

        .. deprecated:: 0.0.105
            Use ``settings=BaseOpenAILLMService.Settings(...)`` instead of
            ``params=InputParams(...)``.
            Will be removed in 2.0.0.

        Parameters:
            frequency_penalty: Penalty for frequent tokens (-2.0 to 2.0).
            presence_penalty: Penalty for new tokens (-2.0 to 2.0).
            seed: Random seed for deterministic outputs.
            temperature: Sampling temperature (0.0 to 2.0).
            top_k: Top-k sampling parameter (currently ignored by OpenAI).
            top_p: Top-p (nucleus) sampling parameter (0.0 to 1.0).
            max_tokens: Maximum tokens in response (deprecated, use max_completion_tokens).
            max_completion_tokens: Maximum completion tokens to generate.
            service_tier: Service tier to use (e.g., "auto", "flex", "priority").
            extra: Additional model-specific parameters.
        """

        frequency_penalty: float | None = Field(default_factory=lambda: NOT_GIVEN, ge=-2.0, le=2.0)
        presence_penalty: float | None = Field(default_factory=lambda: NOT_GIVEN, ge=-2.0, le=2.0)
        seed: int | None = Field(default_factory=lambda: NOT_GIVEN, ge=0)
        temperature: float | None = Field(default_factory=lambda: NOT_GIVEN, ge=0.0, le=2.0)
        # Note: top_k is currently not supported by the OpenAI client library,
        # so top_k is ignored right now.
        top_k: int | None = Field(default=None, ge=0)
        top_p: float | None = Field(default_factory=lambda: NOT_GIVEN, ge=0.0, le=1.0)
        max_tokens: int | None = Field(default_factory=lambda: NOT_GIVEN, ge=1)
        max_completion_tokens: int | None = Field(default_factory=lambda: NOT_GIVEN, ge=1)
        service_tier: str | None = Field(default_factory=lambda: NOT_GIVEN)
        extra: dict[str, Any] | None = Field(default_factory=dict)

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key=None,
        base_url=None,
        organization=None,
        project=None,
        default_headers: Mapping[str, str] | None = None,
        service_tier: str | None = None,
        params: InputParams | None = None,
        settings: Settings | None = None,
        retry_timeout_secs: float | None = 5.0,
        retry_on_timeout: bool | None = False,
        **kwargs,
    ):
        """Initialize the BaseOpenAILLMService.

        Args:
            model: The OpenAI model name to use (e.g., "gpt-4.1", "gpt-4o").

                .. deprecated:: 0.0.105
                    Use ``settings=BaseOpenAILLMService.Settings(model=...)`` instead.
                    Will be removed in 2.0.0.

            api_key: OpenAI API key. If None, uses environment variable.
            base_url: Custom base URL for OpenAI API. If None, uses default.
            organization: OpenAI organization ID.
            project: OpenAI project ID.
            default_headers: Additional HTTP headers to include in requests.
            service_tier: Service tier to use (e.g., "auto", "flex", "priority").
            params: Input parameters for model configuration and behavior.

                .. deprecated:: 0.0.105
                    Use ``settings=BaseOpenAILLMService.Settings(...)`` instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            retry_timeout_secs: Request timeout in seconds. Defaults to 5.0 seconds.
            retry_on_timeout: Whether to retry the request once if it times out.
            **kwargs: Additional arguments passed to the parent LLMService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="gpt-4.1",
            system_instruction=None,
            frequency_penalty=NOT_GIVEN,
            presence_penalty=NOT_GIVEN,
            seed=NOT_GIVEN,
            temperature=NOT_GIVEN,
            top_p=NOT_GIVEN,
            top_k=None,
            max_tokens=NOT_GIVEN,
            max_completion_tokens=NOT_GIVEN,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
            extra={},
        )

        # 2. Apply direct init arg overrides (no warnings in base class)
        if model is not None:
            default_settings.model = model

        # 3. Apply params overrides — only if settings not provided
        if params is not None and not settings:
            default_settings.frequency_penalty = params.frequency_penalty
            default_settings.presence_penalty = params.presence_penalty
            default_settings.seed = params.seed
            default_settings.temperature = params.temperature
            default_settings.top_p = params.top_p
            default_settings.max_tokens = params.max_tokens
            default_settings.max_completion_tokens = params.max_completion_tokens
            if isinstance(params.extra, dict):
                default_settings.extra = params.extra

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            settings=default_settings,
            **kwargs,
        )
        self._service_tier = service_tier
        self._retry_timeout_secs = retry_timeout_secs
        self._retry_on_timeout = retry_on_timeout
        self._full_model_name: str = ""
        self._client = self.create_client(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            project=project,
            default_headers=default_headers,
            **kwargs,
        )

        if self._settings.system_instruction:
            logger.debug(f"{self}: Using system instruction: {self._settings.system_instruction}")

        # Store pending function calls that need to be executed after TTS
        self._pending_function_calls = []

        # --- Tool-call loop guard ---------------------------------------
        # A failing/empty tool call (e.g. web_search with no query → Tavily
        # 400) gets its error result written back to the context, which
        # re-prompts the LLM, which can emit the same bad call again — an
        # infinite tool-call loop that freezes the turn. We cap how many
        # *consecutive* tool-call rounds a single user turn may trigger.
        # Once the cap is hit, get_chat_completions() strips tools/tool_choice
        # for the follow-up completion, forcing a plain text answer that can
        # never loop. The counter resets on each new user turn.
        self._max_tool_call_rounds_per_turn = 2
        self._tool_call_rounds_this_turn = 0

    def create_client(
        self,
        api_key=None,
        base_url=None,
        organization=None,
        project=None,
        default_headers=None,
        **kwargs,
    ):
        """Create an AsyncOpenAI client instance.

        Args:
            api_key: OpenAI API key.
            base_url: Custom base URL for the API.
            organization: OpenAI organization ID.
            project: OpenAI project ID.
            default_headers: Additional HTTP headers.
            **kwargs: Additional client configuration arguments.

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
        """Check if this service can generate processing metrics.

        Returns:
            True, as OpenAI service supports metrics generation.
        """
        return True

    def set_full_model_name(self, full_model_name: str):
        """Set the full AI model name.

        Args:
            full_model_name: The full name of the AI model to use.
        """
        self._full_model_name = full_model_name

    def get_full_model_name(self):
        """Get the current full model name.

        Returns:
            The full name of the AI model being used.
        """
        return self._full_model_name

    async def get_chat_completions(self, context: LLMContext) -> AsyncStream[ChatCompletionChunk]:
        """Get streaming chat completions from OpenAI API with optional timeout and retry.

        Args:
            context: Context to use for the chat completion.
                Contains messages, tools, and tool choice.

        Returns:
            Async stream of chat completion chunks.
        """
        adapter = self.get_llm_adapter()
        logger.debug(
            f"{self}: Generating chat from context {adapter.get_messages_for_logging(context)}"
        )

        params_from_context = adapter.get_llm_invocation_params(
            context,
            system_instruction=assert_given(self._settings.system_instruction),
            convert_developer_to_user=not self.supports_developer_role,
        )

        params = self.build_chat_completion_params(params_from_context)

        # Force-tool only on the FIRST completion of a turn. tool_choice="required"
        # (a per-node routing backstop) lives in settings.extra, so it persists
        # across re-prompts — meaning the follow-up completion *after* a tool
        # result is ALSO compelled to call a tool, and so on, spinning a routing
        # node through 3 dispatches/turn (set_caller_gender / transitions re-fired
        # every round → huge latency + stuck call). Once this turn has already
        # produced one tool call, downgrade "required" → "auto" so the follow-up
        # can answer in text and END the turn. The first completion still honors
        # "required" (the forced transition), so reliability is unchanged.
        if self._tool_call_rounds_this_turn >= 1 and params.get("tool_choice") == "required":
            params["tool_choice"] = "auto"

        # Tool-call loop guard: once a single user turn has triggered the max
        # number of consecutive tool-call rounds, disable tools for the
        # follow-up completion so the model is forced to answer in plain text.
        # This guarantees a failing/empty tool (e.g. web_search with no query)
        # can never spin in an infinite re-prompt loop.
        if self._tool_call_rounds_this_turn >= self._max_tool_call_rounds_per_turn:
            if params.get("tools"):
                logger.warning(
                    f"{self}: tool-call loop guard hit "
                    f"({self._tool_call_rounds_this_turn} consecutive rounds this "
                    f"turn); disabling tools for this completion to force a text answer"
                )
            params.pop("tools", None)
            params.pop("tool_choice", None)

        # An empty completion with tools enabled is almost always the server's
        # tool parser swallowing a malformed textual call (vLLM pythonic buffers
        # `[`-prefixed output, fails to parse e.g. `['func()']`, and streams
        # NOTHING — run 636's 11.7s dead turn). Re-sending the identical request
        # just reproduces the swallow, so the retry drops tools: prose comes
        # through, and a call the model still leaks as text is executed by
        # _recover_leaked_tool_call. tool_choice="required" (routing nodes) is
        # exempt — those force the structured path server-side and need the call.
        if (
            getattr(self, "_empty_retry_depth", 0) > 0
            and params.get("tools")
            and params.get("tool_choice") != "required"
        ):
            logger.warning(
                f"{self}: retrying an empty completion without tools so the "
                f"server tool parser cannot swallow the output again"
            )
            params.pop("tools", None)
            params.pop("tool_choice", None)

        if self._retry_on_timeout:
            try:
                chunks = await asyncio.wait_for(
                    self._client.chat.completions.create(**params), timeout=self._retry_timeout_secs
                )
                return chunks
            except (TimeoutError, APITimeoutError):
                # Retry, this time without a timeout so we get a response
                logger.debug(f"{self}: Retrying chat completion due to timeout")
                chunks = await self._client.chat.completions.create(**params)
                return chunks
        else:
            chunks = await self._client.chat.completions.create(**params)
            return chunks

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Build parameters for chat completion request.

        Subclasses can override this to customize parameters for different providers.

        Args:
            params_from_context: Parameters, derived from the LLM context, to
                use for the chat completion. Contains messages, tools, and tool
                choice.

        Returns:
            Dictionary of parameters for the chat completion request.
        """
        params = {
            "model": self._settings.model,
            "stream": True,
            "stream_options": {"include_usage": True},
            "frequency_penalty": self._settings.frequency_penalty,
            "presence_penalty": self._settings.presence_penalty,
            "seed": self._settings.seed,
            "temperature": self._settings.temperature,
            "top_p": self._settings.top_p,
            "max_tokens": self._settings.max_tokens,
            "max_completion_tokens": self._settings.max_completion_tokens,
            "service_tier": self._service_tier if self._service_tier is not None else NOT_GIVEN,
        }

        # Messages, tools, tool_choice
        params.update(params_from_context)

        params.update(self._settings.extra)

        return params

    async def run_inference(
        self,
        context: LLMContext,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
    ) -> str | None:
        """Run a one-shot, out-of-band (i.e. out-of-pipeline) inference with the given LLM context.

        Args:
            context: The LLM context containing conversation history.
            max_tokens: Optional maximum number of tokens to generate. If provided,
                overrides the service's default max_tokens/max_completion_tokens setting.
            system_instruction: Optional system instruction to use for this inference.
                If provided, overrides any system instruction in the context.

        Returns:
            The LLM's response as a string, or None if no response is generated.
        """
        effective_instruction = system_instruction or self._settings.system_instruction
        adapter = self.get_llm_adapter()
        invocation_params = adapter.get_llm_invocation_params(
            context,
            system_instruction=effective_instruction,
            convert_developer_to_user=not self.supports_developer_role,
        )

        # Build params using the same method as streaming completions
        params = self.build_chat_completion_params(invocation_params)

        # Override for non-streaming
        params["stream"] = False
        params.pop("stream_options", None)

        # Override max_tokens if provided
        if max_tokens is not None:
            # Use max_completion_tokens for newer models, fallback to max_tokens
            if "max_completion_tokens" in params:
                params["max_completion_tokens"] = max_tokens
            else:
                params["max_tokens"] = max_tokens

        # LLM completion
        response = await self._client.chat.completions.create(**params)

        return response.choices[0].message.content

    @traced_llm
    async def _process_context(self, context: LLMContext):
        functions_list = []
        arguments_list = []
        tool_id_list = []
        func_idx = 0
        function_name = ""
        arguments = ""
        tool_call_id = ""
        # Accumulate streamed content so we can recover a tool call the model
        # emitted as plain text instead of a structured tool_calls delta.
        content_buffer = ""

        # Reset pending function calls when processing a new context
        self._pending_function_calls = []

        # Flag to store whether some text was generated in the current generation
        text_generated_signal = False

        await self.start_ttfb_metrics()

        # Generate chat completions from LLMContext
        chunk_stream = await self.get_chat_completions(context)

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
                # Bounded: this finally also runs on cancellation, and over a
                # wedged connection aclose() itself can hang forever — which
                # makes the task un-cancellable, blocks _start_interruption's
                # inline cancel, and freezes the whole pipeline (run 350).
                # Past the bound the socket is leaked; process teardown reaps.
                async def _cleanup():
                    # Close the iterator first to cascade cleanup through
                    # nested async generators (httpx/httpcore internals).
                    if hasattr(chunk_iter, "aclose"):
                        await chunk_iter.aclose()
                    # Then close the stream to release HTTP resources.
                    if hasattr(stream, "close"):
                        await stream.close()
                    elif hasattr(stream, "aclose"):
                        await stream.aclose()

                try:
                    await asyncio.wait_for(_cleanup(), STREAM_CLEANUP_TIMEOUT_SECS)
                except TimeoutError:
                    logger.warning(
                        f"{self}: stream cleanup timed out after "
                        f"{STREAM_CLEANUP_TIMEOUT_SECS}s — abandoning the connection"
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning(f"{self}: stream cleanup failed: {e}")

        async def _timeout_guarded(it):
            # A server that accepts the request but then stops streaming (the
            # response never produces another chunk and never terminates)
            # would otherwise hang this generation forever: heartbeats queue
            # behind it, the pipeline stalls, and the call goes silent until
            # hangup (run 350). Bound the wait per chunk and treat a timeout
            # as end-of-stream — if nothing was generated, the empty-completion
            # retry below re-runs the turn on a fresh connection.
            while True:
                try:
                    yield await asyncio.wait_for(it.__anext__(), STREAM_CHUNK_TIMEOUT_SECS)
                except StopAsyncIteration:
                    return
                except TimeoutError:
                    logger.warning(
                        f"{self}: no stream chunk for {STREAM_CHUNK_TIMEOUT_SECS}s — "
                        f"server stopped streaming mid-response; treating as end of stream"
                    )
                    return

        async with _closing(chunk_stream) as chunk_iter:
            async for chunk in _timeout_guarded(chunk_iter):
                if chunk.usage:
                    cached_tokens = (
                        chunk.usage.prompt_tokens_details.cached_tokens
                        if chunk.usage.prompt_tokens_details
                        else None
                    )
                    reasoning_tokens = (
                        chunk.usage.completion_tokens_details.reasoning_tokens
                        if chunk.usage.completion_tokens_details
                        else None
                    )
                    tokens = LLMTokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                        cache_read_input_tokens=cached_tokens,
                        reasoning_tokens=reasoning_tokens,
                    )
                    await self.start_llm_usage_metrics(tokens)

                if chunk.model and self.get_full_model_name() != chunk.model:
                    self.set_full_model_name(chunk.model)

                if chunk.choices is None or len(chunk.choices) == 0:
                    continue

                await self.stop_ttfb_metrics()

                if not chunk.choices[0].delta:
                    continue

                if chunk.choices[0].delta.tool_calls:
                    # We're streaming the LLM response to enable the fastest response times.
                    # For text, we just yield each chunk as we receive it and count on consumers
                    # to do whatever coalescing they need (eg. to pass full sentences to TTS)
                    #
                    # If the LLM is a function call, we'll do some coalescing here.
                    # If the response contains a function name, we'll yield a frame to tell consumers
                    # that they can start preparing to call the function with that name.
                    # We accumulate all the arguments for the rest of the streamed response, then when
                    # the response is done, we package up all the arguments and the function name and
                    # yield a frame containing the function name and the arguments.

                    tool_call = chunk.choices[0].delta.tool_calls[0]
                    if tool_call.index != func_idx:
                        functions_list.append(function_name)
                        arguments_list.append(arguments or "{}")
                        tool_id_list.append(tool_call_id)
                        function_name = ""
                        arguments = ""
                        tool_call_id = ""
                        func_idx += 1
                    if tool_call.function and tool_call.function.name:
                        function_name += tool_call.function.name
                        tool_call_id = tool_call.id
                    if tool_call.function and tool_call.function.arguments:
                        # Keep iterating through the response to collect all the argument fragments
                        arguments += tool_call.function.arguments
                elif chunk.choices[0].delta.content:
                    text_generated_signal = True
                    content_buffer += chunk.choices[0].delta.content
                    await self._push_llm_text(chunk.choices[0].delta.content)

                # When gpt-4o-audio / gpt-4o-mini-audio is used for llm or stt+llm
                # we need to get LLMTextFrame for the transcript
                elif (
                    hasattr(chunk.choices[0].delta, "audio")
                    and chunk.choices[0].delta.audio
                    and chunk.choices[0].delta.audio.get("transcript")
                ):
                    await self.push_frame(LLMTextFrame(chunk.choices[0].delta.audio["transcript"]))

        # Recover a tool call the model emitted as plain text instead of a
        # structured tool_calls delta (e.g. Gemma-3 via vLLM's pythonic parser
        # writes `[func(args)]` into the content). Only fires when no structured
        # call was produced AND the content contains a call to a *registered*
        # tool, so the normal path is untouched. The recovered call is fed into
        # the same variables the block below consumes, reusing all of its
        # tested defer/run logic.
        #
        # Respect the per-turn loop guard: once it has tripped, get_chat_completions
        # strips `tools` from the request to force a plain-text answer — but a model
        # like Gemma keeps LEAKING the call as text regardless, and recovering it
        # would re-execute a call that keeps getting rejected (missing-slot /
        # waiting-for-user / stray transition), spinning the exact infinite loop the
        # guard exists to stop. So when the guard is active, do NOT recover: let the
        # leaked syntax stay plain text (stripped from speech downstream) and end the
        # turn on the model's words.
        loop_guard_tripped = (
            self._tool_call_rounds_this_turn >= self._max_tool_call_rounds_per_turn
        )
        # EMPTY-GENERATION RETRY: no text and no tool call = a dead turn — the
        # caller hears silence and unanswered messages pile up (two consecutive
        # empty completions swallowed a booking slot and a phone number on a live
        # flow; wf15 run 1893 stalled at the party node after ONE empty + one
        # failed retry). A single retry is not enough for a model that empties
        # transiently (Gemma-4): retry up to MAX_EMPTY_COMPLETION_RETRIES, each a
        # fresh attempt on the same context, before giving up to idle handling.
        # Healthy turns never reach here, so this adds latency only to a turn that
        # would otherwise have gone silent.
        if (
            not function_name
            and not content_buffer.strip()
            and getattr(self, "_empty_retry_depth", 0) < MAX_EMPTY_COMPLETION_RETRIES
        ):
            self._empty_retry_depth = getattr(self, "_empty_retry_depth", 0) + 1
            try:
                logger.warning(
                    f"{self}: empty completion (no text, no tool calls) — "
                    f"retry {self._empty_retry_depth}/{MAX_EMPTY_COMPLETION_RETRIES}"
                )
                await self._process_context(context)
            finally:
                self._empty_retry_depth -= 1
            return

        # What the caller actually HEARS this turn. For a recovered leaked call
        # this is the buffer minus the call syntax — the raw buffer ends with
        # "[transition_to_x()]", which masks the real tail of the spoken reply
        # (a trailing "?" that downstream open-question guards key on).
        spoken_text = content_buffer
        if not function_name and content_buffer and not loop_guard_tripped:
            recovered = _recover_leaked_tool_call(
                content_buffer, [n for n in self._functions.keys() if n]
            )
            if recovered:
                rec_name, rec_args, rec_start, rec_end = recovered
                function_name = rec_name
                arguments = json.dumps(rec_args)
                tool_call_id = f"recovered_{func_idx}"
                logger.warning(
                    f"{self}: recovered tool call '{rec_name}' emitted as plain "
                    f"text (structured tool_calls was empty)"
                )
                # If the call was the only SPEAKABLE content, downstream stripping
                # leaves nothing to play — so no BotStoppedSpeakingFrame will fire
                # to flush a deferred call and the transition would be silently
                # dropped at the next generation. Judge the remainder's
                # speakability, not mere non-emptiness: a markdown fence /
                # "tool_code" label / stray punctuation around the recovered call
                # all strip to silence downstream (run 96 fence precedent).
                remainder = (content_buffer[:rec_start] + content_buffer[rec_end:]).strip()
                # Fence/label residue around the recovered call is never
                # spoken — exclude it from the spoken-text signal too, or a
                # trailing ``` masks the reply's real tail (the engine's
                # open-question end guard keys on a trailing "?").
                spoken_text = re.sub(
                    r"`{2,}[ \t]*[A-Za-z_][\w-]*|`{2,}|\btool_code\b", "", remainder
                ).strip()
                speakable = re.sub(r"[\s.,;:!?\-–—'\"()\[\]]+", "", spoken_text)
                if not speakable:
                    text_generated_signal = False

        # if we got a function name and arguments, check to see if it's a function with
        # a registered handler. If so, run the registered callback, save the result to
        # the context, and re-prompt to get a chat answer. If we don't have a registered
        # handler, raise an exception.
        if function_name:
            # This completion produced at least one tool call: count it as one
            # round against the per-turn loop guard. The next completion (the
            # re-prompt after the tool result) will have tools stripped once the
            # cap is reached (see get_chat_completions).
            self._tool_call_rounds_this_turn += 1

            # added to the list as last function name and arguments not added to the list
            functions_list.append(function_name)
            arguments_list.append(arguments or "{}")
            tool_id_list.append(tool_call_id)

            function_calls = []

            for function_name, arguments, tool_id in zip(
                functions_list, arguments_list, tool_id_list
            ):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    # A generation that hits its token limit mid-arguments
                    # leaves unterminated JSON. Repair (close the dangling
                    # string/braces) and retry — a partially captured call
                    # beats a dropped one, and blank fields are validated by
                    # the application's own handlers.
                    repaired = _repair_truncated_json(arguments)
                    if repaired is None:
                        logger.warning(
                            f"{self}: Failed to parse function call arguments: {arguments}"
                        )
                        continue
                    logger.warning(
                        f"{self}: repaired truncated function call arguments: "
                        f"{str(arguments)[:80]!r}"
                    )
                    arguments = repaired

                # Drop a malformed web_search call with an empty/missing query
                # before it can run (Tavily would 400 and the error result would
                # re-prompt the model into the same bad call). Skipping it here
                # lets the model answer in plain text instead of looping.
                if function_name == "web_search" and not str(
                    (arguments or {}).get("query", "")
                ).strip():
                    logger.warning(
                        f"{self}: dropping web_search call with empty/missing 'query' "
                        f"argument (malformed); model must answer without it"
                    )
                    continue

                function_calls.append(
                    FunctionCallFromLLM(
                        context=context,
                        tool_call_id=tool_id,
                        function_name=function_name,
                        arguments=arguments,
                    )
                )

            # Every collected call was dropped (e.g. only a malformed
            # web_search). Nothing to run or defer — let the turn proceed with
            # whatever text was generated.
            if not function_calls:
                return

            # Send the info frame with function calls so that it can be traced by service_decorators
            await self.push_frame(
                FunctionCallsFromLLMInfoFrame(function_calls=function_calls),
                direction=FrameDirection.DOWNSTREAM,
            )

            # Reliable co-emission signal for downstream consumers (the workflow
            # engine's destination-generation suppression). Reading it here — the
            # moment we know whether THIS generation produced spoken prose — is
            # race-free, unlike inferring it from the aggregated assistant message
            # in the shared context, which on a real (WebRTC) call is often not yet
            # committed when the deferred transition handler runs (→ the engine
            # sees "silent", lets the destination node generate, and the caller
            # hears TWO replies for one turn). True only for real, non-whitespace
            # prose, matching the defer condition below.
            self._last_generation_had_text = bool(
                text_generated_signal and content_buffer.strip()
            )
            # The words themselves, for guards that need more than the bool —
            # e.g. the workflow engine's open-question end guard reads the
            # trailing "?" of the co-spoken reply. Reading the aggregated
            # assistant message from the shared context has the same race as
            # above (not yet committed when a deferred handler runs).
            self._last_generation_text = (
                spoken_text.strip() if self._last_generation_had_text else ""
            )

            # If text was generated, defer function calls until after TTS plays
            # Otherwise, execute them immediately. Whitespace-only content never
            # reaches TTS, so no BotStoppedSpeakingFrame would ever flush the
            # deferred calls — treat it as no text and run them now.
            if text_generated_signal and content_buffer.strip():
                self._pending_function_calls = function_calls
                logger.debug(
                    f"{self}: Deferring {len(function_calls)} function calls until after TTS"
                )
            else:
                logger.debug(f"{self}: Executing {len(function_calls)} function calls")
                await self.run_function_calls(function_calls)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for LLM completion requests.

        Handles LLMContextFrame to trigger LLM completions.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        # A new user turn resets the tool-call loop guard: the user has spoken
        # again, so the per-turn budget of consecutive tool-call rounds starts
        # fresh.
        if isinstance(frame, UserStartedSpeakingFrame):
            self._tool_call_rounds_this_turn = 0

        # Handle BotStoppedSpeakingFrame to execute pending function calls
        if isinstance(frame, BotStoppedSpeakingFrame):
            if self._pending_function_calls:
                logger.debug(
                    f"{self}: Executing {len(self._pending_function_calls)} deferred function calls after TTS"
                )
                await self.run_function_calls(self._pending_function_calls)
                self._pending_function_calls = []
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMContextFrame):
            try:
                await self.push_frame(LLMFullResponseStartFrame())
                await self.start_processing_metrics()
                await self._process_context(frame.context)
            except httpx.TimeoutException as e:
                await self._call_event_handler("on_completion_timeout")
                await self.push_error(error_msg="LLM completion timeout", exception=e)
            except Exception as e:
                await self.push_error(error_msg=f"Error during completion: {e}", exception=e)
            finally:
                await self.stop_processing_metrics()
                await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)
