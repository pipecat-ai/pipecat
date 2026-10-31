#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Google Gemini LLM service using the Interactions API."""

import base64
import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, cast

from loguru import logger

from pipecat.adapters.llm.base_llm_adapter import LLMContextConversionError
from pipecat.adapters.llm.services.gemini_interactions_adapter import (
    GeminiInteractionsLLMAdapter,
    GeminiInteractionsLLMInvocationParams,
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
from pipecat.services.google.llm import (
    GoogleLLMSettings,
    GoogleThinkingConfig,
)
from pipecat.services.google.utils import (
    _thinking_default,
    stream_with_retry,
    update_google_client_http_options,
)
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.tracing.service_decorators import traced_llm
from pipecat.utils.types import assert_given

try:
    import google.genai as genai
    from google.genai.types import HttpOptions, SafetySetting
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Google AI, you need to `uv add "pipecat-ai[google]"`.')
    raise ImportError(f"Missing module: {e}") from e

# The Interactions API's current schema requires google-genai >= 2.0.0; older
# SDKs send a schema the API rejects. Checked at construction time so the rest
# of the Google services stay usable with older google-genai versions.
try:
    from google.genai.interactions import StepStart
except ImportError:
    StepStart = None


@dataclass
class GoogleInteractionsLLMSettings(GoogleLLMSettings):
    """Settings for GoogleInteractionsLLMService.

    The same settings as :class:`~pipecat.services.google.llm.GoogleLLMSettings`,
    so a service's settings carry over between the two Gemini services. The
    Interactions API has no ``top_k``, ``frequency_penalty`` or
    ``presence_penalty``, and controls thinking by level only.

    Parameters:
        thinking: Thinking configuration. ``thinking_level`` and
            ``include_thoughts`` apply; a ``thinking_budget`` is ignored. If this
            is not provided, Pipecat sends the lowest thinking level Google
            documents for the model, for the fastest responses, and logs what it
            sent. Models Pipecat doesn't list, and any model when
            ``use_provider_thinking_default`` is set, run at Google's default.
        safety_settings: Content safety filters, as a list of
            :class:`~google.genai.types.SafetySetting`. Each entry pairs a harm
            category with the threshold at which content is blocked. Categories
            left unspecified keep the Gemini API defaults.
    """


class GoogleInteractionsLLMService(LLMService[GeminiInteractionsLLMAdapter]):
    """Google Gemini LLM service using the Interactions API.

    Sends the full context on every turn, with ``store=False``, so Google keeps
    no conversation state. Takes the same settings as
    :class:`~pipecat.services.google.llm.GoogleLLMService`, and the two read
    each other's thought signatures from the context.

    Example::

        llm = GoogleInteractionsLLMService(
            api_key=os.getenv("GOOGLE_API_KEY"),
            settings=GoogleInteractionsLLMService.Settings(
                system_instruction="You are a helpful assistant.",
            ),
        )
    """

    Settings = GoogleInteractionsLLMSettings
    _settings: Settings

    ThinkingConfig = GoogleThinkingConfig

    adapter_class = GeminiInteractionsLLMAdapter

    supports_response_schema: bool = True

    def __init__(
        self,
        *,
        api_key: str,
        settings: Settings | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_config: dict[str, Any] | None = None,
        http_options: HttpOptions | None = None,
        stream_idle_timeout_secs: float | None = 20.0,
        retry_timeout_secs: float | None = 5.0,
        retry_on_timeout: bool = False,
        **kwargs,
    ):
        """Initialize the service.

        Args:
            api_key: Google AI API key for authentication.
            settings: Runtime-updatable settings for this service.
            tools: Tools to offer the model when the context has none, in
                Gemini's ``generateContent`` format or the Interactions API's.
            tool_config: Configuration for tool usage, in Gemini's
                ``generateContent`` format (``function_calling_config``).
            http_options: HTTP options for the client.
            stream_idle_timeout_secs: How long to wait for the next event of a
                streamed response before giving up on it. Bounds the wait when the
                API accepts a request and then stops producing without closing the
                stream. Set to ``None`` to wait indefinitely.
            retry_timeout_secs: How long to wait for the first event before giving
                up on the request and re-issuing it, when ``retry_on_timeout`` is
                set.
            retry_on_timeout: Whether to re-issue the request once if the first
                event doesn't arrive within ``retry_timeout_secs``.
            **kwargs: Additional arguments passed to parent class.

        Raises:
            ImportError: If the installed google-genai version is older than 2.0.0.
        """
        if StepStart is None:
            raise ImportError(
                "GoogleInteractionsLLMService requires google-genai >= 2.0.0. "
                'Upgrade with `uv add "google-genai>=2.0.0"`.'
            )

        default_settings = self.Settings(
            model="gemini-3.6-flash",
            system_instruction=None,
            max_tokens=4096,
            temperature=None,
            top_k=None,
            top_p=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            filter_incomplete_user_turns=False,
            user_turn_completion_config=None,
            thinking=None,
            safety_settings=None,
            extra={},
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(settings=default_settings, **kwargs)

        self._api_key = api_key
        self._http_options = update_google_client_http_options(http_options)
        self._tools = tools
        self._tool_config = tool_config
        self._stream_idle_timeout_secs = stream_idle_timeout_secs
        self._retry_timeout_secs = retry_timeout_secs
        self._retry_on_timeout = retry_on_timeout
        self._warned_unsupported: set[str] = set()

        self._client: genai.Client | None = genai.Client(
            api_key=self._api_key, http_options=self._http_options
        )

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate usage metrics.

        Returns:
            True, as the Interactions API reports token usage.
        """
        return True

    @staticmethod
    def model_supports_response_schema(model: str) -> bool:
        """Whether a model can enforce a response schema.

        Args:
            model: The model name.
        """
        return True

    async def run_inference(
        self,
        context: LLMContext,
        max_tokens: int | None = None,
        system_instruction: str | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> str | None:
        """Run a one-shot, out-of-band (i.e. out-of-pipeline) inference with the given LLM context.

        Args:
            context: The LLM context containing conversation history.
            max_tokens: Optional maximum number of tokens to generate. If provided,
                overrides the service's default max_tokens setting.
            system_instruction: Optional system instruction to use for this inference.
                If provided, overrides any system instruction in the context.
            response_schema: Optional JSON schema the reply must follow. The
                service asks the provider to enforce it, so the reply is JSON
                text matching the schema.

        Returns:
            The LLM's response as a string, or None if no response is generated.
        """
        params = await self.get_llm_adapter().get_llm_invocation_params(
            context,
            system_instruction=system_instruction
            or assert_given(self._settings.system_instruction),
        )
        request = self._build_request(params)
        if max_tokens is not None:
            request.setdefault("generation_config", {})["max_output_tokens"] = max_tokens

        response_schema = self._check_response_schema(response_schema)
        if response_schema is not None:
            request["response_format"] = {
                "type": "text",
                "mime_type": "application/json",
                "schema": response_schema,
            }

        assert self._client is not None
        interaction = await self._client.aio.interactions.create(**request)

        status = getattr(interaction, "status", None)
        if status not in ("completed", "requires_action"):
            logger.warning(f"{self}: response incomplete, the interaction ended {status}")

        text = "".join(
            item.text
            for step in getattr(interaction, "steps", None) or []
            if step.type == "model_output"
            for item in step.content or []
            if getattr(item, "text", None)
        )
        return text or None

    @traced_llm
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle different frame types.

        Args:
            frame: The frame to process.
            direction: Direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            await self._process_context(frame.context)
        else:
            await self.push_frame(frame, direction)

    async def stop(self, frame):
        """Override stop to gracefully close the client."""
        await super().stop(frame)
        await self._close_client()

    async def cancel(self, frame):
        """Override cancel to gracefully close the client."""
        await super().cancel(frame)
        await self._close_client()

    async def cleanup(self):
        """Release resources held by the service."""
        await super().cleanup()
        await self._close_client()

    async def _close_client(self):
        if not self._client:
            return
        try:
            await self._client.aio.aclose()
        except Exception:
            # Do nothing - we're shutting down anyway
            pass
        finally:
            self._client = None

    def _build_request(self, params: GeminiInteractionsLLMInvocationParams) -> dict[str, Any]:
        """Build an Interactions API request from the context's invocation params."""
        model = assert_given(self._settings.model)
        request: dict[str, Any] = {"model": model, "input": params["input"], "store": False}

        if params["system_instruction"]:
            request["system_instruction"] = params["system_instruction"]

        tools = params["tools"]
        if isinstance(tools, list) and tools:
            request["tools"] = tools
        elif self._tools:
            request["tools"] = GeminiInteractionsLLMAdapter.to_interactions_tools(self._tools)

        generation_config = {
            key: value
            for key, value in {
                "temperature": self._settings.temperature,
                "top_p": self._settings.top_p,
                "max_output_tokens": self._settings.max_tokens,
                "seed": self._settings.seed,
            }.items()
            if value is not None
        }
        for name in ("top_k", "frequency_penalty", "presence_penalty"):
            if getattr(self._settings, name) is not None:
                self._warn_unsupported(name, f"the Interactions API has no {name}")

        thinking = assert_given(self._settings.thinking)
        if thinking:
            if thinking.thinking_level is not None:
                generation_config["thinking_level"] = thinking.thinking_level
            if thinking.include_thoughts is not None:
                generation_config["thinking_summaries"] = (
                    "auto" if thinking.include_thoughts else "none"
                )
            if thinking.thinking_budget is not None:
                self._warn_unsupported(
                    "thinking_budget",
                    "the Interactions API controls thinking by thinking_level only",
                )

        if self._tool_config:
            generation_config["tool_choice"] = _tool_choice(self._tool_config)

        if generation_config:
            request["generation_config"] = generation_config

        safety_settings = assert_given(self._settings.safety_settings)
        if safety_settings:
            request["safety_settings"] = [_safety_setting(s) for s in safety_settings]

        for key, value in (self._settings.extra or {}).items():
            if key == "generation_config" and isinstance(value, dict):
                request["generation_config"] = {**request.get("generation_config", {}), **value}
            else:
                request[key] = value

        # Applied last, so an explicit thinking level from the settings or from
        # extra wins over the low-latency default.
        self._maybe_apply_thinking_default(request)

        return request

    def _maybe_apply_thinking_default(self, request: dict[str, Any]):
        """Send the model's lowest thinking level when none is configured.

        Applies the model's entry in ``_THINKING_DEFAULTS``. Gemini 2.5 models,
        which take a budget, run at Google's default, since the Interactions API
        takes no budget.

        Args:
            request: The request dict (modified in place).
        """
        if "thinking_level" in request.get("generation_config", {}):
            return
        model = assert_given(self._settings.model) or ""
        thinking, can_think = _thinking_default(model, levels_only=True)
        self._apply_thinking_default(
            request,
            model,
            {"generation_config": thinking} if thinking is not None else None,
            can_think=can_think,
            provider="Google",
            settings_fields="`thinking`",
        )

    def _warn_unsupported(self, name: str, reason: str):
        if name in self._warned_unsupported:
            return
        self._warned_unsupported.add(name)
        logger.warning(f"{self}: ignoring {name}, since {reason}")

    async def _stream_events(self, request: dict[str, Any]) -> AsyncGenerator[Any, None]:
        """Yield a request's stream events, re-issuing the request once if it stalls."""
        assert self._client is not None
        client = self._client

        async def open_stream() -> AsyncIterator[Any]:
            # With stream=True, the call returns the event stream.
            stream = await client.aio.interactions.create(**request, stream=True)
            return cast(AsyncIterator[Any], stream)

        async with aclosing(
            stream_with_retry(
                open_stream,
                idle_timeout=self._stream_idle_timeout_secs,
                retry_timeout=self._retry_timeout_secs,
                retry_on_timeout=self._retry_on_timeout,
                label=str(self),
            )
        ) as stream:
            async for event in stream:
                yield event

    async def _process_context(self, context: LLMContext):
        await self.push_frame(LLMFullResponseStartFrame())

        usage = None
        try:
            adapter = self.get_llm_adapter()
            params = await adapter.get_llm_invocation_params(
                context, system_instruction=assert_given(self._settings.system_instruction)
            )
            self._log_llm_response(context)
            request = self._build_request(params)

            await self.start_ttfb_metrics()
            response = _ResponseState(context)
            async with aclosing(self._stream_events(request)) as events:
                async for event in events:
                    usage = await self._handle_event(event, response) or usage
            await self.run_function_calls(response.function_calls)
        except TimeoutError as e:
            await self._call_event_handler("on_completion_timeout")
            await self.push_error(error_msg="LLM completion timeout", exception=e)
        except LLMContextConversionError as e:
            await self.push_error(error_msg=str(e), exception=e)
            # A conversion failure (e.g. corrupt base64 data) can't reach the
            # API at all, but is just as much evidence of an invalid file as a
            # rejection from Gemini itself, so it gets the same best-effort
            # cleanup.
            context.remove_invalid_file_message()
        except Exception as e:
            # The API rejects a bad request (unsupported MIME type, corrupt
            # bytes, etc.) with a 400 that doesn't say which field was invalid,
            # which is grounds to remove a pending file message on a best-effort
            # basis. When one is removed, the fault lay in application-supplied
            # content and the context is repaired, so the error is pushed as
            # APPLICATION.
            removed = (
                getattr(e, "status_code", None) == 400 and context.remove_invalid_file_message()
            )
            await self.push_error(
                error_msg=f"Unknown error occurred: {e}",
                exception=e,
                category=ErrorCategory.APPLICATION if removed else None,
            )
        finally:
            await self.start_llm_usage_metrics(_token_usage(usage))
            await self.push_frame(LLMFullResponseEndFrame())

    async def _handle_event(self, event: Any, response: "_ResponseState") -> Any:
        """Act on one stream event. Returns the response's usage when the event carries it."""
        event_type = getattr(event, "event_type", None)

        if event_type == "step.start":
            # A leading lifecycle event carries no model output, so TTFB ends
            # at the first step.
            await self.stop_ttfb_metrics()
            step = event.step
            response.steps[event.index] = step.type
            if step.type == "function_call":
                # A turn that only calls tools produces no answer text, so the
                # call itself is what the caller gets and TTFAT ends here rather
                # than going unmeasured.
                await self.stop_ttfat_metrics()
                response.calls[event.index] = {
                    "id": step.id,
                    "name": step.name,
                    "arguments": "",
                    "initial": step.arguments or {},
                }
            await self._bookmark_pending_signature(response, event.index)

        elif event_type == "step.delta":
            delta = event.delta
            if delta.type == "text":
                response.text[event.index] = response.text.get(event.index, "") + delta.text
                await self._push_llm_text(delta.text)
            elif delta.type == "arguments_delta":
                if event.index in response.calls:
                    response.calls[event.index]["arguments"] += delta.arguments or ""
            elif delta.type == "thought_summary":
                text = getattr(delta.content, "text", None)
                if text:
                    # The summary arrives whole, so it is bracketed on its own.
                    await self.push_frame(LLMThoughtStartFrame())
                    await self.push_frame(LLMThoughtTextFrame(text))
                    await self.push_frame(LLMThoughtEndFrame())
            elif delta.type == "thought_signature" and delta.signature:
                response.pending_signature = delta.signature

        elif event_type == "step.stop":
            call = response.calls.pop(event.index, None)
            if call is not None:
                response.function_calls.append(self._function_call(response.context, call))
            elif event.index == response.text_bookmark_index:
                await self._store_signature(response, {"text": response.text.get(event.index, "")})

        elif event_type == "interaction.completed":
            interaction = event.interaction
            self._full_model_name = interaction.model
            if interaction.status not in ("completed", "requires_action"):
                logger.warning(
                    f"{self}: response incomplete, the interaction ended {interaction.status}"
                )
            elif not response.function_calls and not "".join(response.text.values()).strip():
                await self._report_empty_response(response)
            return interaction.usage

        elif event_type == "error":
            error = event.error
            message = getattr(error, "message", None) or "Interactions API error"
            await self.push_error(error_msg=f"LLM response error: {message}")

        return None

    async def _report_empty_response(self, response: "_ResponseState"):
        """Report a completed response that produced no text and no function call.

        Nothing reaches the TTS for such a response, so the bot stays silent
        until the user speaks again. The warning lists the steps the response
        produced instead, and the non-fatal error lets the application react.
        """
        steps = [response.steps[index] for index in sorted(response.steps)]
        logger.warning(f"{self}: response completed with no text or function call (steps: {steps})")
        await self.push_error(error_msg="LLM response completed with no text or function call")

    async def _bookmark_pending_signature(self, response: "_ResponseState", index: int):
        """Tie a signature received in a thought step to the step that follows it.

        A function call is named by its id right away. Text is named by what it
        says, so its signature is stored when the text step stops.
        """
        if response.pending_signature is None:
            return
        step_type = response.steps[index]
        if step_type == "function_call":
            await self._store_signature(response, {"function_call": response.calls[index]["id"]})
        elif step_type == "model_output":
            response.text_bookmark_index = index

    async def _store_signature(self, response: "_ResponseState", bookmark: dict[str, Any]):
        """Append the pending thought signature to the context, in GoogleLLMService's form."""
        signature = response.pending_signature
        response.pending_signature = None
        response.text_bookmark_index = None
        if signature is None or not any(bookmark.values()):
            return
        await self.push_frame(
            LLMMessagesAppendFrame(
                [
                    self.get_llm_adapter().create_llm_specific_message(
                        {
                            "type": "thought_signature",
                            "signature": _decode_signature(signature),
                            "bookmark": bookmark,
                        }
                    )
                ]
            )
        )

    def _function_call(self, context: LLMContext, call: dict[str, Any]) -> FunctionCallFromLLM:
        arguments = call["initial"]
        if call["arguments"]:
            try:
                arguments = json.loads(call["arguments"])
            except json.JSONDecodeError:
                logger.warning(
                    f"{self}: Failed to parse function call arguments: {call['arguments']}"
                )
        logger.debug(f"Function call: {call['name']}:{call['id']}")
        return FunctionCallFromLLM(
            context=context,
            tool_call_id=call["id"],
            function_name=call["name"],
            arguments=arguments,
        )


class _ResponseState:
    """What one streamed response has produced so far."""

    def __init__(self, context: LLMContext):
        self.context = context
        # Step index -> step type, call in progress, and text so far.
        self.steps: dict[int, str] = {}
        self.calls: dict[int, dict[str, Any]] = {}
        self.text: dict[int, str] = {}
        self.function_calls: list[FunctionCallFromLLM] = []
        # A thought signature waiting for the step it belongs to.
        self.pending_signature: str | None = None
        self.text_bookmark_index: int | None = None


def _token_usage(usage: Any) -> LLMTokenUsage:
    """Pipecat's token usage for an interaction's usage.

    The API counts tool results fed back to the model apart from the input, and
    thinking apart from the output, but they are input and output too.
    """
    if usage is None:
        return LLMTokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    reasoning_tokens = usage.total_thought_tokens or 0
    prompt_tokens = (usage.total_input_tokens or 0) + (usage.total_tool_use_tokens or 0)
    completion_tokens = (usage.total_output_tokens or 0) + reasoning_tokens
    return LLMTokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cache_read_input_tokens=usage.total_cached_tokens or 0,
        reasoning_tokens=reasoning_tokens,
    )


def _decode_signature(signature: str) -> bytes | str:
    """A thought signature as GoogleLLMService stores it: raw bytes."""
    try:
        return base64.b64decode(signature, validate=True)
    except ValueError:
        return signature


def _tool_choice(tool_config: dict[str, Any]) -> Any:
    """The Interactions API's ``tool_choice`` for a ``generateContent`` tool config."""
    config = tool_config.get("function_calling_config")
    if not isinstance(config, dict):
        return tool_config
    mode = str(config.get("mode") or "auto").lower()
    names = config.get("allowed_function_names")
    if names:
        return {"allowed_tools": {"mode": mode, "tools": list(names)}}
    return mode


def _safety_setting(setting: SafetySetting | dict[str, Any]) -> dict[str, Any]:
    """The Interactions API's form of a ``generateContent`` safety setting.

    ``HARM_CATEGORY_HARASSMENT`` becomes ``harassment``, ``BLOCK_ONLY_HIGH``
    becomes ``block_only_high``, and so on.
    """
    if isinstance(setting, dict):
        return setting

    def name(value: Any) -> str:
        return str(getattr(value, "value", value)).lower()

    result = {
        "type": name(setting.category).removeprefix("harm_category_"),
        "threshold": name(setting.threshold),
    }
    if setting.method is not None:
        result["method"] = name(setting.method)
    return result
