#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Google Gemini integration for Pipecat.

This module provides Google Gemini integration for the Pipecat framework,
including LLM services, context management, and message aggregation.
"""

import base64
import io
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Union

from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter, GeminiLLMInvocationParams
from pipecat.frames.frames import (
    AssistantImageRawFrame,
    AudioRawFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
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
from pipecat.services.google.frames import LLMSearchResponseFrame
from pipecat.services.google.utils import update_google_client_http_options
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.services.settings import (
    NOT_GIVEN,
    LLMSettings,
    _NotGiven,
    is_given,
)
from pipecat.utils.tracing.service_decorators import traced_llm

# Suppress gRPC fork warnings
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

try:
    from google import genai
    from google.api_core.exceptions import DeadlineExceeded
    from google.genai.types import (
        Blob,
        Content,
        FunctionCall,
        FunctionResponse,
        GenerateContentConfig,
        GenerateContentResponse,
        HttpOptions,
        Part,
    )

    # Temporary hack to be able to process Nano Banana returned images.
    genai._api_client.READ_BUFFER_SIZE = 5 * 1024 * 1024
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Google AI, you need to `pip install pipecat-ai[google]`.")
    raise Exception(f"Missing module: {e}")


class GoogleThinkingConfig(BaseModel):
    """Configuration for controlling the model's internal "thinking" process used before generating a response.

    Gemini 2.5 and 3 series models have this thinking process.

    Parameters:
        thinking_level: Thinking level for Gemini 3 models.
            For Gemini 3 Pro, this can be "low" or "high".
            For Gemini 3 Flash, this can be "minimal", "low", "medium", or "high".
            If not provided, Gemini 3 models default to "high".
            Note: Gemini 2.5 series must use thinking_budget instead.
        thinking_budget: Token budget for thinking, for Gemini 2.5 series.
            -1 for dynamic thinking (model decides), 0 to disable thinking,
            or a specific token count (e.g., 128-32768 for 2.5 Pro).
            If not provided, most models today default to dynamic thinking.
            See https://ai.google.dev/gemini-api/docs/thinking#set-budget
            for default values and allowed ranges.
            Note: Gemini 3 models must use thinking_level instead.
        include_thoughts: Whether to include thought summaries in the response.
            Today's models default to not including thoughts (False).
    """

    thinking_budget: Optional[int] = Field(default=None)

    # Why `| str` here? To not break compatibility in case Google adds more
    # levels in the future.
    thinking_level: Optional[Literal["low", "high", "medium", "minimal"] | str] = Field(
        default=None
    )

    include_thoughts: Optional[bool] = Field(default=None)


@dataclass
class GoogleLLMSettings(LLMSettings):
    """Settings for GoogleLLMService.

    Parameters:
        thinking: Thinking configuration.
    """

    thinking: Union["GoogleLLMService.ThinkingConfig", _NotGiven] = field(
        default_factory=lambda: NOT_GIVEN
    )

    @classmethod
    def from_mapping(cls, settings):
        """Convert a plain dict to settings, coercing thinking dicts.

        For backward compatibility, a ``thinking`` value that is a plain dict
        is converted to a :class:`GoogleLLMService.ThinkingConfig`.
        """
        instance = super().from_mapping(settings)
        if is_given(instance.thinking) and isinstance(instance.thinking, dict):
            instance.thinking = GoogleLLMService.ThinkingConfig(**instance.thinking)
        return instance


class GoogleLLMService(LLMService):
    """Google AI (Gemini) LLM service implementation.

    This class implements inference with Google's AI models, translating internally
    from an LLMContext to the messages format expected by the Google AI model.
    """

    Settings = GoogleLLMSettings
    _settings: Settings

    # Overriding the default adapter to use the Gemini one.
    adapter_class = GeminiLLMAdapter

    # Backward compatibility: ThinkingConfig used to be defined inline here.
    ThinkingConfig = GoogleThinkingConfig

    class InputParams(BaseModel):
        """Input parameters for Google AI models.

        .. deprecated:: 0.0.105
            Use ``settings=GoogleLLMService.Settings(...)`` instead.

        Parameters:
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature between 0.0 and 2.0.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter between 0.0 and 1.0.
            thinking: Thinking configuration with thinking_budget, thinking_level, and include_thoughts.
                Used to control the model's internal "thinking" process used before generating a response.
                Gemini 2.5 series models use thinking_budget; Gemini 3 models use thinking_level.
                If this is not provided, Pipecat disables thinking for all
                models where that's possible (the 2.5 series, except 2.5 Pro),
                to reduce latency.
            extra: Additional parameters as a dictionary.
        """

        max_tokens: Optional[int] = Field(default=4096, ge=1)
        temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
        top_k: Optional[int] = Field(default=None, ge=0)
        top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
        thinking: Optional["GoogleLLMService.ThinkingConfig"] = Field(default=None)
        extra: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def __init__(
        self,
        *,
        api_key: str,
        model: Optional[str] = None,
        params: Optional[InputParams] = None,
        settings: Optional[Settings] = None,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_config: Optional[Dict[str, Any]] = None,
        http_options: Optional[HttpOptions] = None,
        **kwargs,
    ):
        """Initialize the Google LLM service.

        Args:
            api_key: Google AI API key for authentication.
            model: Model name to use.

                .. deprecated:: 0.0.105
                    Use ``settings=GoogleLLMService.Settings(model=...)`` instead.

            params: Optional model parameters for inference.

                .. deprecated:: 0.0.105
                    Use ``settings=GoogleLLMService.Settings(...)`` instead.

            settings: Runtime-updatable settings for this service.  When both
                deprecated parameters and *settings* are provided, *settings*
                values take precedence.
            system_instruction: System instruction/prompt for the model.

                .. deprecated:: 0.0.105
                    Use ``settings=GoogleLLMService.Settings(system_instruction=...)`` instead.
            tools: List of available tools/functions.
            tool_config: Configuration for tool usage.
            http_options: HTTP options for the client.
            **kwargs: Additional arguments passed to parent class.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="gemini-2.5-flash",
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
            extra={},
        )

        # 2. Apply direct init arg overrides (deprecated)
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model
        if system_instruction is not None:
            self._warn_init_param_moved_to_settings("system_instruction", "system_instruction")
            default_settings.system_instruction = system_instruction

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                default_settings.max_tokens = params.max_tokens
                default_settings.temperature = params.temperature
                default_settings.top_k = params.top_k
                default_settings.top_p = params.top_p
                default_settings.thinking = params.thinking
                if isinstance(params.extra, dict):
                    default_settings.extra = params.extra

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(settings=default_settings, **kwargs)

        self._api_key = api_key
        self._http_options = update_google_client_http_options(http_options)
        self._tools = tools
        self._tool_config = tool_config

        # Initialize the API client. Subclasses can override this if needed.
        self.create_client()

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate usage metrics.

        Returns:
            True, as Google AI provides token usage metrics.
        """
        return True

    def create_client(self):
        """Create the Gemini client instance. Subclasses can override this."""
        self._client = genai.Client(api_key=self._api_key, http_options=self._http_options)

    async def run_inference(
        self,
        context: LLMContext,
        max_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
    ) -> Optional[str]:
        """Run a one-shot, out-of-band (i.e. out-of-pipeline) inference with the given LLM context.

        Args:
            context: The LLM context containing conversation history.
            max_tokens: Optional maximum number of tokens to generate. If provided,
                overrides the service's default max_tokens setting.
            system_instruction: Optional system instruction to use for this inference.
                If provided, overrides any system instruction in the context.

        Returns:
            The LLM's response as a string, or None if no response is generated.
        """
        messages = []
        system = []
        tools = []
        effective_instruction = system_instruction or self._settings.system_instruction
        adapter = self.get_llm_adapter()
        params: GeminiLLMInvocationParams = adapter.get_llm_invocation_params(
            context, system_instruction=effective_instruction
        )
        messages = params["messages"]
        system = params["system_instruction"]
        tools = params["tools"]

        # Build generation config using the same method as streaming
        generation_params = self._build_generation_params(
            system_instruction=system, tools=tools if tools else None
        )

        # Override max_output_tokens if provided
        if max_tokens is not None:
            generation_params["max_output_tokens"] = max_tokens

        generation_config = GenerateContentConfig(**generation_params)

        # Use the new google-genai client's async method
        response = await self._client.aio.models.generate_content(
            model=self._settings.model,
            contents=messages,
            config=generation_config,
        )

        # Extract text from response
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if part.text:
                    return part.text

        return None

    def _build_generation_params(
        self,
        system_instruction: Optional[str] = None,
        tools: Optional[List] = None,
        tool_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build generation parameters for Google AI API.

        Args:
            system_instruction: Optional system instruction to use.
            tools: Optional list of tools to include.
            tool_config: Optional tool configuration.

        Returns:
            Dictionary of generation parameters with None values filtered out.
        """
        # Filter out None values and create GenerationContentConfig
        generation_params = {
            k: v
            for k, v in {
                "system_instruction": system_instruction,
                "temperature": self._settings.temperature,
                "top_p": self._settings.top_p,
                "top_k": self._settings.top_k,
                "max_output_tokens": self._settings.max_tokens,
                "tools": tools,
                "tool_config": tool_config,
            }.items()
            if v is not None
        }

        # Add thinking parameters if configured
        if self._settings.thinking:
            generation_params["thinking_config"] = self._settings.thinking.model_dump(
                exclude_unset=True
            )

        if self._settings.extra:
            generation_params.update(self._settings.extra)

        return generation_params

    def _maybe_unset_thinking_budget(self, generation_params: Dict[str, Any]):
        try:
            # There's no way to introspect on model capabilities, so
            # to check for models that we know default to thinkin on
            # and can be configured to turn it off.
            if not self._settings.model.startswith("gemini-2.5-flash"):
                return
            # If we have an image model, we don't use a budget either.
            if "image" in self._settings.model:
                return
            # If thinking_config is already set, don't override it.
            if "thinking_config" in generation_params:
                return
            generation_params.setdefault("thinking_config", {})["thinking_budget"] = 0
        except Exception as e:
            logger.error(f"Failed to unset thinking budget: {e}")

    async def _stream_content(self, context: LLMContext) -> AsyncIterator[GenerateContentResponse]:
        adapter = self.get_llm_adapter()
        params: GeminiLLMInvocationParams = adapter.get_llm_invocation_params(
            context, system_instruction=self._settings.system_instruction
        )

        logger.debug(
            f"{self}: Generating chat from context [{params['system_instruction']}] | {adapter.get_messages_for_logging(context)}"
        )

        messages = params["messages"]

        # The adapter already resolved system_instruction vs context system message.
        system_instruction = params["system_instruction"]

        tools = []
        if params["tools"]:
            tools = params["tools"]
        elif self._tools:
            tools = self._tools
        tool_config = None
        if self._tool_config:
            tool_config = self._tool_config

        # Build generation parameters
        generation_params = self._build_generation_params(
            system_instruction=system_instruction,
            tools=tools,
            tool_config=tool_config,
        )

        # possibly modify generation_params (in place) to set thinking to off by default
        self._maybe_unset_thinking_budget(generation_params)

        generation_config = GenerateContentConfig(**generation_params)

        await self.start_ttfb_metrics()
        return await self._client.aio.models.generate_content_stream(
            model=self._settings.model,
            contents=messages,
            config=generation_config,
        )

    @traced_llm
    async def _process_context(self, context: LLMContext):
        await self.push_frame(LLMFullResponseStartFrame())

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        cache_read_input_tokens = 0
        reasoning_tokens = 0

        grounding_metadata = None
        accumulated_text = ""

        try:
            # Generate content from LLMContext
            response = await self._stream_content(context)

            function_calls = []
            async for chunk in response:
                # Stop TTFB metrics after the first chunk
                await self.stop_ttfb_metrics()
                # Gemini may send usage_metadata in multiple chunks with varying behavior:
                # - Sometimes a single chunk, sometimes multiple chunks
                # - Token counts may be cumulative (growing) or may change between chunks
                # - Early chunks may include estimates/overhead that gets refined
                # We use assignment (not accumulation) because the final chunk always contains
                # the authoritative, billable token usage for the entire response.
                if chunk.usage_metadata:
                    prompt_tokens = chunk.usage_metadata.prompt_token_count or 0
                    completion_tokens = chunk.usage_metadata.candidates_token_count or 0
                    total_tokens = chunk.usage_metadata.total_token_count or 0
                    cache_read_input_tokens = chunk.usage_metadata.cached_content_token_count or 0
                    reasoning_tokens = chunk.usage_metadata.thoughts_token_count or 0

                if not chunk.candidates:
                    continue

                for candidate in chunk.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            function_call_id = None
                            if part.text:
                                if part.thought:
                                    # Gemini emits fully-formed thoughts rather
                                    # than chunks so bracket each thought in
                                    # start/end
                                    await self.push_frame(LLMThoughtStartFrame())
                                    await self.push_frame(LLMThoughtTextFrame(part.text))
                                    await self.push_frame(LLMThoughtEndFrame())
                                else:
                                    accumulated_text += part.text
                                    await self._push_llm_text(part.text)
                            elif part.function_call:
                                function_call = part.function_call
                                function_call_id = function_call.id or str(uuid.uuid4())
                                logger.debug(
                                    f"Function call: {function_call.name}:{function_call_id}"
                                )
                                function_calls.append(
                                    FunctionCallFromLLM(
                                        context=context,
                                        tool_call_id=function_call_id,
                                        function_name=function_call.name,
                                        arguments=function_call.args or {},
                                    )
                                )
                            elif part.inline_data and part.inline_data.data:
                                # Here we assume that inline_data is an image.
                                image = Image.open(io.BytesIO(part.inline_data.data))
                                await self.push_frame(
                                    AssistantImageRawFrame(
                                        image=image.tobytes(),
                                        size=image.size,
                                        format="RGB",
                                        original_data=part.inline_data.data,
                                        original_mime_type=part.inline_data.mime_type,
                                    )
                                )

                            # Handle Gemini thought signatures.
                            #
                            # - Gemini 2.5: they appear on function_call Parts,
                            # and then (surprisingly) on the last(*) Part of
                            # model responses following the first function_call
                            # in a conversation.
                            # - Gemini 3 Pro: they appear on the last(*) Part
                            # of model responses, regardless of Part type.
                            #
                            # (*) Since we're using the streaming API, though,
                            # where text Parts may be split across multiple
                            # chunks (each represented by a Part, confusingly),
                            # signatures may actually appear with the first
                            # chunk (Gemini 2.5) or in a trailing empty-text
                            # chunk (Gemini 3 Pro).
                            if part.thought_signature:
                                # Save a "bookmark" for the signature, so we
                                # can later be sure we've put it in the right
                                # place in context when sending the context
                                # back to the LLM to continue the conversation.
                                bookmark = {}
                                if part.function_call:
                                    bookmark["function_call"] = function_call_id
                                elif part.inline_data and part.inline_data.data:
                                    bookmark["inline_data"] = part.inline_data
                                elif part.text is not None:
                                    # Account for Gemini 3 Pro trailing
                                    # empty-text chunk by using all the text
                                    # seen so far in this response's chunks.
                                    bookmark["text"] = accumulated_text
                                else:
                                    logger.warning("Thought signature found on unhandled Part type")
                                if bookmark:
                                    await self.push_frame(
                                        LLMMessagesAppendFrame(
                                            [
                                                self.get_llm_adapter().create_llm_specific_message(
                                                    {
                                                        "type": "thought_signature",
                                                        "signature": part.thought_signature,
                                                        "bookmark": bookmark,
                                                    }
                                                )
                                            ]
                                        )
                                    )

                    if (
                        candidate.grounding_metadata
                        and candidate.grounding_metadata.grounding_chunks
                    ):
                        m = candidate.grounding_metadata
                        rendered_content = (
                            m.search_entry_point.rendered_content if m.search_entry_point else None
                        )
                        origins = [
                            {
                                "site_uri": grounding_chunk.web.uri
                                if grounding_chunk.web
                                else None,
                                "site_title": grounding_chunk.web.title
                                if grounding_chunk.web
                                else None,
                                "results": [
                                    {
                                        "text": grounding_support.segment.text
                                        if grounding_support.segment
                                        else "",
                                        "confidence": grounding_support.confidence_scores,
                                    }
                                    for grounding_support in (
                                        m.grounding_supports if m.grounding_supports else []
                                    )
                                    if grounding_support.grounding_chunk_indices
                                    and index in grounding_support.grounding_chunk_indices
                                ],
                            }
                            for index, grounding_chunk in enumerate(
                                m.grounding_chunks if m.grounding_chunks else []
                            )
                        ]
                        grounding_metadata = {
                            "rendered_content": rendered_content,
                            "origins": origins,
                        }

            await self.run_function_calls(function_calls)
        except DeadlineExceeded:
            await self._call_event_handler("on_completion_timeout")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            if grounding_metadata and isinstance(grounding_metadata, dict):
                llm_search_frame = LLMSearchResponseFrame(
                    search_result=accumulated_text,
                    origins=grounding_metadata["origins"],
                    rendered_content=grounding_metadata["rendered_content"],
                )
                await self.push_frame(llm_search_frame)

            await self.start_llm_usage_metrics(
                LLMTokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cache_read_input_tokens=cache_read_input_tokens,
                    reasoning_tokens=reasoning_tokens,
                )
            )
            await self.push_frame(LLMFullResponseEndFrame())

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

    async def _close_client(self):
        try:
            await self._client.aio.aclose()
        except Exception:
            # Do nothing - we're shutting down anyway
            pass
