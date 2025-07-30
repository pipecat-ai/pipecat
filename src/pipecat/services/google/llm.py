#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Google Gemini integration for Pipecat.

This module provides Google Gemini integration for the Pipecat framework,
including LLM services, context management, and message aggregation.
"""

import os
import uuid
from typing import Any, Dict, List, Optional

from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter, GeminiLLMInvocationParams
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    VisionImageRawFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext, LLMContextFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.google.frames import LLMSearchResponseFrame
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.utils.asyncio.watchdog_async_iterator import WatchdogAsyncIterator
from pipecat.utils.tracing.service_decorators import traced_llm

# Suppress gRPC fork warnings
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

try:
    from google import genai
    from google.api_core.exceptions import DeadlineExceeded
    from google.genai.types import (
        GenerateContentConfig,
        HttpOptions,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Google AI, you need to `pip install pipecat-ai[google]`.")
    raise Exception(f"Missing module: {e}")


class GoogleLLMService(LLMService):
    """Google AI (Gemini) LLM service implementation.

    This class implements inference with Google's AI models, translating internally
    from the universal LLMContext to the message format expected by the Google
    AI model.
    """

    # Overriding the default adapter to use the Gemini one.
    adapter_class = GeminiLLMAdapter

    class InputParams(BaseModel):
        """Input parameters for Google AI models.

        Parameters:
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature between 0.0 and 2.0.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter between 0.0 and 1.0.
            extra: Additional parameters as a dictionary.
        """

        max_tokens: Optional[int] = Field(default=4096, ge=1)
        temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
        top_k: Optional[int] = Field(default=None, ge=0)
        top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
        extra: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-2.0-flash",
        params: Optional[InputParams] = None,
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_config: Optional[Dict[str, Any]] = None,
        http_options: Optional[HttpOptions] = None,
        **kwargs,
    ):
        """Initialize the Google LLM service.

        Args:
            api_key: Google AI API key for authentication.
            model: Model name to use. Defaults to "gemini-2.0-flash".
            params: Input parameters for the model.
            system_instruction: System instruction/prompt for the model.
            tools: List of available tools/functions.
            tool_config: Configuration for tool usage.
            http_options: HTTP options for the client.
            **kwargs: Additional arguments passed to parent class.
        """
        super().__init__(**kwargs)

        params = params or GoogleLLMService.InputParams()

        self.set_model_name(model)
        self._api_key = api_key
        self._system_instruction = system_instruction
        self._http_options = http_options
        self._create_client(api_key, http_options)
        self._settings = {
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_k": params.top_k,
            "top_p": params.top_p,
            "extra": params.extra if isinstance(params.extra, dict) else {},
        }
        self._tools = tools
        self._tool_config = tool_config

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate usage metrics.

        Returns:
            True, as Google AI provides token usage metrics.
        """
        return True

    def _create_client(self, api_key: str, http_options: Optional[HttpOptions] = None):
        self._client = genai.Client(api_key=api_key, http_options=http_options)

    def needs_mcp_alternate_schema(self) -> bool:
        """Check if this LLM service requires alternate MCP schema.

        Google/Gemini has stricter JSON schema validation and requires
        certain properties to be removed or modified for compatibility.

        Returns:
            True for Google/Gemini services.
        """
        return True

    def _maybe_unset_thinking_budget(self, generation_params: Dict[str, Any]):
        try:
            # There's no way to introspect on model capabilities, so
            # to check for models that we know default to thinkin on
            # and can be configured to turn it off.
            if not self._model_name.startswith("gemini-2.5-flash"):
                return
            # If thinking_config is already set, don't override it.
            if "thinking_config" in generation_params:
                return
            generation_params.setdefault("thinking_config", {})["thinking_budget"] = 0
        except Exception as e:
            logger.exception(f"Failed to unset thinking budget: {e}")

    @traced_llm
    async def _process_context(self, context: LLMContext):
        await self.push_frame(LLMFullResponseStartFrame())

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        cache_read_input_tokens = 0
        reasoning_tokens = 0

        grounding_metadata = None
        search_result = ""

        try:
            adapter = self.get_llm_adapter()
            llm_invocation_params: GeminiLLMInvocationParams = adapter.get_llm_invocation_params(
                context
            )

            logger.debug(
                # TODO: figure out a nice way to also log system instruction
                # f"{self}: Generating chat [{self._system_instruction}] | [{adapter.get_messages_for_logging(context)}]"
                f"{self}: Generating chat [{adapter.get_messages_for_logging(context)}]"
            )

            messages = llm_invocation_params["messages"]
            if (
                llm_invocation_params.get("system_instruction")
                and self._system_instruction != llm_invocation_params["system_instruction"]
            ):
                logger.debug(
                    f"System instruction changed: {llm_invocation_params['system_instruction']}"
                )
                self._system_instruction = llm_invocation_params["system_instruction"]

            # TODO: test what happens when there are no tools
            tools = []
            if llm_invocation_params.get("tools"):
                tools = llm_invocation_params["tools"]
            elif self._tools:
                tools = self._tools
            tool_config = None
            if self._tool_config:
                tool_config = self._tool_config

            # Filter out None values and create GenerationContentConfig
            generation_params = {
                k: v
                for k, v in {
                    "system_instruction": self._system_instruction,
                    "temperature": self._settings["temperature"],
                    "top_p": self._settings["top_p"],
                    "top_k": self._settings["top_k"],
                    "max_output_tokens": self._settings["max_tokens"],
                    "tools": tools,
                    "tool_config": tool_config,
                }.items()
                if v is not None
            }

            if self._settings["extra"]:
                generation_params.update(self._settings["extra"])

            # possibly modify generation_params (in place) to set thinking to off by default
            self._maybe_unset_thinking_budget(generation_params)

            generation_config = (
                GenerateContentConfig(**generation_params) if generation_params else None
            )

            await self.start_ttfb_metrics()
            response = await self._client.aio.models.generate_content_stream(
                model=self._model_name,
                contents=messages,
                config=generation_config,
            )

            function_calls = []
            async for chunk in WatchdogAsyncIterator(response, manager=self.task_manager):
                # Stop TTFB metrics after the first chunk
                await self.stop_ttfb_metrics()
                if chunk.usage_metadata:
                    prompt_tokens += chunk.usage_metadata.prompt_token_count or 0
                    completion_tokens += chunk.usage_metadata.candidates_token_count or 0
                    total_tokens += chunk.usage_metadata.total_token_count or 0
                    cache_read_input_tokens += chunk.usage_metadata.cached_content_token_count or 0
                    reasoning_tokens += chunk.usage_metadata.thoughts_token_count or 0

                if not chunk.candidates:
                    continue

                for candidate in chunk.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if not part.thought and part.text:
                                search_result += part.text
                                await self.push_frame(LLMTextFrame(part.text))
                            elif part.function_call:
                                function_call = part.function_call
                                id = function_call.id or str(uuid.uuid4())
                                logger.debug(f"Function call: {function_call.name}:{id}")
                                function_calls.append(
                                    FunctionCallFromLLM(
                                        context=context,
                                        tool_call_id=id,
                                        function_name=function_call.name,
                                        arguments=function_call.args or {},
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
            logger.exception(f"{self} exception: {e}")
        finally:
            if grounding_metadata and isinstance(grounding_metadata, dict):
                llm_search_frame = LLMSearchResponseFrame(
                    search_result=search_result,
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

        context = None

        if isinstance(frame, LLMContextFrame):
            context = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            context = LLMContext(messages=frame.messages)
        elif isinstance(frame, VisionImageRawFrame):
            context = LLMContext()
            context.add_image_frame_message(
                format=frame.format, size=frame.size, image=frame.image, text=frame.text
            )
        elif isinstance(frame, LLMUpdateSettingsFrame):
            await self._update_settings(frame.settings)
        else:
            await self.push_frame(frame, direction)

        if context:
            await self._process_context(context)
