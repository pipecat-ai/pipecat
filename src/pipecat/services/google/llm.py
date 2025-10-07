#
# Copyright (c) 2024–2025, Daily
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
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional

from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter, GeminiLLMInvocationParams
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    OutputImageRawFrame,
    UserImageRawFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantAggregatorParams,
    LLMUserAggregatorParams,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.google.frames import LLMSearchResponseFrame
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.services.openai.llm import (
    OpenAIAssistantContextAggregator,
    OpenAIUserContextAggregator,
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


class GoogleUserContextAggregator(OpenAIUserContextAggregator):
    """Google-specific user context aggregator.

    Extends OpenAI user context aggregator to handle Google AI's specific
    Content and Part message format for user messages.
    """

    async def handle_aggregation(self, aggregation: str):
        """Add the aggregated user text to the context as a Google Content message.

        Args:
            aggregation: The aggregated user text to add as a user message.
        """
        self._context.add_message(Content(role="user", parts=[Part(text=aggregation)]))


class GoogleAssistantContextAggregator(OpenAIAssistantContextAggregator):
    """Google-specific assistant context aggregator.

    Extends OpenAI assistant context aggregator to handle Google AI's specific
    Content and Part message format for assistant responses and function calls.
    """

    async def handle_aggregation(self, aggregation: str):
        """Handle aggregated assistant text response.

        Args:
            aggregation: The aggregated text response from the assistant.
        """
        self._context.add_message(Content(role="model", parts=[Part(text=aggregation)]))

    async def handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame):
        """Handle function call in progress frame.

        Args:
            frame: Frame containing function call details.
        """
        self._context.add_message(
            Content(
                role="model",
                parts=[
                    Part(
                        function_call=FunctionCall(
                            id=frame.tool_call_id, name=frame.function_name, args=frame.arguments
                        )
                    )
                ],
            )
        )
        self._context.add_message(
            Content(
                role="user",
                parts=[
                    Part(
                        function_response=FunctionResponse(
                            id=frame.tool_call_id,
                            name=frame.function_name,
                            response={"response": "IN_PROGRESS"},
                        )
                    )
                ],
            )
        )

    async def handle_function_call_result(self, frame: FunctionCallResultFrame):
        """Handle function call result frame.

        Args:
            frame: Frame containing function call result.
        """
        if frame.result:
            await self._update_function_call_result(
                frame.function_name, frame.tool_call_id, frame.result
            )
        else:
            await self._update_function_call_result(
                frame.function_name, frame.tool_call_id, "COMPLETED"
            )

    async def handle_function_call_cancel(self, frame: FunctionCallCancelFrame):
        """Handle function call cancellation frame.

        Args:
            frame: Frame containing function call cancellation details.
        """
        await self._update_function_call_result(
            frame.function_name, frame.tool_call_id, "CANCELLED"
        )

    async def _update_function_call_result(
        self, function_name: str, tool_call_id: str, result: Any
    ):
        for message in self._context.messages:
            if message.role == "user":
                for part in message.parts:
                    if part.function_response and part.function_response.id == tool_call_id:
                        part.function_response.response = {"value": json.dumps(result)}

    async def handle_user_image_frame(self, frame: UserImageRawFrame):
        """Handle user image frame.

        Args:
            frame: Frame containing user image data and request context.
        """
        await self._update_function_call_result(
            frame.request.function_name, frame.request.tool_call_id, "COMPLETED"
        )
        self._context.add_image_frame_message(
            format=frame.format,
            size=frame.size,
            image=frame.image,
            text=frame.request.context,
        )


@dataclass
class GoogleContextAggregatorPair:
    """Pair of Google context aggregators for user and assistant messages.

    Parameters:
        _user: User context aggregator for handling user messages.
        _assistant: Assistant context aggregator for handling assistant responses.
    """

    _user: GoogleUserContextAggregator
    _assistant: GoogleAssistantContextAggregator

    def user(self) -> GoogleUserContextAggregator:
        """Get the user context aggregator.

        Returns:
            The user context aggregator instance.
        """
        return self._user

    def assistant(self) -> GoogleAssistantContextAggregator:
        """Get the assistant context aggregator.

        Returns:
            The assistant context aggregator instance.
        """
        return self._assistant


class GoogleLLMContext(OpenAILLMContext):
    """Google AI LLM context that extends OpenAI context for Google-specific formatting.

    This class handles conversion between OpenAI-style messages and Google AI's
    Content/Part format, including system messages, function calls, and media.
    """

    def __init__(
        self,
        messages: Optional[List[dict]] = None,
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[dict] = None,
    ):
        """Initialize GoogleLLMContext.

        Args:
            messages: Initial messages in OpenAI format.
            tools: Available tools/functions for the model.
            tool_choice: Tool choice configuration.
        """
        super().__init__(messages=messages, tools=tools, tool_choice=tool_choice)
        self.system_message = None

    @staticmethod
    def upgrade_to_google(obj: OpenAILLMContext) -> "GoogleLLMContext":
        """Upgrade an OpenAI context to a Google context.

        Args:
            obj: OpenAI LLM context to upgrade.

        Returns:
            GoogleLLMContext instance with converted messages.
        """
        if isinstance(obj, OpenAILLMContext) and not isinstance(obj, GoogleLLMContext):
            logger.debug(f"Upgrading to Google: {obj}")
            obj.__class__ = GoogleLLMContext
            obj._restructure_from_openai_messages()
        return obj

    def set_messages(self, messages: List):
        """Set messages and restructure them for Google format.

        Args:
            messages: List of messages to set.
        """
        self._messages[:] = messages
        self._restructure_from_openai_messages()

    def add_messages(self, messages: List):
        """Add messages to the context, converting to Google format as needed.

        Args:
            messages: List of messages to add (can be mixed formats).
        """
        # Convert each message individually
        converted_messages = []
        for msg in messages:
            if isinstance(msg, Content):
                # Already in Gemini format
                converted_messages.append(msg)
            else:
                # Convert from standard format to Gemini format
                converted = self.from_standard_message(msg)
                if converted is not None:
                    converted_messages.append(converted)

        # Add the converted messages to our existing messages
        self._messages.extend(converted_messages)

    def get_messages_for_logging(self) -> List[Dict[str, Any]]:
        """Get messages formatted for logging with sensitive data redacted.

        Returns:
            List of messages in a format ready for logging.
        """
        msgs = []
        for message in self.messages:
            obj = message.to_json_dict()
            try:
                if "parts" in obj:
                    for part in obj["parts"]:
                        if "inline_data" in part:
                            part["inline_data"]["data"] = "..."
            except Exception as e:
                logger.debug(f"Error: {e}")
            msgs.append(obj)
        return msgs

    def add_image_frame_message(
        self, *, format: str, size: tuple[int, int], image: bytes, text: str = None
    ):
        """Add an image message to the context.

        Args:
            format: Image format (e.g., 'RGB', 'RGBA').
            size: Image dimensions as (width, height).
            image: Raw image bytes.
            text: Optional text to accompany the image.
        """
        buffer = io.BytesIO()
        Image.frombytes(format, size, image).save(buffer, format="JPEG")

        parts = []
        if text:
            parts.append(Part(text=text))
        parts.append(Part(inline_data=Blob(mime_type="image/jpeg", data=buffer.getvalue())))

        self.add_message(Content(role="user", parts=parts))

    def add_audio_frames_message(
        self, *, audio_frames: list[AudioRawFrame], text: str = "Audio follows"
    ):
        """Add audio frames as a message to the context.

        Args:
            audio_frames: List of audio frames to add.
            text: Text description of the audio content.
        """
        if not audio_frames:
            return

        sample_rate = audio_frames[0].sample_rate
        num_channels = audio_frames[0].num_channels

        parts = []
        data = b"".join(frame.audio for frame in audio_frames)
        # NOTE(aleix): According to the docs only text or inline_data should be needed.
        # (see https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference)
        parts.append(Part(text=text))
        parts.append(
            Part(
                inline_data=Blob(
                    mime_type="audio/wav",
                    data=(
                        bytes(
                            self.create_wav_header(sample_rate, num_channels, 16, len(data)) + data
                        )
                    ),
                )
            ),
        )
        self.add_message(Content(role="user", parts=parts))
        # message = {"mime_type": "audio/mp3", "data": bytes(data + create_wav_header(sample_rate, num_channels, 16, len(data)))}
        # self.add_message(message)

    def from_standard_message(self, message):
        """Convert standard format message to Google Content object.

        Handles conversion of text, images, and function calls to Google's format.
        System messages are stored separately and return None.

        Args:
            message: Message in standard format.

        Returns:
            Content object with role and parts, or None for system messages.

        Examples:
            Standard text message::

                {
                    "role": "user",
                    "content": "Hello there"
                }

            Converts to Google Content with::

                Content(
                    role="user",
                    parts=[Part(text="Hello there")]
                )

            Standard function call message::

                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search",
                                "arguments": '{"query": "test"}'
                            }
                        }
                    ]
                }

            Converts to Google Content with::

                Content(
                    role="model",
                    parts=[Part(function_call=FunctionCall(name="search", args={"query": "test"}))]
                )

            System message returns None and stores content in self.system_message.
        """
        role = message["role"]
        content = message.get("content", [])
        if role == "system":
            # System instructions are returned as plain text
            if isinstance(content, str):
                self.system_message = content
            elif isinstance(content, list):
                # If content is a list, we assume it's a list of text parts, per the standard
                self.system_message = " ".join(
                    part["text"] for part in content if part.get("type") == "text"
                )
            return None
        elif role == "assistant":
            role = "model"

        parts = []
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                parts.append(
                    Part(
                        function_call=FunctionCall(
                            name=tc["function"]["name"],
                            args=json.loads(tc["function"]["arguments"]),
                        )
                    )
                )
        elif role == "tool":
            role = "model"
            try:
                response = json.loads(message["content"])
                if isinstance(response, dict):
                    response_dict = response
                else:
                    response_dict = {"value": response}
            except Exception as e:
                # Response might not be JSON-deserializable (e.g. plain text).
                response_dict = {"value": message["content"]}
            parts.append(
                Part(
                    function_response=FunctionResponse(
                        name="tool_call_result",  # seems to work to hard-code the same name every time
                        response=response_dict,
                    )
                )
            )
        elif isinstance(content, str):
            parts.append(Part(text=content))
        elif isinstance(content, list):
            for c in content:
                if c["type"] == "text":
                    parts.append(Part(text=c["text"]))
                elif c["type"] == "image_url":
                    parts.append(
                        Part(
                            inline_data=Blob(
                                mime_type="image/jpeg",
                                data=base64.b64decode(c["image_url"]["url"].split(",")[1]),
                            )
                        )
                    )

        message = Content(role=role, parts=parts)
        return message

    def to_standard_messages(self, obj) -> list:
        """Convert Google Content object to standard structured format.

        Handles text, images, and function calls from Google's Content/Part objects.

        Args:
            obj: Google Content object with role and parts.

        Returns:
            List containing a single message in standard format.

        Examples:
            Google Content with text::

                Content(
                    role="user",
                    parts=[Part(text="Hello")]
                )

            Converts to::

                [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Hello"}]
                    }
                ]

            Google Content with function call::

                Content(
                    role="model",
                    parts=[Part(function_call=FunctionCall(name="search", args={"q": "test"}))]
                )

            Converts to::

                [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "search",
                                "type": "function",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q": "test"}'
                                }
                            }
                        ]
                    }
                ]

            Google Content with image::

                Content(
                    role="user",
                    parts=[Part(inline_data=Blob(mime_type="image/jpeg", data=bytes_data))]
                )

            Converts to::

                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/jpeg;base64,<encoded_data>"}
                            }
                        ]
                    }
                ]
        """
        msg = {"role": obj.role, "content": []}
        if msg["role"] == "model":
            msg["role"] = "assistant"

        for part in obj.parts:
            if part.text:
                msg["content"].append({"type": "text", "text": part.text})
            elif part.inline_data:
                encoded = base64.b64encode(part.inline_data.data).decode("utf-8")
                msg["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{part.inline_data.mime_type};base64,{encoded}"},
                    }
                )
            elif part.function_call:
                args = part.function_call.args if hasattr(part.function_call, "args") else {}
                msg["tool_calls"] = [
                    {
                        "id": part.function_call.name,
                        "type": "function",
                        "function": {
                            "name": part.function_call.name,
                            "arguments": json.dumps(args),
                        },
                    }
                ]

            elif part.function_response:
                msg["role"] = "tool"
                resp = (
                    part.function_response.response
                    if hasattr(part.function_response, "response")
                    else {}
                )
                msg["tool_call_id"] = part.function_response.name
                msg["content"] = json.dumps(resp)

        # there might be no content parts for tool_calls messages
        if not msg["content"]:
            del msg["content"]
        return [msg]

    def _restructure_from_openai_messages(self):
        """Restructures messages to ensure proper Google format and message ordering.

        This method handles conversion of OpenAI-formatted messages to Google format,
        with special handling for function calls, function responses, and system messages.
        System messages are added back to the context as user messages when needed.

        The final message order is preserved as:
        1. Function calls (from model)
        2. Function responses (from user)
        3. Text messages (converted from system messages)

        Note:
            System messages are only added back when there are no regular text
            messages in the context, ensuring proper conversation continuity
            after function calls.
        """
        self.system_message = None
        converted_messages = []

        # Process each message, preserving Google-formatted messages and converting others
        for message in self._messages:
            if isinstance(message, Content):
                # Keep existing Google-formatted messages (e.g., function calls/responses)
                converted_messages.append(message)
                continue

            # Convert OpenAI format to Google format, system messages return None
            converted = self.from_standard_message(message)
            if converted is not None:
                converted_messages.append(converted)

        # Update message list
        self._messages[:] = converted_messages

        # Check if we only have function-related messages (no regular text)
        has_regular_messages = any(
            len(msg.parts) == 1
            and getattr(msg.parts[0], "text", None)
            and not getattr(msg.parts[0], "function_call", None)
            and not getattr(msg.parts[0], "function_response", None)
            for msg in self._messages
        )

        # Add system message back as a user message if we only have function messages
        if self.system_message and not has_regular_messages:
            self._messages.append(Content(role="user", parts=[Part(text=self.system_message)]))

        # Remove any empty messages
        self._messages = [m for m in self._messages if m.parts]


class GoogleLLMService(LLMService):
    """Google AI (Gemini) LLM service implementation.

    This class implements inference with Google's AI models, translating internally
    from an OpenAILLMContext or a universal LLMContext to the messages format
    expected by the Google AI model.
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
        model: str = "gemini-2.5-flash",
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

    async def run_inference(self, context: LLMContext | OpenAILLMContext) -> Optional[str]:
        """Run a one-shot, out-of-band (i.e. out-of-pipeline) inference with the given LLM context.

        Args:
            context: The LLM context containing conversation history.

        Returns:
            The LLM's response as a string, or None if no response is generated.
        """
        messages = []
        system = []
        if isinstance(context, LLMContext):
            adapter = self.get_llm_adapter()
            params: GeminiLLMInvocationParams = adapter.get_llm_invocation_params(context)
            messages = params["messages"]
            system = params["system_instruction"]
        else:
            context = GoogleLLMContext.upgrade_to_google(context)
            messages = context.messages
            system = getattr(context, "system_message", None)

        generation_config = GenerateContentConfig(system_instruction=system)

        # Use the new google-genai client's async method
        response = await self._client.aio.models.generate_content(
            model=self._model_name,
            contents=messages,
            config=generation_config,
        )

        # Extract text from response
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if part.text:
                    return part.text

        return None

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
            # If we have an image model, we don't use a budget either.
            if "image" in self._model_name:
                return
            # If thinking_config is already set, don't override it.
            if "thinking_config" in generation_params:
                return
            generation_params.setdefault("thinking_config", {})["thinking_budget"] = 0
        except Exception as e:
            logger.exception(f"Failed to unset thinking budget: {e}")

    async def _stream_content(
        self, params_from_context: GeminiLLMInvocationParams
    ) -> AsyncIterator[GenerateContentResponse]:
        messages = params_from_context["messages"]
        if (
            params_from_context["system_instruction"]
            and self._system_instruction != params_from_context["system_instruction"]
        ):
            logger.debug(f"System instruction changed: {params_from_context['system_instruction']}")
            self._system_instruction = params_from_context["system_instruction"]

        tools = []
        if params_from_context["tools"]:
            tools = params_from_context["tools"]
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
        return await self._client.aio.models.generate_content_stream(
            model=self._model_name,
            contents=messages,
            config=generation_config,
        )

    async def _stream_content_specific_context(
        self, context: OpenAILLMContext
    ) -> AsyncIterator[GenerateContentResponse]:
        logger.debug(
            f"{self}: Generating chat from LLM-specific context [{context.system_message}] | {context.get_messages_for_logging()}"
        )

        params = GeminiLLMInvocationParams(
            messages=context.messages,
            system_instruction=context.system_message,
            tools=context.tools,
        )

        return await self._stream_content(params)

    async def _stream_content_universal_context(
        self, context: LLMContext
    ) -> AsyncIterator[GenerateContentResponse]:
        adapter = self.get_llm_adapter()
        params: GeminiLLMInvocationParams = adapter.get_llm_invocation_params(context)

        logger.debug(
            f"{self}: Generating chat from universal context [{params['system_instruction']}] | {adapter.get_messages_for_logging(context)}"
        )

        return await self._stream_content(params)

    @traced_llm
    async def _process_context(self, context: OpenAILLMContext | LLMContext):
        await self.push_frame(LLMFullResponseStartFrame())

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        cache_read_input_tokens = 0
        reasoning_tokens = 0

        grounding_metadata = None
        search_result = ""

        try:
            # Generate content using either OpenAILLMContext or universal LLMContext
            response = await (
                self._stream_content_specific_context(context)
                if isinstance(context, OpenAILLMContext)
                else self._stream_content_universal_context(context)
            )

            function_calls = []
            async for chunk in response:
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
                            elif part.inline_data and part.inline_data.data:
                                image = Image.open(io.BytesIO(part.inline_data.data))
                                frame = OutputImageRawFrame(
                                    image=image.tobytes(), size=image.size, format="RGB"
                                )
                                await self.push_frame(frame)

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

        if isinstance(frame, OpenAILLMContextFrame):
            context = GoogleLLMContext.upgrade_to_google(frame.context)
        elif isinstance(frame, LLMContextFrame):
            # Handle universal (LLM-agnostic) LLM context frames
            context = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            # NOTE: LLMMessagesFrame is deprecated, so we don't support the newer universal
            # LLMContext with it
            context = GoogleLLMContext(frame.messages)
        elif isinstance(frame, LLMUpdateSettingsFrame):
            await self._update_settings(frame.settings)
        else:
            await self.push_frame(frame, direction)

        if context:
            await self._process_context(context)

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(),
    ) -> GoogleContextAggregatorPair:
        """Create Google-specific context aggregators.

        Creates a pair of context aggregators optimized for Google's message format,
        including support for function calls, tool usage, and image handling.

        Args:
            context: The LLM context to create aggregators for.
            user_params: Parameters for user message aggregation.
            assistant_params: Parameters for assistant message aggregation.

        Returns:
            GoogleContextAggregatorPair: A pair of context aggregators, one for
            the user and one for the assistant, encapsulated in an
            GoogleContextAggregatorPair.

        """
        context.set_llm_adapter(self.get_llm_adapter())

        if isinstance(context, OpenAILLMContext):
            context = GoogleLLMContext.upgrade_to_google(context)
        user = GoogleUserContextAggregator(context, params=user_params)
        assistant = GoogleAssistantContextAggregator(context, params=assistant_params)
        return GoogleContextAggregatorPair(_user=user, _assistant=assistant)
