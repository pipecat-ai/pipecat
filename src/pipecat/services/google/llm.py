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
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Literal, Optional

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
    LLMMessagesFrame,
    LLMTextFrame,
    LLMThoughtEndFrame,
    LLMThoughtStartFrame,
    LLMThoughtTextFrame,
    LLMUpdateSettingsFrame,
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
from pipecat.services.google.utils import update_google_client_http_options
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

    .. deprecated:: 0.0.99
        `OpenAIUserContextAggregator` is deprecated and will be removed in a future version.
        Use the universal `LLMContext` and `LLMContextAggregatorPair` instead.
        See `OpenAILLMContext` docstring for migration guide.
    """

    # Super handles deprecation warning

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

    .. deprecated:: 0.0.99
        `GoogleAssistantContextAggregator` is deprecated and will be removed in a future version.
        Use the universal `LLMContext` and `LLMContextAggregatorPair` instead.
        See `OpenAILLMContext` docstring for migration guide.
    """

    # Super handles deprecation warning

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


@dataclass
class GoogleContextAggregatorPair:
    """Pair of Google context aggregators for user and assistant messages.

    .. deprecated:: 0.0.99
        `GoogleContextAggregatorPair` is deprecated and will be removed in a future version.
        Use the universal `LLMContext` and `LLMContextAggregatorPair` instead.
        See `OpenAILLMContext` docstring for migration guide.

    Parameters:
        _user: User context aggregator for handling user messages.
        _assistant: Assistant context aggregator for handling assistant responses.
    """

    # Aggregators handle deprecation warnings
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

    .. deprecated:: 0.0.99
        `GoogleLLMContext` is deprecated and will be removed in a future version.
        Use the universal `LLMContext` and `LLMContextAggregatorPair` instead.
        See `OpenAILLMContext` docstring for migration guide.
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
        # Super handles deprecation warning
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
                    # Extract MIME type from data URL (format: "data:image/jpeg;base64,...")
                    url = c["image_url"]["url"]
                    mime_type = (
                        url.split(":")[1].split(";")[0] if url.startswith("data:") else "image/jpeg"
                    )
                    parts.append(
                        Part(
                            inline_data=Blob(
                                mime_type=mime_type,
                                data=base64.b64decode(url.split(",")[1]),
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

    class ThinkingConfig(BaseModel):
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

    class InputParams(BaseModel):
        """Input parameters for Google AI models.

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
        self._http_options = update_google_client_http_options(http_options)

        self._settings = {
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_k": params.top_k,
            "top_p": params.top_p,
            "thinking": params.thinking,
            "extra": params.extra if isinstance(params.extra, dict) else {},
        }
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

    async def run_inference(self, context: LLMContext | OpenAILLMContext) -> Optional[str]:
        """Run a one-shot, out-of-band (i.e. out-of-pipeline) inference with the given LLM context.

        Args:
            context: The LLM context containing conversation history.

        Returns:
            The LLM's response as a string, or None if no response is generated.
        """
        messages = []
        system = []
        tools = []
        if isinstance(context, LLMContext):
            adapter = self.get_llm_adapter()
            params: GeminiLLMInvocationParams = adapter.get_llm_invocation_params(context)
            messages = params["messages"]
            system = params["system_instruction"]
            tools = params["tools"]
        else:
            context = GoogleLLMContext.upgrade_to_google(context)
            messages = context.messages
            system = getattr(context, "system_message", None)
            tools = context.tools or []

        # Build generation config using the same method as streaming
        generation_params = self._build_generation_params(
            system_instruction=system, tools=tools if tools else None
        )

        generation_config = GenerateContentConfig(**generation_params)

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
                "temperature": self._settings["temperature"],
                "top_p": self._settings["top_p"],
                "top_k": self._settings["top_k"],
                "max_output_tokens": self._settings["max_tokens"],
                "tools": tools,
                "tool_config": tool_config,
            }.items()
            if v is not None
        }

        # Add thinking parameters if configured
        if self._settings["thinking"]:
            generation_params["thinking_config"] = self._settings["thinking"].model_dump(
                exclude_unset=True
            )

        if self._settings["extra"]:
            generation_params.update(self._settings["extra"])

        return generation_params

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
            logger.error(f"Failed to unset thinking budget: {e}")

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

        # Build generation parameters
        generation_params = self._build_generation_params(
            system_instruction=self._system_instruction, tools=tools, tool_config=tool_config
        )

        # possibly modify generation_params (in place) to set thinking to off by default
        self._maybe_unset_thinking_budget(generation_params)

        generation_config = GenerateContentConfig(**generation_params)

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
        accumulated_text = ""

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

    async def _update_settings(self, settings):
        """Override to handle ThinkingConfig validation."""
        # Convert thinking dict to ThinkingConfig if needed
        if "thinking" in settings and isinstance(settings["thinking"], dict):
            settings = dict(settings)  # Make a copy to avoid modifying the original
            settings["thinking"] = self.ThinkingConfig(**settings["thinking"])
        await super()._update_settings(settings)

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

        .. deprecated:: 0.0.99
            `create_context_aggregator()` is deprecated and will be removed in a future version.
            Use the universal `LLMContext` and `LLMContextAggregatorPair` instead.
            See `OpenAILLMContext` docstring for migration guide.
        """
        context.set_llm_adapter(self.get_llm_adapter())

        if isinstance(context, OpenAILLMContext):
            context = GoogleLLMContext.upgrade_to_google(context)

        # Aggregators handle deprecation warnings
        user = GoogleUserContextAggregator(context, params=user_params)
        assistant = GoogleAssistantContextAggregator(context, params=assistant_params)

        return GoogleContextAggregatorPair(_user=user, _assistant=assistant)
