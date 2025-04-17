#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import io
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    UserImageRawFrame,
    VisionImageRawFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
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
from pipecat.services.llm_service import LLMService
from pipecat.services.openai.llm import (
    OpenAIAssistantContextAggregator,
    OpenAIUserContextAggregator,
)

# Suppress gRPC fork warnings
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

try:
    import google.ai.generativelanguage as glm
    import google.generativeai as gai
    from google.api_core.exceptions import DeadlineExceeded
    from google.generativeai.types import GenerationConfig
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Google AI, you need to `pip install pipecat-ai[google]`.")
    raise Exception(f"Missing module: {e}")


class GoogleUserContextAggregator(OpenAIUserContextAggregator):
    async def push_aggregation(self):
        if len(self._aggregation) > 0:
            self._context.add_message(
                glm.Content(role="user", parts=[glm.Part(text=self._aggregation)])
            )

            # Reset the aggregation. Reset it before pushing it down, otherwise
            # if the tasks gets cancelled we won't be able to clear things up.
            self._aggregation = ""

            # Push context frame
            frame = OpenAILLMContextFrame(self._context)
            await self.push_frame(frame)

            # Reset our accumulator state.
            self.reset()


class GoogleAssistantContextAggregator(OpenAIAssistantContextAggregator):
    async def handle_aggregation(self, aggregation: str):
        self._context.add_message(glm.Content(role="model", parts=[glm.Part(text=aggregation)]))

    async def handle_function_call_in_progress(self, frame: FunctionCallInProgressFrame):
        self._context.add_message(
            glm.Content(
                role="model",
                parts=[
                    glm.Part(
                        function_call=glm.FunctionCall(
                            id=frame.tool_call_id, name=frame.function_name, args=frame.arguments
                        )
                    )
                ],
            )
        )
        self._context.add_message(
            glm.Content(
                role="user",
                parts=[
                    glm.Part(
                        function_response=glm.FunctionResponse(
                            id=frame.tool_call_id,
                            name=frame.function_name,
                            response={"response": "IN_PROGRESS"},
                        )
                    )
                ],
            )
        )

    async def handle_function_call_result(self, frame: FunctionCallResultFrame):
        if frame.result:
            await self._update_function_call_result(
                frame.function_name, frame.tool_call_id, frame.result
            )
        else:
            await self._update_function_call_result(
                frame.function_name, frame.tool_call_id, "COMPLETED"
            )

    async def handle_function_call_cancel(self, frame: FunctionCallCancelFrame):
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
    _user: GoogleUserContextAggregator
    _assistant: GoogleAssistantContextAggregator

    def user(self) -> GoogleUserContextAggregator:
        return self._user

    def assistant(self) -> GoogleAssistantContextAggregator:
        return self._assistant


class GoogleLLMContext(OpenAILLMContext):
    def __init__(
        self,
        messages: Optional[List[dict]] = None,
        tools: Optional[List[dict]] = None,
        tool_choice: Optional[dict] = None,
    ):
        super().__init__(messages=messages, tools=tools, tool_choice=tool_choice)
        self.system_message = None

    @staticmethod
    def upgrade_to_google(obj: OpenAILLMContext) -> "GoogleLLMContext":
        if isinstance(obj, OpenAILLMContext) and not isinstance(obj, GoogleLLMContext):
            logger.debug(f"Upgrading to Google: {obj}")
            obj.__class__ = GoogleLLMContext
            obj._restructure_from_openai_messages()
        return obj

    def set_messages(self, messages: List):
        self._messages[:] = messages
        self._restructure_from_openai_messages()

    def add_messages(self, messages: List):
        # Convert each message individually
        converted_messages = []
        for msg in messages:
            if isinstance(msg, glm.Content):
                # Already in Gemini format
                converted_messages.append(msg)
            else:
                # Convert from standard format to Gemini format
                converted = self.from_standard_message(msg)
                if converted is not None:
                    converted_messages.append(converted)

        # Add the converted messages to our existing messages
        self._messages.extend(converted_messages)

    def get_messages_for_logging(self):
        msgs = []
        for message in self.messages:
            obj = glm.Content.to_dict(message)
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
        buffer = io.BytesIO()
        Image.frombytes(format, size, image).save(buffer, format="JPEG")

        parts = []
        if text:
            parts.append(glm.Part(text=text))
        parts.append(glm.Part(inline_data=glm.Blob(mime_type="image/jpeg", data=buffer.getvalue())))

        self.add_message(glm.Content(role="user", parts=parts))

    def add_audio_frames_message(
        self, *, audio_frames: list[AudioRawFrame], text: str = "Audio follows"
    ):
        if not audio_frames:
            return

        sample_rate = audio_frames[0].sample_rate
        num_channels = audio_frames[0].num_channels

        parts = []
        data = b"".join(frame.audio for frame in audio_frames)
        # NOTE(aleix): According to the docs only text or inline_data should be needed.
        # (see https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference)
        parts.append(glm.Part(text=text))
        parts.append(
            glm.Part(
                inline_data=glm.Blob(
                    mime_type="audio/wav",
                    data=(
                        bytes(
                            self.create_wav_header(sample_rate, num_channels, 16, len(data)) + data
                        )
                    ),
                )
            ),
        )
        self.add_message(glm.Content(role="user", parts=parts))
        # message = {"mime_type": "audio/mp3", "data": bytes(data + create_wav_header(sample_rate, num_channels, 16, len(data)))}
        # self.add_message(message)

    def from_standard_message(self, message):
        """Convert standard format message to Google Content object.

        Handles conversion of text, images, and function calls to Google's format.
        System messages are stored separately and return None.

        Args:
            message: Message in standard format:
                {
                    "role": "user/assistant/system/tool",
                    "content": str | [{"type": "text/image_url", ...}] | None,
                    "tool_calls": [{"function": {"name": str, "arguments": str}}]
                }

        Returns:
            glm.Content object with:
                - role: "user" or "model" (converted from "assistant")
                - parts: List[Part] containing text, inline_data, or function calls
            Returns None for system messages.
        """
        role = message["role"]
        content = message.get("content", [])
        if role == "system":
            self.system_message = content
            return None
        elif role == "assistant":
            role = "model"

        parts = []
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                parts.append(
                    glm.Part(
                        function_call=glm.FunctionCall(
                            name=tc["function"]["name"],
                            args=json.loads(tc["function"]["arguments"]),
                        )
                    )
                )
        elif role == "tool":
            role = "model"
            parts.append(
                glm.Part(
                    function_response=glm.FunctionResponse(
                        name="tool_call_result",  # seems to work to hard-code the same name every time
                        response=json.loads(message["content"]),
                    )
                )
            )
        elif isinstance(content, str):
            parts.append(glm.Part(text=content))
        elif isinstance(content, list):
            for c in content:
                if c["type"] == "text":
                    parts.append(glm.Part(text=c["text"]))
                elif c["type"] == "image_url":
                    parts.append(
                        glm.Part(
                            inline_data=glm.Blob(
                                mime_type="image/jpeg",
                                data=base64.b64decode(c["image_url"]["url"].split(",")[1]),
                            )
                        )
                    )

        message = glm.Content(role=role, parts=parts)
        return message

    def to_standard_messages(self, obj) -> list:
        """Convert Google Content object to standard structured format.

        Handles text, images, and function calls from Google's Content/Part objects.

        Args:
            obj: Google Content object with:
                - role: "model" (converted to "assistant") or "user"
                - parts: List[Part] containing text, inline_data, or function calls

        Returns:
            List of messages in standard format:
            [
                {
                    "role": "user/assistant/tool",
                    "content": [
                        {"type": "text", "text": str} |
                        {"type": "image_url", "image_url": {"url": str}}
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
                args = type(part.function_call).to_dict(part.function_call).get("args", {})
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
                    type(part.function_response).to_dict(part.function_response).get("response", {})
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
            if isinstance(message, glm.Content):
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
            and not getattr(msg.parts[0], "text", None)
            and getattr(msg.parts[0], "function_call", None)
            and getattr(msg.parts[0], "function_response", None)
            for msg in self._messages
        )

        # Add system message back as a user message if we only have function messages
        if self.system_message and not has_regular_messages:
            self._messages.append(
                glm.Content(role="user", parts=[glm.Part(text=self.system_message)])
            )

        # Remove any empty messages
        self._messages = [m for m in self._messages if m.parts]


class GoogleLLMService(LLMService):
    """This class implements inference with Google's AI models.

    This service translates internally from OpenAILLMContext to the messages format
    expected by the Google AI model. We are using the OpenAILLMContext as a lingua
    franca for all LLM services, so that it is easy to switch between different LLMs.
    """

    # Overriding the default adapter to use the Gemini one.
    adapter_class = GeminiLLMAdapter

    class InputParams(BaseModel):
        max_tokens: Optional[int] = Field(default=4096, ge=1)
        temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
        top_k: Optional[int] = Field(default=None, ge=0)
        top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
        extra: Optional[Dict[str, Any]] = Field(default_factory=dict)

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gemini-2.0-flash-001",
        params: InputParams = InputParams(),
        system_instruction: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        gai.configure(api_key=api_key)
        self.set_model_name(model)
        self._system_instruction = system_instruction
        self._create_client()
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
        return True

    def _create_client(self):
        self._client = gai.GenerativeModel(
            self._model_name, system_instruction=self._system_instruction
        )

    async def _process_context(self, context: OpenAILLMContext):
        await self.push_frame(LLMFullResponseStartFrame())

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        grounding_metadata = None
        search_result = ""

        try:
            logger.debug(
                # f"{self}: Generating chat [{self._system_instruction}] | [{context.get_messages_for_logging()}]"
                f"{self}: Generating chat [{context.get_messages_for_logging()}]"
            )

            messages = context.messages
            if context.system_message and self._system_instruction != context.system_message:
                logger.debug(f"System instruction changed: {context.system_message}")
                self._system_instruction = context.system_message
                self._create_client()

            # Filter out None values and create GenerationConfig
            generation_params = {
                k: v
                for k, v in {
                    "temperature": self._settings["temperature"],
                    "top_p": self._settings["top_p"],
                    "top_k": self._settings["top_k"],
                    "max_output_tokens": self._settings["max_tokens"],
                }.items()
                if v is not None
            }

            generation_config = GenerationConfig(**generation_params) if generation_params else None

            await self.start_ttfb_metrics()
            tools = []
            if context.tools:
                tools = context.tools
            elif self._tools:
                tools = self._tools
            tool_config = None
            if self._tool_config:
                tool_config = self._tool_config
            response = await self._client.generate_content_async(
                contents=messages,
                tools=tools,
                stream=True,
                generation_config=generation_config,
                tool_config=tool_config,
            )
            await self.stop_ttfb_metrics()

            if response.usage_metadata:
                # Use only the prompt token count from the response object
                prompt_tokens = response.usage_metadata.prompt_token_count
                total_tokens = prompt_tokens

            async for chunk in response:
                if chunk.usage_metadata:
                    # Use only the completion_tokens from the chunks. Prompt tokens are already counted and
                    # are repeated here.
                    completion_tokens += chunk.usage_metadata.candidates_token_count
                    total_tokens += chunk.usage_metadata.candidates_token_count
                try:
                    for c in chunk.parts:
                        if c.text:
                            search_result += c.text
                            await self.push_frame(LLMTextFrame(c.text))
                        elif c.function_call:
                            logger.debug(f"Function call: {c.function_call}")
                            args = type(c.function_call).to_dict(c.function_call).get("args", {})
                            await self.call_function(
                                context=context,
                                tool_call_id=str(uuid.uuid4()),
                                function_name=c.function_call.name,
                                arguments=args,
                            )
                    # Handle grounding metadata
                    # It seems only the last chunk that we receive may contain this information
                    # If the response doesn't include groundingMetadata, this means the response wasn't grounded.
                    if chunk.candidates:
                        for candidate in chunk.candidates:
                            # logger.debug(f"candidate received: {candidate}")
                            # Extract grounding metadata
                            grounding_metadata = (
                                {
                                    "rendered_content": getattr(
                                        getattr(candidate, "grounding_metadata", None),
                                        "search_entry_point",
                                        None,
                                    ).rendered_content
                                    if hasattr(
                                        getattr(candidate, "grounding_metadata", None),
                                        "search_entry_point",
                                    )
                                    else None,
                                    "origins": [
                                        {
                                            "site_uri": getattr(grounding_chunk.web, "uri", None),
                                            "site_title": getattr(
                                                grounding_chunk.web, "title", None
                                            ),
                                            "results": [
                                                {
                                                    "text": getattr(
                                                        grounding_support.segment, "text", ""
                                                    ),
                                                    "confidence": getattr(
                                                        grounding_support, "confidence_scores", None
                                                    ),
                                                }
                                                for grounding_support in getattr(
                                                    getattr(candidate, "grounding_metadata", None),
                                                    "grounding_supports",
                                                    [],
                                                )
                                                if index
                                                in getattr(
                                                    grounding_support, "grounding_chunk_indices", []
                                                )
                                            ],
                                        }
                                        for index, grounding_chunk in enumerate(
                                            getattr(
                                                getattr(candidate, "grounding_metadata", None),
                                                "grounding_chunks",
                                                [],
                                            )
                                        )
                                    ],
                                }
                                if getattr(candidate, "grounding_metadata", None)
                                else None
                            )
                except Exception as e:
                    # Google LLMs seem to flag safety issues a lot!
                    if chunk.candidates[0].finish_reason == 3:
                        logger.debug(
                            f"LLM refused to generate content for safety reasons - {messages}."
                        )
                    else:
                        logger.exception(f"{self} error: {e}")

        except DeadlineExceeded:
            await self._call_event_handler("on_completion_timeout")
        except Exception as e:
            logger.exception(f"{self} exception: {e}")
        finally:
            if grounding_metadata is not None and isinstance(grounding_metadata, dict):
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
                )
            )
            await self.push_frame(LLMFullResponseEndFrame())

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        context = None

        if isinstance(frame, OpenAILLMContextFrame):
            context = GoogleLLMContext.upgrade_to_google(frame.context)
        elif isinstance(frame, LLMMessagesFrame):
            context = GoogleLLMContext(frame.messages)
        elif isinstance(frame, VisionImageRawFrame):
            context = GoogleLLMContext()
            context.add_image_frame_message(
                format=frame.format, size=frame.size, image=frame.image, text=frame.text
            )
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
        """Create an instance of GoogleContextAggregatorPair from an
        OpenAILLMContext. Constructor keyword arguments for both the user and
        assistant aggregators can be provided.

        Args:
            context (OpenAILLMContext): The LLM context.
            user_params (LLMUserAggregatorParams, optional): User aggregator
                parameters.
            assistant_params (LLMAssistantAggregatorParams, optional): User
                aggregator parameters.

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
