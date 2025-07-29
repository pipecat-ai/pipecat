#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import inspect
import os
import re

"""Google Gemini Multimodal Live API service implementation.

This module provides real-time conversational AI capabilities using Google's
Gemini Multimodal Live API, supporting both text and audio modalities with
voice transcription, streaming responses, and tool usage.
"""

import base64
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field

from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    InputImageRawFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    StartFrame,
    StartInterruptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserImageRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    LLMMessagesFrame,
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
from pipecat.services.google.frames import LLMSearchOrigin, LLMSearchResponseFrame, LLMSearchResult
from pipecat.services.llm_service import FunctionCallFromLLM, LLMService
from pipecat.services.openai.llm import (
    OpenAIAssistantContextAggregator,
    OpenAIUserContextAggregator,
)
from pipecat.services.google.google import GoogleLLMContext

from pipecat.transcriptions.language import Language
from pipecat.utils.asyncio.watchdog_async_iterator import WatchdogAsyncIterator
from pipecat.utils.string import match_endofsentence
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_gemini_live, traced_stt

from . import events
from .file_api import GeminiFileAPI

try:
    from websockets.asyncio.client import connect as websocket_connect
    from google import genai
    from google.genai import types
    from google.genai.types import Content, LiveConnectConfig, Part, LiveClientContent, Modality
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Google AI, you need to `pip install pipecat-ai[google]`.")
    raise Exception(f"Missing module: {e}")


def language_to_gemini_language(language: Language) -> Optional[str]:
    """Maps a Language enum value to a Gemini Live supported language code.

    Source:
    https://ai.google.dev/api/generate-content#MediaResolution

    Args:
        language: The language enum value to convert.

    Returns:
        The Gemini language code string, or None if the language is not supported.
    """
    language_map = {
        # Arabic
        Language.AR: "ar-XA",
        # Bengali
        Language.BN_IN: "bn-IN",
        # Chinese (Mandarin)
        Language.CMN: "cmn-CN",
        Language.CMN_CN: "cmn-CN",
        Language.ZH: "cmn-CN",  # Map general Chinese to Mandarin for Gemini
        Language.ZH_CN: "cmn-CN",  # Map Simplified Chinese to Mandarin for Gemini
        # German
        Language.DE: "de-DE",
        Language.DE_DE: "de-DE",
        # English
        Language.EN: "en-US",  # Default to US English (though not explicitly listed in supported codes)
        Language.EN_US: "en-US",
        Language.EN_AU: "en-AU",
        Language.EN_GB: "en-GB",
        Language.EN_IN: "en-IN",
        # Spanish
        Language.ES: "es-ES",  # Default to Spain Spanish
        Language.ES_ES: "es-ES",
        Language.ES_US: "es-US",
        # French
        Language.FR: "fr-FR",  # Default to France French
        Language.FR_FR: "fr-FR",
        Language.FR_CA: "fr-CA",
        # Gujarati
        Language.GU: "gu-IN",
        Language.GU_IN: "gu-IN",
        # Hindi
        Language.HI: "hi-IN",
        Language.HI_IN: "hi-IN",
        # Indonesian
        Language.ID: "id-ID",
        Language.ID_ID: "id-ID",
        # Italian
        Language.IT: "it-IT",
        Language.IT_IT: "it-IT",
        # Japanese
        Language.JA: "ja-JP",
        Language.JA_JP: "ja-JP",
        # Kannada
        Language.KN: "kn-IN",
        Language.KN_IN: "kn-IN",
        # Korean
        Language.KO: "ko-KR",
        Language.KO_KR: "ko-KR",
        # Malayalam
        Language.ML: "ml-IN",
        Language.ML_IN: "ml-IN",
        # Marathi
        Language.MR: "mr-IN",
        Language.MR_IN: "mr-IN",
        # Dutch
        Language.NL: "nl-NL",
        Language.NL_NL: "nl-NL",
        # Polish
        Language.PL: "pl-PL",
        Language.PL_PL: "pl-PL",
        # Portuguese (Brazil)
        Language.PT_BR: "pt-BR",
        # Russian
        Language.RU: "ru-RU",
        Language.RU_RU: "ru-RU",
        # Tamil
        Language.TA: "ta-IN",
        Language.TA_IN: "ta-IN",
        # Telugu
        Language.TE: "te-IN",
        Language.TE_IN: "te-IN",
        # Thai
        Language.TH: "th-TH",
        Language.TH_TH: "th-TH",
        # Turkish
        Language.TR: "tr-TR",
        Language.TR_TR: "tr-TR",
        # Vietnamese
        Language.VI: "vi-VN",
        Language.VI_VN: "vi-VN",
    }
    return language_map.get(language)


class GeminiMultimodalLiveContext(OpenAILLMContext):
    """Extended OpenAI context for Gemini Multimodal Live API.

    Provides Gemini-specific context management including system instruction
    extraction and message format conversion for the Live API.
    """

    @staticmethod
    def upgrade(obj: OpenAILLMContext) -> "GeminiMultimodalLiveContext":
        """Upgrade an OpenAI context to Gemini context.

        Args:
            obj: The OpenAI context to upgrade.

        Returns:
            The upgraded Gemini context instance.
        """
        if isinstance(obj, OpenAILLMContext) and not isinstance(obj, GeminiMultimodalLiveContext):
            logger.debug(f"Upgrading to Gemini Multimodal Live Context: {obj}")
            obj.__class__ = GeminiMultimodalLiveContext
            obj._restructure_from_openai_messages()
        return obj

    def set_messages(self, messages: List):
        self._messages[:] = messages
        self._restructure_from_openai_messages()

    def add_messages(self, messages: List):
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
        # self._messages.append(converted_messages)
        self._messages.extend(converted_messages)
        self._restructure_from_openai_messages()

    def from_standard_message(self, message):
        """Convert standard format message to Google Content object.

        Handles conversion of text, images, and function calls to Google's format.

        Args:
            message: Message in standard format:
                {
                    "role": "user/assistant/system/tool",
                    "content": str | [{"type": "text/image_url", ...}] | None,
                    "tool_calls": [{"function": {"name": str, "arguments": str}}]
                }

        Returns:
            Content object with:
                - role: "user" or "model" (converted from "assistant")
                - parts: List[Part] containing text, inline_data, or function calls
            Returns None for system messages.
        """
        role = message["role"]
        content = message.get("content", [])
        if role == "system":
            # don't think this is needed anymore
            self.system_message = content
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
            parts.append(
                Part(
                    function_response=FunctionResponse(
                        name="tool_call_result",  # seems to work to hard-code the same name every time
                        response=json.loads(message["content"]),
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

        ## this is broken... ?
        # # Check if we only have function-related messages (no regular text)
        # has_regular_messages = any(
        #     len(msg.parts) == 1
        #     and not getattr(msg.parts[0], "text", None)
        #     and getattr(msg.parts[0], "function_call", None)
        #     and getattr(msg.parts[0], "function_response", None)
        #     for msg in self._messages
        # )
        # # Add system message back as a user message if we only have function messages
        # if self.system_message and not has_regular_messages:
        #     self._messages.append(Content(role="user", parts=[Part(text=self.system_message)]))

        # Remove any empty messages
        self._messages = [m for m in self._messages if m.parts]

    def extract_system_instructions(self):
        """Extract system instructions from context messages.

        Returns:
            Combined system instruction text from all system messages.
        """
        system_instruction = ""
        for item in self.messages:
            if item.get("role") == "system":
                content = item.get("content", "")
                if content:
                    if system_instruction and not system_instruction.endswith("\n"):
                        system_instruction += "\n"
                    system_instruction += str(content)
        return system_instruction

    def add_file_reference(self, file_uri: str, mime_type: str, text: Optional[str] = None):
        """Add a file reference to the context.

        This adds a user message with a file reference that will be sent during context initialization.

        Args:
            file_uri: URI of the uploaded file
            mime_type: MIME type of the file
            text: Optional text prompt to accompany the file
        """
        # Create parts list with file reference
        parts = []
        if text:
            parts.append({"type": "text", "text": text})

        # Add file reference part
        parts.append(
            {"type": "file_data", "file_data": {"mime_type": mime_type, "file_uri": file_uri}}
        )

        # Add to messages
        message = {"role": "user", "content": parts}
        self.messages.append(message)
        logger.info(f"Added file reference to context: {file_uri}")

    def get_messages_for_initializing_history(self):
        """Get messages formatted for Gemini history initialization.

        Returns:
            List of messages in Gemini format for conversation history.
        """
        messages = []
        for item in self.messages:
            role = item.get("role")

            if role == "system":
                continue

            elif role == "assistant":
                role = "model"

            content = item.get("content")
            parts = []
            if isinstance(content, str):
                parts = [{"text": content}]
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        parts.append({"text": part.get("text")})
                    elif part.get("type") == "file_data":
                        file_data = part.get("file_data", {})

                        parts.append(
                            {
                                "fileData": {
                                    "mimeType": file_data.get("mime_type"),
                                    "fileUri": file_data.get("file_uri"),
                                }
                            }
                        )
                    else:
                        logger.warning(f"Unsupported content type: {str(part)[:80]}")
            else:
                logger.warning(f"Unsupported content type: {str(content)[:80]}")
            messages.append({"role": role, "parts": parts})
        return messages


class GeminiMultimodalLiveUserContextAggregator(OpenAIUserContextAggregator):
    """User context aggregator for Gemini Multimodal Live.

    Extends OpenAI user aggregator to handle Gemini-specific message passing
    while maintaining compatibility with the standard aggregation pipeline.
    """

    async def handle_aggregation(self, aggregation: str):
        """Add the aggregated user text to the context.

        Args:
            aggregation: The aggregated user text to add as a user message.
        """
        turn = {"role": self.role, "content": aggregation}
        converted = self._context.from_standard_message(turn)
        self._context.add_message(converted)

    async def process_frame(self, frame, direction):
        """Process incoming frames for user context aggregation.

        Args:
            frame: The frame to process.
            direction: The frame processing direction.
        """
        await super().process_frame(frame, direction)
        # kind of a hack just to pass the LLMMessagesAppendFrame through, but it's fine for now
        if isinstance(frame, LLMMessagesAppendFrame):
            await self.push_frame(frame, direction)


class GeminiMultimodalLiveAssistantContextAggregator(OpenAIAssistantContextAggregator):
    """Assistant context aggregator for Gemini Multimodal Live.

    Handles assistant response aggregation while filtering out LLMTextFrames
    to prevent duplicate context entries, as Gemini Live pushes both
    LLMTextFrames and TTSTextFrames.
    """

    async def handle_aggregation(self, aggregation: str):
        """Add the aggregated user text to the context.

        Args:
            aggregation: The aggregated user text to add as a user message.
        """
        turn = {"role": "assistant", "content": aggregation}
        converted = self._context.from_standard_message(turn)
        self._context.add_message(converted)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames for assistant context aggregation.

        Args:
            frame: The frame to process.
            direction: The frame processing direction.
        """
        # The LLMAssistantContextAggregator uses TextFrames to aggregate the LLM output,
        # but the GeminiMultimodalLiveAssistantContextAggregator pushes LLMTextFrames and TTSTextFrames. We
        # need to override this proces_frame for LLMTextFrame, so that only the TTSTextFrames
        # are process. This ensures that the context gets only one set of messages.
        if not isinstance(frame, LLMTextFrame):
            await super().process_frame(frame, direction)

    async def handle_user_image_frame(self, frame: UserImageRawFrame):
        """Handle user image frames.

        Args:
            frame: The user image frame to handle.
        """
        # We don't want to store any images in the context. Revisit this later
        # when the API evolves.
        pass


@dataclass
class GeminiMultimodalLiveContextAggregatorPair:
    """Pair of user and assistant context aggregators for Gemini Multimodal Live.

    Parameters:
        _user: The user context aggregator instance.
        _assistant: The assistant context aggregator instance.
    """

    _user: GeminiMultimodalLiveUserContextAggregator
    _assistant: GeminiMultimodalLiveAssistantContextAggregator

    def user(self) -> GeminiMultimodalLiveUserContextAggregator:
        """Get the user context aggregator.

        Returns:
            The user context aggregator instance.
        """
        return self._user

    def assistant(self) -> GeminiMultimodalLiveAssistantContextAggregator:
        """Get the assistant context aggregator.

        Returns:
            The assistant context aggregator instance.
        """
        return self._assistant


class GeminiMultimodalModalities(Enum):
    """Supported modalities for Gemini Multimodal Live.

    Parameters:
        TEXT: Text responses.
        AUDIO: Audio responses.
    """

    TEXT = "TEXT"
    AUDIO = "AUDIO"


class GeminiMediaResolution(str, Enum):
    """Media resolution options for Gemini Multimodal Live.

    Parameters:
        UNSPECIFIED: Use default resolution setting.
        LOW: Low resolution with 64 tokens.
        MEDIUM: Medium resolution with 256 tokens.
        HIGH: High resolution with zoomed reframing and 256 tokens.
    """

    UNSPECIFIED = "MEDIA_RESOLUTION_UNSPECIFIED"  # Use default
    LOW = "MEDIA_RESOLUTION_LOW"  # 64 tokens
    MEDIUM = "MEDIA_RESOLUTION_MEDIUM"  # 256 tokens
    HIGH = "MEDIA_RESOLUTION_HIGH"  # Zoomed reframing with 256 tokens


class GeminiVADParams(BaseModel):
    """Voice Activity Detection parameters for Gemini Live.

    Parameters:
        disabled: Whether to disable VAD. Defaults to None.
        start_sensitivity: Sensitivity for speech start detection. Defaults to None.
        end_sensitivity: Sensitivity for speech end detection. Defaults to None.
        prefix_padding_ms: Prefix padding in milliseconds. Defaults to None.
        silence_duration_ms: Silence duration threshold in milliseconds. Defaults to None.
    """

    disabled: Optional[bool] = Field(default=None)
    start_sensitivity: Optional[events.StartSensitivity] = Field(default=None)
    end_sensitivity: Optional[events.EndSensitivity] = Field(default=None)
    prefix_padding_ms: Optional[int] = Field(default=None)
    silence_duration_ms: Optional[int] = Field(default=None)


class ContextWindowCompressionParams(BaseModel):
    """Parameters for context window compression in Gemini Live.

    Parameters:
        enabled: Whether compression is enabled. Defaults to False.
        trigger_tokens: Token count to trigger compression. None uses 80% of context window.
    """

    enabled: bool = Field(default=False)
    trigger_tokens: Optional[int] = Field(
        default=None
    )  # None = use default (80% of context window)


class GoogleVertexMultimodalLiveLLMService(LLMService):
    """Provides access to Google's Gemini Multimodal Live API via Vertex AI.

    This service enables real-time conversations with Gemini, supporting both
    text and audio modalities. It handles voice transcription, streaming audio
    responses, and tool usage.

    Args:
        api_key (str): Google AI API key
        model (str, optional): Model identifier to use. Defaults to
            "models/gemini-2.0-flash-live-001".
        voice_id (str, optional): TTS voice identifier. Defaults to "Charon".
        start_audio_paused (bool, optional): Whether to start with audio input paused.
            Defaults to False.
        start_video_paused (bool, optional): Whether to start with video input paused.
            Defaults to False.
        system_instruction (str, optional): System prompt for the model. Defaults to None.
        tools (Union[List[dict], ToolsSchema], optional): Tools/functions available to the model.
            Defaults to None.
        params (InputParams, optional): Configuration parameters for the model.
            Defaults to InputParams().
        inference_on_context_initialization (bool, optional): Whether to generate a response
            when context is first set. Defaults to True.
    """

    class InputParams(BaseModel):
        """Input parameters for Gemini Multimodal Live generation in Vertex AI.

        Parameters:
            project_id: [required] Google Cloud project ID.
            context_window_compression: Context compression settings. Defaults to None.
            extra: Additional parameters. Defaults to empty dict.
            frequency_penalty: Frequency penalty for generation (0.0-2.0). Defaults to None.
            language: Language for generation. Defaults to EN_US.
            location: GCP region for Vertex AI endpoint. Defaults to "us-east4". https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations
            max_tokens: Maximum tokens to generate. Must be >= 1. Defaults to 4096.
            media_resolution: Media resolution setting. Defaults to UNSPECIFIED.
            modalities: Response modalities. Defaults to AUDIO.
            presence_penalty: Presence penalty for generation (0.0-2.0). Defaults to None.
            temperature: Sampling temperature (0.0-2.0). Defaults to None.
            top_k: Top-k sampling parameter. Must be >= 0. Defaults to None.
            top_p: Top-p sampling parameter (0.0-1.0). Defaults to None.
            vad: Voice activity detection parameters. Defaults to None.
        """

        project_id: str
        context_window_compression: Optional[ContextWindowCompressionParams] = Field(default=None)
        extra: Optional[Dict[str, Any]] = Field(default_factory=dict)
        frequency_penalty: Optional[float] = Field(default=None, ge=0.0, le=2.0)
        language: Optional[Language] = Field(default=Language.EN_US)
        location: str = "us-east4"
        max_tokens: Optional[int] = Field(default=4096, ge=1)
        media_resolution: Optional[GeminiMediaResolution] = Field(
            default=GeminiMediaResolution.UNSPECIFIED
        )
        modalities: Optional[GeminiMultimodalModalities] = Field(
            default=GeminiMultimodalModalities.AUDIO
        )
        presence_penalty: Optional[float] = Field(default=None, ge=0.0, le=2.0)
        temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
        top_k: Optional[int] = Field(default=None, ge=0)
        top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
        vad: Optional[GeminiVADParams] = Field(default=None)

    # Overriding the default adapter to use the Gemini one.
    adapter_class = GeminiLLMAdapter

    def __init__(
        self,
        *,
        api_key: str,
        model="models/gemini-2.0-flash-live-001",
        voice_id: str = "Charon",
        start_audio_paused: bool = False,
        start_video_paused: bool = False,
        system_instruction: Optional[str] = None,
        tools: Optional[Union[List[dict], ToolsSchema]] = None,
        params: Optional[InputParams] = None,
        inference_on_context_initialization: bool = True,
        file_api_base_url: str = "https://generativelanguage.googleapis.com/v1beta/files",
        **kwargs,
    ):
        """Initialize the Gemini Multimodal Live LLM service.

        Args:
            api_key: Google AI API key for authentication.
            base_url: API endpoint base URL. Defaults to the official Gemini Live endpoint.
            model: Model identifier to use. Defaults to "models/gemini-2.0-flash-live-001".
            voice_id: TTS voice identifier. Defaults to "Charon".
            start_audio_paused: Whether to start with audio input paused. Defaults to False.
            start_video_paused: Whether to start with video input paused. Defaults to False.
            system_instruction: System prompt for the model. Defaults to None.
            tools: Tools/functions available to the model. Defaults to None.
            params: Configuration parameters for the model. Defaults to InputParams().
            inference_on_context_initialization: Whether to generate a response when context
                is first set. Defaults to True.
            file_api_base_url: Base URL for the Gemini File API. Defaults to the official endpoint.
            **kwargs: Additional arguments passed to parent LLMService.
        """
        super().__init__(**kwargs)

        params = params or InputParams()

        self._last_sent_time = 0
        self._api_key = api_key
        self._client = self._create_client(api_key)
        self.set_model_name(model)
        self._voice_id = voice_id
        self._language_code = params.language

        self._system_instruction = system_instruction
        self._tools = tools
        self._inference_on_context_initialization = inference_on_context_initialization
        self._needs_turn_complete_message = False

        self._audio_input_paused = start_audio_paused
        self._video_input_paused = start_video_paused
        self._context = None
        self._websocket = None
        self._receive_task = None

        self._disconnecting = False
        self._api_session_ready = False
        self._run_llm_when_api_session_ready = False

        self._user_is_speaking = False
        self._bot_is_speaking = False
        self._user_audio_buffer = bytearray()
        self._user_transcription_buffer = ""
        self._last_transcription_sent = ""
        self._bot_audio_buffer = bytearray()
        self._bot_text_buffer = ""
        self._llm_output_buffer = ""

        self._sample_rate = 16000
        # self._sample_rate = 24000

        self._language = params.language
        self._language_code = (
            language_to_gemini_language(params.language) if params.language else "en-US"
        )
        self._vad_params = params.vad

        self._settings = {
            "frequency_penalty": params.frequency_penalty,
            "max_tokens": params.max_tokens,
            "presence_penalty": params.presence_penalty,
            "temperature": params.temperature,
            "top_k": params.top_k,
            "top_p": params.top_p,
            "modalities": params.modalities,
            "language": self._language_code,
            "media_resolution": params.media_resolution,
            "vad": params.vad,
            "context_window_compression": params.context_window_compression.model_dump()
            if params.context_window_compression
            else {},
            "extra": params.extra if isinstance(params.extra, dict) else {},
        }

        # Initialize the File API client
        self.file_api = GeminiFileAPI(api_key=api_key, base_url=file_api_base_url)

        # Grounding metadata tracking
        self._search_result_buffer = ""
        self._accumulated_grounding_metadata = None

        self._config = LiveConnectConfig(
            response_modalities=[params.modalities],
            #         speech_config=SpeechConfig(
            #         voice_config=VoiceConfig(
            #             prebuilt_voice_config=PrebuiltVoiceConfig(
            #             voice_name=self._voice_id,
            #             )
            #         ),
            #     ),
        )

        ## probably not needed
        self._lcc = LiveClientContent()

    def set_model_name(self, model: str):
        sanitized_model = re.sub("models/", "", model)
        super().set_model_name(sanitized_model)

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate usage metrics.

        Returns:
            True as Gemini Live supports token usage metrics.
        """
        return True

    def needs_mcp_alternate_schema(self) -> bool:
        """Check if this LLM service requires alternate MCP schema.

        Google/Gemini has stricter JSON schema validation and requires
        certain properties to be removed or modified for compatibility.

        Returns:
            True for Google/Gemini services.
        """
        return True

    def _create_client(self, api_key: str):
        return genai.Client(
            vertexai=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT_ID"),
            location="us-central1",
            # http_options={"api_version": "v1beta"}
        )

    def set_audio_input_paused(self, paused: bool):
        """Set the audio input pause state.

        Args:
            paused: Whether to pause audio input.
        """
        self._audio_input_paused = paused

    def set_video_input_paused(self, paused: bool):
        """Set the video input pause state.

        Args:
            paused: Whether to pause video input.
        """
        self._video_input_paused = paused

    def set_model_modalities(self, modalities: GeminiMultimodalModalities):
        """Set the model response modalities.

        Args:
            modalities: The modalities to use for responses.
        """
        self._settings["modalities"] = modalities

    def set_language(self, language: Language):
        """Set the language for generation.

        Args:
            language: The language to use for generation.
        """
        self._language = language
        self._language_code = language_to_gemini_language(language) or "en-US"
        self._settings["language"] = self._language_code
        logger.info(f"Set Gemini language to: {self._language_code}")

    async def set_context(self, context: OpenAILLMContext):
        """Set the context explicitly from outside the pipeline.

        This is useful when initializing a conversation because in server-side VAD mode we might not have a
        way to trigger the pipeline. This sends the history to the server. The `inference_on_context_initialization`
        flag controls whether to set the turnComplete flag when we do this. Without that flag, the model will
        not respond. This is often what we want when setting the context at the beginning of a conversation.

        Args:
            context: The OpenAI LLM context to set.
        """
        if self._context:
            logger.error(
                "Context already set. Can only set up Gemini Multimodal Live context once."
            )
            return
        self._context = GeminiMultimodalLiveContext.upgrade(context)
        await self._create_initial_response()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        # await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def _disconnect(self):
        self._disconnecting = True
        self._api_session_ready = False
        await self.stop_all_metrics()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        self._disconnecting = False

    #
    # speech and interruption handling
    #

    async def _handle_interruption(self):
        self._bot_is_speaking = False
        await self.push_frame(TTSStoppedFrame())
        await self.push_frame(LLMFullResponseEndFrame())

    async def _handle_user_started_speaking(self, frame):
        self._user_is_speaking = True
        pass

    async def _handle_user_stopped_speaking(self, frame):
        # print(f"_____gemini_vertex.py * UserStoppedSpeakingFrame  self._context.messages: {self._context.messages}")
        self._receive_task = self.create_task(
            self._receive_task_handler(self._context, "$ $ $ UserStoppedSpeakingFrame three")
        )
        self._user_is_speaking = False
        self._user_audio_buffer = bytearray()
        await self.start_ttfb_metrics()
        if self._needs_turn_complete_message:
            self._needs_turn_complete_message = False
            evt = events.ClientContentMessage.model_validate(
                {"clientContent": {"turnComplete": True}}
            )

    async def send_client_event(self, evt):
        pass

    #
    # frame processing
    #
    # StartFrame, StopFrame, CancelFrame implemented in base class
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames for the Gemini Live service.

        Args:
            frame: The frame to process.
            direction: The frame processing direction.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            await self.push_frame(frame, direction)

        elif isinstance(frame, OpenAILLMContextFrame):
            # print(f"_____gemini.py * OpenAILLMContextFrame: {frame.context}")
            context: GeminiMultimodalLiveContext = GeminiMultimodalLiveContext.upgrade(
                frame.context
            )

            # For now, we'll only trigger inference here when either:
            #   1. We have not seen a context frame before
            #   2. The last message is a tool call result
            if not self._context:
                self._context = context
                if not self._receive_task:
                    self._receive_task = self.create_task(
                        self._receive_task_handler(self._context, "$ $ $ OpenAILLMContextFrame one")
                    )
                if frame.context.tools:
                    self._tools = frame.context.tools
                await self._create_initial_response()
            elif context.messages and context.messages[-1] and context.messages[-1].role == "tool":
                # Support just one tool call per context frame for now
                tool_result_message = context.messages[-1]
                await self._tool_result(tool_result_message)
            elif context.messages and context.messages[-1] and context.messages[-1].role == "model":
                self._context = context
                # await self.push_frame(frame, direction)

        elif isinstance(frame, LLMMessagesFrame):
            if frame.messages and frame.messages[-1] and frame.messages[-1].role == "system":
                self._context.add_messages(frame.messages)
                # self._context.set_messages(frame.messages)
                self._receive_task = self.create_task(
                    self._receive_task_handler(self._context, "$ $ $ LLMMessagesFrame two")
                )
            await self.push_frame(frame, direction)

        elif isinstance(frame, InputAudioRawFrame):
            await self._send_user_audio(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, InputImageRawFrame):
            await self._send_user_video(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, StartInterruptionFrame):
            await self._handle_interruption()
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStartedSpeakingFrame):
            await self._handle_user_started_speaking(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._handle_user_stopped_speaking(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, BotStartedSpeakingFrame):
            # Ignore this frame. Use the serverContent API message instead
            await self.push_frame(frame, direction)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            # ignore this frame. Use the serverContent.turnComplete API message
            await self.push_frame(frame, direction)
        elif isinstance(frame, LLMMessagesAppendFrame):
            await self._create_single_response(frame.messages)
        elif isinstance(frame, LLMUpdateSettingsFrame):
            await self._update_settings(frame.settings)
        elif isinstance(frame, LLMSetToolsFrame):
            await self._update_settings()
        else:
            await self.push_frame(frame, direction)

    # https://github.com/google-gemini/cookbook/issues/781
    # https://github.com/google-gemini/cookbook/blob/cb04a04359ac7937c4b22e8b4c381451ba1e5d93/quickstarts/Get_started_LiveAPI.py

    ### audio task handler
    async def _receive_task_handler_audio(self, context, blame, frame_audio, frame_sample_rate):
        # print(f"_____gemini.py * _receive_task_handler:::: {blame}")

        async with self._client.aio.live.connect(
            model=self._model_name,
            config=self._config,
        ) as session:
            try:
                if not frame_audio or GeminiMultimodalModalities.TEXT == self._settings["modalities"]:
                    print(
                        f"_____gemini_vertex.py * self._context.messages: {self._context.messages}"
                    )
                    await session.send_client_content(turns=self._context.messages)

                elif GeminiMultimodalModalities.AUDIO == self._settings["modalities"]:
                    await session.send_realtime_input(
                        media=types.Blob(
                            data=frame_audio, mime_type=f"audio/pcm;rate={frame_sample_rate}"
                        )
                        # audio=types.Blob(data=self._user_audio_buffer, mime_type=f"audio/pcm;rate={self._sample_rate}")
                    )

                else:
                    pass

                async for message in session.receive():
                    print(f"_________________________________________________gemini_vertex.py * message: {message}")
                    #   TODO  # don't forget to Check for grounding metadata in server content
                    #     if evt.serverContent and evt.serverContent.groundingMetadata:
                    #         self._accumulated_grounding_metadata = evt.serverContent.groundingMetadata

                    if message.text:
                        print(f"_____gemini.py * message.text::::::: {message.text}")

                        if not self._bot_is_speaking:
                            self._bot_is_speaking = True
                            await self.push_frame(TTSStartedFrame())
                            await self.push_frame(LLMFullResponseStartFrame())
                        await self.push_frame(LLMTextFrame(message.text))
                        # await self.push_frame(LLMTextFrame("something else."))

                    ## WIP audio
                    elif message.data:
                        print(f"_____gemini_vertex.py * audio:::")
                        # https://cloud.google.com/vertex-ai/generative-ai/docs/live-api#:~:text=Vertex%20AI%20Studio.-,Context%20window,inputs%2C%20model%20outputs%2C%20etc.
                        if (
                            message.server_content.model_turn
                            and message.server_content.model_turn.parts
                        ):
                            audio_data = []

                            for part in message.server_content.model_turn.parts:
                                if part.inline_data:
                                    inline_data = part.inline_data
                                    if not inline_data:
                                        return
                                    # if inline_data.mime_type != f"audio/pcm;rate={self._sample_rate}":
                                    if inline_data.mime_type != f"audio/pcm":
                                        logger.warning(
                                            f"Unrecognized server_content format {inline_data.mime_type}"
                                        )
                                        return

                                    audio = base64.b64decode(inline_data.data)
                                    if not audio:
                                        return

                                    if not self._bot_is_speaking:
                                        self._bot_is_speaking = True
                                        await self.push_frame(TTSStartedFrame())
                                        await self.push_frame(LLMFullResponseStartFrame())

                                    self._bot_audio_buffer.extend(audio)
                                    frame = TTSAudioRawFrame(
                                        audio=audio,
                                        sample_rate=self._sample_rate,
                                        num_channels=1,
                                    )
                                    print(f"_____gemini_vertex.py * frame: {frame}")
                                    await self.push_frame(frame)

            except Exception as e:
                print(f"_TODO throw exception?__eeee__gemini.py * e: {e}")

            await self.push_frame(LLMFullResponseEndFrame())
            self._bot_is_speaking = False
            print(f"::::::::::::_____gemini_vertex.py * _receive_task_handler end:::::::")

    #### og
    async def _receive_task_handler(self, context, blame):
        print(f"_____gemini.py * _receive_task_handler:::: {blame}")

        async with self._client.aio.live.connect(
            model=self._model_name,
            config=self._config,
        ) as session:
            try:
                # force text usage for system message setting... how to output audio though :thinking:
                if True:
                # if GeminiMultimodalModalities.TEXT == self._settings["modalities"]:
                    print(
                        f"_____gemini_vertex.py * self._context.messages: {self._context.messages}"
                    )
                    await session.send_client_content(turns=self._context.messages)

                elif GeminiMultimodalModalities.AUDIO == self._settings["modalities"]:
                    await session.send_realtime_input(
                        media=types.Blob(
                            data=self._user_audio_buffer, mime_type=f"audio/pcm;rate=1600"
                        )
                        # audio=types.Blob(data=self._user_audio_buffer, mime_type=f"audio/pcm;rate={self._sample_rate}")
                    )

                else:
                    pass

                async for message in session.receive():
                    print(f"_________________________________________________gemini_vertex.py * message: {message}")
                    #   TODO  # don't forget to Check for grounding metadata in server content
                    #     if evt.serverContent and evt.serverContent.groundingMetadata:
                    #         self._accumulated_grounding_metadata = evt.serverContent.groundingMetadata

                    if message.text:
                        print(f"_____gemini.py * message.text::::::: {message.text}")

                        if not self._bot_is_speaking:
                            self._bot_is_speaking = True
                            await self.push_frame(TTSStartedFrame())
                            await self.push_frame(LLMFullResponseStartFrame())
                        await self.push_frame(LLMTextFrame(message.text))
                        # await self.push_frame(LLMTextFrame("something else."))

                    ## WIP audio
                    elif message.data:
                        print(f"_____gemini_vertex.py * audio:::")
                        # https://cloud.google.com/vertex-ai/generative-ai/docs/live-api#:~:text=Vertex%20AI%20Studio.-,Context%20window,inputs%2C%20model%20outputs%2C%20etc.
                        if (
                            message.server_content.model_turn
                            and message.server_content.model_turn.parts
                        ):
                            audio_data = []

                            for part in message.server_content.model_turn.parts:
                                if part.inline_data:
                                    inline_data = part.inline_data
                                    if not inline_data:
                                        return
                                    # if inline_data.mime_type != f"audio/pcm;rate={self._sample_rate}":
                                    if inline_data.mime_type != f"audio/pcm":
                                        logger.warning(
                                            f"Unrecognized server_content format {inline_data.mime_type}"
                                        )
                                        return

                                    audio = base64.b64decode(inline_data.data)
                                    if not audio:
                                        return

                                    if not self._bot_is_speaking:
                                        self._bot_is_speaking = True
                                        await self.push_frame(TTSStartedFrame())
                                        await self.push_frame(LLMFullResponseStartFrame())

                                    self._bot_audio_buffer.extend(audio)
                                    frame = TTSAudioRawFrame(
                                        audio=audio,
                                        sample_rate=self._sample_rate,
                                        num_channels=1,
                                    )
                                    print(f"_____gemini_vertex.py * frame: {frame}")
                                    await self.push_frame(frame)
                                    await self.push_frame(frame)

            except Exception as e:
                print(f"_TODO throw exception?__eeee__gemini.py * e: {e}")

            await self.push_frame(LLMFullResponseEndFrame())
            self._bot_is_speaking = False
            print(f"::::::::::::_____gemini_vertex.py * _receive_task_handler end:::::::")

    #
    #
    #

    async def _send_user_audio(self, frame):
        """Send user audio frame to Gemini Live API."""
        if self._audio_input_paused:
            return
        
        #### meh, how to send audio... I think I just need to keep the session open
        # and handle all responses, including turns
        self._receive_task = self.create_task(self._receive_task_handler_audio(self._context, "$ $ $ _send_user_audio four", frame.audio, frame.sample_rate))
        
        # # Send all audio to Gemini
        # evt = events.AudioInputMessage.from_raw_audio(frame.audio, frame.sample_rate)
        # # await self.send_client_event(evt)

        # Manage a buffer of audio to use for transcription
        audio = frame.audio
        if self._user_is_speaking:
            self._user_audio_buffer.extend(audio)
        else:
            # Keep 1/2 second of audio in the buffer even when not speaking.
            self._user_audio_buffer.extend(audio)
            length = int((frame.sample_rate * frame.num_channels * 2) * 0.5)
            self._user_audio_buffer = self._user_audio_buffer[-length:]


    async def _send_user_video(self, frame):
        """Send user video frame to Gemini Live API."""
        if self._video_input_paused:
            return

        now = time.time()
        if now - self._last_sent_time < 1:
            return  # Ignore if less than 1 second has passed

        self._last_sent_time = now  # Update last sent time
        logger.debug(f"Sending video frame to Gemini: {frame}")
        evt = events.VideoInputMessage.from_image_frame(frame)
        await self.send_client_event(evt)

    async def _create_initial_response(self):
        """Create initial response based on context history."""
        if not self._api_session_ready:
            self._run_llm_when_api_session_ready = True
            return

        messages = self._context.get_messages_for_initializing_history()
        if not messages:
            return

        logger.debug(f"Creating initial response: {messages}")

        await self.start_ttfb_metrics()

        evt = events.ClientContentMessage.model_validate(
            {
                "clientContent": {
                    "turns": messages,
                    "turnComplete": self._inference_on_context_initialization,
                }
            }
        )
        await self.send_client_event(evt)
        if not self._inference_on_context_initialization:
            self._needs_turn_complete_message = True

    async def _create_single_response(self, messages_list):
        """Create a single response from a list of messages."""
        # Refactor to combine this logic with same logic in GeminiMultimodalLiveContext
        messages = []
        for item in messages_list:
            role = item.get("role")

            if role == "system":
                continue

            elif role == "assistant":
                role = "model"

            content = item.get("content")
            parts = []
            if isinstance(content, str):
                parts = [{"text": content}]
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        parts.append({"text": part.get("text")})
                    elif part.get("type") == "file_data":
                        file_data = part.get("file_data", {})

                        parts.append(
                            {
                                "fileData": {
                                    "mimeType": file_data.get("mime_type"),
                                    "fileUri": file_data.get("file_uri"),
                                }
                            }
                        )
                    else:
                        logger.warning(f"Unsupported content type: {str(part)[:80]}")
            else:
                logger.warning(f"Unsupported content type: {str(content)[:80]}")
            messages.append({"role": role, "parts": parts})
        if not messages:
            return
        logger.debug(f"Creating response: {messages}")

        await self.start_ttfb_metrics()

        evt = events.ClientContentMessage.model_validate(
            {
                "clientContent": {
                    "turns": messages,
                    "turnComplete": True,
                }
            }
        )
        # await self.send_client_event(evt)

    @traced_gemini_live(operation="llm_tool_result")
    async def _tool_result(self, tool_result_message):
        """Send tool result back to the API."""
        # For now we're shoving the name into the tool_call_id field, so this
        # will work until we revisit that.
        id = tool_result_message.get("tool_call_id")
        name = tool_result_message.get("tool_call_name")
        result = json.loads(tool_result_message.get("content") or "")
        response_message = json.dumps(
            {
                "toolResponse": {
                    "functionResponses": [
                        {
                            "id": id,
                            "name": name,
                            "response": {
                                "result": result,
                            },
                        }
                    ],
                }
            }
        )
        await self._websocket.send(response_message)
        # await self._websocket.send(json.dumps({"clientContent": {"turnComplete": True}}))

    @traced_gemini_live(operation="llm_setup")
    async def _handle_evt_setup_complete(self, evt):
        """Handle the setup complete event."""
        # If this is our first context frame, run the LLM
        self._api_session_ready = True
        # Now that we've configured the session, we can run the LLM if we need to.
        if self._run_llm_when_api_session_ready:
            self._run_llm_when_api_session_ready = False
            await self._create_initial_response()

    # async def _handle_evt_model_turn(self, evt):
    #     """Handle the model turn event."""
    #     part = evt.serverContent.modelTurn.parts[0]
    #     if not part:
    #         return

    #     await self.stop_ttfb_metrics()

    #     # part.text is added when `modalities` is set to TEXT; otherwise, it's None
    #     text = part.text
    #     if text:
    #         if not self._bot_text_buffer:
    #             await self.push_frame(LLMFullResponseStartFrame())

    #         self._bot_text_buffer += text
    #         self._search_result_buffer += text  # Also accumulate for grounding
    #         await self.push_frame(LLMTextFrame(text=text))

    #     # Check for grounding metadata in server content
    #     if evt.serverContent and evt.serverContent.groundingMetadata:
    #         self._accumulated_grounding_metadata = evt.serverContent.groundingMetadata

    #     inline_data = part.inlineData
    #     if not inline_data:
    #         return
    #     if inline_data.mimeType != f"audio/pcm;rate={self._sample_rate}":
    #         logger.warning(f"Unrecognized server_content format {inline_data.mimeType}")
    #         return

    #     audio = base64.b64decode(inline_data.data)
    #     if not audio:
    #         return

    #     if not self._bot_is_speaking:
    #         self._bot_is_speaking = True
    #         await self.push_frame(TTSStartedFrame())
    #         await self.push_frame(LLMFullResponseStartFrame())

    #     self._bot_audio_buffer.extend(audio)
    #     frame = TTSAudioRawFrame(
    #         audio=audio,
    #         sample_rate=self._sample_rate,
    #         num_channels=1,
    #     )
    #     await self.push_frame(frame)

    @traced_gemini_live(operation="llm_tool_call")
    async def _handle_evt_tool_call(self, evt):
        """Handle tool call events."""
        function_calls = evt.toolCall.functionCalls
        if not function_calls:
            return
        if not self._context:
            logger.error("Function calls are not supported without a context object.")

        function_calls_llm = [
            FunctionCallFromLLM(
                context=self._context,
                tool_call_id=f.id,
                function_name=f.name,
                arguments=f.args,
            )
            for f in function_calls
        ]

        await self.run_function_calls(function_calls_llm)

    @traced_gemini_live(operation="llm_response")
    async def _handle_evt_turn_complete(self, evt):
        """Handle the turn complete event."""
        self._bot_is_speaking = False
        text = self._bot_text_buffer

        # Determine output and modality for tracing
        if text:
            # TEXT modality
            output_text = text
            output_modality = "TEXT"
        else:
            # AUDIO modality
            output_text = self._llm_output_buffer
            output_modality = "AUDIO"

        # Trace the complete LLM response (this will be handled by the decorator)
        # The decorator will extract the output text and usage metadata from the event

        self._bot_text_buffer = ""
        self._llm_output_buffer = ""

        # Process grounding metadata if we have accumulated any
        if self._accumulated_grounding_metadata:
            await self._process_grounding_metadata(
                self._accumulated_grounding_metadata, self._search_result_buffer
            )

        # Reset grounding tracking for next response
        self._search_result_buffer = ""
        self._accumulated_grounding_metadata = None

        # Only push the TTSStoppedFrame if the bot is outputting audio
        # when text is found, modalities is set to TEXT and no audio
        # is produced.
        if not text:
            await self.push_frame(TTSStoppedFrame())

        await self.push_frame(LLMFullResponseEndFrame())

    @traced_stt
    async def _handle_user_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def _handle_evt_input_transcription(self, evt):
        """Handle the input transcription event.

        Gemini Live sends user transcriptions in either single words or multi-word
        phrases. As a result, we have to aggregate the input transcription. This handler
        aggregates into sentences, splitting on the end of sentence markers.
        """
        if not evt.serverContent.inputTranscription:
            return

        text = evt.serverContent.inputTranscription.text

        if not text:
            return

        # Strip leading space from sentence starts if buffer is empty
        if text.startswith(" ") and not self._user_transcription_buffer:
            text = text.lstrip()

        # Accumulate text in the buffer
        self._user_transcription_buffer += text

        # Check for complete sentences
        while True:
            eos_end_marker = match_endofsentence(self._user_transcription_buffer)
            if not eos_end_marker:
                break

            # Extract the complete sentence
            complete_sentence = self._user_transcription_buffer[:eos_end_marker]
            # Keep the remainder for the next chunk
            self._user_transcription_buffer = self._user_transcription_buffer[eos_end_marker:]

            # Send a TranscriptionFrame with the complete sentence
            logger.debug(f"[Transcription:user] [{complete_sentence}]")
            await self._handle_user_transcription(
                complete_sentence, True, self._settings["language"]
            )
            await self.push_frame(
                TranscriptionFrame(
                    text=complete_sentence,
                    user_id="",
                    timestamp=time_now_iso8601(),
                    result=evt,
                ),
                FrameDirection.UPSTREAM,
            )

    async def _handle_evt_output_transcription(self, evt):
        """Handle the output transcription event."""
        if not evt.serverContent.outputTranscription:
            return

        # This is the output transcription text when modalities is set to AUDIO.
        # In this case, we push LLMTextFrame and TTSTextFrame to be handled by the
        # downstream assistant context aggregator.
        text = evt.serverContent.outputTranscription.text

        if not text:
            return

        # Accumulate text for grounding as well
        self._search_result_buffer += text

        # Check for grounding metadata in server content
        if evt.serverContent and evt.serverContent.groundingMetadata:
            self._accumulated_grounding_metadata = evt.serverContent.groundingMetadata
        # Collect text for tracing
        self._llm_output_buffer += text

        await self.push_frame(LLMTextFrame(text=text))
        await self.push_frame(TTSTextFrame(text=text))

    async def _handle_evt_grounding_metadata(self, evt):
        """Handle dedicated grounding metadata events."""
        if evt.serverContent and evt.serverContent.groundingMetadata:
            grounding_metadata = evt.serverContent.groundingMetadata
            # Process the grounding metadata immediately
            await self._process_grounding_metadata(grounding_metadata, self._search_result_buffer)

    async def _process_grounding_metadata(
        self, grounding_metadata: events.GroundingMetadata, search_result: str = ""
    ):
        """Process grounding metadata and emit LLMSearchResponseFrame."""
        if not grounding_metadata:
            return

        # Extract rendered content for search suggestions
        rendered_content = None
        if (
            grounding_metadata.searchEntryPoint
            and grounding_metadata.searchEntryPoint.renderedContent
        ):
            rendered_content = grounding_metadata.searchEntryPoint.renderedContent

        # Convert grounding chunks and supports to LLMSearchOrigin format
        origins = []

        if grounding_metadata.groundingChunks and grounding_metadata.groundingSupports:
            # Create a mapping of chunk indices to origins
            chunk_to_origin = {}

            for index, chunk in enumerate(grounding_metadata.groundingChunks):
                if chunk.web:
                    origin = LLMSearchOrigin(
                        site_uri=chunk.web.uri, site_title=chunk.web.title, results=[]
                    )
                    chunk_to_origin[index] = origin
                    origins.append(origin)

            # Add grounding support results to the appropriate origins
            for support in grounding_metadata.groundingSupports:
                if support.segment and support.groundingChunkIndices:
                    text = support.segment.text or ""
                    confidence_scores = support.confidenceScores or []

                    # Add this result to all origins referenced by this support
                    for chunk_index in support.groundingChunkIndices:
                        if chunk_index in chunk_to_origin:
                            result = LLMSearchResult(text=text, confidence=confidence_scores)
                            chunk_to_origin[chunk_index].results.append(result)

        # Create and push the search response frame
        search_frame = LLMSearchResponseFrame(
            search_result=search_result, origins=origins, rendered_content=rendered_content
        )

        await self.push_frame(search_frame)

    async def _handle_evt_usage_metadata(self, evt):
        """Handle the usage metadata event."""
        if not evt.usageMetadata:
            return

        usage = evt.usageMetadata

        # Ensure we have valid integers for all token counts
        prompt_tokens = usage.promptTokenCount or 0
        completion_tokens = usage.responseTokenCount or 0
        total_tokens = usage.totalTokenCount or (prompt_tokens + completion_tokens)

        tokens = LLMTokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        await self.start_llm_usage_metrics(tokens)

    def create_context_aggregator(
        self,
        context: OpenAILLMContext,
        *,
        user_params: LLMUserAggregatorParams = LLMUserAggregatorParams(),
        assistant_params: LLMAssistantAggregatorParams = LLMAssistantAggregatorParams(
            expect_stripped_words=True
        ),
    ) -> GeminiMultimodalLiveContextAggregatorPair:
        """Create an instance of GeminiMultimodalLiveContextAggregatorPair from an OpenAILLMContext.

        Constructor keyword arguments for both the user and assistant aggregators can be provided.

        Args:
            context: The LLM context to use.
            user_params: User aggregator parameters. Defaults to LLMUserAggregatorParams().
            assistant_params: Assistant aggregator parameters. Defaults to LLMAssistantAggregatorParams().

        Returns:
            GeminiMultimodalLiveContextAggregatorPair: A pair of context
            aggregators, one for the user and one for the assistant,
            encapsulated in an GeminiMultimodalLiveContextAggregatorPair.
        """
        context.set_llm_adapter(self.get_llm_adapter())

        GeminiMultimodalLiveContext.upgrade(context)
        user = GeminiMultimodalLiveUserContextAggregator(context, params=user_params)

        assistant = GeminiMultimodalLiveAssistantContextAggregator(context, params=assistant_params)
        return GeminiMultimodalLiveContextAggregatorPair(_user=user, _assistant=assistant)
