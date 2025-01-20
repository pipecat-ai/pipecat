#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import io
import json
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional

from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    AudioRawFrame,
    ErrorFrame,
    Frame,
    FunctionCallResultProperties,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    OpenAILLMContextAssistantTimestampFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    VisionImageRawFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import LLMService, TTSService
from pipecat.services.google.frames import LLMSearchResponseFrame
from pipecat.services.openai import (
    OpenAIAssistantContextAggregator,
    OpenAIUserContextAggregator,
)
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

try:
    import google.ai.generativelanguage as glm
    import google.generativeai as gai
    from google.cloud import texttospeech_v1
    from google.generativeai.types import GenerationConfig
    from google.oauth2 import service_account
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Google AI, you need to `pip install pipecat-ai[google]`. Also, set the environment variable GOOGLE_API_KEY for the GoogleLLMService and GOOGLE_APPLICATION_CREDENTIALS for the GoogleTTSService`."
    )
    raise Exception(f"Missing module: {e}")


def language_to_google_language(language: Language) -> str | None:
    language_map = {
        # Afrikaans
        Language.AF: "af-ZA",
        Language.AF_ZA: "af-ZA",
        # Arabic
        Language.AR: "ar-XA",
        # Bengali
        Language.BN: "bn-IN",
        Language.BN_IN: "bn-IN",
        # Bulgarian
        Language.BG: "bg-BG",
        Language.BG_BG: "bg-BG",
        # Catalan
        Language.CA: "ca-ES",
        Language.CA_ES: "ca-ES",
        # Chinese (Mandarin and Cantonese)
        Language.ZH: "cmn-CN",
        Language.ZH_CN: "cmn-CN",
        Language.ZH_TW: "cmn-TW",
        Language.ZH_HK: "yue-HK",
        # Czech
        Language.CS: "cs-CZ",
        Language.CS_CZ: "cs-CZ",
        # Danish
        Language.DA: "da-DK",
        Language.DA_DK: "da-DK",
        # Dutch
        Language.NL: "nl-NL",
        Language.NL_BE: "nl-BE",
        Language.NL_NL: "nl-NL",
        # English
        Language.EN: "en-US",
        Language.EN_US: "en-US",
        Language.EN_AU: "en-AU",
        Language.EN_GB: "en-GB",
        Language.EN_IN: "en-IN",
        # Estonian
        Language.ET: "et-EE",
        Language.ET_EE: "et-EE",
        # Filipino
        Language.FIL: "fil-PH",
        Language.FIL_PH: "fil-PH",
        # Finnish
        Language.FI: "fi-FI",
        Language.FI_FI: "fi-FI",
        # French
        Language.FR: "fr-FR",
        Language.FR_CA: "fr-CA",
        Language.FR_FR: "fr-FR",
        # Galician
        Language.GL: "gl-ES",
        Language.GL_ES: "gl-ES",
        # German
        Language.DE: "de-DE",
        Language.DE_DE: "de-DE",
        # Greek
        Language.EL: "el-GR",
        Language.EL_GR: "el-GR",
        # Gujarati
        Language.GU: "gu-IN",
        Language.GU_IN: "gu-IN",
        # Hebrew
        Language.HE: "he-IL",
        Language.HE_IL: "he-IL",
        # Hindi
        Language.HI: "hi-IN",
        Language.HI_IN: "hi-IN",
        # Hungarian
        Language.HU: "hu-HU",
        Language.HU_HU: "hu-HU",
        # Icelandic
        Language.IS: "is-IS",
        Language.IS_IS: "is-IS",
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
        # Latvian
        Language.LV: "lv-LV",
        Language.LV_LV: "lv-LV",
        # Lithuanian
        Language.LT: "lt-LT",
        Language.LT_LT: "lt-LT",
        # Malay
        Language.MS: "ms-MY",
        Language.MS_MY: "ms-MY",
        # Malayalam
        Language.ML: "ml-IN",
        Language.ML_IN: "ml-IN",
        # Marathi
        Language.MR: "mr-IN",
        Language.MR_IN: "mr-IN",
        # Norwegian
        Language.NO: "nb-NO",
        Language.NB: "nb-NO",
        Language.NB_NO: "nb-NO",
        # Polish
        Language.PL: "pl-PL",
        Language.PL_PL: "pl-PL",
        # Portuguese
        Language.PT: "pt-PT",
        Language.PT_BR: "pt-BR",
        Language.PT_PT: "pt-PT",
        # Punjabi
        Language.PA: "pa-IN",
        Language.PA_IN: "pa-IN",
        # Romanian
        Language.RO: "ro-RO",
        Language.RO_RO: "ro-RO",
        # Russian
        Language.RU: "ru-RU",
        Language.RU_RU: "ru-RU",
        # Serbian
        Language.SR: "sr-RS",
        Language.SR_RS: "sr-RS",
        # Slovak
        Language.SK: "sk-SK",
        Language.SK_SK: "sk-SK",
        # Spanish
        Language.ES: "es-ES",
        Language.ES_ES: "es-ES",
        Language.ES_US: "es-US",
        # Swedish
        Language.SV: "sv-SE",
        Language.SV_SE: "sv-SE",
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
        # Ukrainian
        Language.UK: "uk-UA",
        Language.UK_UA: "uk-UA",
        # Vietnamese
        Language.VI: "vi-VN",
        Language.VI_VN: "vi-VN",
    }

    return language_map.get(language)


class GoogleUserContextAggregator(OpenAIUserContextAggregator):
    async def _push_aggregation(self):
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
            self._reset()


class GoogleAssistantContextAggregator(OpenAIAssistantContextAggregator):
    async def _push_aggregation(self):
        if not (
            self._aggregation or self._function_call_result or self._pending_image_frame_message
        ):
            return

        run_llm = False
        properties: Optional[FunctionCallResultProperties] = None

        aggregation = self._aggregation
        self._reset()

        try:
            if self._function_call_result:
                frame = self._function_call_result
                properties = frame.properties
                self._function_call_result = None
                if frame.result:
                    logger.debug(f"FunctionCallResultFrame result: {frame.arguments}")
                    self._context.add_message(
                        glm.Content(
                            role="model",
                            parts=[
                                glm.Part(
                                    function_call=glm.FunctionCall(
                                        name=frame.function_name, args=frame.arguments
                                    )
                                )
                            ],
                        )
                    )
                    response = frame.result
                    if isinstance(response, str):
                        response = {"response": response}
                    self._context.add_message(
                        glm.Content(
                            role="user",
                            parts=[
                                glm.Part(
                                    function_response=glm.FunctionResponse(
                                        name=frame.function_name, response=response
                                    )
                                )
                            ],
                        )
                    )
                    if properties and properties.run_llm is not None:
                        # If the tool call result has a run_llm property, use it
                        run_llm = properties.run_llm
                    else:
                        # Default behavior is to run the LLM if there are no function calls in progress
                        run_llm = not bool(self._function_calls_in_progress)
            else:
                if aggregation.strip():
                    self._context.add_message(
                        glm.Content(role="model", parts=[glm.Part(text=aggregation)])
                    )

            if self._pending_image_frame_message:
                frame = self._pending_image_frame_message
                self._pending_image_frame_message = None
                self._context.add_image_frame_message(
                    format=frame.user_image_raw_frame.format,
                    size=frame.user_image_raw_frame.size,
                    image=frame.user_image_raw_frame.image,
                    text=frame.text,
                )
                run_llm = True

            if run_llm:
                await self._user_context_aggregator.push_context_frame()

            # Emit the on_context_updated callback once the function call result is added to the context
            if properties and properties.on_context_updated is not None:
                await properties.on_context_updated()

            # Push context frame
            frame = OpenAILLMContextFrame(self._context)
            await self.push_frame(frame)

            # Push timestamp frame with current time
            timestamp_frame = OpenAILLMContextAssistantTimestampFrame(timestamp=time_now_iso8601())
            await self.push_frame(timestamp_frame)

        except Exception as e:
            logger.exception(f"Error processing frame: {e}")


@dataclass
class GoogleContextAggregatorPair:
    _user: "GoogleUserContextAggregator"
    _assistant: "GoogleAssistantContextAggregator"

    def user(self) -> "GoogleUserContextAggregator":
        return self._user

    def assistant(self) -> "GoogleAssistantContextAggregator":
        return self._assistant


class GoogleLLMContext(OpenAILLMContext):
    def __init__(
        self,
        messages: list[dict] | None = None,
        tools: list[dict] | None = None,
        tool_choice: dict | None = None,
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

    def add_audio_frames_message(self, *, audio_frames: list[AudioRawFrame], text: str = None):
        if not audio_frames:
            return

        sample_rate = audio_frames[0].sample_rate
        num_channels = audio_frames[0].num_channels

        parts = []
        data = b"".join(frame.audio for frame in audio_frames)
        if text:
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
        self.system_message = None
        # first, map across self._messages calling self.from_standard_message(m) to modify messages in place
        try:
            self._messages[:] = [
                msg
                for msg in (self.from_standard_message(m) for m in self._messages)
                if msg is not None
            ]
            # We might have been given a messages list with only a system message. If so, let's put that back in
            # the messages list as a user message.
            if self.system_message and not self._messages:
                self.add_message(
                    glm.Content(role="user", parts=[glm.Part(text=self.system_message)])
                )
        except Exception as e:
            logger.error(f"Error mapping messages: {e}")
        # iterate over messages and remove any messages that have an empty content list
        self._messages = [m for m in self._messages if m.parts]


class GoogleLLMService(LLMService):
    """This class implements inference with Google's AI models

    This service translates internally from OpenAILLMContext to the messages format
    expected by the Google AI model. We are using the OpenAILLMContext as a lingua
    franca for all LLM services, so that it is easy to switch between different LLMs.
    """

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
        model: str = "gemini-1.5-flash-latest",
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
                # f"Generating chat: {self._system_instruction} | {context.get_messages_for_logging()}"
                f"Generating chat: {context.get_messages_for_logging()}"
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
                            logger.debug(f"!!! Function call: {c.function_call}")
                            args = type(c.function_call).to_dict(c.function_call).get("args", {})
                            await self.call_function(
                                context=context,
                                tool_call_id="what_should_this_be",
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

    @staticmethod
    def create_context_aggregator(
        context: OpenAILLMContext, *, assistant_expect_stripped_words: bool = True
    ) -> GoogleContextAggregatorPair:
        user = GoogleUserContextAggregator(context)
        assistant = GoogleAssistantContextAggregator(
            user, expect_stripped_words=assistant_expect_stripped_words
        )
        return GoogleContextAggregatorPair(_user=user, _assistant=assistant)


class GoogleTTSService(TTSService):
    class InputParams(BaseModel):
        pitch: Optional[str] = None
        rate: Optional[str] = None
        volume: Optional[str] = None
        emphasis: Optional[Literal["strong", "moderate", "reduced", "none"]] = None
        language: Optional[Language] = Language.EN
        gender: Optional[Literal["male", "female", "neutral"]] = None
        google_style: Optional[Literal["apologetic", "calm", "empathetic", "firm", "lively"]] = None

    def __init__(
        self,
        *,
        credentials: Optional[str] = None,
        credentials_path: Optional[str] = None,
        voice_id: str = "en-US-Neural2-A",
        sample_rate: int = 24000,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._settings = {
            "sample_rate": sample_rate,
            "pitch": params.pitch,
            "rate": params.rate,
            "volume": params.volume,
            "emphasis": params.emphasis,
            "language": self.language_to_service_language(params.language)
            if params.language
            else "en-US",
            "gender": params.gender,
            "google_style": params.google_style,
        }
        self.set_voice(voice_id)
        self._client: texttospeech_v1.TextToSpeechAsyncClient = self._create_client(
            credentials, credentials_path
        )

    def _create_client(
        self, credentials: Optional[str], credentials_path: Optional[str]
    ) -> texttospeech_v1.TextToSpeechAsyncClient:
        creds: Optional[service_account.Credentials] = None

        # Create a Google Cloud service account for the Cloud Text-to-Speech API
        # Using either the provided credentials JSON string or the path to a service account JSON
        # file, create a Google Cloud service account and use it to authenticate with the API.
        if credentials:
            # Use provided credentials JSON string
            json_account_info = json.loads(credentials)
            creds = service_account.Credentials.from_service_account_info(json_account_info)
        elif credentials_path:
            # Use service account JSON file if provided
            creds = service_account.Credentials.from_service_account_file(credentials_path)

        return texttospeech_v1.TextToSpeechAsyncClient(credentials=creds)

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        return language_to_google_language(language)

    def _construct_ssml(self, text: str) -> str:
        ssml = "<speak>"

        # Voice tag
        voice_attrs = [f"name='{self._voice_id}'"]

        language = self._settings["language"]
        voice_attrs.append(f"language='{language}'")

        if self._settings["gender"]:
            voice_attrs.append(f"gender='{self._settings['gender']}'")
        ssml += f"<voice {' '.join(voice_attrs)}>"

        # Prosody tag
        prosody_attrs = []
        if self._settings["pitch"]:
            prosody_attrs.append(f"pitch='{self._settings['pitch']}'")
        if self._settings["rate"]:
            prosody_attrs.append(f"rate='{self._settings['rate']}'")
        if self._settings["volume"]:
            prosody_attrs.append(f"volume='{self._settings['volume']}'")

        if prosody_attrs:
            ssml += f"<prosody {' '.join(prosody_attrs)}>"

        # Emphasis tag
        if self._settings["emphasis"]:
            ssml += f"<emphasis level='{self._settings['emphasis']}'>"

        # Google style tag
        if self._settings["google_style"]:
            ssml += f"<google:style name='{self._settings['google_style']}'>"

        ssml += text

        # Close tags
        if self._settings["google_style"]:
            ssml += "</google:style>"
        if self._settings["emphasis"]:
            ssml += "</emphasis>"
        if prosody_attrs:
            ssml += "</prosody>"
        ssml += "</voice></speak>"

        return ssml

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        try:
            await self.start_ttfb_metrics()

            is_journey_voice = "journey" in self._voice_id.lower()

            # Create synthesis input based on voice_id
            if is_journey_voice:
                synthesis_input = texttospeech_v1.SynthesisInput(text=text)
            else:
                ssml = self._construct_ssml(text)
                synthesis_input = texttospeech_v1.SynthesisInput(ssml=ssml)

            voice = texttospeech_v1.VoiceSelectionParams(
                language_code=self._settings["language"], name=self._voice_id
            )
            audio_config = texttospeech_v1.AudioConfig(
                audio_encoding=texttospeech_v1.AudioEncoding.LINEAR16,
                sample_rate_hertz=self._settings["sample_rate"],
            )

            request = texttospeech_v1.SynthesizeSpeechRequest(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )

            response = await self._client.synthesize_speech(request=request)

            await self.start_tts_usage_metrics(text)

            yield TTSStartedFrame()

            # Skip the first 44 bytes to remove the WAV header
            audio_content = response.audio_content[44:]

            # Read and yield audio data in chunks
            chunk_size = 8192
            for i in range(0, len(audio_content), chunk_size):
                chunk = audio_content[i : i + chunk_size]
                if not chunk:
                    break
                await self.stop_ttfb_metrics()
                frame = TTSAudioRawFrame(chunk, self._settings["sample_rate"], 1)
                yield frame
                await asyncio.sleep(0)  # Allow other tasks to run

            yield TTSStoppedFrame()

        except Exception as e:
            logger.exception(f"{self} error generating TTS: {e}")
            error_message = f"TTS generation error: {str(e)}"
            yield ErrorFrame(error=error_message)
        finally:
            yield TTSStoppedFrame()
