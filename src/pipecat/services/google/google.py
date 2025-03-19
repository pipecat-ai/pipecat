#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import io
import json
import os
import time
import uuid

from google.api_core.exceptions import DeadlineExceeded
from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk

from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter

# Suppress gRPC fork warnings
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Literal, Mapping, Optional, Union

from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field, field_validator

from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMTextFrame,
    LLMUpdateSettingsFrame,
    StartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    URLImageRawFrame,
    UserImageRawFrame,
    VisionImageRawFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import ImageGenService, LLMService, STTService, TTSService
from pipecat.services.google.frames import LLMSearchResponseFrame
from pipecat.services.openai import (
    OpenAIAssistantContextAggregator,
    OpenAILLMService,
    OpenAIUnhandledFunctionException,
    OpenAIUserContextAggregator,
)
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

try:
    import google.ai.generativelanguage as glm
    import google.generativeai as gai
    from google import genai
    from google.api_core.client_options import ClientOptions
    from google.auth.transport.requests import Request
    from google.cloud import speech_v2, texttospeech_v1
    from google.cloud.speech_v2.types import cloud_speech
    from google.genai import types
    from google.generativeai.types import GenerationConfig
    from google.oauth2 import service_account

except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Google AI, you need to `pip install pipecat-ai[google]`. Also, set the environment variable GOOGLE_API_KEY for the GoogleLLMService and GOOGLE_APPLICATION_CREDENTIALS for the GoogleTTSService`."
    )
    raise Exception(f"Missing module: {e}")


def language_to_google_tts_language(language: Language) -> Optional[str]:
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


def language_to_google_stt_language(language: Language) -> Optional[str]:
    """Maps Language enum to Google Speech-to-Text V2 language codes.

    Args:
        language: Language enum value.

    Returns:
        Optional[str]: Google STT language code or None if not supported.
    """
    language_map = {
        # Afrikaans
        Language.AF: "af-ZA",
        Language.AF_ZA: "af-ZA",
        # Albanian
        Language.SQ: "sq-AL",
        Language.SQ_AL: "sq-AL",
        # Amharic
        Language.AM: "am-ET",
        Language.AM_ET: "am-ET",
        # Arabic
        Language.AR: "ar-EG",  # Default to Egypt
        Language.AR_AE: "ar-AE",
        Language.AR_BH: "ar-BH",
        Language.AR_DZ: "ar-DZ",
        Language.AR_EG: "ar-EG",
        Language.AR_IQ: "ar-IQ",
        Language.AR_JO: "ar-JO",
        Language.AR_KW: "ar-KW",
        Language.AR_LB: "ar-LB",
        Language.AR_MA: "ar-MA",
        Language.AR_OM: "ar-OM",
        Language.AR_QA: "ar-QA",
        Language.AR_SA: "ar-SA",
        Language.AR_SY: "ar-SY",
        Language.AR_TN: "ar-TN",
        Language.AR_YE: "ar-YE",
        # Armenian
        Language.HY: "hy-AM",
        Language.HY_AM: "hy-AM",
        # Azerbaijani
        Language.AZ: "az-AZ",
        Language.AZ_AZ: "az-AZ",
        # Basque
        Language.EU: "eu-ES",
        Language.EU_ES: "eu-ES",
        # Bengali
        Language.BN: "bn-IN",  # Default to India
        Language.BN_BD: "bn-BD",
        Language.BN_IN: "bn-IN",
        # Bosnian
        Language.BS: "bs-BA",
        Language.BS_BA: "bs-BA",
        # Bulgarian
        Language.BG: "bg-BG",
        Language.BG_BG: "bg-BG",
        # Burmese
        Language.MY: "my-MM",
        Language.MY_MM: "my-MM",
        # Catalan
        Language.CA: "ca-ES",
        Language.CA_ES: "ca-ES",
        # Chinese
        Language.ZH: "cmn-Hans-CN",  # Default to Simplified Chinese
        Language.ZH_CN: "cmn-Hans-CN",
        Language.ZH_HK: "cmn-Hans-HK",
        Language.ZH_TW: "cmn-Hant-TW",
        Language.YUE: "yue-Hant-HK",  # Cantonese
        Language.YUE_CN: "yue-Hant-HK",
        # Croatian
        Language.HR: "hr-HR",
        Language.HR_HR: "hr-HR",
        # Czech
        Language.CS: "cs-CZ",
        Language.CS_CZ: "cs-CZ",
        # Danish
        Language.DA: "da-DK",
        Language.DA_DK: "da-DK",
        # Dutch
        Language.NL: "nl-NL",  # Default to Netherlands
        Language.NL_BE: "nl-BE",
        Language.NL_NL: "nl-NL",
        # English
        Language.EN: "en-US",  # Default to US
        Language.EN_AU: "en-AU",
        Language.EN_CA: "en-CA",
        Language.EN_GB: "en-GB",
        Language.EN_GH: "en-GH",
        Language.EN_HK: "en-HK",
        Language.EN_IN: "en-IN",
        Language.EN_IE: "en-IE",
        Language.EN_KE: "en-KE",
        Language.EN_NG: "en-NG",
        Language.EN_NZ: "en-NZ",
        Language.EN_PH: "en-PH",
        Language.EN_SG: "en-SG",
        Language.EN_TZ: "en-TZ",
        Language.EN_US: "en-US",
        Language.EN_ZA: "en-ZA",
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
        Language.FR: "fr-FR",  # Default to France
        Language.FR_BE: "fr-BE",
        Language.FR_CA: "fr-CA",
        Language.FR_CH: "fr-CH",
        Language.FR_FR: "fr-FR",
        # Galician
        Language.GL: "gl-ES",
        Language.GL_ES: "gl-ES",
        # Georgian
        Language.KA: "ka-GE",
        Language.KA_GE: "ka-GE",
        # German
        Language.DE: "de-DE",  # Default to Germany
        Language.DE_AT: "de-AT",
        Language.DE_CH: "de-CH",
        Language.DE_DE: "de-DE",
        # Greek
        Language.EL: "el-GR",
        Language.EL_GR: "el-GR",
        # Gujarati
        Language.GU: "gu-IN",
        Language.GU_IN: "gu-IN",
        # Hebrew
        Language.HE: "iw-IL",
        Language.HE_IL: "iw-IL",
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
        Language.IT_CH: "it-CH",
        # Japanese
        Language.JA: "ja-JP",
        Language.JA_JP: "ja-JP",
        # Javanese
        Language.JV: "jv-ID",
        Language.JV_ID: "jv-ID",
        # Kannada
        Language.KN: "kn-IN",
        Language.KN_IN: "kn-IN",
        # Kazakh
        Language.KK: "kk-KZ",
        Language.KK_KZ: "kk-KZ",
        # Khmer
        Language.KM: "km-KH",
        Language.KM_KH: "km-KH",
        # Korean
        Language.KO: "ko-KR",
        Language.KO_KR: "ko-KR",
        # Lao
        Language.LO: "lo-LA",
        Language.LO_LA: "lo-LA",
        # Latvian
        Language.LV: "lv-LV",
        Language.LV_LV: "lv-LV",
        # Lithuanian
        Language.LT: "lt-LT",
        Language.LT_LT: "lt-LT",
        # Macedonian
        Language.MK: "mk-MK",
        Language.MK_MK: "mk-MK",
        # Malay
        Language.MS: "ms-MY",
        Language.MS_MY: "ms-MY",
        # Malayalam
        Language.ML: "ml-IN",
        Language.ML_IN: "ml-IN",
        # Marathi
        Language.MR: "mr-IN",
        Language.MR_IN: "mr-IN",
        # Mongolian
        Language.MN: "mn-MN",
        Language.MN_MN: "mn-MN",
        # Nepali
        Language.NE: "ne-NP",
        Language.NE_NP: "ne-NP",
        # Norwegian
        Language.NO: "no-NO",
        Language.NB: "no-NO",
        Language.NB_NO: "no-NO",
        # Persian
        Language.FA: "fa-IR",
        Language.FA_IR: "fa-IR",
        # Polish
        Language.PL: "pl-PL",
        Language.PL_PL: "pl-PL",
        # Portuguese
        Language.PT: "pt-PT",  # Default to Portugal
        Language.PT_BR: "pt-BR",
        Language.PT_PT: "pt-PT",
        # Punjabi
        Language.PA: "pa-Guru-IN",
        Language.PA_IN: "pa-Guru-IN",
        # Romanian
        Language.RO: "ro-RO",
        Language.RO_RO: "ro-RO",
        # Russian
        Language.RU: "ru-RU",
        Language.RU_RU: "ru-RU",
        # Serbian
        Language.SR: "sr-RS",
        Language.SR_RS: "sr-RS",
        # Sinhala
        Language.SI: "si-LK",
        Language.SI_LK: "si-LK",
        # Slovak
        Language.SK: "sk-SK",
        Language.SK_SK: "sk-SK",
        # Slovenian
        Language.SL: "sl-SI",
        Language.SL_SI: "sl-SI",
        # Spanish
        Language.ES: "es-ES",  # Default to Spain
        Language.ES_AR: "es-AR",
        Language.ES_BO: "es-BO",
        Language.ES_CL: "es-CL",
        Language.ES_CO: "es-CO",
        Language.ES_CR: "es-CR",
        Language.ES_DO: "es-DO",
        Language.ES_EC: "es-EC",
        Language.ES_ES: "es-ES",
        Language.ES_GT: "es-GT",
        Language.ES_HN: "es-HN",
        Language.ES_MX: "es-MX",
        Language.ES_NI: "es-NI",
        Language.ES_PA: "es-PA",
        Language.ES_PE: "es-PE",
        Language.ES_PR: "es-PR",
        Language.ES_PY: "es-PY",
        Language.ES_SV: "es-SV",
        Language.ES_US: "es-US",
        Language.ES_UY: "es-UY",
        Language.ES_VE: "es-VE",
        # Sundanese
        Language.SU: "su-ID",
        Language.SU_ID: "su-ID",
        # Swahili
        Language.SW: "sw-TZ",  # Default to Tanzania
        Language.SW_KE: "sw-KE",
        Language.SW_TZ: "sw-TZ",
        # Swedish
        Language.SV: "sv-SE",
        Language.SV_SE: "sv-SE",
        # Tamil
        Language.TA: "ta-IN",  # Default to India
        Language.TA_IN: "ta-IN",
        Language.TA_MY: "ta-MY",
        Language.TA_SG: "ta-SG",
        Language.TA_LK: "ta-LK",
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
        # Urdu
        Language.UR: "ur-IN",  # Default to India
        Language.UR_IN: "ur-IN",
        Language.UR_PK: "ur-PK",
        # Uzbek
        Language.UZ: "uz-UZ",
        Language.UZ_UZ: "uz-UZ",
        # Vietnamese
        Language.VI: "vi-VN",
        Language.VI_VN: "vi-VN",
        # Xhosa
        Language.XH: "xh-ZA",
        # Zulu
        Language.ZU: "zu-ZA",
        Language.ZU_ZA: "zu-ZA",
    }

    return language_map.get(language)


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
            if not isinstance(frame.result, str):
                return

            response = {"response": frame.result}

            await self._update_function_call_result(
                frame.function_name, frame.tool_call_id, response
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
                        part.function_response.response = {"response": result}

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
    _user: "GoogleUserContextAggregator"
    _assistant: "GoogleAssistantContextAggregator"

    def user(self) -> "GoogleUserContextAggregator":
        return self._user

    def assistant(self) -> "GoogleAssistantContextAggregator":
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
        user_kwargs: Mapping[str, Any] = {},
        assistant_kwargs: Mapping[str, Any] = {},
    ) -> GoogleContextAggregatorPair:
        """Create an instance of GoogleContextAggregatorPair from an
        OpenAILLMContext. Constructor keyword arguments for both the user and
        assistant aggregators can be provided.

        Args:
            context (OpenAILLMContext): The LLM context.
            user_kwargs (Mapping[str, Any], optional): Additional keyword
                arguments for the user context aggregator constructor. Defaults
                to an empty mapping.
            assistant_kwargs (Mapping[str, Any], optional): Additional keyword
                arguments for the assistant context aggregator
                constructor. Defaults to an empty mapping.

        Returns:
            GoogleContextAggregatorPair: A pair of context aggregators, one for
            the user and one for the assistant, encapsulated in an
            GoogleContextAggregatorPair.

        """
        context.set_llm_adapter(self.get_llm_adapter())

        if isinstance(context, OpenAILLMContext):
            context = GoogleLLMContext.upgrade_to_google(context)
        user = GoogleUserContextAggregator(context, **user_kwargs)
        assistant = GoogleAssistantContextAggregator(context, **assistant_kwargs)
        return GoogleContextAggregatorPair(_user=user, _assistant=assistant)


class GoogleLLMOpenAIBetaService(OpenAILLMService):
    """This class implements inference with Google's AI LLM models using the OpenAI format.
    Ref - https://ai.google.dev/gemini-api/docs/openai
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        model: str = "gemini-2.0-flash",
        **kwargs,
    ):
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

    async def _process_context(self, context: OpenAILLMContext):
        functions_list = []
        arguments_list = []
        tool_id_list = []
        func_idx = 0
        function_name = ""
        arguments = ""
        tool_call_id = ""

        await self.start_ttfb_metrics()

        chunk_stream: AsyncStream[ChatCompletionChunk] = await self._stream_chat_completions(
            context
        )

        async for chunk in chunk_stream:
            if chunk.usage:
                tokens = LLMTokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                )
                await self.start_llm_usage_metrics(tokens)

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
                logger.debug(f"Tool call: {chunk.choices[0].delta.tool_calls}")
                tool_call = chunk.choices[0].delta.tool_calls[0]
                if tool_call.index != func_idx:
                    functions_list.append(function_name)
                    arguments_list.append(arguments)
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
                await self.push_frame(LLMTextFrame(chunk.choices[0].delta.content))

        # if we got a function name and arguments, check to see if it's a function with
        # a registered handler. If so, run the registered callback, save the result to
        # the context, and re-prompt to get a chat answer. If we don't have a registered
        # handler, raise an exception.
        if function_name and arguments:
            # added to the list as last function name and arguments not added to the list
            functions_list.append(function_name)
            arguments_list.append(arguments)
            tool_id_list.append(tool_call_id)

            logger.debug(
                f"Function list: {functions_list}, Arguments list: {arguments_list}, Tool ID list: {tool_id_list}"
            )
            for index, (function_name, arguments, tool_id) in enumerate(
                zip(functions_list, arguments_list, tool_id_list), start=1
            ):
                if function_name == "":
                    # TODO: Remove the _process_context method once Google resolves the bug
                    # where the index is incorrectly set to None instead of returning the actual index,
                    # which currently results in an empty function name('').
                    continue
                if self.has_function(function_name):
                    run_llm = False
                    arguments = json.loads(arguments)
                    await self.call_function(
                        context=context,
                        function_name=function_name,
                        arguments=arguments,
                        tool_call_id=tool_id,
                        run_llm=run_llm,
                    )
                else:
                    raise OpenAIUnhandledFunctionException(
                        f"The LLM tried to call a function named '{function_name}', but there isn't a callback registered for that function."
                    )


class GoogleVertexLLMService(OpenAILLMService):
    """Implements inference with Google's AI models via Vertex AI while
    maintaining OpenAI API compatibility.

    Reference:
    https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-vertex-using-openai-library

    """

    class InputParams(OpenAILLMService.InputParams):
        """Input parameters specific to Vertex AI."""

        # https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations
        location: str = "us-east4"
        project_id: str

    def __init__(
        self,
        *,
        credentials: Optional[str] = None,
        credentials_path: Optional[str] = None,
        model: str = "google/gemini-2.0-flash-001",
        params: InputParams = OpenAILLMService.InputParams(),
        **kwargs,
    ):
        """Initializes the VertexLLMService.
        Args:
            credentials (Optional[str]): JSON string of service account credentials.
            credentials_path (Optional[str]): Path to the service account JSON file.
            model (str): Model identifier. Defaults to "google/gemini-2.0-flash-001".
            params (InputParams): Vertex AI input parameters.
            **kwargs: Additional arguments for OpenAILLMService.
        """
        base_url = self._get_base_url(params)
        self._api_key = self._get_api_token(credentials, credentials_path)

        super().__init__(api_key=self._api_key, base_url=base_url, model=model, **kwargs)

    @staticmethod
    def _get_base_url(params: InputParams) -> str:
        """Constructs the base URL for Vertex AI API."""
        return (
            f"https://{params.location}-aiplatform.googleapis.com/v1/"
            f"projects/{params.project_id}/locations/{params.location}/endpoints/openapi"
        )

    @staticmethod
    def _get_api_token(credentials: Optional[str], credentials_path: Optional[str]) -> str:
        """Retrieves an authentication token using Google service account credentials.
        Args:
            credentials (Optional[str]): JSON string of service account credentials.
            credentials_path (Optional[str]): Path to the service account JSON file.
        Returns:
            str: OAuth token for API authentication.
        """
        creds: Optional[service_account.Credentials] = None

        if credentials:
            # Parse and load credentials from JSON string
            creds = service_account.Credentials.from_service_account_info(
                json.loads(credentials), scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        elif credentials_path:
            # Load credentials from JSON file
            creds = service_account.Credentials.from_service_account_file(
                credentials_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )

        if not creds:
            raise ValueError("No valid credentials provided.")

        creds.refresh(Request())  # Ensure token is up-to-date, lifetime is 1 hour.

        return creds.token


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
        sample_rate: Optional[int] = None,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._settings = {
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

    def language_to_service_language(self, language: Language) -> Optional[str]:
        return language_to_google_tts_language(language)

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
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics()

            # Check if the voice is a Chirp voice (including Chirp 3) or Journey voice
            is_chirp_voice = "chirp" in self._voice_id.lower()
            is_journey_voice = "journey" in self._voice_id.lower()

            # Create synthesis input based on voice_id
            if is_chirp_voice or is_journey_voice:
                # Chirp and Journey voices don't support SSML, use plain text
                synthesis_input = texttospeech_v1.SynthesisInput(text=text)
            else:
                ssml = self._construct_ssml(text)
                synthesis_input = texttospeech_v1.SynthesisInput(ssml=ssml)

            voice = texttospeech_v1.VoiceSelectionParams(
                language_code=self._settings["language"], name=self._voice_id
            )
            audio_config = texttospeech_v1.AudioConfig(
                audio_encoding=texttospeech_v1.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
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
                frame = TTSAudioRawFrame(chunk, self.sample_rate, 1)
                yield frame
                await asyncio.sleep(0)  # Allow other tasks to run

            yield TTSStoppedFrame()

        except Exception as e:
            logger.exception(f"{self} error generating TTS: {e}")
            error_message = f"TTS generation error: {str(e)}"
            yield ErrorFrame(error=error_message)
        finally:
            yield TTSStoppedFrame()


class GoogleImageGenService(ImageGenService):
    class InputParams(BaseModel):
        number_of_images: int = Field(default=1, ge=1, le=8)
        model: str = Field(default="imagen-3.0-generate-002")
        negative_prompt: str = Field(default=None)

    def __init__(
        self,
        *,
        params: InputParams = InputParams(),
        api_key: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.set_model_name(params.model)
        self._params = params
        self._client = genai.Client(api_key=api_key)

    def can_generate_metrics(self) -> bool:
        return True

    async def run_image_gen(self, prompt: str) -> AsyncGenerator[Frame, None]:
        """Generate images from a text prompt using Google's Imagen model.

        Args:
            prompt (str): The text description to generate images from.

        Yields:
            Frame: Generated image frames or error frames.
        """
        logger.debug(f"Generating image from prompt: {prompt}")
        await self.start_ttfb_metrics()

        try:
            response = await self._client.aio.models.generate_images(
                model=self._params.model,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=self._params.number_of_images,
                    negative_prompt=self._params.negative_prompt,
                ),
            )
            await self.stop_ttfb_metrics()

            if not response or not response.generated_images:
                logger.error(f"{self} error: image generation failed")
                yield ErrorFrame("Image generation failed")
                return

            for img_response in response.generated_images:
                # Google returns the image data directly
                image_bytes = img_response.image.image_bytes
                image = Image.open(io.BytesIO(image_bytes))

                frame = URLImageRawFrame(
                    url=None,  # Google doesn't provide URLs, only image data
                    image=image.tobytes(),
                    size=image.size,
                    format=image.format,
                )
                yield frame

        except Exception as e:
            logger.error(f"{self} error generating image: {e}")
            yield ErrorFrame(f"Image generation error: {str(e)}")


class GoogleSTTService(STTService):
    """Google Cloud Speech-to-Text V2 service implementation.

    Provides real-time speech recognition using Google Cloud's Speech-to-Text V2 API
    with streaming support. Handles audio transcription and optional voice activity detection.

    Attributes:
        InputParams: Configuration parameters for the STT service.
    """

    # Google Cloud's STT service has a connection time limit of 5 minutes per stream.
    # They've shared an "endless streaming" example that guided this implementation:
    # https://cloud.google.com/speech-to-text/docs/transcribe-streaming-audio#endless-streaming

    STREAMING_LIMIT = 240000  # 4 minutes in milliseconds

    class InputParams(BaseModel):
        """Configuration parameters for Google Speech-to-Text.

        Attributes:
            languages: Single language or list of recognition languages. First language is primary.
            model: Speech recognition model to use.
            use_separate_recognition_per_channel: Process each audio channel separately.
            enable_automatic_punctuation: Add punctuation to transcripts.
            enable_spoken_punctuation: Include spoken punctuation in transcript.
            enable_spoken_emojis: Include spoken emojis in transcript.
            profanity_filter: Filter profanity from transcript.
            enable_word_time_offsets: Include timing information for each word.
            enable_word_confidence: Include confidence scores for each word.
            enable_interim_results: Stream partial recognition results.
            enable_voice_activity_events: Detect voice activity in audio.
        """

        languages: Union[Language, List[Language]] = Field(default_factory=lambda: [Language.EN_US])
        model: Optional[str] = "latest_long"
        use_separate_recognition_per_channel: Optional[bool] = False
        enable_automatic_punctuation: Optional[bool] = True
        enable_spoken_punctuation: Optional[bool] = False
        enable_spoken_emojis: Optional[bool] = False
        profanity_filter: Optional[bool] = False
        enable_word_time_offsets: Optional[bool] = False
        enable_word_confidence: Optional[bool] = False
        enable_interim_results: Optional[bool] = True
        enable_voice_activity_events: Optional[bool] = False

        @field_validator("languages", mode="before")
        @classmethod
        def validate_languages(cls, v) -> List[Language]:
            if isinstance(v, Language):
                return [v]
            return v

        @property
        def language_list(self) -> List[Language]:
            """Get languages as a guaranteed list."""
            assert isinstance(self.languages, list)
            return self.languages

    def __init__(
        self,
        *,
        credentials: Optional[str] = None,
        credentials_path: Optional[str] = None,
        location: str = "global",
        sample_rate: Optional[int] = None,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        """Initialize the Google STT service.

        Args:
            credentials: JSON string containing Google Cloud service account credentials.
            credentials_path: Path to service account credentials JSON file.
            location: Google Cloud location (e.g., "global", "us-central1").
            sample_rate: Audio sample rate in Hertz.
            params: Configuration parameters for the service.
            **kwargs: Additional arguments passed to STTService.

        Raises:
            ValueError: If neither credentials nor credentials_path is provided.
            ValueError: If project ID is not found in credentials.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._location = location
        self._stream = None
        self._config = None
        self._request_queue = asyncio.Queue()
        self._streaming_task = None

        # Used for keep-alive logic
        self._stream_start_time = 0
        self._last_audio_input = []
        self._audio_input = []
        self._result_end_time = 0
        self._is_final_end_time = 0
        self._final_request_end_time = 0
        self._bridging_offset = 0
        self._last_transcript_was_final = False
        self._new_stream = True
        self._restart_counter = 0

        # Configure client options based on location
        client_options = None
        if self._location != "global":
            client_options = ClientOptions(api_endpoint=f"{self._location}-speech.googleapis.com")

        # Extract project ID and create client
        if credentials:
            json_account_info = json.loads(credentials)
            self._project_id = json_account_info.get("project_id")
            creds = service_account.Credentials.from_service_account_info(json_account_info)
        elif credentials_path:
            with open(credentials_path) as f:
                json_account_info = json.load(f)
                self._project_id = json_account_info.get("project_id")
            creds = service_account.Credentials.from_service_account_file(credentials_path)
        else:
            raise ValueError("Either credentials or credentials_path must be provided")

        if not self._project_id:
            raise ValueError("Project ID not found in credentials")

        self._client = speech_v2.SpeechAsyncClient(credentials=creds, client_options=client_options)

        self._settings = {
            "language_codes": [
                self.language_to_service_language(lang) for lang in params.language_list
            ],
            "model": params.model,
            "use_separate_recognition_per_channel": params.use_separate_recognition_per_channel,
            "enable_automatic_punctuation": params.enable_automatic_punctuation,
            "enable_spoken_punctuation": params.enable_spoken_punctuation,
            "enable_spoken_emojis": params.enable_spoken_emojis,
            "profanity_filter": params.profanity_filter,
            "enable_word_time_offsets": params.enable_word_time_offsets,
            "enable_word_confidence": params.enable_word_confidence,
            "enable_interim_results": params.enable_interim_results,
            "enable_voice_activity_events": params.enable_voice_activity_events,
        }

    def language_to_service_language(self, language: Language | List[Language]) -> str | List[str]:
        """Convert Language enum(s) to Google STT language code(s).

        Args:
            language: Single Language enum or list of Language enums.

        Returns:
            str | List[str]: Google STT language code(s).
        """
        if isinstance(language, list):
            return [language_to_google_stt_language(lang) or "en-US" for lang in language]
        return language_to_google_stt_language(language) or "en-US"

    async def _reconnect_if_needed(self):
        """Reconnect the stream if it's currently active."""
        if self._streaming_task:
            logger.debug("Reconnecting stream due to configuration changes")
            await self._disconnect()
            await self._connect()

    async def set_language(self, language: Language):
        """Update the service's recognition language.

        A convenience method for setting a single language.

        Args:
            language: New language for recognition.
        """
        logger.debug(f"Switching STT language to: {language}")
        await self.set_languages([language])

    async def set_languages(self, languages: List[Language]):
        """Update the service's recognition languages.

        Args:
            languages: List of languages for recognition. First language is primary.
        """
        logger.debug(f"Switching STT languages to: {languages}")
        self._settings["language_codes"] = [
            self.language_to_service_language(lang) for lang in languages
        ]
        # Recreate stream with new languages
        await self._reconnect_if_needed()

    async def set_model(self, model: str):
        """Update the service's recognition model."""
        logger.debug(f"Switching STT model to: {model}")
        await super().set_model(model)
        self._settings["model"] = model
        # Recreate stream with new model
        await self._reconnect_if_needed()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def update_options(
        self,
        *,
        languages: Optional[List[Language]] = None,
        model: Optional[str] = None,
        enable_automatic_punctuation: Optional[bool] = None,
        enable_spoken_punctuation: Optional[bool] = None,
        enable_spoken_emojis: Optional[bool] = None,
        profanity_filter: Optional[bool] = None,
        enable_word_time_offsets: Optional[bool] = None,
        enable_word_confidence: Optional[bool] = None,
        enable_interim_results: Optional[bool] = None,
        enable_voice_activity_events: Optional[bool] = None,
        location: Optional[str] = None,
    ) -> None:
        """Update service options dynamically.

        Args:
            languages: New list of recongition languages.
            model: New recognition model.
            enable_automatic_punctuation: Enable/disable automatic punctuation.
            enable_spoken_punctuation: Enable/disable spoken punctuation.
            enable_spoken_emojis: Enable/disable spoken emojis.
            profanity_filter: Enable/disable profanity filter.
            enable_word_time_offsets: Enable/disable word timing info.
            enable_word_confidence: Enable/disable word confidence scores.
            enable_interim_results: Enable/disable interim results.
            enable_voice_activity_events: Enable/disable voice activity detection.
            location: New Google Cloud location.

        Note:
            Changes that affect the streaming configuration will cause
            the stream to be reconnected.
        """
        # Update settings with new values
        if languages is not None:
            logger.debug(f"Updating language to: {languages}")
            self._settings["language_codes"] = [
                self.language_to_service_language(lang) for lang in languages
            ]

        if model is not None:
            logger.debug(f"Updating model to: {model}")
            self._settings["model"] = model

        if enable_automatic_punctuation is not None:
            logger.debug(f"Updating automatic punctuation to: {enable_automatic_punctuation}")
            self._settings["enable_automatic_punctuation"] = enable_automatic_punctuation

        if enable_spoken_punctuation is not None:
            logger.debug(f"Updating spoken punctuation to: {enable_spoken_punctuation}")
            self._settings["enable_spoken_punctuation"] = enable_spoken_punctuation

        if enable_spoken_emojis is not None:
            logger.debug(f"Updating spoken emojis to: {enable_spoken_emojis}")
            self._settings["enable_spoken_emojis"] = enable_spoken_emojis

        if profanity_filter is not None:
            logger.debug(f"Updating profanity filter to: {profanity_filter}")
            self._settings["profanity_filter"] = profanity_filter

        if enable_word_time_offsets is not None:
            logger.debug(f"Updating word time offsets to: {enable_word_time_offsets}")
            self._settings["enable_word_time_offsets"] = enable_word_time_offsets

        if enable_word_confidence is not None:
            logger.debug(f"Updating word confidence to: {enable_word_confidence}")
            self._settings["enable_word_confidence"] = enable_word_confidence

        if enable_interim_results is not None:
            logger.debug(f"Updating interim results to: {enable_interim_results}")
            self._settings["enable_interim_results"] = enable_interim_results

        if enable_voice_activity_events is not None:
            logger.debug(f"Updating voice activity events to: {enable_voice_activity_events}")
            self._settings["enable_voice_activity_events"] = enable_voice_activity_events

        if location is not None:
            logger.debug(f"Updating location to: {location}")
            self._location = location

        # Reconnect the stream for updates
        await self._reconnect_if_needed()

    async def _connect(self):
        """Initialize streaming recognition config and stream."""
        logger.debug("Connecting to Google Speech-to-Text")

        # Set stream start time
        self._stream_start_time = int(time.time() * 1000)
        self._new_stream = True

        self._config = cloud_speech.StreamingRecognitionConfig(
            config=cloud_speech.RecognitionConfig(
                explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                    encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self.sample_rate,
                    audio_channel_count=1,
                ),
                language_codes=self._settings["language_codes"],
                model=self._settings["model"],
                features=cloud_speech.RecognitionFeatures(
                    enable_automatic_punctuation=self._settings["enable_automatic_punctuation"],
                    enable_spoken_punctuation=self._settings["enable_spoken_punctuation"],
                    enable_spoken_emojis=self._settings["enable_spoken_emojis"],
                    profanity_filter=self._settings["profanity_filter"],
                    enable_word_time_offsets=self._settings["enable_word_time_offsets"],
                    enable_word_confidence=self._settings["enable_word_confidence"],
                ),
            ),
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                enable_voice_activity_events=self._settings["enable_voice_activity_events"],
                interim_results=self._settings["enable_interim_results"],
            ),
        )

        self._streaming_task = self.create_task(self._stream_audio())

    async def _disconnect(self):
        """Clean up streaming recognition resources."""
        if self._streaming_task:
            logger.debug("Disconnecting from Google Speech-to-Text")
            # Send sentinel value to stop request generator
            await self._request_queue.put(None)
            await self.cancel_task(self._streaming_task)
            self._streaming_task = None
            # Clear any remaining items in the queue
            while not self._request_queue.empty():
                try:
                    self._request_queue.get_nowait()
                    self._request_queue.task_done()
                except asyncio.QueueEmpty:
                    break

    async def _request_generator(self):
        """Generates requests for the streaming recognize method."""
        recognizer_path = f"projects/{self._project_id}/locations/{self._location}/recognizers/_"
        logger.trace(f"Using recognizer path: {recognizer_path}")

        try:
            # Send initial config
            yield cloud_speech.StreamingRecognizeRequest(
                recognizer=recognizer_path,
                streaming_config=self._config,
            )

            while True:
                try:
                    audio_data = await self._request_queue.get()
                    if audio_data is None:  # Sentinel value to stop
                        break

                    # Check streaming limit
                    if (int(time.time() * 1000) - self._stream_start_time) > self.STREAMING_LIMIT:
                        logger.debug("Streaming limit reached, initiating graceful reconnection")
                        # Instead of immediate reconnection, we'll break and let the stream close naturally
                        self._last_audio_input = self._audio_input
                        self._audio_input = []
                        self._restart_counter += 1
                        # Put the current audio chunk back in the queue
                        await self._request_queue.put(audio_data)
                        break

                    self._audio_input.append(audio_data)
                    yield cloud_speech.StreamingRecognizeRequest(audio=audio_data)

                except asyncio.CancelledError:
                    break
                finally:
                    self._request_queue.task_done()

        except Exception as e:
            logger.error(f"Error in request generator: {e}")
            raise

    async def _stream_audio(self):
        """Handle bi-directional streaming with Google STT."""
        try:
            while True:
                try:
                    # Start bi-directional streaming
                    streaming_recognize = await self._client.streaming_recognize(
                        requests=self._request_generator()
                    )

                    # Process responses
                    await self._process_responses(streaming_recognize)

                    # If we're here, check if we need to reconnect
                    if (int(time.time() * 1000) - self._stream_start_time) > self.STREAMING_LIMIT:
                        logger.debug("Reconnecting stream after timeout")
                        # Reset stream start time
                        self._stream_start_time = int(time.time() * 1000)
                        continue
                    else:
                        # Normal stream end
                        break

                except Exception as e:
                    logger.warning(f"{self} Reconnecting: {e}")

                    await asyncio.sleep(1)  # Brief delay before reconnecting
                    self._stream_start_time = int(time.time() * 1000)
                    continue

        except Exception as e:
            logger.error(f"Error in streaming task: {e}")
            await self.push_frame(ErrorFrame(str(e)))

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process an audio chunk for STT transcription."""
        if self._streaming_task:
            # Queue the audio data
            await self._request_queue.put(audio)
        yield None

    async def _process_responses(self, streaming_recognize):
        """Process streaming recognition responses."""
        try:
            async for response in streaming_recognize:
                # Check streaming limit
                if (int(time.time() * 1000) - self._stream_start_time) > self.STREAMING_LIMIT:
                    logger.debug("Stream timeout reached in response processing")
                    break

                if not response.results:
                    continue

                for result in response.results:
                    if not result.alternatives:
                        continue

                    transcript = result.alternatives[0].transcript
                    if not transcript:
                        continue

                    primary_language = self._settings["language_codes"][0]

                    if result.is_final:
                        self._last_transcript_was_final = True
                        await self.push_frame(
                            TranscriptionFrame(transcript, "", time_now_iso8601(), primary_language)
                        )
                    else:
                        self._last_transcript_was_final = False
                        await self.push_frame(
                            InterimTranscriptionFrame(
                                transcript, "", time_now_iso8601(), primary_language
                            )
                        )

        except Exception as e:
            logger.error(f"Error processing Google STT responses: {e}")

            # Re-raise the exception to let it propagate (e.g. in the case of a timeout, propagate to _stream_audio to reconnect)
            raise
