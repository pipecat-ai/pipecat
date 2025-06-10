#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import json
import warnings
from typing import Any, AsyncGenerator, Dict, Optional

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    TranslationFrame,
)
from pipecat.services.gladia.config import GladiaInputParams
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Gladia, you need to `pip install pipecat-ai[gladia]`.")
    raise Exception(f"Missing module: {e}")


def language_to_gladia_language(language: Language) -> Optional[str]:
    """Convert a Language enum to Gladia's language code format.

    Args:
        language: The Language enum value to convert

    Returns:
        The Gladia language code string or None if not supported
    """
    BASE_LANGUAGES = {
        Language.AF: "af",
        Language.AM: "am",
        Language.AR: "ar",
        Language.AS: "as",
        Language.AZ: "az",
        Language.BA: "ba",
        Language.BE: "be",
        Language.BG: "bg",
        Language.BN: "bn",
        Language.BO: "bo",
        Language.BR: "br",
        Language.BS: "bs",
        Language.CA: "ca",
        Language.CS: "cs",
        Language.CY: "cy",
        Language.DA: "da",
        Language.DE: "de",
        Language.EL: "el",
        Language.EN: "en",
        Language.ES: "es",
        Language.ET: "et",
        Language.EU: "eu",
        Language.FA: "fa",
        Language.FI: "fi",
        Language.FO: "fo",
        Language.FR: "fr",
        Language.GL: "gl",
        Language.GU: "gu",
        Language.HA: "ha",
        Language.HAW: "haw",
        Language.HE: "he",
        Language.HI: "hi",
        Language.HR: "hr",
        Language.HT: "ht",
        Language.HU: "hu",
        Language.HY: "hy",
        Language.ID: "id",
        Language.IS: "is",
        Language.IT: "it",
        Language.JA: "ja",
        Language.JV: "jv",
        Language.KA: "ka",
        Language.KK: "kk",
        Language.KM: "km",
        Language.KN: "kn",
        Language.KO: "ko",
        Language.LA: "la",
        Language.LB: "lb",
        Language.LN: "ln",
        Language.LO: "lo",
        Language.LT: "lt",
        Language.LV: "lv",
        Language.MG: "mg",
        Language.MI: "mi",
        Language.MK: "mk",
        Language.ML: "ml",
        Language.MN: "mn",
        Language.MR: "mr",
        Language.MS: "ms",
        Language.MT: "mt",
        Language.MY_MR: "mymr",
        Language.NE: "ne",
        Language.NL: "nl",
        Language.NN: "nn",
        Language.NO: "no",
        Language.OC: "oc",
        Language.PA: "pa",
        Language.PL: "pl",
        Language.PS: "ps",
        Language.PT: "pt",
        Language.RO: "ro",
        Language.RU: "ru",
        Language.SA: "sa",
        Language.SD: "sd",
        Language.SI: "si",
        Language.SK: "sk",
        Language.SL: "sl",
        Language.SN: "sn",
        Language.SO: "so",
        Language.SQ: "sq",
        Language.SR: "sr",
        Language.SU: "su",
        Language.SV: "sv",
        Language.SW: "sw",
        Language.TA: "ta",
        Language.TE: "te",
        Language.TG: "tg",
        Language.TH: "th",
        Language.TK: "tk",
        Language.TL: "tl",
        Language.TR: "tr",
        Language.TT: "tt",
        Language.UK: "uk",
        Language.UR: "ur",
        Language.UZ: "uz",
        Language.VI: "vi",
        Language.YI: "yi",
        Language.YO: "yo",
        Language.ZH: "zh",
    }

    result = BASE_LANGUAGES.get(language)

    # If not found in base languages, try to find the base language from a variant
    if not result:
        # Convert enum value to string and get the base language part (e.g. es-ES -> es)
        lang_str = str(language.value)
        base_code = lang_str.split("-")[0].lower()
        # Look up the base code in our supported languages
        result = base_code if base_code in BASE_LANGUAGES.values() else None

    return result


# Deprecation warning for nested InputParams
class _InputParamsDescriptor:
    """Descriptor for backward compatibility with deprecation warning."""

    def __get__(self, obj, objtype=None):
        warnings.warn(
            "GladiaSTTService.InputParams is deprecated and will be removed in a future version. "
            "Import and use GladiaInputParams directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return GladiaInputParams


class GladiaSTTService(STTService):
    """Speech-to-Text service using Gladia's API.

    This service connects to Gladia's WebSocket API for real-time transcription
    with support for multiple languages, custom vocabulary, and various processing options.

    For complete API documentation, see: https://docs.gladia.io/api-reference/v2/live/init
    """

    # Maintain backward compatibility
    InputParams = _InputParamsDescriptor()

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "https://api.gladia.io/v2/live",
        confidence: float = 0.5,
        sample_rate: Optional[int] = None,
        model: str = "solaria-1",
        params: Optional[GladiaInputParams] = None,
        **kwargs,
    ):
        """Initialize the Gladia STT service.

        Args:
            api_key: Gladia API key
            url: Gladia API URL
            confidence: Minimum confidence threshold for transcriptions
            sample_rate: Audio sample rate in Hz
            model: Model to use ("solaria-1", "solaria-mini-1", "fast",
                or "accurate")
            params: Additional configuration parameters
            **kwargs: Additional arguments passed to the STTService
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or GladiaInputParams()

        # Warn about deprecated language parameter if it's used
        if params.language is not None:
            warnings.warn(
                "The 'language' parameter is deprecated and will be removed in a future version. "
                "Use 'language_config' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        self._api_key = api_key
        self._url = url
        self.set_model_name(model)
        self._confidence = confidence
        self._params = params
        self._websocket = None
        self._receive_task = None
        self._keepalive_task = None
        self._settings = {}

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert pipecat Language enum to Gladia's language code."""
        return language_to_gladia_language(language)

    def _prepare_settings(self) -> Dict[str, Any]:
        settings = {
            "encoding": self._params.encoding or "wav/pcm",
            "bit_depth": self._params.bit_depth or 16,
            "sample_rate": self.sample_rate,
            "channels": self._params.channels or 1,
            "model": self._model_name,
        }

        # Add custom_metadata if provided
        if self._params.custom_metadata:
            settings["custom_metadata"] = self._params.custom_metadata

        # Add endpointing parameters if provided
        if self._params.endpointing is not None:
            settings["endpointing"] = self._params.endpointing
        if self._params.maximum_duration_without_endpointing is not None:
            settings["maximum_duration_without_endpointing"] = (
                self._params.maximum_duration_without_endpointing
            )

        # Add language configuration (prioritize language_config over deprecated language)
        if self._params.language_config:
            settings["language_config"] = self._params.language_config.model_dump(exclude_none=True)
        elif self._params.language:  # Backward compatibility for deprecated parameter
            language_code = self.language_to_service_language(self._params.language)
            if language_code:
                settings["language_config"] = {
                    "languages": [language_code],
                    "code_switching": False,
                }

        # Add pre_processing configuration if provided
        if self._params.pre_processing:
            settings["pre_processing"] = self._params.pre_processing.model_dump(exclude_none=True)

        # Add realtime_processing configuration if provided
        if self._params.realtime_processing:
            settings["realtime_processing"] = self._params.realtime_processing.model_dump(
                exclude_none=True
            )

        # Add messages_config if provided
        if self._params.messages_config:
            settings["messages_config"] = self._params.messages_config.model_dump(exclude_none=True)

        # Store settings for tracing
        self._settings = settings

        return settings

    async def start(self, frame: StartFrame):
        """Start the Gladia STT websocket connection."""
        await super().start(frame)
        if self._websocket:
            return
        settings = self._prepare_settings()
        response = await self._setup_gladia(settings)
        self._websocket = await websockets.connect(response["url"])
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler())
        if self._websocket and not self._keepalive_task:
            self._keepalive_task = self.create_task(self._keepalive_task_handler())

    async def stop(self, frame: EndFrame):
        """Stop the Gladia STT websocket connection."""
        await super().stop(frame)
        await self._send_stop_recording()

        if self._keepalive_task:
            await self.cancel_task(self._keepalive_task)
            self._keepalive_task = None

        if self._websocket:
            await self._websocket.close()
            self._websocket = None

        if self._receive_task:
            await self.wait_for_task(self._receive_task)
            self._receive_task = None

    async def cancel(self, frame: CancelFrame):
        """Cancel the Gladia STT websocket connection."""
        await super().cancel(frame)

        if self._keepalive_task:
            await self.cancel_task(self._keepalive_task)
            self._keepalive_task = None

        if self._websocket:
            await self._websocket.close()
            self._websocket = None

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Run speech-to-text on audio data."""
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()
        await self._send_audio(audio)
        yield None

    async def _setup_gladia(self, settings: Dict[str, Any]):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._url,
                headers={"X-Gladia-Key": self._api_key, "Content-Type": "application/json"},
                json=settings,
            ) as response:
                if response.ok:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Gladia error: {response.status}: {error_text or response.reason}"
                    )
                    raise Exception(
                        f"Failed to initialize Gladia session: {response.status} - {error_text}"
                    )

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[str] = None
    ):
        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()

    async def _send_audio(self, audio: bytes):
        data = base64.b64encode(audio).decode("utf-8")
        message = {"type": "audio_chunk", "data": {"chunk": data}}
        await self._websocket.send(json.dumps(message))

    async def _send_stop_recording(self):
        if self._websocket and not self._websocket.closed:
            await self._websocket.send(json.dumps({"type": "stop_recording"}))

    async def _keepalive_task_handler(self):
        """Send periodic empty audio chunks to keep the connection alive."""
        try:
            while True:
                # Send keepalive every 20 seconds (Gladia times out after 30 seconds)
                await asyncio.sleep(20)
                if self._websocket and not self._websocket.closed:
                    # Send an empty audio chunk as keepalive
                    empty_audio = b""
                    await self._send_audio(empty_audio)
                else:
                    logger.debug("Websocket closed, stopping keepalive")
                    break
        except websockets.exceptions.ConnectionClosed:
            logger.debug("Connection closed during keepalive")
        except Exception as e:
            logger.error(f"Error in Gladia keepalive task: {e}")

    async def _receive_task_handler(self):
        try:
            async for message in self._websocket:
                content = json.loads(message)
                if content["type"] == "transcript":
                    utterance = content["data"]["utterance"]
                    confidence = utterance.get("confidence", 0)
                    language = utterance["language"]
                    transcript = utterance["text"]
                    is_final = content["data"]["is_final"]
                    if confidence >= self._confidence:
                        if is_final:
                            await self.push_frame(
                                TranscriptionFrame(
                                    transcript,
                                    "",
                                    time_now_iso8601(),
                                    language,
                                    result=content,
                                )
                            )
                            await self._handle_transcription(
                                transcript=transcript,
                                is_final=is_final,
                                language=language,
                            )
                        else:
                            await self.push_frame(
                                InterimTranscriptionFrame(
                                    transcript,
                                    "",
                                    time_now_iso8601(),
                                    language,
                                    result=content,
                                )
                            )
                elif content["type"] == "translation":
                    translated_utterance = content["data"]["translated_utterance"]
                    original_language = content["data"]["original_language"]
                    translated_language = translated_utterance["language"]
                    confidence = translated_utterance.get("confidence", 0)
                    translation = translated_utterance["text"]
                    if translated_language != original_language and confidence >= self._confidence:
                        await self.push_frame(
                            TranslationFrame(
                                translation, "", time_now_iso8601(), translated_language
                            )
                        )
        except websockets.exceptions.ConnectionClosed:
            # Expected when closing the connection
            pass
        except Exception as e:
            logger.error(f"Error in Gladia WebSocket handler: {e}")
