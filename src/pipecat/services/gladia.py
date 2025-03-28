#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
import warnings
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import aiohttp
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.services.ai_services import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Gladia, you need to `pip install pipecat-ai[gladia]`. Also, set `GLADIA_API_KEY` environment variable."
    )
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
        Language.BG: "bg",
        Language.BN: "bn",
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
        Language.FR: "fr",
        Language.GA: "ga",
        Language.GL: "gl",
        Language.GU: "gu",
        Language.HE: "he",
        Language.HI: "hi",
        Language.HR: "hr",
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
        Language.LO: "lo",
        Language.LT: "lt",
        Language.LV: "lv",
        Language.MK: "mk",
        Language.ML: "ml",
        Language.MN: "mn",
        Language.MR: "mr",
        Language.MS: "ms",
        Language.MT: "mt",
        Language.MY: "my",
        Language.NE: "ne",
        Language.NL: "nl",
        Language.NO: "no",
        Language.OR: "or",
        Language.PA: "pa",
        Language.PL: "pl",
        Language.PS: "ps",
        Language.PT: "pt",
        Language.RO: "ro",
        Language.RU: "ru",
        Language.SI: "si",
        Language.SK: "sk",
        Language.SL: "sl",
        Language.SO: "so",
        Language.SQ: "sq",
        Language.SR: "sr",
        Language.SU: "su",
        Language.SV: "sv",
        Language.SW: "sw",
        Language.TA: "ta",
        Language.TE: "te",
        Language.TH: "th",
        Language.TR: "tr",
        Language.UK: "uk",
        Language.UR: "ur",
        Language.UZ: "uz",
        Language.VI: "vi",
        Language.ZH: "zh",
        Language.ZU: "zu",
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


# Configurations supported by Gladia
# Refer to the docs for more information:
# https://docs.gladia.io/api-reference/v2/live/init


class LanguageConfig(BaseModel):
    """Configuration for language detection and handling.

    Attributes:
        languages: List of language codes to use for transcription
        code_switching: Whether to auto-detect language changes during transcription
    """

    languages: Optional[List[str]] = None
    code_switching: Optional[bool] = None


class PreProcessingConfig(BaseModel):
    """Configuration for audio pre-processing options.

    Attributes:
        audio_enhancer: Whether to apply audio enhancement
        speech_threshold: Sensitivity for speech detection (0-1)
    """

    audio_enhancer: Optional[bool] = None
    speech_threshold: Optional[float] = None


class CustomVocabularyItem(BaseModel):
    """Represents a custom vocabulary item with an intensity value.

    Attributes:
        value: The vocabulary word or phrase
        intensity: The bias intensity for this vocabulary item (0-1)
    """

    value: str
    intensity: float


class CustomVocabularyConfig(BaseModel):
    """Configuration for custom vocabulary.

    Attributes:
        vocabulary: List of words/phrases or CustomVocabularyItem objects
        default_intensity: Default intensity for simple string vocabulary items
    """

    vocabulary: Optional[List[Union[str, CustomVocabularyItem]]] = None
    default_intensity: Optional[float] = None


class CustomSpellingConfig(BaseModel):
    """Configuration for custom spelling rules.

    Attributes:
        spelling_dictionary: Mapping of correct spellings to phonetic variations
    """

    spelling_dictionary: Optional[Dict[str, List[str]]] = None


class TranslationConfig(BaseModel):
    """Configuration for real-time translation.

    Attributes:
        target_languages: List of target language codes for translation
        model: Translation model to use ("base" or "enhanced")
        match_original_utterances: Whether to align translations with original utterances
    """

    target_languages: Optional[List[str]] = None
    model: Optional[str] = None
    match_original_utterances: Optional[bool] = None


class RealtimeProcessingConfig(BaseModel):
    """Configuration for real-time processing features.

    Attributes:
        words_accurate_timestamps: Whether to provide per-word timestamps
        custom_vocabulary: Whether to enable custom vocabulary
        custom_vocabulary_config: Custom vocabulary configuration
        custom_spelling: Whether to enable custom spelling
        custom_spelling_config: Custom spelling configuration
        translation: Whether to enable translation
        translation_config: Translation configuration
        named_entity_recognition: Whether to enable named entity recognition
        sentiment_analysis: Whether to enable sentiment analysis
    """

    words_accurate_timestamps: Optional[bool] = None
    custom_vocabulary: Optional[bool] = None
    custom_vocabulary_config: Optional[CustomVocabularyConfig] = None
    custom_spelling: Optional[bool] = None
    custom_spelling_config: Optional[CustomSpellingConfig] = None
    translation: Optional[bool] = None
    translation_config: Optional[TranslationConfig] = None
    named_entity_recognition: Optional[bool] = None
    sentiment_analysis: Optional[bool] = None


class MessagesConfig(BaseModel):
    """Configuration for controlling which message types are sent via WebSocket.

    Attributes:
        receive_partial_transcripts: Whether to receive intermediate transcription results
        receive_final_transcripts: Whether to receive final transcription results
        receive_speech_events: Whether to receive speech begin/end events
        receive_pre_processing_events: Whether to receive pre-processing events
        receive_realtime_processing_events: Whether to receive real-time processing events
        receive_post_processing_events: Whether to receive post-processing events
        receive_acknowledgments: Whether to receive acknowledgment messages
        receive_errors: Whether to receive error messages
        receive_lifecycle_events: Whether to receive lifecycle events
    """

    receive_partial_transcripts: Optional[bool] = None
    receive_final_transcripts: Optional[bool] = None
    receive_speech_events: Optional[bool] = None
    receive_pre_processing_events: Optional[bool] = None
    receive_realtime_processing_events: Optional[bool] = None
    receive_post_processing_events: Optional[bool] = None
    receive_acknowledgments: Optional[bool] = None
    receive_errors: Optional[bool] = None
    receive_lifecycle_events: Optional[bool] = None


class GladiaSTTService(STTService):
    """Speech-to-Text service using Gladia's API.

    This service connects to Gladia's WebSocket API for real-time transcription
    with support for multiple languages, custom vocabulary, and various processing options.

    For complete API documentation, see: https://docs.gladia.io/api-reference/v2/live/init
    """

    class InputParams(BaseModel):
        """Configuration parameters for the Gladia STT service.

        Attributes:
            encoding: Audio encoding format
            bit_depth: Audio bit depth
            channels: Number of audio channels
            custom_metadata: Additional metadata to include with requests
            endpointing: Silence duration in seconds to mark end of speech
            maximum_duration_without_endpointing: Maximum utterance duration without silence
            language: DEPRECATED - Use language_config instead
            language_config: Detailed language configuration
            pre_processing: Audio pre-processing options
            realtime_processing: Real-time processing features
            messages_config: WebSocket message filtering options
        """

        encoding: Optional[str] = "wav/pcm"
        bit_depth: Optional[int] = 16
        channels: Optional[int] = 1
        custom_metadata: Optional[Dict[str, Any]] = None
        endpointing: Optional[float] = None
        maximum_duration_without_endpointing: Optional[int] = None
        language: Optional[Language] = None  # Deprecated
        language_config: Optional[LanguageConfig] = None
        pre_processing: Optional[PreProcessingConfig] = None
        realtime_processing: Optional[RealtimeProcessingConfig] = None
        messages_config: Optional[MessagesConfig] = None

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "https://api.gladia.io/v2/live",
        confidence: float = 0.5,
        sample_rate: Optional[int] = None,
        model: str = "fast",
        params: InputParams = InputParams(),
        **kwargs,
    ):
        """Initialize the Gladia STT service.

        Args:
            api_key: Gladia API key
            url: Gladia API URL
            confidence: Minimum confidence threshold for transcriptions
            sample_rate: Audio sample rate in Hz
            model: Model to use ("fast" or "accurate")
            params: Additional configuration parameters
            **kwargs: Additional arguments passed to the STTService
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

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
            settings["language_config"] = self._params.language_config.dict(exclude_none=True)
        elif self._params.language:  # Backward compatibility for deprecated parameter
            language_code = self.language_to_service_language(self._params.language)
            if language_code:
                settings["language_config"] = {
                    "languages": [language_code],
                    "code_switching": False,
                }

        # Add pre_processing configuration if provided
        if self._params.pre_processing:
            settings["pre_processing"] = self._params.pre_processing.dict(exclude_none=True)

        # Add realtime_processing configuration if provided
        if self._params.realtime_processing:
            settings["realtime_processing"] = self._params.realtime_processing.dict(
                exclude_none=True
            )

        # Add messages_config if provided
        if self._params.messages_config:
            settings["messages_config"] = self._params.messages_config.dict(exclude_none=True)

        return settings

    async def start(self, frame: StartFrame):
        """Start the Gladia STT websocket connection."""
        await super().start(frame)
        if self._websocket:
            return
        settings = self._prepare_settings()
        response = await self._setup_gladia(settings)
        self._websocket = await websockets.connect(response["url"])
        if not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler())

    async def stop(self, frame: EndFrame):
        """Stop the Gladia STT websocket connection."""
        await super().stop(frame)
        await self._send_stop_recording()
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
        if self._receive_task:
            await self.wait_for_task(self._receive_task)
            self._receive_task = None

    async def cancel(self, frame: CancelFrame):
        """Cancel the Gladia STT websocket connection."""
        await super().cancel(frame)
        await self._websocket.close()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Run speech-to-text on audio data."""
        await self.start_processing_metrics()
        await self._send_audio(audio)
        await self.stop_processing_metrics()
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
                    logger.error(
                        f"Gladia error: {response.status}: {response.text or response.reason}"
                    )
                    raise Exception(f"Failed to initialize Gladia session: {response.status}")

    async def _send_audio(self, audio: bytes):
        data = base64.b64encode(audio).decode("utf-8")
        message = {"type": "audio_chunk", "data": {"chunk": data}}
        await self._websocket.send(json.dumps(message))

    async def _send_stop_recording(self):
        if self._websocket and not self._websocket.closed:
            await self._websocket.send(json.dumps({"type": "stop_recording"}))

    async def _receive_task_handler(self):
        async for message in self._websocket:
            content = json.loads(message)
            if content["type"] == "transcript":
                utterance = content["data"]["utterance"]
                confidence = utterance.get("confidence", 0)
                transcript = utterance["text"]
                if confidence >= self._confidence:
                    if content["data"]["is_final"]:
                        await self.push_frame(
                            TranscriptionFrame(transcript, "", time_now_iso8601())
                        )
                    else:
                        await self.push_frame(
                            InterimTranscriptionFrame(transcript, "", time_now_iso8601())
                        )
