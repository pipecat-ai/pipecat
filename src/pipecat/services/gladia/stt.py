#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gladia Speech-to-Text (STT) service implementation.

This module provides a Speech-to-Text service using Gladia's real-time WebSocket API,
supporting multiple languages, custom vocabulary, and various audio processing options.
"""

import asyncio
import base64
import json
import warnings
from typing import Any, AsyncGenerator, Dict, Literal, Optional

import aiohttp
from loguru import logger

from pipecat import __version__ as pipecat_version
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
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Gladia, you need to `pip install pipecat-ai[gladia]`.")
    raise Exception(f"Missing module: {e}")


def language_to_gladia_language(language: Language) -> Optional[str]:
    """Convert a Language enum to Gladia's language code format.

    Args:
        language: The Language enum value to convert.

    Returns:
        The Gladia language code string or None if not supported.
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
        with warnings.catch_warnings():
            warnings.simplefilter("always")
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
    Provides automatic reconnection, audio buffering, and comprehensive error handling.

    For complete API documentation, see: https://docs.gladia.io/api-reference/v2/live/init

    .. deprecated:: 0.0.62
        Use :class:`~pipecat.services.gladia.config.GladiaInputParams` directly instead.
    """

    # Maintain backward compatibility
    InputParams = _InputParamsDescriptor()

    def __init__(
        self,
        *,
        api_key: str,
        region: Literal["us-west", "eu-west"] | None = None,
        url: str = "https://api.gladia.io/v2/live",
        confidence: Optional[float] = None,
        sample_rate: Optional[int] = None,
        model: str = "solaria-1",
        params: Optional[GladiaInputParams] = None,
        max_reconnection_attempts: int = 5,
        reconnection_delay: float = 1.0,
        max_buffer_size: int = 1024 * 1024 * 20,  # 20MB default buffer
        **kwargs,
    ):
        """Initialize the Gladia STT service.

        Args:
            api_key: Gladia API key for authentication.
            region: Region used to process audio. eu-west or us-west. Defaults to eu-west.
            url: Gladia API URL. Defaults to "https://api.gladia.io/v2/live".
            confidence: Minimum confidence threshold for transcriptions (0.0-1.0).

                .. deprecated:: 0.0.86
                    The 'confidence' parameter is deprecated and will be removed in a future version.
                    No confidence threshold is applied.

            sample_rate: Audio sample rate in Hz. If None, uses service default.
            model: Model to use for transcription. Defaults to "solaria-1".
            params: Additional configuration parameters for Gladia service.
            max_reconnection_attempts: Maximum number of reconnection attempts. Defaults to 5.
            reconnection_delay: Initial delay between reconnection attempts in seconds.
            max_buffer_size: Maximum size of audio buffer in bytes. Defaults to 20MB.
            **kwargs: Additional arguments passed to the STTService parent class.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or GladiaInputParams()

        if params.language is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "The 'language' parameter is deprecated and will be removed in a future version. "
                    "Use 'language_config' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        if confidence:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "The 'confidence' parameter is deprecated and will be removed in a future version. "
                    "No confidence threshold is applied.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        self._api_key = api_key
        self._region = region
        self._url = url
        self.set_model_name(model)
        self._params = params
        self._websocket = None
        self._receive_task = None
        self._keepalive_task = None
        self._settings = {}

        # Reconnection settings
        self._max_reconnection_attempts = max_reconnection_attempts
        self._reconnection_delay = reconnection_delay
        self._reconnection_attempts = 0
        self._session_url = None
        self._connection_active = False

        # Audio buffer management
        self._audio_buffer = bytearray()
        self._bytes_sent = 0
        self._max_buffer_size = max_buffer_size
        self._buffer_lock = asyncio.Lock()

        # Connection management
        self._connection_task = None
        self._should_reconnect = True

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate performance metrics.

        Returns:
            True, indicating this service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert pipecat Language enum to Gladia's language code.

        Args:
            language: The Language enum value to convert.

        Returns:
            The Gladia language code string or None if not supported.
        """
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
        settings["custom_metadata"] = dict(self._params.custom_metadata or {})
        settings["custom_metadata"]["pipecat"] = pipecat_version

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
        """Start the Gladia STT websocket connection.

        Args:
            frame: The start frame triggering service startup.
        """
        await super().start(frame)
        if self._connection_task:
            return

        self._should_reconnect = True
        self._connection_task = self.create_task(self._connection_handler())

    async def stop(self, frame: EndFrame):
        """Stop the Gladia STT websocket connection.

        Args:
            frame: The end frame triggering service shutdown.
        """
        await super().stop(frame)
        self._should_reconnect = False
        await self._send_stop_recording()

        if self._connection_task:
            await self.cancel_task(self._connection_task)
            self._connection_task = None

        await self._cleanup_connection()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Gladia STT websocket connection.

        Args:
            frame: The cancel frame triggering service cancellation.
        """
        await super().cancel(frame)
        self._should_reconnect = False

        if self._connection_task:
            await self.cancel_task(self._connection_task)
            self._connection_task = None

        await self._cleanup_connection()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Run speech-to-text on audio data.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            None (processing is handled asynchronously via WebSocket).
        """
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()

        # Add audio to buffer
        async with self._buffer_lock:
            self._audio_buffer.extend(audio)
            # Trim buffer if it exceeds max size
            if len(self._audio_buffer) > self._max_buffer_size:
                trim_size = len(self._audio_buffer) - self._max_buffer_size
                self._audio_buffer = self._audio_buffer[trim_size:]
                self._bytes_sent = max(0, self._bytes_sent - trim_size)
                logger.warning(f"Audio buffer exceeded max size, trimmed {trim_size} bytes")

        # Send audio if connected
        if self._connection_active and self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._send_audio(audio)
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"Websocket closed while sending audio chunk: {e}")
                self._connection_active = False

        yield None

    async def _connection_handler(self):
        """Handle WebSocket connection with automatic reconnection."""
        while self._should_reconnect:
            try:
                # Initialize session if needed
                if not self._session_url:
                    settings = self._prepare_settings()
                    response = await self._setup_gladia(settings)
                    self._session_url = response["url"]
                    self._reconnection_attempts = 0
                    logger.info(f"Session URL : {self._session_url}")

                # Connect with automatic reconnection
                async with websocket_connect(self._session_url) as websocket:
                    try:
                        self._websocket = websocket
                        self._connection_active = True
                        logger.debug(f"{self} Connected to Gladia WebSocket")

                        # Send buffered audio if any
                        await self._send_buffered_audio()

                        # Start tasks
                        self._receive_task = self.create_task(self._receive_task_handler())
                        self._keepalive_task = self.create_task(self._keepalive_task_handler())

                        # Wait for tasks to complete
                        await asyncio.gather(self._receive_task, self._keepalive_task)

                    except websockets.exceptions.ConnectionClosed as e:
                        logger.warning(f"WebSocket connection closed: {e}")
                        self._connection_active = False

                        # Clean up tasks
                        if self._receive_task:
                            await self.cancel_task(self._receive_task)
                        if self._keepalive_task:
                            await self.cancel_task(self._keepalive_task)

                        # Attempt reconnect using helper
                        if not await self._maybe_reconnect():
                            break

            except Exception as e:
                logger.error(f"Error in connection handler: {e}")
                self._connection_active = False

                if not self._should_reconnect:
                    break

                # Reset session URL to get a new one
                self._session_url = None
                await asyncio.sleep(self._reconnection_delay)

    async def _cleanup_connection(self):
        """Clean up connection resources."""
        self._connection_active = False

        if self._keepalive_task:
            await self.cancel_task(self._keepalive_task)
            self._keepalive_task = None

        if self._websocket:
            await self._websocket.close()
            self._websocket = None

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

    async def _setup_gladia(self, settings: Dict[str, Any]):
        async with aiohttp.ClientSession() as session:
            params = {}
            if self._region:
                params["region"] = self._region
            async with session.post(
                self._url,
                headers={"X-Gladia-Key": self._api_key},
                json=settings,
                params=params,
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
        """Send audio chunk with proper message format."""
        if self._websocket and self._websocket.state is State.OPEN:
            data = base64.b64encode(audio).decode("utf-8")
            message = {"type": "audio_chunk", "data": {"chunk": data}}
            await self._websocket.send(json.dumps(message))

    async def _send_buffered_audio(self):
        """Send any buffered audio after reconnection."""
        async with self._buffer_lock:
            if self._audio_buffer:
                logger.debug(f"{self} Sending {len(self._audio_buffer)} bytes of buffered audio")
                await self._send_audio(bytes(self._audio_buffer))

    async def _send_stop_recording(self):
        if self._websocket and self._websocket.state is State.OPEN:
            await self._websocket.send(json.dumps({"type": "stop_recording"}))

    async def _keepalive_task_handler(self):
        """Send periodic empty audio chunks to keep the connection alive."""
        try:
            KEEPALIVE_SLEEP = 20
            while self._connection_active:
                # Send keepalive (Gladia times out after 30 seconds)
                await asyncio.sleep(KEEPALIVE_SLEEP)
                if self._websocket and self._websocket.state is State.OPEN:
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

                # Handle audio chunk acknowledgments
                if content["type"] == "audio_chunk" and content.get("acknowledged"):
                    byte_range = content["data"]["byte_range"]
                    async with self._buffer_lock:
                        # Update bytes sent and trim acknowledged data from buffer
                        end_byte = byte_range[1]
                        if end_byte > self._bytes_sent:
                            trim_size = end_byte - self._bytes_sent
                            self._audio_buffer = self._audio_buffer[trim_size:]
                            self._bytes_sent = end_byte

                elif content["type"] == "transcript":
                    utterance = content["data"]["utterance"]
                    language = utterance["language"]
                    transcript = utterance["text"]
                    is_final = content["data"]["is_final"]
                    if is_final:
                        await self.push_frame(
                            TranscriptionFrame(
                                transcript,
                                self._user_id,
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
                                self._user_id,
                                time_now_iso8601(),
                                language,
                                result=content,
                            )
                        )
                elif content["type"] == "translation":
                    translated_utterance = content["data"]["translated_utterance"]
                    original_language = content["data"]["original_language"]
                    translated_language = translated_utterance["language"]
                    translation = translated_utterance["text"]
                    if translated_language != original_language:
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

    async def _maybe_reconnect(self) -> bool:
        """Handle exponential backoff reconnection logic."""
        if not self._should_reconnect:
            return False
        self._reconnection_attempts += 1
        if self._reconnection_attempts > self._max_reconnection_attempts:
            logger.error(f"Max reconnection attempts ({self._max_reconnection_attempts}) reached")
            self._should_reconnect = False
            return False
        delay = self._reconnection_delay * (2 ** (self._reconnection_attempts - 1))
        logger.debug(
            f"{self} Reconnecting in {delay} seconds (attempt {self._reconnection_attempts}/{self._max_reconnection_attempts})"
        )
        await asyncio.sleep(delay)
        return True
