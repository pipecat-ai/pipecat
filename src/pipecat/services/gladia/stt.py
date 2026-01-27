#
# Copyright (c) 2024-2026, Daily
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

from pipecat import version as pipecat_version
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    TranslationFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.services.gladia.config import GladiaInputParams
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language, resolve_language
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
    LANGUAGE_MAP = {
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

    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


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


class GladiaSTTService(WebsocketSTTService):
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
        max_buffer_size: int = 1024 * 1024 * 20,  # 20MB default buffer
        should_interrupt: bool = True,
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
            max_buffer_size: Maximum size of audio buffer in bytes. Defaults to 20MB.
            should_interrupt: Determine whether the bot should be interrupted when
                Gladia VAD detects user speech. Defaults to True.
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
        self._receive_task = None
        self._keepalive_task = None
        self._settings = {}

        # Session management
        self._session_url = None
        self._session_id = None
        self._connection_active = False

        # Audio buffer management
        self._audio_buffer = bytearray()
        self._bytes_sent = 0
        self._max_buffer_size = max_buffer_size
        self._buffer_lock = asyncio.Lock()

        # VAD state tracking
        self._is_speaking = False
        self._should_interrupt = should_interrupt

    def __str__(self):
        return f"{self.name} [{self._session_id}]"

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
        settings["custom_metadata"]["pipecat"] = pipecat_version()

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
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Gladia STT websocket connection.

        Args:
            frame: The end frame triggering service shutdown.
        """
        await super().stop(frame)
        await self._send_stop_recording()
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Gladia STT websocket connection.

        Args:
            frame: The cancel frame triggering service cancellation.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Run speech-to-text on audio data.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            None (processing is handled asynchronously via WebSocket).
        """
        await self.start_processing_metrics()

        # Add audio to buffer
        async with self._buffer_lock:
            self._audio_buffer.extend(audio)
            # Trim buffer if it exceeds max size
            if len(self._audio_buffer) > self._max_buffer_size:
                trim_size = len(self._audio_buffer) - self._max_buffer_size
                self._audio_buffer = self._audio_buffer[trim_size:]
                self._bytes_sent = max(0, self._bytes_sent - trim_size)
                logger.warning(f"{self} Audio buffer exceeded max size, trimmed {trim_size} bytes")

        # Send audio if connected
        if self._connection_active and self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._send_audio(audio)
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"{self} Websocket closed while sending audio chunk: {e}")
                self._connection_active = False

        yield None

    async def _connect(self):
        """Connect to the Gladia service.

        Initializes the session if needed and establishes websocket connection.
        """
        await super()._connect()

        # Initialize session if needed
        if not self._session_url:
            settings = self._prepare_settings()
            response = await self._setup_gladia(settings)
            self._session_url = response["url"]
            self._session_id = response["id"]
            logger.info(f"{self} Session URL: {self._session_url}")

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

        if self._websocket and not self._keepalive_task:
            self._keepalive_task = self.create_task(self._keepalive_task_handler())

    async def _disconnect(self):
        """Disconnect from the Gladia service.

        Cleans up tasks and closes websocket connection.
        """
        await super()._disconnect()

        self._connection_active = False

        if self._keepalive_task:
            await self.cancel_task(self._keepalive_task)
            self._keepalive_task = None

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish the websocket connection to Gladia."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug(f"{self}Connecting to Gladia WebSocket")

            self._websocket = await websocket_connect(self._session_url)
            self._connection_active = True

            # Reset byte tracking for new connection
            async with self._buffer_lock:
                self._bytes_sent = 0

            await self._call_event_handler("on_connected")

            # Send buffered audio if any
            await self._send_buffered_audio()

            logger.debug(f"{self} Connected to Gladia WebSocket")
        except Exception as e:
            await self.push_error(error_msg=f"Unable to connect to Gladia: {e}", exception=e)
            raise

    async def _disconnect_websocket(self):
        """Close the websocket connection to Gladia."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                logger.debug(f"{self} Disconnecting from Gladia WebSocket")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

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
                        f"{self} Gladia error: {response.status}: {error_text or response.reason}"
                    )
                    raise Exception(
                        f"{self} Failed to initialize Gladia session: {response.status} - {error_text}"
                    )

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[str] = None
    ):
        await self.stop_processing_metrics()

    async def _on_speech_started(self):
        """Handle speech start event from Gladia.

        Broadcasts UserStartedSpeakingFrame and optionally triggers interruption
        when VAD is enabled.
        """
        if not self._params.enable_vad or self._is_speaking:
            return

        logger.debug(f"{self} User started speaking")
        self._is_speaking = True

        await self.broadcast_frame(UserStartedSpeakingFrame)
        if self._should_interrupt:
            await self.push_interruption_task_frame_and_wait()

    async def _on_speech_ended(self):
        """Handle speech end event from Gladia.

        Broadcasts UserStoppedSpeakingFrame when VAD is enabled.
        """
        if not self._params.enable_vad or not self._is_speaking:
            return
        self._is_speaking = False
        await self.broadcast_frame(UserStoppedSpeakingFrame)
        logger.debug(f"{self} User stopped speaking")

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

    def _get_websocket(self):
        """Get the current WebSocket connection.

        Returns:
            The WebSocket connection.

        Raises:
            Exception: If WebSocket is not connected.
        """
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        """Receive and process websocket messages.

        Continuously processes messages from the websocket connection.
        """
        async for message in self._get_websocket():
            try:
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
                elif content["type"] == "speech_start":
                    await self._on_speech_started()
                elif content["type"] == "speech_end":
                    await self._on_speech_ended()
            except json.JSONDecodeError:
                logger.warning(f"{self} Received non-JSON message: {message}")

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
                    logger.debug(f"{self} Websocket closed, stopping keepalive")
                    break
        except websockets.exceptions.ConnectionClosed:
            logger.debug(f"{self} Connection closed during keepalive")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
