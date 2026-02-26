#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Smallest STT service implementation with WebSocket-based real-time transcription."""

import json
from enum import Enum
from typing import Any, AsyncGenerator, List, Optional, Union
from urllib.parse import urlencode

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from websockets import ClientConnection
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Smallest, you need to `pip install websockets`.")
    raise Exception(f"Missing module: {e}")


# =============================================================================
# Enums and Models
# =============================================================================


class AudioChannel(int, Enum):
    """Audio channel configuration."""

    MONO = 1
    STEREO = 2


class AudioEncoding(str, Enum):
    """Supported audio encoding formats."""

    LINEAR16 = "linear16"
    FLAC = "flac"
    MULAW = "mulaw"
    OPUS = "opus"


class SensitiveData(str, Enum):
    """Types of sensitive data that can be redacted."""

    PCI = "pci"
    SSN = "ssn"
    NUMBERS = "numbers"


class EventType(str, Enum):
    """WebSocket event types from Smallest API."""

    TRANSCRIPTION = "transcription"
    ERROR = "error"


class TranscriptionResponse(BaseModel):
    """Response model for transcription events."""

    type: EventType = EventType.TRANSCRIPTION
    text: str
    isEndOfTurn: bool
    isFinal: bool


class ErrorResponse(BaseModel):
    """Response model for error events."""

    type: EventType = EventType.ERROR
    message: str
    error: Union[List[str], str]


# =============================================================================
# Language Mapping
# =============================================================================


def language_to_smallest_language(language: Language) -> Optional[str]:
    """Convert a Language enum to Smallest's language code format.

    Args:
        language: The pipecat Language enum value.

    Returns:
        The Smallest language code string, or None if not supported.
    """
    LANGUAGE_MAP = {
        # Primary supported languages
        Language.EN: "en",
        Language.HI: "hi",
        # Extended language support (based on common ASR languages)
        Language.EN_US: "en",
        Language.EN_GB: "en",
        Language.EN_AU: "en",
        Language.EN_IN: "en",
        Language.HI_IN: "hi",
        # Additional languages if supported by Smallest
        Language.BN: "bn",
        Language.TA: "ta",
        Language.TE: "te",
        Language.MR: "mr",
        Language.GU: "gu",
        Language.KN: "kn",
        Language.ML: "ml",
        Language.PA: "pa",
        Language.OR: "or",
    }

    result = LANGUAGE_MAP.get(language)

    if not result:
        # Try extracting base language code
        lang_str = str(language.value)
        base_code = lang_str.split("-")[0].lower()
        supported_bases = set(LANGUAGE_MAP.values())
        result = base_code if base_code in supported_bases else None

    return result


# =============================================================================
# SmallestSTTService
# =============================================================================


class SmallestSTTService(WebsocketSTTService):
    """Speech-to-Text service using Smallest's WebSocket API.

    This service connects to Smallest's real-time ASR API for streaming
    transcription with support for multiple Indian languages and various
    audio processing options.

    Example:
        ```python
        stt = SmallestSTTService(
            api_key="your-api-key",
            params=SmallestSTTService.InputParams(
                language=Language.HI,
                add_punctuation=True,
            ),
        )
        ```

    Attributes:
        _websocket: The active WebSocket connection.
        _receive_task: Background task for receiving messages.
        _settings: Current configuration settings.
    """

    class InputParams(BaseModel):
        """Configuration parameters for the Smallest STT service.

        Attributes:
            encoding: The audio encoding format. Defaults to LINEAR16.
            sample_rate: Audio sample rate in Hz (8000-48000).
            language: The language for transcription.
            channels: Number of audio channels (mono/stereo).
            add_punctuation: Whether to add punctuation to transcripts.
            speech_end_threshold: Duration (ms) to detect end of speech (10-60000).
            emit_voice_activity: Whether to emit VAD events.
            redact_sensitive_data: Types of sensitive data to redact.
            speech_endpointing: Controls speech endpointing behavior.
        """

        encoding: Optional[AudioEncoding] = Field(
            default=AudioEncoding.LINEAR16,
            description="The encoding format of the audio input.",
        )
        sample_rate: Optional[int] = Field(
            default=16000,
            ge=8000,
            le=48000,
            description="The sample rate of the audio in Hz.",
        )
        language: Optional[Language] = Field(
            default=Language.EN,
            description="The language of the audio input.",
        )
        channels: Optional[AudioChannel] = Field(
            default=AudioChannel.MONO,
            description="The number of audio channels.",
        )
        add_punctuation: Optional[bool] = Field(
            default=None,
            description="Whether to add punctuation to the transcript.",
        )
        speech_end_threshold: Optional[int] = Field(
            default=None,
            ge=10,
            le=60000,
            description="Duration in ms to determine end of speech segment.",
        )
        emit_voice_activity: Optional[bool] = Field(
            default=None,
            description="Whether to emit voice activity detection events.",
        )
        redact_sensitive_data: Optional[List[SensitiveData]] = Field(
            default=None,
            description="Types of sensitive data to redact from transcript.",
        )
        speech_endpointing: Optional[int] = Field(
            default=None,
            description="Controls speech endpointing behavior.",
        )

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "wss://waves-api.smallest.ai/api/v1/asr",
        model: Optional[str] = None,
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        ping_interval: Optional[float] = 20,
        ping_timeout: Optional[float] = 20,
        **kwargs,
    ):
        """Initialize the Smallest STT service.

        Args:
            api_key: Smallest API key for authentication.
            url: WebSocket endpoint URL.
            model: Model name to use (if applicable).
            sample_rate: Audio sample rate. Overrides params.sample_rate if set.
            params: Configuration parameters for the service.
            ping_interval: WebSocket ping interval in seconds.
            ping_timeout: WebSocket ping timeout in seconds.
            **kwargs: Additional arguments passed to WebsocketSTTService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._url = url
        self._params = params or self.InputParams()
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout

        # Initialize settings from params
        self._settings = {
            "audioEncoding": self._params.encoding,
            "audioSampleRate": sample_rate or self._params.sample_rate,
            "audioLanguage": self._params.language,
            "audioChannels": self._params.channels,
            "addPunctuation": self._params.add_punctuation,
            "speechEndThreshold": self._params.speech_end_threshold,
            "emitVoiceActivity": self._params.emit_voice_activity,
            "redactSensitiveData": self._params.redact_sensitive_data,
            "speechEndpointing": self._params.speech_endpointing,
        }

        if model:
            self.set_model_name(model)

        # WebSocket state
        self._websocket: ClientConnection | None = None
        self._receive_task = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def language(self) -> Language:
        """Get the current transcription language."""
        return self._settings.get("audioLanguage", Language.EN)

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Smallest service supports metrics generation.
        """
        return True

    async def set_language(self, language: Language):
        """Set the transcription language.

        Args:
            language: The language to use for speech recognition.

        Note:
            Changing language requires reconnecting to the WebSocket.
        """
        logger.info(f"{self}: Switching STT language to: [{language}]")
        self._settings["audioLanguage"] = language
        await self._disconnect()
        await self._connect()

    async def set_model(self, model: str):
        """Set the STT model.

        Args:
            model: The model name to use for transcription.

        Note:
            Changing model requires reconnecting to the WebSocket.
        """
        await super().set_model(model)
        logger.info(f"{self}: Switching STT model to: [{model}]")
        self._settings["model"] = model
        await self._disconnect()
        await self._connect()

    # -------------------------------------------------------------------------
    # Lifecycle Methods
    # -------------------------------------------------------------------------

    async def start(self, frame: StartFrame):
        """Start the STT service and establish WebSocket connection.

        Args:
            frame: Frame indicating service should start.
        """
        await super().start(frame)
        self._settings["audioSampleRate"] = self.sample_rate
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the STT service and close WebSocket connection.

        Args:
            frame: Frame indicating service should stop.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the STT service and close WebSocket connection.

        Args:
            frame: Frame indicating service should be cancelled.
        """
        await super().cancel(frame)
        await self._disconnect()

    # -------------------------------------------------------------------------
    # Frame Processing
    # -------------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle speech events.

        Args:
            frame: The frame to process.
            direction: Direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            # Start metrics when user starts speaking
            await self._start_metrics()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # Send finalize when user stops speaking
            await self._finalize_audio()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process audio data for speech-to-text transcription.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            Frame: ErrorFrame on failure, None otherwise.
                   Transcription results are handled via WebSocket responses.
        """
        # Reconnect if connection is closed
        if not self._websocket or self._websocket.state is State.CLOSED:
            await self._connect()

        if self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._websocket.send(audio)
            except Exception as e:
                logger.error(f"{self}: Error sending audio: {e}")
                yield ErrorFrame(error=f"{self}: Error sending audio: {e}")
                await self._disconnect()

        yield None

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    async def _start_metrics(self):
        """Start performance metrics collection for transcription processing."""
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()

    # -------------------------------------------------------------------------
    # WebSocket Connection
    # -------------------------------------------------------------------------

    def _build_websocket_url(self) -> str:
        """Build the WebSocket URL with query parameters.

        Returns:
            The complete WebSocket URL with encoded parameters.
        """

        def convert_value(v):
            """Convert value to URL-safe format."""
            if isinstance(v, Enum):
                return v.value
            if isinstance(v, list):
                return ",".join(str(item.value if isinstance(item, Enum) else item) for item in v)
            return v

        # Filter out None values and convert enums
        params = {k: convert_value(v) for k, v in self._settings.items() if v is not None}

        query_string = urlencode(params)
        return f"{self._url}?{query_string}"

    async def _connect(self):
        """Establish WebSocket connection to Smallest STT."""
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Close WebSocket connection and cleanup tasks."""
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Connect to Smallest STT WebSocket endpoint."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                logger.debug(f"{self}: Reusing existing connection")
                return

            logger.debug(f"{self}: Connecting to Smallest STT")

            headers = {"Authorization": f"Bearer {self._api_key}"}

            self._websocket = await websocket_connect(
                self._build_websocket_url(),
                additional_headers=headers,
                ping_interval=self._ping_interval,
                ping_timeout=self._ping_timeout,
            )

            await self._call_event_handler("on_connected")
            logger.info(f"{self}: Successfully connected to Smallest STT")

        except Exception as e:
            logger.error(f"{self}: Failed to connect: {e}")
            self._websocket = None
            await self.push_error(error_msg=f"Connection error: {e}", exception=e)
            await self._call_event_handler("on_connection_error", str(e))

    async def _disconnect_websocket(self):
        """Disconnect from Smallest STT WebSocket."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                logger.debug(f"{self}: Disconnecting from Smallest STT")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self}: Error closing WebSocket: {e}")
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Get the current WebSocket connection.

        Returns:
            The WebSocket connection.

        Raises:
            Exception: If WebSocket is not connected.
        """
        if self._websocket:
            return self._websocket
        raise Exception("WebSocket not connected")

    # -------------------------------------------------------------------------
    # Message Handling
    # -------------------------------------------------------------------------

    async def _receive_messages(self):
        """Continuously receive and process WebSocket messages."""
        async for message in self._get_websocket():
            try:
                data = json.loads(message)
                await self._process_response(data)
            except json.JSONDecodeError as e:
                logger.warning(f"{self}: Failed to parse JSON message: {e}")
            except ValidationError as e:
                logger.warning(f"{self}: Failed to validate message: {e}")
            except Exception as e:
                logger.error(f"{self}: Error processing message: {e}")
                await self.push_error(
                    error_msg=f"Error processing message: {e}",
                    exception=e,
                )

    async def _process_response(self, data: dict[str, Any]):
        """Process a response message from Smallest.

        Args:
            data: Parsed JSON response data.
        """
        event_type = data.get("type")

        if event_type == EventType.TRANSCRIPTION.value:
            await self._on_transcription(data)
        elif event_type == EventType.ERROR.value:
            await self._on_error(data)
        else:
            logger.debug(f"{self}: Unknown event type: {event_type}")

    async def _on_transcription(self, data: dict):
        """Handle transcription response.

        Args:
            data: Transcription event data.
        """
        try:
            response = TranscriptionResponse.model_validate(data)
        except ValidationError as e:
            logger.warning(f"{self}: Invalid transcription response: {e}")
            return

        text = response.text.strip()
        if not text:
            return

        # Stop TTFB metrics on first transcription
        await self.stop_ttfb_metrics()

        if response.isFinal:
            # Handle finalize confirmation if this is from our finalize request
            if response.isEndOfTurn:
                self.confirm_finalize()

            await self.push_frame(
                TranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    self.language,
                )
            )
            await self._handle_transcription(text, True, self.language)
            await self.stop_processing_metrics()
        else:
            # Interim transcription
            await self.push_frame(
                InterimTranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    self.language,
                )
            )

    async def _on_error(self, data: dict):
        """Handle error response.

        Args:
            data: Error event data.
        """
        try:
            response = ErrorResponse.model_validate(data)
            error_msg = f"{response.message}: {response.error}"
        except ValidationError:
            error_msg = str(data)

        logger.error(f"{self}: Error from server: {error_msg}")
        await self.push_error(error_msg=error_msg)
        await self.stop_all_metrics()

        # Attempt reconnection
        try:
            await self._connect()
        except Exception as e:
            logger.error(f"{self}: Failed to reconnect after error: {e}")

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing.

        This method is decorated with @traced_stt for observability.

        Args:
            transcript: The transcribed text.
            is_final: Whether this is a final transcription.
            language: The detected/configured language.
        """
        pass

    # -------------------------------------------------------------------------
    # Audio Control
    # -------------------------------------------------------------------------

    async def _finalize_audio(self):
        """Signal end of audio segment to get final transcription."""
        if not self._websocket or self._websocket.state is not State.OPEN:
            return

        try:
            # Mark that we're expecting a finalized response
            self.request_finalize()

            finalize_msg = json.dumps({"type": "Finalize"})
            await self._websocket.send(finalize_msg)
            logger.debug(f"{self}: Sent finalize message")
        except Exception as e:
            logger.error(f"{self}: Error sending finalize message: {e}")
            await self.push_error(
                error_msg=f"Error sending finalize: {e}",
                exception=e,
            )
