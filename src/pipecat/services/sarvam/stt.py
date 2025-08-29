"""Sarvam AI Speech-to-Text service implementation.

This module provides a streaming Speech-to-Text service using Sarvam AI's WebSocket-based
API. It supports real-time transcription with Voice Activity Detection (VAD) and
can handle multiple audio formats for Indian language speech recognition.
"""

import asyncio
import base64
import json
from enum import StrEnum
from typing import Literal, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    StartFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    import httpx
    import websockets
    from sarvamai import AsyncSarvamAI
    from sarvamai.speech_to_text_translate_streaming.socket_client import (
        AsyncSpeechToTextTranslateStreamingSocketClient,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Sarvam, you need to `pip install sarvamai websockets httpx`.")
    raise Exception(f"Missing module: {e}")


class TranscriptionMetrics(BaseModel):
    """Metrics for transcription performance."""

    audio_duration: float
    processing_latency: float


class TranscriptionData(BaseModel):
    """Data structure for transcription results."""

    request_id: str
    transcript: str
    language_code: Optional[str]
    metrics: Optional[TranscriptionMetrics] = None
    is_final: Optional[bool] = None


class TranscriptionResponse(BaseModel):
    """Response structure for transcription data."""

    type: Literal["data"]
    data: TranscriptionData


class VADSignal(StrEnum):
    """Voice Activity Detection signal types."""

    START = "START_SPEECH"
    END = "END_SPEECH"


class EventData(BaseModel):
    """Data structure for VAD events."""

    signal_type: VADSignal
    occured_at: float


class EventResponse(BaseModel):
    """Response structure for VAD events."""

    type: Literal["events"]
    data: EventData


class SarvamSpeechToTextTranslateService(STTService):
    """Sarvam speech-to-text service.

    Provides real-time speech recognition using Sarvam's WebSocket API.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "saaras:v2.5",
        language_code: str = "hi-IN",
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        """Initialize the Sarvam STT service.

        Args:
            api_key: Sarvam API key for authentication.
            model: Sarvam model to use for transcription.
            language_code: Language code for transcription (e.g., "hi-IN", "kn-IN").
            sample_rate: Audio sample rate. Defaults to 16000 if not specified.
            **kwargs: Additional arguments passed to the parent STTService.
        """
        super().__init__(sample_rate=sample_rate or 16000, **kwargs)

        self.set_model_name(model)
        self._api_key = api_key
        self._model = model
        self._language_code = language_code

        self._client = AsyncSarvamAI(api_subscription_key=api_key)
        self._websocket = None
        self._websocket_connection = None
        self._listening_task = None
        self._is_connected = False

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Sarvam service supports metrics generation.
        """
        return True

    async def set_model(self, model: str):
        """Set the Sarvam model and reconnect.

        Args:
            model: The Sarvam model name to use.
        """
        await super().set_model(model)
        logger.info(f"Switching STT model to: [{model}]")
        self._model = model
        await self._disconnect()
        await self._connect()

    async def start(self, frame: StartFrame):
        """Start the Sarvam STT service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Sarvam STT service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Sarvam STT service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes):
        """Send audio data to Sarvam for transcription.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            Frame: None (transcription results come via WebSocket callbacks).
        """
        if not self._is_connected or not self._websocket_connection:
            logger.warning("WebSocket not connected, cannot process audio")
            yield None
            return

        try:
            # Convert audio bytes to base64 for Sarvam API
            audio_base64 = base64.b64encode(audio).decode("utf-8")

            message = {
                "audio": {
                    "data": audio_base64,
                    "encoding": "audio/wav",
                    "sample_rate": self.sample_rate,
                }
            }
            await self._websocket_connection.send(json.dumps(message))

        except websockets.exceptions.ConnectionClosed:
            logger.error("WebSocket connection closed")
            await self.push_error(ErrorFrame("WebSocket connection closed"))
        except Exception as e:
            logger.error(f"Error sending audio to Sarvam: {e}")
            await self.push_error(ErrorFrame(f"Failed to send audio: {e}"))

        yield None

    async def _connect(self):
        """Connect to Sarvam WebSocket API directly."""
        logger.debug("Connecting to Sarvam")

        try:
            # Build WebSocket URL and headers manually
            ws_url = (
                self._client._client_wrapper.get_environment().production
                + "/speech-to-text-translate/ws"
            )

            # Add query parameters
            query_params = httpx.QueryParams()
            query_params = query_params.add("model", self._model)
            query_params = query_params.add("vad_signals", True)

            ws_url = ws_url + f"?{query_params}"

            # Get headers
            headers = self._client._client_wrapper.get_headers()
            headers["Api-Subscription-Key"] = self._api_key

            # Connect to WebSocket directly
            self._websocket_connection = await websockets.connect(
                ws_url, additional_headers=headers
            )

            # Create the socket client wrapper
            self._websocket = AsyncSpeechToTextTranslateStreamingSocketClient(
                websocket=self._websocket_connection
            )

            # Start listening for messages
            self._listening_task = asyncio.create_task(self._listen_for_messages())
            self._is_connected = True

            logger.info("Connected to Sarvam successfully")

        except Exception as e:
            logger.error(f"Failed to connect to Sarvam: {e}")
            self._websocket = None
            self._websocket_connection = None
            self._is_connected = False
            await self.push_error(ErrorFrame(f"Failed to connect to Sarvam: {e}"))

    async def _disconnect(self):
        """Disconnect from Sarvam WebSocket API."""
        self._is_connected = False

        if self._listening_task:
            self._listening_task.cancel()
            try:
                await self._listening_task
            except asyncio.CancelledError:
                pass
            self._listening_task = None

        if self._websocket_connection:
            try:
                await self._websocket_connection.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket connection: {e}")
            finally:
                logger.debug("Disconnected from Sarvam WebSocket")
                self._websocket_connection = None
                self._websocket = None

    async def _listen_for_messages(self):
        """Listen for messages from Sarvam WebSocket."""
        try:
            while self._websocket and self._is_connected:
                try:
                    message = await self._websocket_connection.recv()
                    response = json.loads(message)
                    await self._handle_response(response)

                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket connection closed")
                    break
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error receiving message from Sarvam: {e}")
                    break

            # If we get here, connection was lost
            if self._is_connected:
                logger.warning("Connection lost")
                self._is_connected = False

        except asyncio.CancelledError:
            logger.debug("Message listening cancelled")
        except Exception as e:
            logger.error(f"Unexpected error in message listener: {e}")
            await self.push_error(ErrorFrame(f"Message listener error: {e}"))

    async def _handle_response(self, response):
        """Handle transcription response from Sarvam.

        Args:
            response: The response object from Sarvam WebSocket.
        """
        logger.debug(f"Received response: {response}")

        try:
            if response["type"] == "events":
                parsed = EventResponse(**response)
                signal = parsed.data.signal_type
                timestamp = parsed.data.occured_at
                logger.debug(f"VAD Signal: {signal}, Occurred at: {timestamp}")

                if signal == VADSignal.START:
                    await self.start_metrics()
                    logger.debug("User started speaking")
                    await self.push_frame(UserStartedSpeakingFrame())
                    await self.push_frame(VADUserStartedSpeakingFrame())
                    await self.push_frame(StartInterruptionFrame())

            elif response["type"] == "data":
                await self.stop_ttfb_metrics()
                parsed = TranscriptionResponse(**response)
                transcript = parsed.data.transcript
                language_code = parsed.data.language_code
                if language_code is None:
                    language_code = "hi-IN"
                language = self._map_language_code_to_enum(language_code)
                await self.push_frame(UserStoppedSpeakingFrame())
                await self.push_frame(VADUserStoppedSpeakingFrame())
                await self.push_frame(StopInterruptionFrame())

                if transcript and transcript.strip():
                    await self.push_frame(
                        TranscriptionFrame(
                            transcript,
                            self._user_id,
                            time_now_iso8601(),
                            language,
                            result=response,
                        )
                    )

                # await self._handle_transcription(transcript, is_final, language)
                await self.stop_processing_metrics()

        except Exception as e:
            logger.error(f"Error handling Sarvam response: {e}")
            await self.push_error(ErrorFrame(f"Failed to handle response: {e}"))

    def _map_language_code_to_enum(self, language_code: str) -> Language:
        """Map Sarvam language code (e.g., "hi-IN") to pipecat Language enum."""
        logger.debug(f"Audio language detected as: {language_code}")
        mapping = {
            "bn-IN": Language.BN_IN,
            "gu-IN": Language.GU_IN,
            "hi-IN": Language.HI_IN,
            "kn-IN": Language.KN_IN,
            "ml-IN": Language.ML_IN,
            "mr-IN": Language.MR_IN,
            "ta-IN": Language.TA_IN,
            "te-IN": Language.TE_IN,
            "pa-IN": Language.PA_IN,
            "od-IN": Language.OR_IN,
            "en-US": Language.EN_US,
            "en-IN": Language.EN_IN,
            "as-IN": Language.AS_IN,
        }
        return mapping.get(language_code, Language.HI_IN)

    async def start_metrics(self):
        """Start TTFB and processing metrics collection."""
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass
