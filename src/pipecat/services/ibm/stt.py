#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""IBM Speech Services speech-to-text service implementation.

This module provides integration with IBM's Speech-to-Text API for transcription
using segmented audio processing. The service uploads audio files and receives
transcription results directly.
"""

import base64
import io
import json
import time
from enum import Enum
from typing import AsyncGenerator, List, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel

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
from pipecat.services.stt_latency import IBM_TTFS_P99
from pipecat.services.stt_service import SegmentedSTTService, WebsocketSTTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use IBM Realtime STT, you need to `pip install pipecat-ai[ibm]`."
    )
    raise Exception(f"Missing module: {e}")


def language_to_ibm_model(language: Language) -> Optional[str]:
    """Convert Language enum to IBM STT model name.

    IBM uses model names like 'en-US_BroadbandModel', 'es-ES_BroadbandModel'

    Source: https://cloud.ibm.com/docs/speech-to-text?topic=speech-to-text-models
    """
    LANGUAGE_MODEL_MAP = {
        Language.EN: "en-US",
        Language.EN_US: "en-US",
        Language.EN_GB: "en-GB",
        Language.EN_AU: "en-AU",
        Language.ES: "es-ES",
        Language.ES_ES: "es-ES",
        Language.ES_LA: "es-LA_Telephony_LSM_Alpha",
        Language.FR: "fr-FR",
        Language.FR_CA: "fr-CA",
        Language.DE: "de-DE",
        Language.IT: "it-IT_Telephony",
        Language.JA: "ja-JP",
        Language.KO: "ko-KR_Telephony",
        Language.PT: "pt-BR_Telephony_LSM",
        Language.ZH: "zh-CN_Telephony",
        Language.AR: "ar-MS_Telephony",
        Language.NL: "nl-NL_Telephony",
        Language.SV: "sv-SE_Telephony",
        # Add more as needed
    }
    return LANGUAGE_MODEL_MAP.get(language)


class IBMSTTService(WebsocketSTTService):
    """IBM Speech-to-Text WebSocket service.

    Uses IBM's WebSocket API for real-time transcription.
    Supports interim results, timestamps, and word confidence scores.
    """

    class InputParams(BaseModel):
        """IBM STT configuration parameters.

        Parameters:
            language: Target language (converted to IBM model)
            model: IBM model name (overrides language if provided)
            interim_results: Enable interim transcription results
            smart_formatting: Enable smart formatting (dates, times, etc.)
            profanity_filter: Enable profanity filtering
            timestamps: Include word-level timestamps
            word_confidence: Include word confidence scores
            speaker_labels: Enable speaker diarization
            end_of_phrase_silence_time: Silence duration to end phrase (0.0-2.0s)
            split_transcript_at_phrase_end: Split at phrase boundaries
        """
        language: Optional[Language] = Language.EN_US
        model: Optional[str] = None
        interim_results: bool = True
        smart_formatting: bool = False
        profanity_filter: bool = False
        timestamps: bool = True
        word_confidence: bool = False
        speaker_labels: bool = False
        end_of_phrase_silence_time: float = 0.8
        split_transcript_at_phrase_end: bool = True

        max_alternatives: int = 0
        inactivity_timeout: int = 30
        smart_formatting_version: int = 2
        punctuation: bool = False
        hints: List[str] = []
        enrichments: List[str] = []
        redaction: bool = False
        customization_weight: float = 1.0
        grammar_name: str = None
        grammar_semantics: bool = False
        speech_detector_sensitivity: float = 0.0
        speech_event_sensitivity: float = 0.0
        background_audio_suppression: float = 0.0
        speech_length_soft_limit: int = 0
        speech_begin_event: bool = False
        audio_metrics: bool = False
        processing_metrics: bool = False
        processing_metrics_interval: float = 0.0
        scd: bool = False
        _lambda: float = None
        character_insertion_bias: float = 0.0
        acoustic_weight: float = 0.0
        max_active: int = 0
        low_latency: bool = False
        speedup: float = 0.0
        include_transparent_words: bool = False
        skip_zero_len_words: bool = False
        lattice_generation: bool = False
        max_buffer: float = 0.0
        min_buffer: float = 0.0
        lid_confidence: float = 0.99

    def __init__(
            self,
            *,
            api_key: str,
            url: str,  # e.g., "https://api.us-south.speech-to-text.watson.cloud.ibm.com"
            model: Optional[str] = None,
            sample_rate: Optional[int] = None,
            params: Optional[InputParams] = None,
            ttfs_p99_latency: Optional[float] = None,
            **kwargs,
    ):
        """Initialize the IBM STT service.

        Args:
            api_key: IBM Cloud API key for authentication.
            url: IBM STT service URL (e.g., "https://api.us-south.speech-to-text.watson.cloud.ibm.com").
            model: IBM model name (overrides language-derived model if provided).
            sample_rate: Audio sample rate in Hz.
            params: Optional :class:`InputParams` for STT configuration.
            ttfs_p99_latency: Expected P99 latency for time-to-first-speech metrics.
            **kwargs: Additional keyword arguments forwarded to :class:`~pipecat.services.stt_service.WebsocketSTTService`.
        """
        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency or 0.5,  # IBM is typically fast
            **kwargs,
        )

        self._api_key = api_key
        self._url = url
        self._params = params or IBMSTTService.InputParams()
        self._model = model or language_to_ibm_model(self._params.language)
        self._access_token = None
        self._token_expiry = 0
        self._receive_task = None
        self._websocket = None

        # Latency tracking
        self._connect_start_time = None
        self._first_byte_time = None
        self._first_result_time = None

        # Ensure sample rate is set immediately for WebSocket connection
        if sample_rate and self._sample_rate == 0:
            self._sample_rate = sample_rate

    async def _get_access_token(self) -> str:
        """Get IAM access token using API key.

        IBM requires IAM authentication. Token expires after 1 hour.
        """
        import time

        # Return cached token if still valid
        if self._access_token and time.time() < self._token_expiry:
            return self._access_token

        # Request new token from IBM IAM
        url = "https://iam.cloud.ibm.com/identity/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": self._api_key
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Failed to get IAM token: {error_text}")

                result = await response.json()
                self._access_token = result["access_token"]
                # Token expires in 1 hour (3600s), refresh 5 min (300s) early
                self._token_expiry = time.time() + result.get("expires_in", 3600) - 300

                logger.debug("Obtained new IAM access token")
                return self._access_token

    async def _connect_websocket(self):
        """Connect to IBM STT WebSocket endpoint."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            # Start timing
            self._connect_start_time = time.time()
            logger.debug("Connecting to IBM STT WebSocket")

            # Get access token
            token = await self._get_access_token()

            # Build WebSocket URL
            # Remove https:// and add /v1/recognize
            ws_host = self._url.replace("https://", "").replace("http://", "")
            ws_url = f"wss://{ws_host}/v1/recognize"

            # Add query parameters
            params = [
                f"model={self._model}",
                f"access_token={token}"
            ]

            ws_url = f"{ws_url}?{'&'.join(params)}"

            # Connect
            self._websocket = await websocket_connect(ws_url)
            
            # Log connection latency
            connect_latency = (time.time() - self._connect_start_time) * 1000
            logger.info(f"IBM STT WebSocket connected in {connect_latency:.2f}ms")

            # Send start message
            await self._send_start_message()

            # Reset timing for first byte
            self._first_byte_time = None
            self._first_result_time = None

            await self._call_event_handler("on_connected")
            logger.debug("Connected to IBM STT")

        except Exception as e:
            await self.push_error(
                error_msg=f"Unable to connect to IBM STT: {e}",
                exception=e
            )

    async def _connect(self):
        """Establish WebSocket connection to IBM STT."""
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(
                self._receive_task_handler(self._report_error)
            )

    async def _disconnect(self):
        """Close WebSocket connection and cleanup tasks."""
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

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

    async def start(self, frame: StartFrame):
        """Start the STT service and establish WebSocket connection.

        Args:
            frame: Frame indicating service should start.
        """
        await super().start(frame)
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

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate processing metrics.

        Returns:
            True, as IBM STT service supports metrics generation.
        """
        return True

    async def _disconnect_websocket(self):
        """Disconnect from IBM STT WebSocket."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                logger.debug("Disconnecting from IBM STT")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def set_language(self, language: Language):
        """Set the transcription language.

        Args:
            language: The language to use for speech-to-text transcription.

        Note:
            Changing language requires reconnecting to the WebSocket.
        """
        logger.info(f"Switching STT language to: [{language}]")
        new_model = language_to_ibm_model(language)
        if new_model:
            self._model = new_model
            self._params.language = language
            # Reconnect with new settings
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
        logger.info(f"Switching STT model to: [{model}]")
        self._model = model
        # Reconnect with new settings
        await self._disconnect()
        await self._connect()

    @traced_stt
    async def _handle_transcription(
            self, transcript: str, is_final: bool, language: Optional[str] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def _start_metrics(self):
        """Start performance metrics collection for transcription processing."""
        await self.start_processing_metrics()

    async def _send_start_message(self):
        """Send start message to IBM STT."""
        # Determine content type based on sample rate
        content_type = f"audio/l16;rate={self.sample_rate}"

        start_message = {
            "action": "start",
            "content-type": content_type,
            "model": self._model,
            "interim_results": self._params.interim_results,
            "smart_formatting": self._params.smart_formatting,
            "profanity_filter": self._params.profanity_filter,
            "timestamps": self._params.timestamps,
            "word_confidence": self._params.word_confidence,
            "speaker_labels": self._params.speaker_labels,
            "end_of_phrase_silence_time": self._params.end_of_phrase_silence_time,
            "split_transcript_at_phrase_end": self._params.split_transcript_at_phrase_end,
        }

        await self._websocket.send(json.dumps(start_message))
        logger.debug(f"Sent start message: {start_message}")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Stream audio to IBM STT.

        Args:
            audio: Raw PCM audio bytes (16-bit, mono)

        Yields:
            None - results come via WebSocket responses
        """
        # Reconnect if needed
        if not self._websocket or self._websocket.state is State.CLOSED:
            await self._connect()

        if self._websocket and self._websocket.state is State.OPEN:
            try:
                # IBM expects raw PCM audio, not base64
                await self._websocket.send(audio)
            except Exception as e:
                yield ErrorFrame(f"IBM STT error: {str(e)}")

        yield None

    async def _receive_messages(self):
        """Receive and process messages from IBM STT."""
        async for message in self._get_websocket():
            try:
                # Log first byte received
                if self._first_byte_time is None and self._connect_start_time:
                    self._first_byte_time = time.time()
                    first_byte_latency = (self._first_byte_time - self._connect_start_time) * 1000
                    logger.info(f"IBM STT first byte received in {first_byte_latency:.2f}ms")
                
                data = json.loads(message)
                await self._process_response(data)
            except json.JSONDecodeError:
                logger.warning(f"Received non-JSON message: {message}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")

    async def _process_response(self, data: dict):
        """Process IBM STT response message.

        IBM response format:
        {
            "results": [{
                "alternatives": [{
                    "transcript": "text",
                    "confidence": 0.95,
                    "timestamps": [["word", start, end], ...]
                }],
                "final": true/false
            }],
            "result_index": 0
        }
        """
        # Handle state messages
        if "state" in data:
            state = data["state"]
            if state == "listening":
                logger.debug("IBM STT is listening")
            return

        # Handle error messages
        if "error" in data:
            error_msg = data["error"]
            await self.push_error(error_msg=f"IBM STT error: {error_msg}")
            return

        # Handle transcription results
        if "results" in data:
            results = data["results"]
            if not results:
                return

            for result in results:
                if "alternatives" not in result:
                    continue

                alternatives = result["alternatives"]
                if not alternatives:
                    continue

                # Get best alternative (first one)
                alternative = alternatives[0]
                text = alternative.get("transcript", "").strip()

                if not text:
                    continue

                # Log first transcription result
                if self._first_result_time is None and self._connect_start_time:
                    self._first_result_time = time.time()
                    first_result_latency = (self._first_result_time - self._connect_start_time) * 1000
                    logger.info(f"IBM STT first transcription result in {first_result_latency:.2f}ms")

                is_final = result.get("final", False)
                confidence = alternative.get("confidence")

                # Extract timestamps if available
                timestamps = alternative.get("timestamps", [])

                if is_final:
                    await self.stop_processing_metrics()
                    await self._handle_transcription(text, True)

                    logger.debug(f"Final transcript: [{text}]")

                    await self.push_frame(
                        TranscriptionFrame(
                            text,
                            self._user_id,
                            time_now_iso8601(),
                            self._model,
                            result=data,
                        )
                    )
                else:
                    # Interim result
                    logger.trace(f"Interim transcript: [{text}]")

                    await self.push_frame(
                        InterimTranscriptionFrame(
                            text,
                            self._user_id,
                            time_now_iso8601(),
                            self._model,
                            result=data,
                        )
                    )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and handle VAD events."""
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self._start_metrics()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # Send stop message to finalize transcription
            if self._websocket and self._websocket.state is State.OPEN:
                try:
                    stop_message = {"action": "stop"}
                    await self._websocket.send(json.dumps(stop_message))
                    logger.trace("Sent stop message to IBM STT")
                except Exception as e:
                    logger.warning(f"Failed to send stop: {e}")