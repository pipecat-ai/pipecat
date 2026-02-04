#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gradium's speech-to-text service implementation.

This module provides integration with Gradium's real-time speech-to-text
WebSocket API for streaming audio transcription.
"""

import base64
import json
from typing import AsyncGenerator, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use Gradium, you need to `pip install "pipecat-ai[gradium]"`.')
    raise Exception(f"Missing module: {e}")

SAMPLE_RATE = 24000


def language_to_gradium_language(language: Language) -> Optional[str]:
    """Convert a Language enum to Gradium's language code format.

    Args:
        language: The Language enum value to convert.

    Returns:
        The Gradium language code string or None if not supported.
    """
    LANGUAGE_MAP = {
        Language.DE: "de",
        Language.EN: "en",
        Language.ES: "es",
        Language.FR: "fr",
        Language.PT: "pt",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


class GradiumSTTService(WebsocketSTTService):
    """Gradium real-time speech-to-text service.

    Provides real-time speech transcription using Gradium's WebSocket API.
    Supports both interim and final transcriptions with configurable parameters
    for audio processing and connection management.
    """

    class InputParams(BaseModel):
        """Configuration parameters for Gradium STT API.

        Parameters:
            language: Expected language of the audio (e.g., "en", "es", "fr").
                This helps ground the model to a specific language and improve
                transcription quality.
            delay_in_frames: Delay in audio frames (80ms each) before text is
                generated. Higher delays allow more context but increase latency.
                Allowed values: 7, 8, 10, 12, 14, 16, 20, 24, 36, 48.
                Default is 10 (800ms). Lower values like 7-8 give faster response.
        """

        language: Optional[Language] = None
        delay_in_frames: Optional[int] = None

    def __init__(
        self,
        *,
        api_key: str,
        api_endpoint_base_url: str = "wss://eu.api.gradium.ai/api/speech/asr",
        params: Optional[InputParams] = None,
        json_config: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the Gradium STT service.

        Args:
            api_key: Gradium API key for authentication.
            api_endpoint_base_url: WebSocket endpoint URL. Defaults to Gradium's streaming endpoint.
            params: Configuration parameters for language and delay settings.
            json_config: Optional JSON configuration string for additional model settings.

                .. deprecated:: 0.0.101
                    Use `params` instead for type-safe configuration.

            **kwargs: Additional arguments passed to parent STTService class.
        """
        super().__init__(sample_rate=SAMPLE_RATE, **kwargs)

        if json_config is not None:
            import warnings

            warnings.warn(
                "Parameter 'json_config' is deprecated and will be removed in a future version, use 'params' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        self._api_key = api_key
        self._api_endpoint_base_url = api_endpoint_base_url
        self._websocket = None
        self._params = params or GradiumSTTService.InputParams()
        self._json_config = json_config

        self._receive_task = None

        self._audio_buffer = bytearray()
        self._chunk_size_ms = 80
        self._chunk_size_bytes = 0

        # Set from the ready message when connecting to the service.
        # These values are used for flushing transcription.
        self._delay_in_frames = 0
        self._frame_size = 0

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            True if metrics generation is supported.
        """
        return True

    async def set_language(self, language: Language):
        """Set the recognition language and reconnect.

        Args:
            language: The language to use for speech recognition.
        """
        logger.info(f"Switching STT language to: [{language}]")
        self._params.language = language
        await self._disconnect()
        await self._connect()

    async def start(self, frame: StartFrame):
        """Start the speech-to-text service.

        Args:
            frame: Start frame to begin processing.
        """
        await super().start(frame)
        self._chunk_size_bytes = int(self._chunk_size_ms * self.sample_rate * 2 / 1000)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the speech-to-text service.

        Args:
            frame: End frame to stop processing.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the speech-to-text service.

        Args:
            frame: Cancel frame to abort processing.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with VAD-specific handling.

        When VAD detects the user has stopped speaking, we flush the transcription
        by sending silence frames. This makes the system more reactive by getting
        the final transcription faster without closing the connection.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self.start_processing_metrics()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._flush_transcription()

    async def _flush_transcription(self):
        """Flush the transcription by sending silence frames.

        When VAD detects the user stopped speaking, we send delay_in_frames
        chunks of silence (zeros) to flush the remaining audio from the model's
        buffer. This allows for faster turn-around without closing the connection.

        From Gradium docs: "feed in delay_in_frames chunks of silence (vectors
        of zeros). If those are fed in faster than realtime, the API also has
        a possibility to process them faster."
        """
        if not self._websocket or self._websocket.state is not State.OPEN:
            return

        if self._delay_in_frames <= 0:
            logger.debug("No delay_in_frames set, skipping flush")
            return

        # Create a silence chunk (zeros) of frame_size samples
        # Each sample is 2 bytes (16-bit PCM)
        silence_bytes = bytes(self._frame_size * 2)
        silence_b64 = base64.b64encode(silence_bytes).decode("utf-8")

        logger.debug(f"Flushing Gradium STT with {self._delay_in_frames} silence frames")

        for _ in range(self._delay_in_frames):
            msg = {"type": "audio", "audio": silence_b64}
            try:
                await self._websocket.send(json.dumps(msg))
            except Exception as e:
                logger.warning(f"Failed to send silence frame: {e}")
                break

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process audio data for speech-to-text conversion.

        Args:
            audio: Raw audio bytes to process.

        Yields:
            None (processing handled via WebSocket messages).
        """
        self._audio_buffer.extend(audio)

        while len(self._audio_buffer) >= self._chunk_size_bytes:
            chunk = bytes(self._audio_buffer[: self._chunk_size_bytes])
            self._audio_buffer = self._audio_buffer[self._chunk_size_bytes :]
            chunk = base64.b64encode(chunk).decode("utf-8")
            msg = {"type": "audio", "audio": chunk}
            if self._websocket and self._websocket.state is State.OPEN:
                await self._websocket.send(json.dumps(msg))

        yield None

    @traced_stt
    async def _trace_transcription(self, transcript: str, is_final: bool, language: Language):
        """Record transcription event for tracing."""
        pass

    async def _connect(self):
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Gradium STT")

            ws_url = self._api_endpoint_base_url
            headers = {
                "x-api-key": self._api_key,
                "x-api-source": "pipecat",
            }
            self._websocket = await websocket_connect(
                ws_url,
                additional_headers=headers,
            )
            await self._call_event_handler("on_connected")
            setup_msg = {
                "type": "setup",
                "input_format": "pcm",
            }
            # Build json_config: start with deprecated json_config, then override with params
            json_config = {}
            if self._json_config:
                json_config = json.loads(self._json_config)
            if self._params.language:
                gradium_language = language_to_gradium_language(self._params.language)
                if gradium_language:
                    json_config["language"] = gradium_language
            if self._params.delay_in_frames:
                json_config["delay_in_frames"] = self._params.delay_in_frames
            if json_config:
                setup_msg["json_config"] = json_config
            await self._websocket.send(json.dumps(setup_msg))
            ready_msg = await self._websocket.recv()
            ready_msg = json.loads(ready_msg)
            if ready_msg["type"] == "error":
                raise Exception(f"received error {ready_msg['message']}")
            if ready_msg["type"] != "ready":
                raise Exception(f"unexpected first message type {ready_msg['type']}")

            # Store delay_in_frames and frame_size for silence flushing
            self._delay_in_frames = ready_msg.get("delay_in_frames", 0)
            self._frame_size = ready_msg.get("frame_size", 1920)
            logger.debug(
                f"Connected to Gradium STT (delay_in_frames={self._delay_in_frames}, "
                f"frame_size={self._frame_size})"
            )

        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            raise

    async def _disconnect(self):
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _disconnect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                logger.debug("Disconnecting from Gradium STT")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _process_messages(self):
        async for message in self._get_websocket():
            try:
                data = json.loads(message)
                await self._process_response(data)
            except json.JSONDecodeError:
                logger.warning(f"Received non-JSON message: {message}")

    async def _receive_messages(self):
        while True:
            await self._process_messages()
            logger.debug(f"{self} Gradium connection was disconnected (timeout?), reconnecting")
            await self._connect_websocket()

    async def _process_response(self, msg):
        type_ = msg.get("type", "")
        if type_ == "text":
            await self._handle_text(msg["text"])
        elif type_ == "end_of_stream":
            await self._handle_end_of_stream()
        elif type_ == "error":
            await self.push_error(error_msg=f"Error: {msg}")

    async def _handle_end_of_stream(self):
        """Handle termination message."""
        logger.debug("Received end_of_stream message from server")

    async def _handle_text(self, text: str):
        """Handle transcription results."""
        await self.push_frame(
            TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
            )
        )
        await self._trace_transcription(text, is_final=True, language=None)
        await self.stop_processing_metrics()
