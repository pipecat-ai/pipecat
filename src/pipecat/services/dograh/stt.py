#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Dograh STT Service implementation using WebSocket streaming."""

import asyncio
import json
import time
from typing import AsyncGenerator, Awaitable, Callable, Dict, Optional

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    MetricsFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import STTUsageMetricsData
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.services.websocket_service import WebsocketService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    import websockets
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Dograh STT, you need to `pip install websockets`.")
    raise Exception(f"Missing module: {e}")


class DograhSTTService(STTService, WebsocketService):
    """Dograh speech-to-text service using WebSocket streaming.

    This service provides real-time speech recognition using Dograh's unified WebSocket API.
    Supports streaming transcription, interim results, and VAD events.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "wss://services.dograh.com",
        ws_path: str = "/api/v1/stt/stream",
        model: str = "default",
        language: Language = Language.EN,
        sample_rate: Optional[int] = None,
        interim_results: bool = True,
        vad_events: bool = False,
        **kwargs,
    ):
        """Initialize Dograh STT service.

        Args:
            api_key: The Dograh API key for authentication.
            base_url: WebSocket base URL for Dograh API. Defaults to "wss://services.dograh.com".
            ws_path: WebSocket path for STT streaming. Defaults to "/api/v1/stt/stream".
            model: STT model to use. Options include "default", "fast", "accurate".
                   The actual model used is determined by Dograh backend configuration.
            language: Language for speech recognition. Defaults to English.
            sample_rate: Audio sample rate in Hz. Defaults to None.
            interim_results: Whether to receive interim transcription results.
            vad_events: Whether to receive voice activity detection events.
            **kwargs: Additional arguments passed to the parent services.
        """
        STTService.__init__(self, sample_rate=sample_rate, **kwargs)
        WebsocketService.__init__(self, reconnect_on_error=True, **kwargs)

        self._api_key = api_key
        self._base_url = base_url
        self._ws_path = ws_path
        self._model = model
        self._language = language
        self._interim_results = interim_results
        self._vad_events = vad_events

        self.set_model_name(model)

        self._receive_task = None
        self._keepalive_task = None

        # Session tracking for metrics
        self._session_start_time: Optional[float] = None
        self._start_metadata = None

        # Register event handlers if VAD is enabled
        if self._vad_events:
            self._register_event_handler("on_speech_started")
            self._register_event_handler("on_speech_ended")

    @property
    def vad_enabled(self):
        """Check if VAD events are enabled.

        Returns:
            True if VAD events are enabled.
        """
        return self._vad_events

    async def set_model(self, model: str):
        """Set the speech recognition model.

        Args:
            model: The model identifier to use.
        """
        self._model = model
        self.set_model_name(model)

    async def set_language(self, language: Language):
        """Set the language for speech recognition.

        Args:
            language: The language to use for recognition.
        """
        self._language = language

    async def _connect_websocket(self):
        """Connect to the WebSocket endpoint."""
        try:
            url = f"{self._base_url}{self._ws_path}"
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }

            logger.debug(f"Connecting to Dograh STT WebSocket at {url}")
            self._websocket = await websocket_connect(url, additional_headers=headers)

            # Send initial configuration
            config_msg = {
                "type": "config",
                "model": self._model,
                "language": self._language.value,
                "sample_rate": self.sample_rate,
                "interim_results": self._interim_results,
                "vad_events": self._vad_events,
            }

            # Add workflow_run_id if available from StartFrame metadata
            if self._start_metadata and "workflow_run_id" in self._start_metadata:
                config_msg["correlation_id"] = self._start_metadata["workflow_run_id"]

            await self._websocket.send(json.dumps(config_msg))

            logger.info("Connected to Dograh STT service")

        except Exception as e:
            logger.error(f"Failed to connect to Dograh STT service: {e}")
            raise

    async def _disconnect_websocket(self):
        """Disconnect from the WebSocket endpoint."""
        try:
            if self._websocket:
                # Send end of stream signal
                end_msg = {"type": "end_of_stream"}
                await self._websocket.send(json.dumps(end_msg))

                await self._websocket.close()
                self._websocket = None

            logger.info("Disconnected from Dograh STT service")

        except Exception as e:
            logger.error(f"Error disconnecting from Dograh STT service: {e}")

    async def _reconnect_websocket(self, retry_count: int) -> bool:
        """Reconnect to the WebSocket.

        Args:
            retry_count: Current retry attempt number.

        Returns:
            True if reconnection successful, False otherwise.
        """
        logger.debug(f"Reconnecting to Dograh STT (attempt {retry_count})")
        await self._disconnect_websocket()
        try:
            await self._connect_websocket()
            return True
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            return False

    async def _connect(self):
        """Connect to the service."""
        await self._connect_websocket()

        # Start receive task handler from WebsocketService base class
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

        # Start keepalive task
        if self._websocket and not self._keepalive_task:
            self._keepalive_task = self.create_task(self._keepalive_task_handler())

    async def _disconnect(self):
        """Disconnect from the service."""
        logger.debug(f"{self}: disconnecting")

        # Cancel receive task
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        # Cancel keepalive task
        if self._keepalive_task:
            await self.cancel_task(self._keepalive_task)
            self._keepalive_task = None

        await self._disconnect_websocket()

    async def _report_error(self, frame: ErrorFrame):
        """Report an error to the pipeline.

        Args:
            frame: The error frame to push upstream.
        """
        await self.push_frame(frame, FrameDirection.UPSTREAM)

    async def _receive_messages(self):
        """Handle incoming WebSocket messages from Dograh."""
        # If websocket was closed (e.g., due to quota exceeded), just return
        if not self._websocket:
            return

        async for message in self._websocket:
            try:
                msg = json.loads(message)
                msg_type = msg.get("type")

                if msg_type == "transcription":
                    await self._handle_transcription(msg)

                elif msg_type == "speech_started":
                    await self._handle_speech_started(msg)

                elif msg_type == "speech_ended":
                    await self._handle_speech_ended(msg)

                elif msg_type == "error":
                    error_msg = msg.get("message", "Unknown error")

                    # Check if this is a quota error
                    is_quota_error = (
                        "quota" in error_msg.lower() and "exceeded" in error_msg.lower()
                    )

                    # For quota errors, handle gracefully without error logs
                    if is_quota_error:
                        logger.info(f"STT quota exceeded: {error_msg}")

                        # Push the error frame to trigger pipeline shutdown
                        await self.push_frame(
                            ErrorFrame(
                                error=f"STT service quota exceeded: {error_msg}", fatal=True
                            ),
                            direction=FrameDirection.UPSTREAM,
                        )

                        # Close the websocket gracefully
                        logger.info("Closing websocket connection gracefully due to quota exceeded")
                        try:
                            if self._websocket:
                                await self._websocket.close(
                                    code=1000, reason="Quota exceeded - closing gracefully"
                                )
                                self._websocket = None
                        except Exception as close_error:
                            logger.debug(f"Error while closing websocket: {close_error}")

                        # Raise CancelledError to cleanly cancel the receive task
                        # This will cancel the _receive_task without any error logs
                        raise asyncio.CancelledError("Quota exceeded - cancelling receive task")
                    else:
                        # For non-quota errors, raise exception to trigger retry logic
                        raise Exception(f"Dograh STT error: {error_msg}")

                elif msg_type == "ready":
                    logger.debug("Dograh STT service is ready")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode message from Dograh: {e}")
                raise
            except Exception as e:
                logger.error(f"Error processing Dograh STT message: {e}")
                raise

    async def _keepalive_task_handler(self):
        """Send periodic keepalive messages to maintain WebSocket connection."""
        KEEPALIVE_SLEEP = 5
        while True:
            await asyncio.sleep(KEEPALIVE_SLEEP)
            try:
                if self._websocket and self._websocket.state == State.OPEN:
                    keepalive_msg = {"type": "keepalive"}
                    await self._websocket.send(json.dumps(keepalive_msg))
                    logger.trace("Sent STT keepalive")
            except websockets.ConnectionClosed:
                logger.debug("Dograh STT keepalive connection closed")
                break
            except Exception as e:
                logger.error(f"Unexpected STT keepalive error: {e}")

    @traced_stt
    async def _handle_transcription_traced(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def _handle_transcription(self, msg: Dict):
        """Process transcription message from Dograh."""
        transcript = msg.get("text", "")
        is_final = msg.get("is_final", False)
        confidence = msg.get("confidence", 0.0)
        language_code = msg.get("language")

        # Parse language if provided
        language = None
        if language_code:
            try:
                language = Language(language_code)
            except ValueError:
                language = self._language

        if transcript:
            await self.stop_ttfb_metrics()

            if is_final:
                logger.debug(f"Final transcription: {transcript}")
                await self.push_frame(
                    TranscriptionFrame(
                        text=transcript,
                        user_id=self._user_id,
                        timestamp=time_now_iso8601(),
                        language=language,
                        result={"confidence": confidence} if confidence else None,
                    )
                )
                await self._handle_transcription_traced(transcript, is_final, language)
                await self.stop_processing_metrics()
            else:
                # Interim transcription
                await self.push_frame(
                    InterimTranscriptionFrame(
                        text=transcript,
                        user_id=self._user_id,
                        timestamp=time_now_iso8601(),
                        language=language,
                        result={"confidence": confidence} if confidence else None,
                    )
                )

    async def _handle_speech_started(self, msg: Dict):
        """Handle speech started event."""
        logger.debug("Speech started detected")
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()
        await self.push_frame(UserStartedSpeakingFrame())
        await self._call_event_handler("on_speech_started")

    async def _handle_speech_ended(self, msg: Dict):
        """Handle speech ended event."""
        logger.debug("Speech ended detected")
        await self.push_frame(UserStoppedSpeakingFrame())
        await self._call_event_handler("on_speech_ended")

    async def _emit_stt_usage_metrics(self):
        """Emit STT usage metrics."""
        if self._session_start_time:
            session_duration = time.time() - self._session_start_time
            metrics_data = STTUsageMetricsData(
                processor=self.name,
                model=self._model,
                value=session_duration,
            )
            frame = MetricsFrame(data=[metrics_data])
            await self.push_frame(frame)

    async def start(self, frame: StartFrame):
        """Start the STT service.

        Args:
            frame: The start frame containing initialization data.
        """
        await super().start(frame)
        self._session_start_time = time.time()
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the STT service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._emit_stt_usage_metrics()
        await self._disconnect()
        self._session_start_time = None

    async def cancel(self, frame: CancelFrame):
        """Cancel the STT service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._emit_stt_usage_metrics()
        await self._disconnect()
        self._session_start_time = None

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Send audio data to Dograh for transcription.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            Frame: None (transcription results come via WebSocket callbacks).
        """
        if self._websocket and self._websocket.state == State.OPEN:
            # Send audio as binary frame
            await self._websocket.send(audio)
        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with Dograh-specific handling.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        # Capture StartFrame metadata for workflow_run_id
        if isinstance(frame, StartFrame):
            self._start_metadata = frame.metadata

        await super().process_frame(frame, direction)

        # Handle specific frame types if needed
        if isinstance(frame, StartFrame):
            # Reinitialize on new start if needed
            pass
