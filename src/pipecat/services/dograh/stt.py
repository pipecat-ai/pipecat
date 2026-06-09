#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Dograh STT Service implementation using WebSocket streaming."""

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.dograh.mps_billing import (
    MPS_BILLING_VERSION_KEY,
    MPS_BILLING_VERSION_V2,
    get_correlation_id,
    uses_mps_billing_v2,
)
from pipecat.services.settings import STTSettings
from pipecat.services.stt_latency import DOGRAH_TTFS_P99
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
    logger.error("In order to use STT, you need to `pip install websockets`.")
    raise Exception(f"Missing module: {e}")


@dataclass
class DograhSTTSettings(STTSettings):
    """Settings for DograhSTTService."""

    pass


class DograhSTTService(STTService, WebsocketService):
    """Dograh speech-to-text service using WebSocket streaming.

    This service provides real-time speech recognition using Dograh's unified WebSocket API.
    Supports streaming transcription, interim results, and VAD events.
    """

    Settings = DograhSTTSettings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "wss://services.dograh.com",
        ws_path: str = "/api/v1/stt/stream",
        correlation_id: str | None = None,
        sample_rate: int | None = None,
        interim_results: bool = True,
        vad_events: bool = False,
        keyterms: list[str] | None = None,
        settings: DograhSTTSettings | None = None,
        ttfs_p99_latency: float | None = DOGRAH_TTFS_P99,
        **kwargs,
    ):
        """Initialize STT service.

        Args:
            api_key: The Dograh API key for authentication.
            base_url: WebSocket base URL for Dograh API. Defaults to "wss://services.dograh.com".
            ws_path: WebSocket path for STT streaming. Defaults to "/api/v1/stt/stream".
            correlation_id: Optional server-generated correlation ID for MPS billing v2.
            sample_rate: Audio sample rate in Hz. Defaults to None.
            interim_results: Whether to receive interim transcription results.
            vad_events: Whether to receive voice activity detection events.
            keyterms: Optional list of keyterms for speech recognition boosting.
            settings: STT settings including model and language.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to the parent services.
        """
        default_settings = DograhSTTSettings(model="default", language="multi")
        if settings is not None:
            default_settings.apply_update(settings)

        STTService.__init__(
            self,
            sample_rate=sample_rate,
            settings=default_settings,
            ttfs_p99_latency=ttfs_p99_latency,
            **kwargs,
        )
        WebsocketService.__init__(self, reconnect_on_error=True, **kwargs)

        self._api_key = api_key
        self._base_url = base_url
        self._ws_path = ws_path
        self._correlation_id = correlation_id
        self._interim_results = interim_results
        self._vad_events = vad_events
        self._keyterms = keyterms or []

        self._receive_task = None
        self._keepalive_task = None

        # Session tracking for metrics
        self._session_start_time: float | None = None
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

    async def set_language(self, language: Language):
        """Set the language for speech recognition.

        Args:
            language: The language to use for recognition.
        """
        await self._update_settings(STTSettings(language=language))

    def _get_correlation_id(self) -> str | None:
        return get_correlation_id(
            explicit_correlation_id=self._correlation_id,
            start_metadata=self._start_metadata,
        )

    def _uses_mps_billing_v2(self) -> bool:
        return uses_mps_billing_v2(
            explicit_correlation_id=self._correlation_id,
            start_metadata=self._start_metadata,
        )

    async def _connect_websocket(self):
        """Connect to the WebSocket endpoint."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            url = f"{self._base_url}{self._ws_path}"
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }

            logger.debug(f"Connecting to STT WebSocket at {url}")
            ws = await websocket_connect(url, additional_headers=headers)
            self._websocket = ws

            # Send initial configuration
            config_msg = {
                "type": "config",
                "model": self._settings.model,
                "language": self._settings.language,
                "sample_rate": self.sample_rate,
                "interim_results": self._interim_results,
                "vad_events": self._vad_events,
            }

            # Add keyterms if provided
            if self._keyterms:
                config_msg["keyterms"] = self._keyterms

            correlation_id = self._get_correlation_id()
            if correlation_id:
                config_msg["correlation_id"] = correlation_id
                if self._uses_mps_billing_v2():
                    config_msg[MPS_BILLING_VERSION_KEY] = MPS_BILLING_VERSION_V2

            await ws.send(json.dumps(config_msg))

            logger.info("Connected to STT service")

        except Exception as e:
            self._websocket = None
            logger.error(f"Failed to connect to STT service: {e}")
            raise

    async def _disconnect_websocket(self):
        """Disconnect from the WebSocket endpoint."""
        try:
            if self._websocket:
                logger.debug("Disconnecting from STT service")
                # Send end of stream signal
                end_msg = {"type": "end_of_stream"}
                await self._websocket.send(json.dumps(end_msg))
                await self._websocket.close()
                logger.debug("Disconnected from STT service")
        except Exception as e:
            logger.error(f"Error disconnecting from STT service: {e}")
        finally:
            self._websocket = None

    async def _connect(self):
        """Connect to the service."""
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

        if self._websocket and not self._keepalive_task:
            self._keepalive_task = self.create_task(self._keepalive_task_handler())

    async def _disconnect(self):
        """Disconnect from the service."""
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

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
                        # Push error frame so observers (e.g. RealtimeFeedbackObserver)
                        # can surface it to the frontend before reconnect swallows it
                        await self.push_frame(
                            ErrorFrame(error=f"STT error: {error_msg}"),
                            direction=FrameDirection.UPSTREAM,
                        )
                        # Raise to trigger reconnect logic
                        raise Exception(f"STT error: {error_msg}")

                elif msg_type == "ready":
                    logger.debug("STT service is ready")

            except asyncio.CancelledError:
                raise
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode message from Dograh: {e}")
                raise
            except Exception as e:
                logger.error(f"Error processing STT message: {e}")
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
                logger.debug("STT keepalive connection closed")
                break
            except Exception as e:
                logger.error(f"Unexpected STT keepalive error: {e}")

    @traced_stt
    async def _handle_transcription_traced(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def _send_finalize(self):
        """Send finalize message to Dograh server to flush the current transcript."""
        if self._websocket and self._websocket.state == State.OPEN:
            finalize_msg = json.dumps({"type": "finalize"})
            await self._websocket.send(finalize_msg)
            logger.trace("Sent finalize to STT server")

    async def _handle_transcription(self, msg: dict):
        """Process transcription message from Dograh."""
        transcript = msg.get("text", "")
        is_final = msg.get("is_final", False)
        from_finalize = msg.get("from_finalize", False)
        confidence = msg.get("confidence", 0.0)
        language_code = msg.get("language")

        # Parse language if provided
        language: Language | None = None
        if language_code:
            try:
                language = Language(language_code)
            except ValueError:
                settings_language = self._settings.language
                if isinstance(settings_language, Language):
                    language = settings_language

        if transcript:
            if is_final:
                # Check if this response is from a finalize() call.
                # Only mark as finalized when both we requested it AND the server confirms it.
                if from_finalize:
                    self.confirm_finalize()
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

    async def _handle_speech_started(self, msg: dict):
        """Handle speech started event."""
        logger.debug("Speech started detected")
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()
        await self.push_frame(UserStartedSpeakingFrame())
        await self._call_event_handler("on_speech_started")

    async def _handle_speech_ended(self, msg: dict):
        """Handle speech ended event."""
        logger.debug("Speech ended detected")
        await self.push_frame(UserStoppedSpeakingFrame())
        await self._call_event_handler("on_speech_ended")

    async def start(self, frame: StartFrame):
        """Start the STT service.

        Args:
            frame: The start frame containing initialization data.
        """
        await super().start(frame)
        self._start_metadata = frame.metadata
        self._session_start_time = time.time()
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the STT service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()
        self._session_start_time = None

    async def cancel(self, frame: CancelFrame):
        """Cancel the STT service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()
        self._session_start_time = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with Dograh-specific handling.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            # Send finalize to flush the current transcript from Deepgram (via Dograh server)
            if self._websocket and self._websocket.state == State.OPEN:
                self.request_finalize()
                await self._send_finalize()
                logger.trace(f"Triggered finalize event on: {frame.name=}, {direction=}")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
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
