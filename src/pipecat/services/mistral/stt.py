#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Mistral Speech-to-Text service implementation.

This module provides a real-time STT service that integrates with Mistral's
Voxtral Realtime transcription API using the Mistral SDK's RealtimeConnection.
"""

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_latency import MISTRAL_TTFS_P99
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from mistralai.client import Mistral
    from mistralai.client.models import (
        AudioFormat,
        RealtimeTranscriptionError,
        RealtimeTranscriptionSessionCreated,
        TranscriptionStreamDone,
        TranscriptionStreamLanguage,
        TranscriptionStreamTextDelta,
    )
    from mistralai.extra.realtime import RealtimeConnection, UnknownRealtimeEvent
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Mistral STT, you need to `pip install pipecat-ai[mistral]`.")
    raise Exception(f"Missing module: {e}")


@dataclass
class MistralSTTSettings(STTSettings):
    """Settings for MistralSTTService.

    Parameters:
        model: STT model identifier.
        language: Language hint for transcription.
    """

    pass


class MistralSTTService(STTService):
    """Mistral Speech-to-Text service using the Voxtral Realtime API.

    This service uses the Mistral SDK's RealtimeConnection to stream audio
    and receive transcription events over WebSocket. It extends STTService
    directly (rather than WebsocketSTTService) because the SDK manages
    the WebSocket connection internally.

    Event handlers available:

    - on_connected: Called when a transcription session is created.
    - on_disconnected: Called when the connection is closed.
    - on_connection_error: Called when a transcription error occurs.

    Example::

        @stt.event_handler("on_connected")
        async def on_connected(stt):
            logger.info("Mistral STT connected")
    """

    Settings = MistralSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        sample_rate: int | None = None,
        target_streaming_delay_ms: int | None = None,
        ttfs_p99_latency: float | None = MISTRAL_TTFS_P99,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize Mistral STT service.

        Args:
            api_key: Mistral API key for authentication.
            base_url: Custom API endpoint URL.
            sample_rate: Audio sample rate in Hz. If None, uses the pipeline
                sample rate.
            target_streaming_delay_ms: Streaming delay for accuracy/latency
                tradeoff. Higher values may improve accuracy at the cost of
                latency.
            ttfs_p99_latency: P99 latency from speech end to final transcript
                in seconds. Override for your deployment.
            settings: Runtime-updatable settings.
            **kwargs: Additional keyword arguments passed to STTService.
        """
        default_settings = self.Settings(
            model="voxtral-mini-transcribe-realtime-2602",
            language=None,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            settings=default_settings,
            **kwargs,
        )

        self._client = Mistral(api_key=api_key, server_url=base_url)
        self._target_streaming_delay_ms = target_streaming_delay_ms
        self._connection: RealtimeConnection | None = None
        self._receive_task = None
        self._accumulated_text = ""
        self._detected_language: str | None = None

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate processing metrics.

        Returns:
            True, indicating metrics are supported.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the STT service and establish connection.

        Args:
            frame: Frame indicating service should start.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the STT service and close connection.

        Args:
            frame: Frame indicating service should stop.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the STT service and close connection.

        Args:
            frame: Frame indicating service should be cancelled.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle speech events.

        Args:
            frame: The frame to process.
            direction: Direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._accumulated_text = ""
            await self._start_metrics()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._connection and not self._connection.is_closed:
                await self._connection.flush_audio()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Send audio data to Mistral for transcription.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            None - transcription results arrive via the receive events task.
        """
        if not self._connection or self._connection.is_closed:
            await self._connect()

        await self._connection.send_audio(audio)
        yield None

    async def _start_metrics(self):
        """Start performance metrics collection for transcription processing."""
        await self.start_processing_metrics()

    async def _connect(self):
        """Establish a connection to the Mistral Realtime API."""
        try:
            logger.debug(f"{self}: Connecting to Mistral STT")

            audio_format = AudioFormat(
                encoding="pcm_s16le",
                sample_rate=self.sample_rate,
            )

            self._connection = await self._client.audio.realtime.connect(
                model=self._settings.model,
                audio_format=audio_format,
                target_streaming_delay_ms=self._target_streaming_delay_ms,
            )

            self._receive_task = self.create_task(
                self._receive_events(), name="mistral_stt_receive"
            )
        except Exception as e:
            await self.push_error(error_msg=f"Error connecting to Mistral STT: {e}", exception=e)

    async def _disconnect(self):
        """Close the connection and cancel the receive task."""
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        if self._connection and not self._connection.is_closed:
            try:
                logger.debug(f"{self}: Disconnecting from Mistral STT")
                await self._connection.close()
            except Exception as e:
                logger.warning(f"{self}: Error closing connection: {e}")
            finally:
                self._connection = None
                await self._call_event_handler("on_disconnected")

    async def _receive_events(self):
        """Background task: iterate connection events and handle them."""
        try:
            async for event in self._connection.events():
                if isinstance(event, RealtimeTranscriptionSessionCreated):
                    logger.debug(f"{self}: Session created: {event.session}")
                    await self._call_event_handler("on_connected")

                elif isinstance(event, TranscriptionStreamTextDelta):
                    self._accumulated_text += event.text
                    await self.push_frame(
                        InterimTranscriptionFrame(
                            self._accumulated_text,
                            self._user_id,
                            time_now_iso8601(),
                        )
                    )

                elif isinstance(event, TranscriptionStreamDone):
                    if event.text:
                        await self.push_frame(
                            TranscriptionFrame(
                                event.text,
                                self._user_id,
                                time_now_iso8601(),
                                language=self._detected_language,
                            )
                        )
                        await self._handle_transcription(event.text, True, self._detected_language)
                    await self.stop_processing_metrics()
                    self._accumulated_text = ""

                elif isinstance(event, TranscriptionStreamLanguage):
                    self._detected_language = event.audio_language

                elif isinstance(event, RealtimeTranscriptionError):
                    error_msg = event.error.message if event.error else "Unknown error"
                    await self.push_error(error_msg=f"Mistral STT error: {error_msg}")
                    await self._call_event_handler("on_connection_error", error_msg)

                elif isinstance(event, UnknownRealtimeEvent):
                    logger.warning(f"{self}: Unknown realtime event: {event}")

        except Exception as e:
            await self.push_error(error_msg=f"Mistral STT receive error: {e}", exception=e)
            await self._call_event_handler("on_connection_error", str(e))
        finally:
            self._connection = None

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: str | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta, reconnecting if model or language changes.

        Args:
            delta: An STT settings delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)

        if changed:
            await self._disconnect()
            await self._connect()

        return changed
