#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Together AI speech-to-text service implementation.

This module provides a STT service using Together AI's WebSocket API:

- ``TogetherSTTService``: WebSocket-based real-time STT using an
  OpenAI-compatible realtime transcription endpoint.
"""

import base64
import json
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Optional

from loguru import logger

from pipecat.services.settings import STTSettings
from pipecat.services.stt_latency import TOGETHER_TTFS_P99

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Together, you need to `pip install pipecat-ai[together]`.")
    raise Exception(f"Missing module: {e}")

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
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt


@dataclass
class TogetherSTTSettings(STTSettings):
    """Settings for the Together AI STT service."""

    pass


class TogetherSTTService(WebsocketSTTService):
    """Together AI speech-to-text service.

    Provides real-time speech recognition using Together AI's WebSocket API
    with OpenAI-compatible speech-to-text endpoints.

    Example::

        stt = TogetherSTTService(
            api_key="your-api-key",
            settings=TogetherSTTService.Settings(
                model="openai/whisper-large-v3",
                language=Language.EN,
            ),
        )
    """

    Settings = TogetherSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        sample_rate: Optional[int] = None,
        base_url: str = "wss://api.together.ai/v1",
        settings: Optional[Settings] = None,
        ttfs_p99_latency: float = TOGETHER_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Together AI STT service.

        Args:
            api_key: Together AI API key for authentication.
            sample_rate: Audio sample rate in Hz. If None, uses the pipeline's rate.
            base_url: The URL of the Together AI WebSocket API.
            settings: Runtime-updatable settings for the STT service.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to WebsocketSTTService.
        """
        default_settings = self.Settings(
            model="openai/whisper-large-v3",
            language=Language.EN,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url
        self._receive_task = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Together AI language format.

        Args:
            language: The language to convert.

        Returns:
            The language code string, or None if not supported.
        """
        return str(language)

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect if anything changed."""
        changed = await super()._update_settings(delta)

        if changed:
            await self._disconnect()
            await self._connect()

        return changed

    async def start(self, frame: StartFrame):
        """Start the Together AI STT service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Together AI STT service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Together AI STT service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Send audio data to Together AI for transcription.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            Frame: None (transcription results come via WebSocket callbacks).
        """
        await self._send_audio(audio)
        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with Together AI-specific handling.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self.start_processing_metrics()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._commit_audio_buffer()

    # ------------------------------------------------------------------
    # WebSocket connection management
    # ------------------------------------------------------------------

    async def _connect(self):
        """Connect to the transcription endpoint and start receiving."""
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Disconnect and clean up background tasks."""
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task, timeout=1.0)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish the WebSocket connection to the Together AI endpoint."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            url = (
                f"{self._base_url}/realtime?intent=transcription"
                f"&model={self._settings.model}"
                f"&input_audio_format=pcm_s16le_16000"
            )
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "OpenAI-Beta": "realtime=v1",
            }

            self._websocket = await websocket_connect(url, additional_headers=headers)
            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(
                error_msg=f"Error connecting to Together AI STT: {e}",
                exception=e,
            )
            self._websocket = None

    async def _disconnect_websocket(self):
        """Close the WebSocket connection."""
        try:
            if self._websocket:
                await self._websocket.close()
        except Exception as e:
            await self.push_error(
                error_msg=f"Error disconnecting: {e}",
                exception=e,
            )
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    # ------------------------------------------------------------------
    # Client events
    # ------------------------------------------------------------------

    async def _send_audio(self, audio: bytes):
        """Send audio data via ``input_audio_buffer.append``.

        Args:
            audio: Raw audio bytes at the pipeline sample rate.
        """
        try:
            if not self._disconnecting and self._websocket:
                payload = base64.b64encode(audio).decode("utf-8")
                await self._websocket.send(
                    json.dumps({"type": "input_audio_buffer.append", "audio": payload})
                )
        except Exception as e:
            if self._disconnecting or not self._websocket:
                return
            await self.push_error(
                error_msg=f"Error sending audio: {e}",
                exception=e,
            )

    async def _commit_audio_buffer(self):
        """Commit the current audio buffer for transcription."""
        try:
            if not self._disconnecting and self._websocket:
                await self._websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        except Exception as e:
            if self._disconnecting or not self._websocket:
                return
            await self.push_error(
                error_msg=f"Error committing audio buffer: {e}",
                exception=e,
            )

    # ------------------------------------------------------------------
    # Server event handling
    # ------------------------------------------------------------------

    async def _receive_messages(self):
        """Receive and dispatch server events from the transcription session.

        Called by ``WebsocketService._receive_task_handler`` which wraps
        this method with automatic reconnection on connection errors.
        """
        async for message in self._websocket:
            try:
                evt = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"{self} failed to parse WebSocket message")
                continue

            evt_type = evt.get("type", "")

            if evt_type == "session.created":
                logger.debug(f"{self} session created")
            elif evt_type == "conversation.item.input_audio_transcription.delta":
                await self._handle_transcription_delta(evt)
            elif evt_type == "conversation.item.input_audio_transcription.completed":
                await self._handle_transcription_completed(evt)
            elif evt_type == "conversation.item.input_audio_transcription.failed":
                await self._handle_transcription_failed(evt)
            elif evt_type == "input_audio_buffer.committed":
                logger.trace(f"Audio buffer committed: item_id={evt.get('item_id', '')}")
            elif evt_type == "error":
                await self._handle_error(evt)
            else:
                logger.trace(f"{self} unhandled event: {evt_type}")

    async def _handle_transcription_delta(self, evt: dict):
        """Handle incremental transcription text.

        Args:
            evt: The delta event from the server.
        """
        delta = evt.get("transcript", "")
        if delta.strip():
            await self.push_frame(
                InterimTranscriptionFrame(
                    delta,
                    self._user_id,
                    time_now_iso8601(),
                    result=evt,
                )
            )

    async def _handle_transcription_completed(self, evt: dict):
        """Handle a completed transcription for a speech segment.

        Args:
            evt: The completed event containing the full transcript.
        """
        transcript = evt.get("transcript", "").strip()
        if transcript:
            await self.push_frame(
                TranscriptionFrame(
                    transcript,
                    self._user_id,
                    time_now_iso8601(),
                    result=evt,
                )
            )
            await self._handle_transcription_trace(transcript, True, self._settings.language)
            await self.stop_processing_metrics()

    @traced_stt
    async def _handle_transcription_trace(
        self, transcript: str, is_final: bool, language: Optional[Language] = None
    ):
        """Record transcription result for tracing.

        Args:
            transcript: The transcribed text.
            is_final: Whether this is a final transcription result.
            language: Optional language of the transcription.
        """
        pass

    async def _handle_transcription_failed(self, evt: dict):
        """Handle a transcription failure for a speech segment.

        Args:
            evt: The failed event containing error details.
        """
        error_info = evt.get("error", {})
        await self.push_error(error_msg=f"Transcription failed: {error_info}")

    async def _handle_error(self, evt: dict):
        """Handle a fatal error from the transcription session.

        Raises an exception so that ``WebsocketService`` can decide
        whether to attempt reconnection.

        Args:
            evt: The error event.
        """
        error_info = evt.get("error", {})
        error_msg = error_info.get("message", "Unknown error")
        error_code = error_info.get("code", "")
        msg = f"Together AI STT error [{error_code}]: {error_msg}"
        await self.push_error(error_msg=msg)
        raise Exception(msg)
