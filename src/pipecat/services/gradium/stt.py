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
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

from loguru import logger
from pydantic import BaseModel

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
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven
from pipecat.services.stt_latency import GRADIUM_TTFS_P99
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


@dataclass
class GradiumSTTSettings(STTSettings):
    """Settings for GradiumSTTService.

    Parameters:
        delay_in_frames: Delay in audio frames (80ms each) before text is
            generated. Higher delays allow more context but increase latency.
            Allowed values: 7, 8, 10, 12, 14, 16, 20, 24, 36, 48.
            Default is 10 (800ms). Lower values like 7-8 give faster response.
    """

    delay_in_frames: Optional[int] | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class GradiumSTTService(WebsocketSTTService):
    """Gradium real-time speech-to-text service.

    Provides real-time speech transcription using Gradium's WebSocket API.
    Supports both interim and final transcriptions with configurable parameters
    for audio processing and connection management.
    """

    Settings = GradiumSTTSettings
    _settings: Settings

    class InputParams(BaseModel):
        """Configuration parameters for Gradium STT API.

        .. deprecated:: 0.0.105
            Use ``settings=GradiumSTTService.Settings(...)`` instead.

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
        encoding: str = "pcm",
        params: Optional[InputParams] = None,
        json_config: Optional[str] = None,
        settings: Optional[Settings] = None,
        ttfs_p99_latency: Optional[float] = GRADIUM_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Gradium STT service.

        Args:
            api_key: Gradium API key for authentication.
            api_endpoint_base_url: WebSocket endpoint URL. Defaults to Gradium's streaming endpoint.
            encoding: Audio input format. One of "pcm", "wav", or "opus". Defaults to "pcm".
            params: Configuration parameters for language and delay settings.

                .. deprecated:: 0.0.105
                    Use ``settings=GradiumSTTService.Settings(...)`` instead.

            json_config: Optional JSON configuration string for additional model settings.

                .. deprecated:: 0.0.101
                    Use `params` instead for type-safe configuration.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to parent STTService class.
        """
        if json_config is not None:
            import warnings

            warnings.warn(
                "Parameter 'json_config' is deprecated and will be removed in a future version, use 'params' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="default",
            language=None,
            delay_in_frames=None,
        )

        # 2. (No step 2, as there are no deprecated direct args)

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                default_settings.language = params.language
                if params.delay_in_frames is not None:
                    default_settings.delay_in_frames = params.delay_in_frames

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=SAMPLE_RATE,
            ttfs_p99_latency=ttfs_p99_latency,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._api_endpoint_base_url = api_endpoint_base_url
        self._encoding = encoding
        self._websocket = None
        self._json_config = json_config

        self._receive_task = None

        self._audio_buffer = bytearray()
        self._chunk_size_ms = 80
        self._chunk_size_bytes = 0

        # Accumulates text fragments within a turn. Each "text" message
        # appends to this list. On "flushed" the full text is joined and
        # pushed as a TranscriptionFrame. Any trailing fragments are
        # flushed when the user starts speaking again.
        self._accumulated_text: list[str] = []
        self._flush_counter = 0

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            True if metrics generation is supported.
        """
        return True

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta, sync params, and reconnect.

        Args:
            delta: A :class:`STTSettings` (or ``GradiumSTTService.Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)
        if not changed:
            return changed

        if self._websocket:
            await self._disconnect()
            await self._connect()
        return changed

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

    async def _start_metrics(self):
        """Start performance metrics collection for transcription processing."""
        await self.start_processing_metrics()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle speech events.

        Args:
            frame: The frame to process.
            direction: Direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self._start_metrics()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._send_flush()

    async def _send_flush(self):
        """Send a flush request to process any buffered audio immediately.

        Sends a flush message to tell the server to process buffered audio.
        The server responds with text fragments followed by a "flushed"
        acknowledgment, which triggers finalization.
        """
        if not self._websocket or self._websocket.state is not State.OPEN:
            return

        self._flush_counter += 1
        flush_id = str(self._flush_counter)
        msg = {"type": "flush", "flush_id": flush_id}
        try:
            await self._websocket.send(json.dumps(msg))
        except Exception as e:
            logger.warning(f"Failed to send flush: {e}")

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
                "model_name": self._settings.model,
                "input_format": self._encoding,
            }
            # Build json_config: start with deprecated json_config, then override with params
            json_config = {}
            if self._json_config:
                json_config = json.loads(self._json_config)
            if self._settings.language:
                gradium_language = language_to_gradium_language(self._settings.language)
                if gradium_language:
                    json_config["language"] = gradium_language
            if self._settings.delay_in_frames:
                json_config["delay_in_frames"] = self._settings.delay_in_frames
            if json_config:
                setup_msg["json_config"] = json_config
            await self._websocket.send(json.dumps(setup_msg))
            ready_msg = await self._websocket.recv()
            ready_msg = json.loads(ready_msg)
            if ready_msg["type"] == "error":
                raise Exception(f"received error {ready_msg['message']}")
            if ready_msg["type"] != "ready":
                raise Exception(f"unexpected first message type {ready_msg['type']}")

            logger.debug("Connected to Gradium STT")

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

    async def _receive_messages(self):
        async for message in self._get_websocket():
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"Received non-JSON message: {message}")
                continue

            type_ = msg.get("type", "")
            if type_ == "text":
                await self._handle_text(msg["text"])
            elif type_ == "flushed":
                await self._handle_flushed()
            elif type_ == "end_of_stream":
                logger.debug("Received end_of_stream message from server")
            elif type_ == "error":
                await self.push_error(error_msg=f"Error: {msg}")

    async def _handle_text(self, text: str):
        """Handle streaming transcription fragment.

        Accumulates text and pushes an InterimTranscriptionFrame with the
        full accumulated text so far.
        """
        self._accumulated_text.append(text)
        accumulated = " ".join(self._accumulated_text)
        await self.push_frame(
            InterimTranscriptionFrame(
                text=accumulated,
                user_id=self._user_id,
                timestamp=time_now_iso8601(),
                language=self._settings.language,
            )
        )
        await self.stop_processing_metrics()

    async def _handle_flushed(self):
        """Handle flush completion by pushing the finalized transcription.

        The "flushed" message confirms that buffered audio has been processed.
        Any trailing text fragments that arrive after this will be caught by
        the TTFB timeout handler.
        """
        if not self._accumulated_text:
            return
        text = " ".join(self._accumulated_text)
        self._accumulated_text.clear()
        logger.debug(f"Final transcription: [{text}]")
        await self.push_frame(
            TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                self._settings.language,
            )
        )
        await self._trace_transcription(text, is_final=True, language=self._settings.language)
