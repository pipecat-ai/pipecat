#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Dograh TTS Service implementation using WebSocket streaming."""

import asyncio
import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.dograh.mps_billing import (
    MPS_BILLING_VERSION_KEY,
    MPS_BILLING_VERSION_V2,
    get_correlation_id,
    uses_mps_billing_v2,
)
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven
from pipecat.services.tts_service import TextAggregationMode, WebsocketTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    import websockets
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Dograh TTS, you need to `pip install websockets`.")
    raise Exception(f"Missing module: {e}")


def calculate_word_times(
    alignment_info: dict[str, Any], cumulative_time: float
) -> list[tuple[str, float]]:
    """Calculate word timestamps from alignment information.

    Args:
        alignment_info: Word alignment data from Dograh API.
        cumulative_time: Base time offset for this chunk.

    Returns:
        List of (word, timestamp) tuples.
    """
    words_data = alignment_info.get("words", [])
    word_times = []

    for word_info in words_data:
        word = word_info.get("word", "")
        start_time_ms = word_info.get("start", 0)
        start_time_seconds = cumulative_time + (start_time_ms / 1000.0)
        word_times.append((word, start_time_seconds))

    return word_times


@dataclass
class DograhTTSSettings(TTSSettings):
    """Settings for DograhTTSService.

    Parameters:
        speed: Speech speed control (0.5 to 2.0).
        pitch: Voice pitch control (-1.0 to 1.0).
        volume: Volume control (0.0 to 1.0).
    """

    speed: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    pitch: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    volume: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class DograhTTSService(WebsocketTTSService):
    """Dograh WebSocket-based TTS service with word timestamps.

    This service provides real-time text-to-speech using Dograh's unified WebSocket API.
    Supports word-level timestamps and audio streaming.
    """

    Settings = DograhTTSSettings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "wss://services.dograh.com",
        ws_path: str = "/api/v1/tts/stream",
        correlation_id: str | None = None,
        sample_rate: int | None = None,
        settings: DograhTTSSettings | None = None,
        text_aggregation_mode: TextAggregationMode | None = None,
        **kwargs,
    ):
        """Initialize Dograh TTS service.

        Args:
            api_key: The Dograh API key for authentication.
            base_url: WebSocket base URL for Dograh API. Defaults to "wss://services.dograh.com".
            ws_path: WebSocket path for TTS streaming. Defaults to "/api/v1/tts/stream".
            correlation_id: Optional server-generated correlation ID for MPS billing v2.
            sample_rate: Output audio sample rate in Hz. Defaults to None.
            settings: TTS settings including model, voice, speed, pitch, volume.
            text_aggregation_mode: How to aggregate incoming text before synthesis.
            **kwargs: Additional arguments passed to the parent service.
        """
        default_settings = DograhTTSSettings(
            model="default",
            voice="default",
            language="en",
            speed=1.0,
            pitch=0.0,
            volume=1.0,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            text_aggregation_mode=text_aggregation_mode,
            push_text_frames=False,
            push_stop_frames=True,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url
        self._ws_path = ws_path
        self._correlation_id = correlation_id
        # Only forward fields the upstream provider recognizes. Unknown fields
        # (e.g. pitch/volume/language) would be echoed verbatim by MPS to the
        # upstream provider, which may reject voice settings that change between
        # contexts on the same connection.
        self._voice_settings: dict[str, float | bool] = {}
        speed = default_settings.speed
        if isinstance(speed, (int, float)):
            self._voice_settings["speed"] = float(speed)

        # WebSocket tasks
        self._receive_task = None
        self._keepalive_task = None

        # State management
        self._cumulative_time = 0
        self._accumulated_text = ""
        self._start_metadata = None
        self._remote_initialized_context_ids: set[str] = set()
        self._finished_context_ids: set[str] = set()
        self._cancelled_context_ids: set[str] = set()

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Dograh service supports metrics generation.
        """
        return True

    def _reset_state(self):
        """Reset internal state variables."""
        self._cumulative_time = 0
        logger.debug(f"{self}: Reset internal state")

    async def set_language(self, language: Language):
        """Set the language for synthesis.

        Args:
            language: The language to use for synthesis.
        """
        self._settings.language = language.value

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
        """Establish the websocket connection to Dograh TTS service."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            url = f"{self._base_url}{self._ws_path}"
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }

            logger.debug(f"Connecting to Dograh TTS WebSocket at {url}")
            ws = await websocket_connect(url, additional_headers=headers)
            self._websocket = ws
            self._remote_initialized_context_ids.clear()
            self._finished_context_ids.clear()
            self._cancelled_context_ids.clear()

            # Send initial configuration
            config_msg = {
                "type": "config",
                "model": self._settings.model,
                "voice": self._settings.voice,
                "sample_rate": self.sample_rate,
            }
            if self._voice_settings:
                config_msg["settings"] = self._voice_settings

            correlation_id = self._get_correlation_id()
            if correlation_id:
                config_msg["correlation_id"] = correlation_id
                if self._uses_mps_billing_v2():
                    config_msg[MPS_BILLING_VERSION_KEY] = MPS_BILLING_VERSION_V2

            await ws.send(json.dumps(config_msg))

            logger.info(f"Connected to Dograh TTS service")

        except Exception as e:
            self._websocket = None
            logger.error(f"Failed to connect to Dograh TTS service: {e}")
            raise

    async def _disconnect_websocket(self):
        """Close the websocket connection to Dograh TTS service."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Dograh TTS service")
                await self._websocket.close()
                logger.debug("Disconnected from Dograh TTS service")
        except Exception as e:
            logger.error(f"Error disconnecting from Dograh TTS service: {e}")
        finally:
            self._remote_initialized_context_ids.clear()
            self._finished_context_ids.clear()
            self._cancelled_context_ids.clear()
            await self.remove_active_audio_context()
            self._websocket = None

    async def _connect(self):
        """Connect to the Dograh TTS service with full initialization."""
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

        if self._websocket and not self._keepalive_task:
            self._keepalive_task = self.create_task(self._keepalive_task_handler())

    async def _disconnect(self):
        """Disconnect from Dograh TTS service and clean up tasks."""
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        if self._keepalive_task:
            await self.cancel_task(self._keepalive_task)
            self._keepalive_task = None

        await self._disconnect_websocket()

    def _get_websocket(self):
        """Get the WebSocket connection.

        Returns:
            The websocket connection.

        Raises:
            Exception: If websocket is not connected.
        """
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        """Handle incoming WebSocket messages from Dograh."""
        async for message in self._get_websocket():
            try:
                msg = json.loads(message)
                msg_type = msg.get("type")
                ctx_id = msg.get("context_id")

                # Handle final messages first. The server sends "final" to signal
                # it is done generating audio for the context. We must call
                # remove_audio_context to signal the parent's _handle_audio_context
                # loop to stop (otherwise it waits for a 3-second timeout).
                if msg_type == "final":
                    logger.trace(f"Received final message for context {ctx_id}")
                    if ctx_id:
                        self._remote_initialized_context_ids.discard(ctx_id)
                        if self.audio_context_available(ctx_id):
                            await self.remove_audio_context(ctx_id)
                    continue

                # Skip messages for unavailable contexts
                if ctx_id and not self.audio_context_available(ctx_id):
                    if self.get_active_audio_context_id() == ctx_id:
                        logger.debug(
                            f"Received a delayed message, recreating the context: {ctx_id}"
                        )
                        await self.create_audio_context(ctx_id)
                    else:
                        logger.debug(f"Ignoring message from unavailable context: {ctx_id}")
                        continue

                if msg_type == "audio":
                    await self.stop_ttfb_metrics()
                    await self.start_word_timestamps()

                    audio_data = msg.get("audio")
                    if audio_data:
                        audio = base64.b64decode(audio_data)
                        frame = TTSAudioRawFrame(audio, self.sample_rate, 1, context_id=ctx_id)

                        # Use context ID from message or current context
                        effective_ctx_id = ctx_id or self.get_active_audio_context_id()
                        if effective_ctx_id:
                            await self.append_to_audio_context(effective_ctx_id, frame)

                elif msg_type == "alignment":
                    # Handle word alignment data
                    alignment = msg.get("data", {})
                    word_times = calculate_word_times(alignment, self._cumulative_time)

                    if word_times:
                        await self.add_word_timestamps(word_times, ctx_id)

                        # Update cumulative time based on last word
                        self._cumulative_time = word_times[-1][1]

                elif msg_type == "error":
                    error_msg = msg.get("message", "Unknown error")
                    await self.push_frame(TTSStoppedFrame())
                    await self.stop_all_metrics()

                    # Check if this is a quota error
                    is_quota_error = (
                        "quota" in error_msg.lower() and "exceeded" in error_msg.lower()
                    )

                    # For quota errors, push a fatal error frame to trigger pipeline shutdown
                    if is_quota_error:
                        logger.info(f"TTS quota exceeded: {error_msg}")

                        # Push the error frame to trigger pipeline shutdown
                        await self.push_frame(
                            ErrorFrame(
                                error=f"TTS service quota exceeded: {error_msg}", fatal=True
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
                        raise asyncio.CancelledError("Quota exceeded - cancelling receive task")
                    else:
                        raise Exception(f"Dograh TTS error: {error_msg}")

            except asyncio.CancelledError:
                raise
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode message from Dograh: {e}")
                raise
            except Exception as e:
                logger.error(f"Error processing Dograh TTS message: {e}")
                raise

    async def _keepalive_task_handler(self):
        """Send periodic keepalive messages to maintain WebSocket connection."""
        KEEPALIVE_SLEEP = 10
        while True:
            await asyncio.sleep(KEEPALIVE_SLEEP)
            try:
                if self._websocket and self._websocket.state is State.OPEN:
                    context_id = self.get_active_audio_context_id()
                    if not context_id:
                        continue
                    if context_id not in self._remote_initialized_context_ids:
                        logger.trace(
                            f"Skipping keepalive for uninitialized remote context {context_id}"
                        )
                        continue
                    keepalive_msg = {
                        "type": "keepalive",
                        "context_id": context_id,
                    }
                    await self._websocket.send(json.dumps(keepalive_msg))
                    logger.trace(f"Sent keepalive for context {context_id}")
            except websockets.ConnectionClosed as e:
                logger.warning(f"{self} keepalive error: {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected keepalive error: {e}")

    async def _send_text(self, text: str, context_id: str):
        """Send text to the WebSocket for synthesis."""
        if self._websocket and context_id:
            msg = {"type": "synthesize", "text": text, "context_id": context_id}
            await self._websocket.send(json.dumps(msg))

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using Dograh's streaming WebSocket API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                if not self.audio_context_available(context_id):
                    await self.create_audio_context(context_id)
                    await self.start_ttfb_metrics()
                    yield TTSStartedFrame(context_id=context_id)
                    self._cumulative_time = 0

                    # Send initial context setup with voice settings
                    context_msg = {
                        "type": "create_context",
                        "context_id": context_id,
                        "voice": self._settings.voice,
                        "model": self._settings.model,
                    }
                    if self._voice_settings:
                        context_msg["settings"] = self._voice_settings

                    correlation_id = self._get_correlation_id()
                    if correlation_id:
                        context_msg["correlation_id"] = correlation_id
                        if self._uses_mps_billing_v2():
                            context_msg[MPS_BILLING_VERSION_KEY] = MPS_BILLING_VERSION_V2

                    await self._get_websocket().send(json.dumps(context_msg))
                    self._remote_initialized_context_ids.add(context_id)
                    self._finished_context_ids.discard(context_id)
                    self._cancelled_context_ids.discard(context_id)
                    logger.trace(f"Created new context {context_id} with voice settings")

                # Send text for synthesis
                await self._send_text(text, context_id)
                self._accumulated_text += text
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                yield TTSStoppedFrame(context_id=context_id)
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
                return

            yield None

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")

    async def _flush_usage_metrics(self):
        """Bill the turn's accumulated text, then clear the buffer."""
        if self._accumulated_text:
            await self.start_tts_usage_metrics(self._accumulated_text)
            self._accumulated_text = ""

    async def _finalize_context_state(self):
        """Flush usage metrics and reset word-timestamp state.

        Used on interruption, where no completion hook fires for the context.
        """
        await self._flush_usage_metrics()
        self._reset_state()

    async def _finish_context(self, context_id: str):
        """Finalize a context at end of turn.

        Sends the ``close_context`` verb so the server finalizes the upstream
        provider and a terminal ``final`` is emitted right after the last audio
        byte, instead of the context completing via the audio-context idle timeout.

        Only usage is billed here; word-timestamp state is reset later in
        ``on_audio_context_completed`` (once the context has drained), because
        alignment can still arrive after this message.

        Args:
            context_id: The context ID to finalize.
        """
        if context_id and self._websocket and context_id not in self._finished_context_ids:
            logger.trace(f"{self}: Finishing context {context_id}")
            self._remote_initialized_context_ids.discard(context_id)
            self._finished_context_ids.add(context_id)
            try:
                await self._websocket.send(
                    json.dumps({"type": "close_context", "context_id": context_id})
                )
            except Exception as e:
                logger.error(f"Error finishing context: {e}")

        await self._flush_usage_metrics()

    async def _cancel_context(self, context_id: str):
        """Abort a context on interruption.

        Sends the ``cancel`` verb so the server stops upstream generation
        immediately rather than letting the provider finish the buffered audio.
        Requires server-side ``cancel`` handling; a server without it ignores the
        message. The base class abandons the interrupted context
        (``on_audio_context_completed`` does not fire for it), so usage is billed
        and word-timestamp state is reset here.

        Args:
            context_id: The context ID to cancel.
        """
        if context_id and self._websocket and context_id not in self._cancelled_context_ids:
            logger.trace(f"{self}: Cancelling context {context_id}")
            self._remote_initialized_context_ids.discard(context_id)
            self._cancelled_context_ids.add(context_id)
            try:
                await self._websocket.send(json.dumps({"type": "cancel", "context_id": context_id}))
            except Exception as e:
                logger.error(f"Error cancelling context: {e}")

        await self._finalize_context_state()

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the Dograh context when the bot is interrupted."""
        await self._cancel_context(context_id)
        await super().on_audio_context_interrupted(context_id)

    async def on_audio_context_completed(self, context_id: str):
        """Reset word-timestamp state after all audio for the context has played."""
        self._reset_state()
        await super().on_audio_context_completed(context_id)

    async def on_turn_context_completed(self):
        """Finish the server-side context at end of turn."""
        context_id = self._turn_context_id
        should_finish = bool(context_id and self.audio_context_available(context_id))
        await super().on_turn_context_completed()
        if should_finish and context_id:
            await self._finish_context(context_id)

    async def flush_audio(self, context_id: str | None = None):
        """Flush any pending audio and finalize the current context.

        Args:
            context_id: The specific context to flush. If None, falls back to the
                currently active context.
        """
        flush_id = context_id or self.get_active_audio_context_id()
        if not flush_id or not self._websocket:
            return
        logger.debug(f"{self}: flushing audio")
        # MPS routes by "type" and translates this generic flush into whatever
        # the upstream provider expects.
        msg = {"type": "flush", "context_id": flush_id}
        await self._websocket.send(json.dumps(msg))

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        """Push a frame and handle state changes.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await super().push_frame(frame, direction)
        if isinstance(frame, (TTSStoppedFrame, InterruptionFrame)):
            if isinstance(frame, TTSStoppedFrame):
                await self.add_word_timestamps([("Reset", 0)], self.get_active_audio_context_id())

    async def start(self, frame: StartFrame):
        """Start the TTS service.

        Args:
            frame: The start frame containing initialization data.
        """
        await super().start(frame)
        self._start_metadata = frame.metadata
        self._reset_state()
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the TTS service and clean up resources.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()
