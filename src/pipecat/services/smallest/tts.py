#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Smallest AI text-to-speech service implementation.

This module provides a WebSocket-based integration with Smallest AI's
Waves API for real-time text-to-speech synthesis.
"""

import asyncio
import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from loguru import logger
from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

from pipecat import version as pipecat_version
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven
from pipecat.services.tts_service import InterruptibleTTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts


class SmallestTTSModel(StrEnum):
    """Available Smallest AI TTS models."""

    LIGHTNING_V3_1 = "lightning_v3.1"
    LIGHTNING_V3_1_PRO = "lightning_v3.1_pro"


_MODEL_DEFAULT_VOICES: dict[SmallestTTSModel, str] = {
    SmallestTTSModel.LIGHTNING_V3_1: "sophia",
    SmallestTTSModel.LIGHTNING_V3_1_PRO: "meher",
}


def language_to_smallest_tts_language(language: Language) -> str:
    """Convert a Language enum to a Smallest TTS language string.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Smallest language code. If ``language`` is not in
        the verified mapping, falls back to the base language code (e.g.,
        ``en`` from ``en-US``) and logs a warning (via
        ``resolve_language(..., use_base_code=True)``).
    """
    LANGUAGE_MAP = {
        Language.AR: "ar",
        Language.BN: "bn",
        Language.DE: "de",
        Language.EN: "en",
        Language.ES: "es",
        Language.FR: "fr",
        Language.GU: "gu",
        Language.HE: "he",
        Language.HI: "hi",
        Language.IT: "it",
        Language.KN: "kn",
        Language.MR: "mr",
        Language.NL: "nl",
        Language.PL: "pl",
        Language.RU: "ru",
        Language.TA: "ta",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


@dataclass
class SmallestTTSSettings(TTSSettings):
    """Settings for SmallestTTSService.

    Parameters:
        speed: Speech speed multiplier (0.5–2.0).
    """

    speed: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class SmallestTTSService(InterruptibleTTSService):
    """Smallest AI real-time text-to-speech service using WebSocket streaming.

    Provides real-time text-to-speech synthesis using Smallest AI's WebSocket API.
    Supports streaming audio generation with configurable voice parameters and
    language settings. Handles interruptions by reconnecting the WebSocket.

    Example::

        tts = SmallestTTSService(
            api_key="your-api-key",
            settings=SmallestTTSService.Settings(
                voice="sophia",
                language=Language.EN,
                speed=1.0,
            ),
        )
    """

    Settings = SmallestTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "wss://api.smallest.ai",
        sample_rate: int | None = None,
        output_format: str = "pcm",
        word_timestamps: bool = True,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Smallest AI WebSocket TTS service.

        Args:
            api_key: Smallest AI API key for authentication.
            base_url: Base WebSocket URL for the Smallest API.
            sample_rate: Audio sample rate in Hz. If None, uses default.
            output_format: Audio format returned by the API. One of ``pcm``,
                ``mp3``, ``wav``, ``ulaw``, ``alaw``. Defaults to ``pcm``,
                which is what Pipecat expects internally. Fixed at init time.
            word_timestamps: Whether to request per-word timing events, enabled by
                default. When ``True``, the server interleaves ``word_timestamp``
                messages and the service emits aligned per-word ``TTSTextFrame``s.
                Supported on base-queue English + Hindi voices (``meher``,
                ``devansh``, ``kartik``, ``maithili``, ``liam``, ``avery``); other
                voices silently emit no word events, so leaving this on is safe
                regardless of voice. Fixed at init time because it determines
                whether text frames are produced from word timing or pushed whole.
            settings: Runtime-updatable settings for the TTS service.
            **kwargs: Additional arguments passed to parent InterruptibleTTSService.
        """
        # Resolve the model early so we can pick the right default voice.
        model = SmallestTTSModel.LIGHTNING_V3_1_PRO
        if settings is not None and settings.model not in (None, NOT_GIVEN):
            try:
                model = SmallestTTSModel(settings.model)
            except ValueError:
                pass

        default_settings = self.Settings(
            model=model.value,
            voice=_MODEL_DEFAULT_VOICES[model],
            language=Language.EN,
            speed=None,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_stop_frames=True,
            push_start_frame=True,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            # When word timestamps are on, per-word TTSTextFrames are emitted from
            # the word events; otherwise the base class pushes the whole text.
            push_text_frames=not word_timestamps,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._output_format = output_format
        self._word_timestamps = word_timestamps
        self._receive_task = None
        self._keepalive_task = None

        # Word-timestamp offset tracking. Smallest sends one request per
        # run_tts() call and reports word timestamps relative to *that request's*
        # audio (resetting to ~0 each request). All requests in an LLM turn share
        # one audio context, so we accumulate each request's duration and offset
        # later requests onto the turn's continuous timeline. Request boundaries
        # are detected by a change in the message ``request_id`` (Smallest emits a
        # single ``complete`` for the whole turn, not one per request). Reset per
        # turn in on_turn_context_created(). This mirrors the cumulative-offset
        # pattern used by the Rime, Inworld, and Hume TTS services.
        self._cumulative_time: float = 0.0  # offset from prior requests in the turn
        self._request_end_time: float = 0.0  # max word end seen in the in-flight request
        self._wt_request_id: str | None = None  # request_id of the in-flight request

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Smallest service supports metrics generation.
        """
        return True

    async def flush_audio(self, context_id: str | None = None):
        """Flush any pending audio data."""
        logger.trace(f"{self}: flushing audio")

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to Smallest service language format.

        Args:
            language: The language to convert.

        Returns:
            The Smallest-specific language code, or None if not supported.
        """
        return language_to_smallest_tts_language(language)

    def _build_msg(self, text: str) -> dict:
        """Build a WebSocket message for the Smallest API.

        Args:
            text: The text to synthesize.

        Returns:
            Dictionary with the API message payload.
        """
        msg = {
            "text": text,
            "voice_id": self._settings.voice,
            "model": self._settings.model,
            "language": self._settings.language,
            "sample_rate": self.sample_rate,
        }

        if self._settings.speed is not None:
            msg["speed"] = self._settings.speed

        if self._word_timestamps:
            msg["word_timestamps"] = True

        msg["output_format"] = self._output_format

        return msg

    def _build_websocket_url(self) -> str:
        """Build the WebSocket URL."""
        return f"{self._base_url}/waves/v1/tts/live"

    async def start(self, frame: StartFrame):
        """Start the Smallest TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Smallest TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Smallest TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta.

        All fields (model, speed, voice, language) take effect on the next
        ``_build_msg`` call without reconnecting.
        """
        return await super()._update_settings(delta)

    async def _connect(self):
        """Connect to Smallest WebSocket and start receive task."""
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

        if self._websocket and not self._keepalive_task:
            self._keepalive_task = self.create_task(self._keepalive_task_handler())

    async def _disconnect(self):
        """Disconnect from Smallest WebSocket and clean up tasks."""
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        if self._keepalive_task:
            await self.cancel_task(self._keepalive_task)
            self._keepalive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish WebSocket connection to the Smallest API."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Smallest TTS")

            self._websocket = await websocket_connect(
                self._build_websocket_url(),
                additional_headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "X-Source": "pipecat",
                    "X-Pipecat-Version": pipecat_version(),
                },
            )

            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Smallest TTS connection error: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close the WebSocket connection and clean up state."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Smallest TTS")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(
                error_msg=f"Smallest TTS error closing websocket: {e}", exception=e
            )
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Get the WebSocket connection if available.

        Returns:
            The active WebSocket connection.

        Raises:
            Exception: If no WebSocket connection is available.
        """
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _keepalive_task_handler(self):
        """Send periodic keepalive messages to prevent idle timeout."""
        KEEPALIVE_INTERVAL = 30
        while True:
            await asyncio.sleep(KEEPALIVE_INTERVAL)
            await self._send_keepalive()

    async def _send_keepalive(self):
        """Send a silent message to keep the WebSocket connection alive."""
        if self._websocket and self._websocket.state is State.OPEN:
            msg = {
                "text": " ",
                "voice_id": self._settings.voice,
                "model": self._settings.model,
                "language": self._settings.language,
            }
            await self._websocket.send(json.dumps(msg))

    def _advance_word_timestamp_request(self, request_id: str | None):
        """Roll the turn offset forward when word timestamps cross into a new request.

        Smallest reports word timestamps relative to each request's own audio and
        does not emit a per-request ``complete``, so a change in ``request_id`` is
        what marks the boundary between requests within a turn. When the boundary
        is crossed, the just-finished request's span (its last word ``end``) is
        folded into the running offset applied to subsequent requests.

        Args:
            request_id: The ``request_id`` of the current message.
        """
        if request_id == self._wt_request_id:
            return
        if self._wt_request_id is not None:
            self._cumulative_time += self._request_end_time
            self._request_end_time = 0.0
        self._wt_request_id = request_id

    async def on_turn_context_created(self, context_id: str):
        """Reset the word-timestamp offset at the start of each turn.

        Each LLM turn gets a fresh audio context, so the per-request offset
        accumulated for the previous turn must not carry over.

        Args:
            context_id: The newly created turn context ID.
        """
        self._cumulative_time = 0.0
        self._request_end_time = 0.0
        self._wt_request_id = None

    async def _receive_messages(self):
        """Receive and process messages from the Smallest WebSocket API."""
        async for message in self._get_websocket():
            msg = json.loads(message)
            status = msg.get("status")

            if status == "complete":
                await self.stop_all_metrics()
            elif status == "chunk":
                await self.stop_ttfb_metrics()
                context_id = self.get_active_audio_context_id()
                frame = TTSAudioRawFrame(
                    audio=base64.b64decode(msg["data"]["audio"]),
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=context_id,
                )
                await self.append_to_audio_context(context_id, frame)
            elif status == "word_timestamp":
                self._advance_word_timestamp_request(msg.get("request_id"))
                data = msg.get("data", {})
                word = data.get("word")
                start = data.get("start")
                end = data.get("end")
                if word is not None and start is not None:
                    context_id = self.get_active_audio_context_id()
                    # Offset this request's relative start onto the turn timeline.
                    # The base class consumes only (word, start); `end` is used
                    # locally to size the offset for the next request.
                    await self.add_word_timestamps(
                        [(word, start + self._cumulative_time)], context_id
                    )
                    if end is not None:
                        self._request_end_time = max(self._request_end_time, end)
            elif status == "error":
                context_id = self.get_active_audio_context_id()
                await self.push_frame(TTSStoppedFrame(context_id=context_id))
                await self.stop_all_metrics()
                await self.push_error(error_msg=f"Smallest TTS error: {msg.get('error', msg)}")
            else:
                logger.warning(f"{self} unknown message status: {msg}")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using Smallest's WebSocket streaming API.

        Args:
            text: The text to synthesize into speech.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: Audio arrives via WebSocket receive task.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                msg = self._build_msg(text=text)
                await self._get_websocket().send(json.dumps(msg))
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                yield ErrorFrame(error=f"Smallest TTS send error: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Smallest TTS error: {e}")
