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
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Optional

from loguru import logger

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

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Smallest, you need to `pip install pipecat-ai[smallest]`.")
    raise Exception(f"Missing module: {e}")


class SmallestTTSModel(str, Enum):
    """Available Smallest AI TTS models."""

    LIGHTNING_V2 = "lightning-v2"
    LIGHTNING_V3_1 = "lightning-v3.1"


def language_to_smallest_tts_language(language: Language) -> Optional[str]:
    """Convert a Language enum to a Smallest TTS language string.

    Args:
        language: The Language enum value to convert.

    Returns:
        The Smallest language code string, or None if unsupported.
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
        speed: Speech speed multiplier.
        consistency: Consistency level for voice generation (0-1).
        similarity: Similarity level for voice generation (0-1).
        enhancement: Enhancement level for voice generation (0-2).
    """

    speed: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    consistency: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    similarity: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    enhancement: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


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
        base_url: str = "wss://waves-api.smallest.ai",
        sample_rate: Optional[int] = None,
        settings: Optional[Settings] = None,
        **kwargs,
    ):
        """Initialize the Smallest AI WebSocket TTS service.

        Args:
            api_key: Smallest AI API key for authentication.
            base_url: Base WebSocket URL for the Smallest API.
            sample_rate: Audio sample rate in Hz. If None, uses default.
            settings: Runtime-updatable settings for the TTS service.
            **kwargs: Additional arguments passed to parent InterruptibleTTSService.
        """
        default_settings = self.Settings(
            model=SmallestTTSModel.LIGHTNING_V3_1.value,
            voice="sophia",
            language=Language.EN,
            speed=None,
            consistency=None,
            similarity=None,
            enhancement=None,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_stop_frames=True,
            push_start_frame=True,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._receive_task = None
        self._keepalive_task = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Smallest service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
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
            "language": self._settings.language,
            "sample_rate": self.sample_rate,
        }

        if self._settings.speed is not None:
            msg["speed"] = self._settings.speed

        # consistency, similarity, enhancement are only supported by lightning-v2
        if self._settings.model == SmallestTTSModel.LIGHTNING_V2.value:
            if self._settings.consistency is not None:
                msg["consistency"] = self._settings.consistency
            if self._settings.similarity is not None:
                msg["similarity"] = self._settings.similarity
            if self._settings.enhancement is not None:
                msg["enhancement"] = self._settings.enhancement

        return msg

    def _build_websocket_url(self) -> str:
        """Build the WebSocket URL from base URL and model."""
        return f"{self._base_url}/api/v1/{self._settings.model}/get_speech/stream"

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
        """Apply a settings delta, reconnecting if model changed.

        Per-message fields (speed, consistency, similarity, enhancement, voice,
        language) apply automatically on the next ``_build_msg`` call. A model
        change requires reconnecting because the model is part of the WebSocket URL.
        """
        changed = await super()._update_settings(delta)

        if not changed:
            return changed

        if "model" in changed:
            await self._disconnect()
            await self._connect()

        return changed

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
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
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
        """Send a flush message to keep the connection alive."""
        if self._websocket and self._websocket.state is State.OPEN:
            msg = {"flush": True}
            await self._websocket.send(json.dumps(msg))

    async def flush_audio(self, context_id: Optional[str] = None):
        """Flush any pending audio synthesis."""
        if not self._websocket or self._websocket.state is State.CLOSED:
            return
        await self._get_websocket().send(json.dumps({"flush": True}))

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
            elif status == "error":
                context_id = self.get_active_audio_context_id()
                await self.push_frame(TTSStoppedFrame(context_id=context_id))
                await self.stop_all_metrics()
                await self.push_error(error_msg=f"Smallest TTS error: {msg.get('error', msg)}")
            else:
                logger.warning(f"{self} unknown message status: {msg}")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
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
