#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Telnyx text-to-speech service.

Provides streaming TTS via the Telnyx WebSocket API at
wss://api.telnyx.com/v2/text-to-speech/speech.

Protocol:
  - Connect with Authorization: Bearer <key> header.
  - Send init frame {"text": " ", "voice_settings": {"voice_speed": <speed>}}.
  - Send text frames {"text": "..."} for each synthesis request.
  - Receive audio frames with base64-encoded PCM audio ({"audio": "<b64>"}).
  - Receive a final frame ({"isFinal": true}) when synthesis is done.
  - Receive an error frame ({"error": "..."}) on failure.
  - Send {"force": true} to interrupt mid-stream.
"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven, assert_given
from pipecat.services.tts_service import WebsocketTTSService
from pipecat.transcriptions.language import Language


@dataclass
class TelnyxTTSSettings(TTSSettings):
    """Settings for TelnyxTTSService.

    Parameters:
        voice: Telnyx voice ID (e.g., "Telnyx.NaturalHD.astra").
        speed: Speech speed multiplier (0.5 to 2.0).
    """

    voice: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    speed: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class TelnyxTTSService(WebsocketTTSService):
    """Telnyx streaming text-to-speech over WebSocket.

    Sends JSON text frames, receives base64-encoded PCM audio. The WebSocket
    stays open for the session so consecutive sentences reuse the connection.

    Telnyx TTS does not multiplex concurrent contexts over a single socket.
    Audio for the most recent run_tts call is tagged with that call's
    context_id. If a second run_tts arrives before the first finishes, the
    service sends a ``{"force": true}`` frame to interrupt the in-flight
    synthesis.

    Example::

        tts = TelnyxTTSService(
            api_key="your-telnyx-api-key",
            voice="Telnyx.NaturalHD.astra",
            settings=TelnyxTTSService.Settings(
                speed=1.0,
            ),
        )
    """

    Settings = TelnyxTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        voice: str = "Telnyx.NaturalHD.astra",
        speed: float = 1.0,
        sample_rate: int = 24000,
        settings: Settings | None = None,
        **kwargs: Any,
    ):
        """Initialize the Telnyx TTS service.

        Args:
            api_key: Telnyx API key for authentication.
            voice: Telnyx voice ID (e.g., "Telnyx.NaturalHD.astra").
            speed: Speech speed multiplier (0.5 to 2.0).
            sample_rate: Audio sample rate in Hz.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to WebsocketTTSService.
        """
        default_settings = self.Settings(
            voice=voice,
            speed=speed,
            language=None,
        )

        if settings:
            default_settings.apply_update(settings)

        super().__init__(
            push_start_frame=True,
            push_stop_frames=False,
            sample_rate=sample_rate,
            **kwargs,
        )
        self._api_key = api_key
        self._voice = voice
        self._speed = speed
        self._sample_rate = sample_rate
        self._receive_task = None
        self._current_context_id: str | None = None

    def _build_url(self) -> str:
        return (
            f"wss://api.telnyx.com/v2/text-to-speech/speech"
            f"?voice={self._voice}"
            f"&audio_format=linear16"
            f"&sample_rate={self._sample_rate}"
        )

    async def start(self, frame: Frame):
        """Start the TTS service and connect to the WebSocket."""
        await super().start(frame)
        await self._connect()

    async def _connect(self):
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            logger.debug("Connecting to Telnyx TTS")
            self._websocket = await websocket_connect(
                self._build_url(),
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
            )
            # Send the init handshake with voice_settings.
            init_frame = {
                "text": " ",
                "voice_settings": {"voice_speed": self._speed},
            }
            await self._websocket.send(json.dumps(init_frame))
            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            if self._websocket:
                logger.debug("Disconnecting from Telnyx TTS")
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
                logger.warning(f"Telnyx TTS: non-JSON message: {message}")
                continue

            ctx_id = self._current_context_id

            if msg.get("error"):
                error_msg = msg["error"]
                logger.error(f"Telnyx TTS error: {error_msg}")
                if ctx_id and self.audio_context_available(ctx_id):
                    await self.append_to_audio_context(ctx_id, TTSStoppedFrame(context_id=ctx_id))
                    await self.remove_audio_context(ctx_id)
                await self.push_error(error_msg=error_msg)
                continue

            if msg.get("isFinal"):
                if ctx_id and self.audio_context_available(ctx_id):
                    await self.append_to_audio_context(ctx_id, TTSStoppedFrame(context_id=ctx_id))
                    await self.remove_audio_context(ctx_id)
                continue

            audio_b64 = msg.get("audio")
            if not audio_b64 or not ctx_id or not self.audio_context_available(ctx_id):
                continue

            audio_data = base64.b64decode(audio_b64)
            frame = TTSAudioRawFrame(
                audio=audio_data,
                sample_rate=self.sample_rate,
                num_channels=1,
                context_id=ctx_id,
            )
            await self.append_to_audio_context(ctx_id, frame)

    async def on_audio_context_interrupted(self, context_id: str):
        """Handle interruption by sending a force frame to stop synthesis."""
        if self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._websocket.send(json.dumps({"force": True}))
            except Exception as e:
                logger.warning(f"Telnyx TTS: failed to send force frame: {e}")
        await super().on_audio_context_interrupted(context_id)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Send text to the Telnyx TTS WebSocket for synthesis."""
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            if self._websocket is None:
                yield ErrorFrame(error="websocket unavailable")
                return

            self._current_context_id = context_id

            try:
                await self._websocket.send(json.dumps({"text": text}))
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()
                return

            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
