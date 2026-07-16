#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Telnyx speech-to-text service.

Provides streaming STT via the Telnyx WebSocket API at
wss://api.telnyx.com/v2/speech-to-text/transcription.

Protocol:
  - Connect with Authorization: Bearer <key> header.
  - Send raw 16-bit PCM audio as binary WebSocket frames.
  - Receive JSON text frames with transcript, is_final, confidence, speech_final.
  - Send {"type": "CloseStream"} to end the session gracefully (Deepgram, Speechmatics, Soniox only).
  - Receive {"errors": [...]} on validation or connection errors.
"""

from __future__ import annotations

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
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601


@dataclass
class TelnyxSTTSettings(STTSettings):
    """Settings for TelnyxSTTService.

    Parameters:
        transcription_engine: Telnyx STT engine (Telnyx, Deepgram, Google, Azure).
        input_format: Audio input encoding (linear16, mulaw, alaw).
    """

    transcription_engine: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    input_format: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class TelnyxSTTService(WebsocketSTTService):
    """Telnyx streaming speech-to-text over WebSocket.

    Sends raw PCM audio as binary frames, receives JSON transcription results.
    The WebSocket stays open for the session. Supports interim and final
    transcripts depending on the engine.

    Example::

        stt = TelnyxSTTService(
            api_key="your-telnyx-api-key",
            settings=TelnyxSTTService.Settings(
                transcription_engine="Telnyx",
                input_format="linear16",
            ),
        )
    """

    Settings = TelnyxSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        transcription_engine: str = "Telnyx",
        input_format: str = "linear16",
        sample_rate: int = 16000,
        language: str = "en-US",
        interim_results: bool = True,
        settings: Settings | None = None,
        **kwargs: Any,
    ):
        """Initialize the Telnyx STT service.

        Args:
            api_key: Telnyx API key for authentication.
            transcription_engine: STT engine (Telnyx, Deepgram, Google, Azure).
            input_format: Audio encoding (linear16, mulaw, alaw).
            sample_rate: Audio sample rate in Hz.
            language: BCP-47 language code (e.g., "en-US").
            interim_results: Whether to request interim (partial) transcripts.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to WebsocketSTTService.
        """
        default_settings = self.Settings(
            transcription_engine=transcription_engine,
            input_format=input_format,
            language=None,
        )

        if settings:
            default_settings.apply_update(settings)

        super().__init__(sample_rate=sample_rate, **kwargs)
        self._api_key = api_key
        self._transcription_engine = transcription_engine
        self._input_format = input_format
        self._sample_rate = sample_rate
        self._language = language
        self._interim_results = interim_results
        self._receive_task = None

    def _build_url(self) -> str:
        return (
            "wss://api.telnyx.com/v2/speech-to-text/transcription"
            f"?transcription_engine={self._transcription_engine}"
            f"&input_format={self._input_format}"
            f"&sample_rate={self._sample_rate}"
            f"&language={self._language}"
            f"&interim_results={'true' if self._interim_results else 'false'}"
        )

    async def start(self, frame: Frame):
        """Start the STT service and connect to the WebSocket."""
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
            logger.debug("Connecting to Telnyx STT")
            self._websocket = await websocket_connect(
                self._build_url(),
                additional_headers={"Authorization": f"Bearer {self._api_key}"},
            )
            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            if self._websocket:
                logger.debug("Disconnecting from Telnyx STT")
                if self._transcription_engine in ("Deepgram", "Speechmatics", "Soniox"):
                    try:
                        await self._websocket.send(json.dumps({"type": "CloseStream"}))
                    except Exception:
                        pass
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
                data = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"Telnyx STT: non-JSON message: {message}")
                continue

            if data.get("errors"):
                errors = data["errors"]
                detail = errors[0].get("detail") if errors else "Unknown Telnyx STT error"
                logger.error(f"Telnyx STT error: {detail}")
                await self.push_error(error_msg=detail)
                continue

            if data.get("utterance_end"):
                continue

            is_final = data.get("is_final", False)
            text = data.get("transcript", "").strip()

            if not text:
                continue

            if is_final:
                await self.stop_processing_metrics()
                logger.debug(f"Telnyx final transcript: [{text}]")
                await self.push_frame(
                    TranscriptionFrame(
                        text,
                        self._user_id,
                        time_now_iso8601(),
                        self._language,
                        result=data,
                    )
                )
            else:
                logger.trace(f"Telnyx interim transcript: [{text}]")
                await self.push_frame(
                    InterimTranscriptionFrame(
                        text,
                        self._user_id,
                        time_now_iso8601(),
                        self._language,
                        result=data,
                    )
                )

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Send audio to the Telnyx STT WebSocket for transcription.

        Args:
            audio: Raw PCM audio bytes.

        Yields:
            None -- transcription results arrive via WebSocket messages.
        """
        await self._connected_event.wait()

        if not self._websocket or self._websocket.state is State.CLOSED:
            await self._connect()

        if self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._websocket.send(audio)
            except Exception as e:
                yield ErrorFrame(error=f"Telnyx STT error: {e}")
                return

        yield None
