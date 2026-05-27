#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""SLNG speech-to-text services."""

import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
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
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven, is_given
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use SLNG STT, you need to `pip install pipecat-ai[slng]`.")
    raise Exception(f"Missing module: {e}")


@dataclass
class SlngSTTSettings(STTSettings):
    """Settings for SlngSTTService.

    Parameters:
        language: Language for speech recognition.
        enable_vad: Whether to enable server-side VAD.
        enable_partials: Whether to receive partial (interim) transcriptions.
    """

    enable_vad: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    enable_partials: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class SlngSTTService(WebsocketSTTService):
    """Speech-to-text service using the SLNG bridge WebSocket API.

    Provides real-time speech transcription through a WebSocket connection
    to the SLNG STT bridge. Supports partial (interim) and final transcriptions,
    server-side VAD, and configurable models and languages.

    The SLNG protocol requires an ``init`` message immediately after connection
    followed by base64-encoded audio frames. Finalization is triggered on
    ``VADUserStoppedSpeakingFrame`` via a ``finalize`` message.
    """

    Settings = SlngSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "slng/deepgram/nova:3-en",
        base_url: str = "api.slng.ai",
        encoding: str = "linear16",
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize SlngSTTService.

        Args:
            api_key: Authentication key for the SLNG API.
            model: The transcription model to use. Defaults to "slng/deepgram/nova:3-en".
            base_url: The API host (without scheme). Defaults to "api.slng.ai".
            encoding: Audio encoding format. Defaults to "linear16".
            sample_rate: Audio sample rate in Hz. If None, uses the pipeline sample rate.
            settings: Runtime-updatable settings override.
            **kwargs: Additional arguments passed to parent WebsocketSTTService.
        """
        default_settings = self.Settings(
            model=model,
            language=Language.EN,
            enable_vad=True,
            enable_partials=True,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            keepalive_timeout=120,
            keepalive_interval=30,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url
        self._encoding = encoding
        self._receive_task = None

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate processing metrics.

        Returns:
            True, indicating metrics are supported.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the STT service and establish the WebSocket connection.

        Args:
            frame: Frame indicating service should start.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the STT service and close the WebSocket connection.

        Args:
            frame: Frame indicating service should stop.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the STT service and close the WebSocket connection.

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
            await self.start_processing_metrics()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._websocket and self._websocket.state is State.OPEN:
                await self._websocket.send(json.dumps({"type": "finalize"}))

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Process audio data for speech-to-text transcription.

        Encodes audio as base64 and sends it over the WebSocket connection.

        Args:
            audio: Raw PCM audio bytes to transcribe.

        Yields:
            None — transcription results are delivered via WebSocket responses.
        """
        if not self._websocket or self._websocket.state is not State.OPEN:
            await self._connect()

        if self._websocket is None:
            logger.warning(f"{self}: websocket unavailable after reconnect, dropping audio")
            yield None
            return

        try:
            msg = json.dumps({"type": "audio", "data": base64.b64encode(audio).decode("utf-8")})
            await self._websocket.send(msg)
        except Exception as e:
            logger.warning(f"{self}: send failed: {e}")
        yield None

    async def _send_keepalive(self, silence: bytes):
        """Send a JSON keepalive message instead of raw silence bytes.

        Args:
            silence: Silent PCM bytes (ignored; a JSON keepalive is sent instead).
        """
        if self._websocket is None:
            return
        try:
            await self._websocket.send(json.dumps({"type": "keepalive"}))
        except Exception as e:
            logger.warning(f"{self}: keepalive send failed: {e}")

    async def _connect(self):
        await self._connect_websocket()
        await super()._connect()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish the WebSocket connection and send the init message."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            logger.debug(f"Connecting to SLNG STT ({self._settings.model})")

            model = self._settings.model or "slng/deepgram/nova:3-en"
            if "://" in self._base_url:
                ws_url = f"{self._base_url}/v1/bridges/unmute/stt/{model}"
            else:
                ws_url = f"wss://{self._base_url}/v1/bridges/unmute/stt/{model}"
            headers = {"Authorization": f"Bearer {self._api_key}"}

            self._websocket = await websocket_connect(ws_url, additional_headers=headers)

            config: dict[str, Any] = {
                "sample_rate": self.sample_rate,
                "encoding": self._encoding,
            }

            if is_given(self._settings.language) and self._settings.language is not None:
                config["language"] = str(self._settings.language)

            if is_given(self._settings.enable_vad):
                config["enable_vad"] = self._settings.enable_vad

            if is_given(self._settings.enable_partials):
                config["enable_partials"] = self._settings.enable_partials

            init_msg = {
                "type": "init",
                "config": config,
            }
            await self._websocket.send(json.dumps(init_msg))
            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            await self.push_error(error_msg=f"Unable to connect to SLNG STT: {e}", exception=e)

    async def _disconnect_websocket(self):
        """Send a close message and shut down the WebSocket."""
        ws = self._websocket
        try:
            if ws and ws.state is State.OPEN:
                logger.debug("Disconnecting from SLNG STT")
                await ws.send(json.dumps({"type": "close"}))
                await ws.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing SLNG STT websocket: {e}", exception=e)
        finally:
            if self._websocket is ws:
                self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("SLNG STT websocket not connected")

    async def _receive_messages(self):
        """Receive and dispatch incoming WebSocket messages."""
        async for message in self._get_websocket():
            try:
                data = json.loads(message)
                await self._process_message(data)
            except json.JSONDecodeError:
                logger.warning(f"{self}: received non-JSON message: {message}")
            except Exception as e:
                logger.error(f"{self}: error processing message: {e}")

    async def _process_message(self, data: dict[str, Any]):
        """Dispatch a decoded server message.

        Args:
            data: Decoded JSON payload from the server.
        """
        msg_type = data.get("type")

        if msg_type == "ready":
            session_id = data.get("session_id", "")
            logger.debug(f"{self}: SLNG STT session ready (id={session_id})")

        elif msg_type == "partial_transcript":
            transcript = data.get("transcript", "")
            if transcript:
                language = None
                if raw_lang := data.get("language"):
                    try:
                        language = Language(raw_lang)
                    except ValueError:
                        pass
                await self.push_frame(
                    InterimTranscriptionFrame(
                        transcript,
                        self._user_id,
                        time_now_iso8601(),
                        language,
                        result=data,
                    )
                )

        elif msg_type == "final_transcript":
            transcript = data.get("transcript", "")
            language = None
            if "language" in data:
                try:
                    language = Language(data["language"])
                except (ValueError, KeyError):
                    pass
            if transcript:
                await self.push_frame(
                    TranscriptionFrame(
                        transcript,
                        self._user_id,
                        time_now_iso8601(),
                        language,
                        result=data,
                    )
                )
                await self._handle_transcription(transcript, True, language)
                await self.stop_processing_metrics()

        elif msg_type == "error":
            error_msg = data.get("message", "Unknown SLNG STT error")
            logger.error(f"{self}: SLNG STT error: {error_msg}")
            await self.push_error(error_msg=error_msg)
            await self.stop_all_metrics()

        else:
            logger.debug(f"{self}: unknown message type: {msg_type}")

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing.

        Args:
            transcript: The transcribed text.
            is_final: Whether this is a final (not interim) transcription.
            language: Detected or configured language.
        """
        pass

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect if needed.

        Args:
            delta: A settings delta to apply.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)
        if not changed:
            return changed
        await self._request_reconnect()
        return changed
