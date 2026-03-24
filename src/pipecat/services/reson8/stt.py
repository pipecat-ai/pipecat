#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Reson8 speech-to-text service implementation."""

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional
from urllib.parse import urlencode

from loguru import logger

from pipecat import version as pipecat_version
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven
from pipecat.services.stt_latency import RESON8_TTFS_P99
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Reson8, you need to `pip install pipecat-ai[reson8]`.")
    raise Exception(f"Missing module: {e}")


@dataclass
class Reson8STTSettings(STTSettings):
    """Settings for Reson8 STT service.

    Parameters:
        include_timestamps: Word-level timestamps.
        include_words: Individual word results.
        include_confidence: Confidence scores.
        include_interim: Interim (partial) transcription results.
        custom_model_id: Custom model ID for transcription bias.
    """

    include_timestamps: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    include_words: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    include_confidence: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    include_interim: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    custom_model_id: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class Reson8STTService(WebsocketSTTService):
    """Speech-to-Text service using Reson8's WebSocket API.

    This service connects to Reson8's WebSocket API for real-time transcription.
    """

    _settings: Reson8STTSettings

    def __init__(
        self,
        *,
        api_key: str,
        api_url: str = "https://api.reson8.dev",
        sample_rate: Optional[int] = None,
        include_timestamps: Optional[bool] = None,
        include_words: Optional[bool] = None,
        include_confidence: Optional[bool] = None,
        include_interim: Optional[bool] = True,
        custom_model_id: Optional[str] = None,
        ttfs_p99_latency: Optional[float] = RESON8_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Reson8 STT service.

        Args:
            api_key: Reson8 API key.
            api_url: Reson8 API base URL.
            sample_rate: Audio sample rate.
            include_timestamps: Enable word-level timestamps.
            include_words: Enable individual word results.
            include_confidence: Enable confidence scores.
            include_interim: Enable interim (partial) transcription results.
            custom_model_id: Custom model ID for transcription bias.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
            **kwargs: Additional arguments passed to the STTService.
        """
        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            settings=Reson8STTSettings(
                model=None,
                language=None,
                include_timestamps=include_timestamps,
                include_words=include_words,
                include_confidence=include_confidence,
                include_interim=include_interim,
                custom_model_id=custom_model_id,
            ),
            **kwargs,
        )

        self._api_key = api_key
        self._api_url = api_url

        self._receive_task = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True

    async def start(self, frame: StartFrame):
        """Start the Reson8 STT websocket connection."""
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Reson8 STT websocket connection."""
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Reson8 STT websocket connection."""
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Send audio data to Reson8 STT service."""
        if self._websocket and self._websocket.state is State.OPEN:
            await self._websocket.send(audio)

        yield None

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Optional[str] = None
    ):
        pass

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, sending flush requests on VAD stop."""
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._websocket and self._websocket.state is State.OPEN:
                flush_id = str(uuid.uuid4())
                flush_msg = json.dumps({"type": "flush_request", "id": flush_id})
                await self._websocket.send(flush_msg)
                logger.debug(f"Sent flush_request (id={flush_id}) on: {frame.name=}, {direction=}")

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
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Reson8 STT")

            ws_url = self._build_ws_url()
            self._websocket = await websocket_connect(
                ws_url,
                additional_headers={
                    "Authorization": f"ApiKey {self._api_key}",
                    "User-Agent": f"pipecat/{pipecat_version()}",
                },
            )

            if not self._websocket:
                await self.push_error(error_msg=f"Unable to connect to Reson8 API at {ws_url}")
                raise Exception(f"Unable to connect to Reson8 API at {ws_url}")

            await self._call_event_handler("on_connected")
            logger.debug("Connected to Reson8 STT")
        except Exception as e:
            await self.push_error(error_msg=f"Unable to connect to Reson8: {e}", exception=e)
            raise

    async def _disconnect_websocket(self):
        try:
            if self._websocket:
                logger.debug("Disconnecting from Reson8 STT")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
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
                if isinstance(message, bytes):
                    continue

                msg = json.loads(message)
                msg_type = msg.get("type")

                if msg_type == "transcript":
                    text = msg.get("text", "")
                    is_final = msg.get("is_final", True)

                    result = {}
                    if "start_ms" in msg:
                        result["start_ms"] = msg["start_ms"]
                    if "duration_ms" in msg:
                        result["duration_ms"] = msg["duration_ms"]
                    if "words" in msg:
                        result["words"] = msg["words"]

                    if is_final:
                        await self.push_frame(
                            TranscriptionFrame(
                                text=text,
                                user_id=self._user_id,
                                timestamp=time_now_iso8601(),
                                result=result or None,
                                finalized=True,
                            )
                        )
                        await self._handle_transcription(text, is_final=True)
                        await self.stop_processing_metrics()
                    else:
                        if not text:
                            continue
                        await self.start_processing_metrics()
                        await self.push_frame(
                            InterimTranscriptionFrame(
                                text=text,
                                user_id=self._user_id,
                                timestamp=time_now_iso8601(),
                                result=result or None,
                            )
                        )
                elif msg_type == "flush_confirmation":
                    flush_id = msg.get("id")
                    logger.debug(f"Received flush_confirmation (id={flush_id})")
                elif msg_type == "error":
                    error_msg = msg.get("message", "Unknown error")
                    status = msg.get("status", "")
                    await self.push_error(error_msg=f"Reson8 error ({status}): {error_msg}")
                else:
                    logger.debug(f"Unhandled Reson8 message type: {msg_type}")

            except json.JSONDecodeError:
                logger.warning(f"Received non-JSON message: {message}")
            except Exception as e:
                logger.warning(f"Error processing message: {e}")

    async def _send_keepalive(self, silence: bytes):
        await self._websocket.send(silence)

    async def _update_settings(self, delta: Reson8STTSettings) -> dict[str, Any]:
        """Apply settings delta, reconnecting to apply changes."""
        changed = await super()._update_settings(delta)

        if not changed:
            return changed

        logger.info(f"Reson8 STT settings changed ({', '.join(changed.keys())}), reconnecting")
        await self._disconnect()
        await self._connect()

        return changed

    def _build_ws_url(self) -> str:
        base = self._api_url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
        s = self._settings
        params: dict[str, str] = {
            "encoding": "pcm_s16le",
            "sample_rate": str(self.sample_rate),
            "channels": "1",
        }
        if s.include_timestamps:
            params["include_timestamps"] = "true"
        if s.include_words:
            params["include_words"] = "true"
        if s.include_confidence:
            params["include_confidence"] = "true"
        if s.include_interim:
            params["include_interim"] = "true"
        if s.custom_model_id:
            params["custom_model_id"] = s.custom_model_id

        return f"{base}/v1/speech-to-text/realtime?{urlencode(params)}"
