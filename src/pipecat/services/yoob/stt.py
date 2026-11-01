#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Yoob speech-to-text (Undertone) for Pipecat.

Streams the user's audio to ``wss://tts-api.yoob.com/v1/audio/transcriptions/stream`` (16 kHz s16le mono) and
pushes interim and final transcriptions. A VAD stop closes the current utterance at once, so the final arrives
without waiting for the server's own end-of-speech. Each final's ``result`` carries Yoob's extra fields: the
tagged caption (sound tags such as ``[laughing]``), ``emotion`` and ``emotion_probs``, and word times.
"""

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from loguru import logger
from websockets.protocol import State

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

YOOB_BASE_URL = "https://tts-api.yoob.com/v1"
YOOB_STT_SAMPLE_RATE = 16000
# Measured 2026-11-01: a final arrives ~0.3 s after the utterance is closed.
YOOB_TTFS_P99 = 0.6
# How long a graceful end waits for the last results after asking for them.
_FINISH_TIMEOUT_S = 3.0
# Fields the stream carries for Yoob's own monitoring; left out of the frames' result.
_INTERNAL_KEYS = ("encoder", "timing", "commit_ms")


@dataclass
class YoobSTTSettings(STTSettings):
    """Settings for YoobSTTService."""


class YoobSTTService(WebsocketSTTService):
    """Yoob live transcription with interim results, emotion and sound tags."""

    Settings = YoobSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        sample_rate: int | None = None,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = YOOB_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Yoob STT service.

        Args:
            api_key: Defaults to ``YOOB_API_KEY``. Keys come from https://app.yoob.com.
            base_url: Defaults to ``YOOB_BASE_URL`` or https://tts-api.yoob.com/v1.
            sample_rate: Input rate. Defaults to the pipeline's; audio is resampled to 16 kHz.
            settings: Runtime settings (model, language).
            ttfs_p99_latency: P99 time from speech end to final transcript, in seconds.
            **kwargs: Passed to ``WebsocketSTTService``.
        """
        default_settings = self.Settings(model="yoob-asr-turbo-v1", language=Language.EN.value)
        if settings is not None:
            default_settings.apply_update(settings)
        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            keepalive_timeout=20,
            keepalive_interval=5,
            settings=default_settings,
            **kwargs,
        )
        self._api_key = api_key or os.environ.get("YOOB_API_KEY") or ""
        if not self._api_key:
            raise ValueError("Yoob API key is required: pass api_key or set YOOB_API_KEY")
        base = (base_url or os.environ.get("YOOB_BASE_URL") or YOOB_BASE_URL).rstrip("/")
        self._ws_url = base.replace("https://", "wss://", 1).replace("http://", "ws://", 1) + (
            "/audio/transcriptions/stream"
        )
        self._receive_task = None
        self._resampler = create_stream_resampler()
        # the newest partial not yet covered by a final
        self._last_partial: dict[str, Any] | None = None
        self._done: asyncio.Event | None = None

    def can_generate_metrics(self) -> bool:
        """Yoob STT reports TTFB and usage metrics."""
        return True

    async def setup(self, setup: FrameProcessorSetup):
        """Connect when the pipeline sets the service up."""
        await super().setup(setup)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Ask for the last results, then disconnect on a graceful end."""
        await self._finish()
        await super().stop(frame)
        await self._disconnect()

    async def _finish(self):
        ws = self._websocket
        if not ws or ws.state is not State.OPEN or not self._receive_task:
            return
        # The server closes the socket after "done"; that close must not trigger a reconnect.
        self._disconnecting = True
        self._done = asyncio.Event()
        try:
            await ws.send(json.dumps({"type": "stop"}))
            await asyncio.wait_for(self._done.wait(), _FINISH_TIMEOUT_S)
        except Exception as e:
            logger.debug(f"{self}: no end of stream from Yoob ({type(e).__name__})")
        await self._flush_partial()

    async def cancel(self, frame: CancelFrame):
        """Disconnect on cancel."""
        await super().cancel(frame)
        await self._disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Close the current utterance when local VAD says the user stopped speaking."""
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._websocket and self._websocket.state is State.OPEN:
                await self._websocket.send(json.dumps({"type": "end"}))

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Send audio to the socket; transcripts arrive on the receive task."""
        if not self._websocket or self._websocket.state is not State.OPEN:
            yield None
            return
        if self.sample_rate != YOOB_STT_SAMPLE_RATE:
            audio = await self._resampler.resample(audio, self.sample_rate, YOOB_STT_SAMPLE_RATE)
        try:
            if audio:
                await self._websocket.send(audio)
        except Exception as e:
            logger.warning(f"{self}: send failed: {e}")
        yield None

    async def _send_keepalive(self, silence: bytes):
        # A heartbeat instead of silent audio: nothing is billed. Only audio resets the API's 3-minute idle close,
        # after which the service reconnects.
        if self._websocket is not None:
            await self._websocket.send(json.dumps({"type": "hb"}))

    async def _connect(self):
        await self._connect_websocket()
        await super()._connect()
        if not self._receive_task:
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
            logger.debug("Connecting to Yoob STT")
            self._websocket = await self._websocket_connect(
                self._ws_url, additional_headers={"Authorization": f"Bearer {self._api_key}"}
            )
            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            await self.push_error(error_msg=f"Unable to connect to Yoob: {e}", exception=e)

    async def _disconnect_websocket(self):
        ws = self._websocket
        try:
            if ws and ws.state is State.OPEN:
                logger.debug(f"{self}: Disconnecting from Yoob STT")
                await ws.send(json.dumps({"type": "stop"}))
                await ws.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            if self._websocket is ws:
                self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        try:
            async for message in self._get_websocket():
                if isinstance(message, bytes):
                    continue
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    logger.warning(f"{self}: non-JSON message from Yoob ({len(message)} chars)")
                    continue
                try:
                    await self._process_response(data)
                except Exception as e:
                    logger.error(f"{self}: error processing a Yoob message: {type(e).__name__}")
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._flush_partial()
            raise
        # The socket closed: keep the words heard so far instead of dropping them with the connection.
        await self._flush_partial()

    async def _flush_partial(self):
        data, self._last_partial = self._last_partial, None
        if data:
            await self._on_transcript(data, is_final=True)

    async def _process_response(self, data: dict[str, Any]):
        kind = data.get("type")
        if kind == "error":
            if data.get("code") == "idle_timeout":
                # The API closes a session after 3 minutes without audio; the service reconnects on its own.
                logger.debug(f"{self}: Yoob closed an idle session; reconnecting")
                return
            await self.push_error(error_msg=f"Yoob: {data.get('message') or data.get('code')}")
        elif kind == "partial":
            if data.get("text"):
                self._last_partial = data
            await self._on_transcript(data, is_final=False)
        elif kind == "final":
            self._last_partial = None
            await self._on_transcript(data, is_final=True)
        elif kind == "done":
            if self._done:
                self._done.set()

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        pass

    async def _on_transcript(self, data: dict[str, Any], *, is_final: bool):
        transcript = str(data.get("text") or "")
        if not transcript:
            return
        result = {k: v for k, v in data.items() if k not in _INTERNAL_KEYS}
        if is_final:
            await self.emit_stt_usage_metrics()
            await self.push_frame(
                TranscriptionFrame(
                    transcript, self._user_id, time_now_iso8601(), Language.EN, result=result
                )
            )
            await self._handle_transcription(transcript, True, Language.EN)
        else:
            await self.push_frame(
                InterimTranscriptionFrame(
                    transcript, self._user_id, time_now_iso8601(), Language.EN, result=result
                )
            )
