#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Palabra AI realtime speech-to-text service implementation."""

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    TranslationFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_latency import PALABRA_TTFS_P99
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.types import NOT_GIVEN, NotGiven, is_given

DEFAULT_URL = "wss://stream.palabra.ai/asr/v1/speech-to-text/stream"
FINALIZE_MESSAGE = json.dumps({"message_type": "finalize"})
FINALIZED_MARKER = "<fin>"


def _to_language(code: str | None) -> Language | None:
    if not code:
        return None
    try:
        return Language(code)
    except ValueError:
        return None


def _strip_finalized_marker(text: str) -> tuple[str, bool]:
    assert FINALIZED_MARKER not in text or text.endswith(FINALIZED_MARKER), (
        "Palabra text contains <fin> before the end of the text"
    )
    if not text.endswith(FINALIZED_MARKER):
        return text, False
    return text.removesuffix(FINALIZED_MARKER), True


@dataclass
class PalabraSTTSettings(STTSettings):
    """Settings for :class:`PalabraSTTService`.

    All Palabra settings are sent as WebSocket query parameters. Updating any
    setting reconnects the stream.

    Parameters:
        translate_languages: Target languages for text translations. Every
            finalized segment is translated into each of them and pushed as a
            :class:`~pipecat.frames.frames.TranslationFrame`. ``None`` disables
            translation.
        enable_filler_filter: Whether Palabra removes filler words.
        finalization_mode: ``"auto"`` or ``"manual"`` finalization mode.
        finalization_timeout: Safety timeout used by manual finalization mode.
    """

    translate_languages: list[Language] | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    enable_filler_filter: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    finalization_mode: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    finalization_timeout: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class PalabraSTTService(WebsocketSTTService):
    """Stream raw PCM audio to Palabra's realtime transcription API.

    Incoming Pipecat audio frames are sent immediately without client-side
    batching. When ``vad_force_turn_endpoint`` is enabled, a local VAD stop
    sends Palabra's ``finalize`` command. Palabra acknowledges each command by
    appending ``<fin>`` to a transcription response.

    API documentation:
    https://docs.palabra.ai/docs/streaming_api/realtime_stt
    """

    Settings = PalabraSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = DEFAULT_URL,
        sample_rate: int | None = None,
        vad_force_turn_endpoint: bool = True,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = PALABRA_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Palabra STT service.

        Args:
            api_key: Palabra API key.
            url: Palabra realtime STT WebSocket URL.
            sample_rate: Audio sample rate. If ``None``, use the pipeline rate.
            vad_force_turn_endpoint: Send ``finalize`` on every local VAD stop.
            settings: Runtime-updatable Palabra settings.
            ttfs_p99_latency: P99 speech-end-to-final-transcript latency.
            **kwargs: Additional arguments passed to ``WebsocketSTTService``.
        """
        default_settings = self.Settings(
            model=None,
            language=None,
            translate_languages=None,
            enable_filler_filter=None,
            finalization_mode=None,
            finalization_timeout=None,
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
        self._url = url
        self._vad_force_turn_endpoint = vad_force_turn_endpoint
        self._receive_task = None

    def can_generate_metrics(self) -> bool:
        """Return whether this service can generate processing metrics."""
        return True

    async def start(self, frame: StartFrame):
        """Start the service and open its WebSocket connection.

        Args:
            frame: The start frame.
        """
        await super().start(frame)
        await self._connect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Send one raw PCM audio frame to Palabra immediately.

        Args:
            audio: Raw 16-bit mono PCM bytes.

        Yields:
            None. Results arrive asynchronously on the receive task.
        """
        if self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._websocket.send(audio)
            except Exception as e:
                logger.warning(f"{self}: audio send failed: {e}")
        yield None

    async def _send_finalize(self):
        if not self._websocket or self._websocket.state is not State.OPEN:
            return
        try:
            self.request_finalize()
            await self._websocket.send(FINALIZE_MESSAGE)
        except Exception as e:
            logger.warning(f"{self}: finalize failed: {e}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and finalize the current segment on VAD stop.

        Args:
            frame: Frame to process.
            direction: Frame direction.
        """
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStoppedSpeakingFrame) and self._vad_force_turn_endpoint:
            await self._send_finalize()

    def _build_url(self) -> str:
        settings = self._settings
        params: dict[str, str] = {
            "token": self._api_key,
            "format": "pcm_s16le",
            "sample_rate": str(self.sample_rate),
        }
        if settings.translate_languages:
            codes = [lang.value.lower() for lang in settings.translate_languages]
            params["translate_languages"] = ",".join(dict.fromkeys(codes))
        if settings.enable_filler_filter is not None:
            params["enable_filler_filter"] = "true" if settings.enable_filler_filter else "false"
        if settings.finalization_mode is not None and is_given(settings.finalization_mode):
            params["finalization_mode"] = settings.finalization_mode
        if settings.finalization_timeout is not None:
            params["finalization_timeout"] = str(settings.finalization_timeout)
        return f"{self._url}?{urlencode(params, safe=',')}"

    async def _connect(self):
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(
                self._receive_task_handler(self._report_error), name="palabra_receive"
            )

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

            logger.debug("Connecting to Palabra STT")
            self._websocket = await self._websocket_connect(self._build_url())
            await self._call_event_handler("on_connected")
            logger.debug("Connected to Palabra STT")
        except Exception as e:
            self._websocket = None
            await self.push_error(error_msg=f"Unable to connect to Palabra STT: {e}", exception=e)
            await self._call_event_handler("on_connection_error", str(e))

    async def _disconnect_websocket(self):
        try:
            if self._websocket:
                logger.debug("Disconnecting from Palabra STT")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing Palabra websocket: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise RuntimeError("Palabra websocket is not connected")

    async def _receive_messages(self):
        async for message in self._get_websocket():
            try:
                content = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"{self}: received non-JSON Palabra message: {message!r}")
                continue

            message_type = content.get("message_type")
            if message_type == "transcription":
                await self._handle_transcription_message(content)
            elif message_type == "translated_transcription":
                await self._handle_translation_message(content)
            elif message_type == "error":
                data = content.get("data") or {}
                await self.push_error(
                    error_msg=f"Palabra STT error {data.get('code')}: {data.get('desc')}"
                )
            else:
                logger.trace(f"{self}: ignoring Palabra message {message_type}")

    async def _handle_transcription_message(self, content: dict[str, Any]):
        segment = content.get("segment") or {}
        text, acknowledged = _strip_finalized_marker(segment.get("text", ""))
        language = _to_language(content.get("language"))

        if text.strip():
            frame_type = TranscriptionFrame if acknowledged else InterimTranscriptionFrame
            frame = frame_type(
                text=text,
                user_id=self._user_id,
                timestamp=time_now_iso8601(),
                language=language,
                result=content,
            )
            if acknowledged:
                self.confirm_finalize()
            await self.push_frame(frame)

    async def _handle_translation_message(self, content: dict[str, Any]):
        segment = content.get("segment") or {}
        text = segment.get("text", "")
        if not text:
            return
        await self.push_frame(
            TranslationFrame(
                text=text,
                user_id=self._user_id,
                timestamp=time_now_iso8601(),
                language=_to_language(content.get("language")),
            )
        )
