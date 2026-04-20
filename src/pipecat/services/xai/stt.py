#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""xAI speech-to-text service implementation.

This module provides integration with xAI's real-time speech-to-text WebSocket
API documented at https://docs.x.ai/developers/rest-api-reference/inference/voice.
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any
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
)
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven
from pipecat.services.stt_latency import XAI_TTFS_P99
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use xAI STT, you need to `pip install "pipecat-ai[xai]"`.')
    raise Exception(f"Missing module: {e}")


def language_to_xai_stt_language(language: Language) -> str | None:
    """Convert a Language enum to the xAI STT language code.

    xAI STT accepts two-letter language codes (e.g. ``en``, ``fr``, ``de``,
    ``ja``). When set, the server applies Inverse Text Normalization.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding xAI STT language code, or None if not supported.
    """
    LANGUAGE_MAP = {
        Language.AR: "ar",
        Language.BN: "bn",
        Language.DE: "de",
        Language.EN: "en",
        Language.ES: "es",
        Language.FR: "fr",
        Language.HI: "hi",
        Language.ID: "id",
        Language.IT: "it",
        Language.JA: "ja",
        Language.KO: "ko",
        Language.PT: "pt",
        Language.RU: "ru",
        Language.TR: "tr",
        Language.VI: "vi",
        Language.ZH: "zh",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


@dataclass
class XAISTTSettings(STTSettings):
    """Settings for XAISTTService.

    Parameters:
        interim_results: When True, partial transcripts are emitted
            approximately every 500ms.
        endpointing: Silence duration in milliseconds that triggers a
            speech-final event. Range 0-5000. Server default is 10ms.
        multichannel: When True, transcribes each interleaved channel
            independently. Requires ``channels`` >= 2.
        channels: Number of interleaved channels (2-8). Required when
            ``multichannel`` is True.
        diarize: When True, the server attaches a ``speaker`` field to each
            word identifying the detected speaker.
    """

    interim_results: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    endpointing: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    multichannel: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    channels: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    diarize: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class XAISTTService(WebsocketSTTService):
    """xAI real-time speech-to-text service.

    Streams audio to xAI's WebSocket STT endpoint and emits interim and final
    transcription frames. The ``XAI_API_KEY`` is passed directly as a Bearer
    token on the WebSocket handshake.

    The connection is persistent: audio is streamed continuously and the
    server emits ``transcript.partial`` events with ``is_final`` and
    ``speech_final`` flags to mark utterance boundaries. If the connection
    drops mid-session, the base class reconnects automatically.
    """

    Settings = XAISTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        ws_url: str = "wss://api.x.ai/v1/stt",
        sample_rate: int = 16000,
        encoding: str = "pcm",
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = XAI_TTFS_P99,
        **kwargs,
    ):
        """Initialize the xAI STT service.

        Args:
            api_key: xAI API key (used as Bearer for the WebSocket handshake).
            ws_url: WebSocket endpoint URL. Defaults to ``wss://api.x.ai/v1/stt``.
            sample_rate: Audio sample rate in Hz. Supported values: 8000,
                16000, 22050, 24000, 44100, 48000. Defaults to 16000.
            encoding: Audio encoding. One of ``"pcm"`` (signed 16-bit LE),
                ``"mulaw"``, or ``"alaw"``. Defaults to ``"pcm"``.
            settings: Runtime-updatable settings overriding defaults.
            ttfs_p99_latency: P99 latency from speech end to final transcript
                in seconds. See https://github.com/pipecat-ai/stt-benchmark.
            **kwargs: Additional arguments passed to WebsocketSTTService.
        """
        default_settings = self.Settings(
            model=None,
            language=Language.EN,
            interim_results=True,
            endpointing=None,
            multichannel=None,
            channels=None,
            diarize=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            settings=default_settings,
            ttfs_p99_latency=ttfs_p99_latency,
            **kwargs,
        )

        self._api_key = api_key
        self._ws_url = ws_url
        self._encoding = encoding

        self._receive_task: asyncio.Task | None = None
        self._session_ready = asyncio.Event()

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            True if metrics generation is supported.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to the xAI STT language code."""
        return language_to_xai_stt_language(language)

    async def _update_settings(self, delta: Settings) -> dict[str, Any]:
        """Apply a settings delta and reconnect to apply changes.

        xAI STT configures the session via WebSocket query parameters, so any
        change requires a fresh connection.
        """
        changed = await super()._update_settings(delta)
        if not changed:
            return changed
        await self._disconnect()
        await self._connect()
        return changed

    async def start(self, frame: StartFrame):
        """Start the speech-to-text service."""
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the speech-to-text service."""
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the speech-to-text service."""
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Forward raw audio bytes to the xAI STT WebSocket.

        Transcription frames are pushed from the receive task, not yielded
        from this coroutine.
        """
        if self._websocket and self._websocket.state is State.OPEN and self._session_ready.is_set():
            try:
                await self._websocket.send(audio)
            except Exception as e:
                await self.push_error(error_msg=f"xAI STT send failed: {e}", exception=e)
        yield None

    def _build_ws_url(self) -> str:
        """Build the WebSocket URL with session query parameters."""
        s = self._settings
        params: dict[str, Any] = {
            "sample_rate": self.sample_rate,
            "encoding": self._encoding,
        }

        if s.language is not None:
            params["language"] = s.language

        optional_fields = {
            "interim_results": s.interim_results,
            "endpointing": s.endpointing,
            "multichannel": s.multichannel,
            "channels": s.channels,
            "diarize": s.diarize,
        }
        for key, val in optional_fields.items():
            if val is None:
                continue
            if isinstance(val, bool):
                params[key] = str(val).lower()
            else:
                params[key] = val

        return f"{self._ws_url}?{urlencode(params)}"

    async def _connect(self):
        """Establish the WebSocket connection and start the receive task."""
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Tear down the WebSocket connection and cancel the receive task."""
        await super()._disconnect()
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                await self._websocket.send(json.dumps({"type": "audio.done"}))
        except Exception as e:
            logger.debug(f"{self} error sending audio.done during disconnect: {e}")

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Open a WebSocket connection to the xAI STT endpoint."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to xAI STT WebSocket")
            self._session_ready.clear()

            ws_url = self._build_ws_url()
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": f"xAI/1.0 (integration=Pipecat/{pipecat_version()})",
            }
            self._websocket = await websocket_connect(ws_url, additional_headers=headers)
            await self._call_event_handler("on_connected")
            logger.debug(f"{self} connected to xAI STT WebSocket")
        except Exception as e:
            await self.push_error(error_msg=f"Unable to connect to xAI STT: {e}", exception=e)
            raise

    async def _disconnect_websocket(self):
        """Close the WebSocket connection."""
        try:
            if self._websocket:
                logger.debug("Disconnecting from xAI STT WebSocket")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing xAI STT websocket: {e}", exception=e)
        finally:
            self._websocket = None
            self._session_ready.clear()
            await self._call_event_handler("on_disconnected")

    async def _receive_messages(self):
        """Receive and dispatch xAI STT WebSocket messages."""
        if not self._websocket:
            raise Exception("Websocket not connected")
        async for message in self._websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"{self} received non-JSON message: {message}")
                continue
            await self._handle_message(data)

    async def _handle_message(self, message: dict[str, Any]):
        """Branch on xAI STT event type."""
        msg_type = message.get("type")

        if msg_type == "transcript.created":
            self._session_ready.set()
            logger.debug(f"{self} xAI STT session ready")
        elif msg_type == "transcript.partial":
            await self._handle_transcript(message)
        elif msg_type == "transcript.done":
            if message.get("text"):
                await self._push_final_transcript(message, speech_final=True)
        elif msg_type == "error":
            await self.push_error(
                error_msg=f"xAI STT error: {message.get('message', message)}",
                exception=Exception(message),
            )
        else:
            logger.debug(f"{self} unhandled xAI STT message: {message}")

    async def _handle_transcript(self, message: dict[str, Any]):
        text = message.get("text", "")
        if not text:
            return

        is_final = bool(message.get("is_final"))
        speech_final = bool(message.get("speech_final"))
        language = self._language_for_frame()

        if is_final:
            await self._push_final_transcript(
                message, speech_final=speech_final, language=language, text=text
            )
        else:
            await self.push_frame(
                InterimTranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    language,
                    result=message,
                )
            )

    async def _push_final_transcript(
        self,
        message: dict[str, Any],
        *,
        speech_final: bool,
        language: Language | None = None,
        text: str | None = None,
    ):
        text = text if text is not None else message.get("text", "")
        if not text:
            return
        language = language if language is not None else self._language_for_frame()

        await self.push_frame(
            TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                language,
                result=message,
                finalized=speech_final,
            )
        )
        await self._trace_transcription(text, True, language)
        if speech_final:
            await self.stop_processing_metrics()

    def _language_for_frame(self) -> Language:
        """Return a Language enum suitable for transcription frames.

        Settings stores the service-specific string (e.g. ``"en"``); frames
        carry the enum value.
        """
        lang = self._settings.language
        if isinstance(lang, Language):
            return lang
        if isinstance(lang, str):
            try:
                return Language(lang)
            except ValueError:
                return Language.EN
        return Language.EN

    @traced_stt
    async def _trace_transcription(self, transcript: str, is_final: bool, language: Language):
        """Record transcription event for tracing."""
        pass
