#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Modulate speech-to-text service implementation.

This module provides integration with Modulate's Velma streaming STT
WebSocket API documented at https://docs.modulate.ai/api-reference/stt.
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

from loguru import logger
from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven, is_given
from pipecat.services.stt_latency import MODULATE_TTFS_P99
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

# Sample rates the streaming API accepts for raw PCM input.
SUPPORTED_SAMPLE_RATES = {8000, 11025, 16000, 22050, 32000, 44100, 48000, 96000}


def language_to_modulate_language(language: Language) -> str:
    """Convert a Language enum to the Modulate STT language code.

    Modulate accepts case-insensitive ISO 639-1 codes and uses only the
    primary language subtag of BCP 47 tags, so region/script subtags are
    stripped (e.g. ``en-US`` becomes ``en``). Modulate does not publish a
    list of supported languages; the code is passed to the service as a
    hint.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding service language code.
    """
    return str(language).split("-")[0].lower()


@dataclass
class ModulateSTTSettings(STTSettings):
    """Settings for ModulateSTTService.

    Parameters:
        partial_results: When True, interim transcripts stream while each
            utterance is in progress. Each interim carries the complete text
            of the current utterance so far and supersedes the previous one.
        speaker_diarization: When True, the server attaches a 1-indexed
            ``speaker`` number to each utterance.
        emotion_signal: When True, utterances include a detected ``emotion``.
        accent_signal: When True, utterances include a detected ``accent``.
        deepfake_signal: When True, utterances include a ``deepfake_score``
            (0.0-1.0; null for utterances shorter than 0.5 seconds).
        pii_phi_tagging: When True, sensitive data in transcripts is wrapped
            in PII/PHI tags.
        custom_terms: Custom vocabulary to bias transcription toward domain
            terms and names. Entries are either plain strings or objects
            with pronunciations, e.g.
            ``["Modulate", {"term": "Velma", "pronunciations": ["VEL-muh"]}]``.
    """

    partial_results: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    speaker_diarization: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    emotion_signal: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    accent_signal: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    deepfake_signal: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    pii_phi_tagging: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    custom_terms: list[Any] | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class ModulateSTTService(WebsocketSTTService):
    """Modulate real-time speech-to-text service.

    Streams raw PCM audio to Modulate's multilingual streaming STT WebSocket
    endpoint over a persistent connection and emits interim and final
    transcription frames.

    The server segments speech into utterances itself: each detected
    utterance produces a final ``utterance`` message shortly after the
    speaker pauses, with the language detected per utterance (or pinned via
    the ``language`` setting). While an utterance is in progress, interim
    ``partial_utterance`` messages carry its complete text so far, so each
    interim frame supersedes the previous one. Session options are sent as a
    JSON configuration frame immediately after connecting; changing settings
    at runtime therefore requires a reconnect.
    """

    Settings = ModulateSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "wss://platform.modulate.ai/api/velma-2-stt-streaming",
        sample_rate: int | None = None,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = MODULATE_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Modulate STT service.

        Args:
            api_key: Modulate API key (sent as a query parameter on the
                WebSocket handshake, per the streaming API).
            url: WebSocket endpoint URL. Defaults to
                ``wss://platform.modulate.ai/api/velma-2-stt-streaming``.
            sample_rate: Audio sample rate in Hz. If None, determined from
                the start frame. Supported values: 8000, 11025, 16000,
                22050, 32000, 44100, 48000, 96000.
            settings: Runtime-updatable settings overriding defaults. By
                default the language is auto-detected per utterance,
                ``partial_results`` is enabled, and ``speaker_diarization``
                is disabled.
            ttfs_p99_latency: P99 latency from speech end to final transcript
                in seconds. See https://github.com/pipecat-ai/stt-benchmark.
            **kwargs: Additional arguments passed to WebsocketSTTService.
        """
        default_settings = self.Settings(
            model="velma-2",
            language=None,
            partial_results=True,
            speaker_diarization=False,
            emotion_signal=None,
            accent_signal=None,
            deepfake_signal=None,
            pii_phi_tagging=None,
            custom_terms=None,
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
        self._url = url

        self._receive_task: asyncio.Task | None = None

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            True if metrics generation is supported.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to the Modulate STT language code."""
        return language_to_modulate_language(language)

    async def start(self, frame: StartFrame):
        """Start the speech-to-text service."""
        await super().start(frame)
        if self.sample_rate not in SUPPORTED_SAMPLE_RATES:
            logger.warning(
                f"{self} sample rate {self.sample_rate} is not supported by Modulate STT "
                f"(supported: {sorted(SUPPORTED_SAMPLE_RATES)})"
            )
        await self._connect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Forward raw audio bytes to the Modulate STT WebSocket.

        Transcription frames are pushed from the receive task, not yielded
        from this coroutine.
        """
        if self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._websocket.send(audio)
            except Exception as e:
                logger.warning(f"{self}: send failed: {e}")
        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, tracking processing metrics per user turn."""
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self.start_processing_metrics()

    async def _update_settings(self, delta: Settings) -> dict[str, Any]:
        """Apply a settings delta and reconnect to apply changes.

        Modulate STT configures the session with a configuration frame sent
        at connection start, so any change requires a fresh connection.
        """
        changed = await super()._update_settings(delta)
        if changed:
            await self._request_reconnect()
        return changed

    def _build_ws_url(self) -> str:
        """Build the WebSocket URL with connection query parameters.

        The URL contains the API key, so it must never be logged.
        """
        params = {
            "api_key": self._api_key,
            "audio_format": "s16le",
            "sample_rate": self.sample_rate,
            "num_channels": 1,
        }
        return f"{self._url}?{urlencode(params)}"

    def _build_config(self) -> dict[str, Any]:
        """Build the JSON configuration frame sent before any audio.

        Only explicitly configured fields are included; omitted fields keep
        the server's defaults.
        """
        s = self._settings
        config: dict[str, Any] = {}
        optional_fields = {
            "partial_results": s.partial_results,
            "speaker_diarization": s.speaker_diarization,
            "emotion_signal": s.emotion_signal,
            "accent_signal": s.accent_signal,
            "deepfake_signal": s.deepfake_signal,
            "pii_phi_tagging": s.pii_phi_tagging,
            "custom_terms": s.custom_terms,
        }
        for key, val in optional_fields.items():
            if is_given(val) and val is not None:
                config[key] = val
        if is_given(s.language) and s.language is not None:
            config["language"] = s.language
        return config

    async def _connect(self):
        """Establish the WebSocket connection and start the receive task."""
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Tear down the WebSocket connection and cancel the receive task."""
        await super()._disconnect()
        await self._send_end_of_stream()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Open a WebSocket connection and send the configuration frame."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug(f"{self} connecting to Modulate STT")
            websocket = await websocket_connect(self._build_ws_url())
            self._websocket = websocket
            config = self._build_config()
            if config:
                await websocket.send(json.dumps(config))
            await self._call_event_handler("on_connected")
            logger.debug(f"{self} connected to Modulate STT")
        except Exception as e:
            self._websocket = None
            await self.push_error(error_msg=f"Unable to connect to Modulate STT: {e}", exception=e)

    async def _disconnect_websocket(self):
        """Close the WebSocket connection."""
        try:
            if self._websocket:
                logger.debug(f"{self} disconnecting from Modulate STT")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing Modulate websocket: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def _receive_messages(self):
        """Receive and dispatch Modulate STT WebSocket messages."""
        if not self._websocket:
            raise Exception("Websocket not connected")
        async for message in self._websocket:
            if isinstance(message, bytes):
                continue
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"{self} received non-JSON message: {message}")
                continue
            await self._handle_message(data)

    async def _handle_message(self, data: dict[str, Any]):
        """Branch on Modulate STT message type."""
        msg_type = data.get("type")

        if msg_type == "partial_utterance":
            text = (data.get("partial_utterance") or {}).get("text", "")
            if text:
                await self.push_frame(
                    InterimTranscriptionFrame(
                        text,
                        self._user_id,
                        time_now_iso8601(),
                        self._language_from_code(None),
                        result=data,
                    )
                )
        elif msg_type == "utterance":
            utterance = data.get("utterance") or {}
            text = utterance.get("text", "")
            language = self._language_from_code(utterance.get("language"))
            if text:
                await self.push_frame(
                    TranscriptionFrame(
                        text,
                        self._user_id,
                        time_now_iso8601(),
                        language,
                        result=data,
                        finalized=True,
                    )
                )
                await self._trace_transcription(text, True, language)
                await self.stop_processing_metrics()
        elif msg_type == "done":
            # Sent only in response to our end-of-stream signal during
            # disconnect; the server closes the connection right after.
            logger.debug(f"{self} stream complete: {data}")
        elif msg_type == "error":
            error = data.get("error", "unknown error")
            await self.push_error(error_msg=f"Modulate STT error: {error}")
        else:
            logger.debug(f"{self} unhandled Modulate STT message: {data}")

    def _language_from_code(self, code: str | None) -> Language | None:
        """Return a Language enum for transcription frames.

        Prefers the language reported in the message (finals carry the
        per-utterance detected language), falling back to the configured
        language, or None when the language is unknown (e.g. interims, which
        carry no language field).
        """
        if not code and isinstance(self._settings.language, str):
            code = self._settings.language
        if not code:
            return None
        try:
            return Language(code)
        except ValueError:
            return None

    async def _send_end_of_stream(self):
        """Signal end-of-stream so the server finalizes any pending utterance.

        The server responds with a ``done`` message and closes the
        connection, so this is only sent during disconnect.
        """
        if self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._websocket.send("")
            except Exception as e:
                logger.debug(f"{self}: end-of-stream send failed: {e}")

    @traced_stt
    async def _trace_transcription(self, transcript: str, is_final: bool, language: Language):
        """Record transcription event for tracing."""
        pass
