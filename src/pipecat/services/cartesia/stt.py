#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Cartesia Speech-to-Text service implementation.

This module provides a WebSocket-based STT service that integrates with
the Cartesia Live transcription API for real-time speech recognition.
"""

import json
import urllib.parse
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from websockets.protocol import State

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
from pipecat.services.stt_latency import CARTESIA_TTFS_P99
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.deprecation import deprecated
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

# Cartesia caps a connection at 100 keyterms totaling 1200 characters.
_MAX_KEYTERMS = 100
_MAX_KEYTERM_CHARS = 1200

# Keyterms are only honored by the ink-2 model family.
_KEYTERM_MODEL_PREFIX = "ink-2"


def _prepare_keyterms(keyterms: list[str] | None | _NotGiven) -> list[str]:
    """Normalize keyterms to the limits Cartesia accepts on a connection.

    Drops blank entries and truncates to :data:`_MAX_KEYTERMS` terms totaling
    :data:`_MAX_KEYTERM_CHARS` characters, warning about whatever is dropped.
    Truncating keeps an oversized list from failing the connection mid-call.

    Args:
        keyterms: Keyterms from settings, which may be unset or ``None``.

    Returns:
        The keyterms to send, in the order given.
    """
    if not is_given(keyterms) or not keyterms:
        return []

    terms = [stripped for term in keyterms if (stripped := term.strip())]

    prepared: list[str] = []
    total_chars = 0
    for term in terms:
        if len(prepared) >= _MAX_KEYTERMS or total_chars + len(term) > _MAX_KEYTERM_CHARS:
            break
        prepared.append(term)
        total_chars += len(term)

    if len(prepared) < len(terms):
        logger.warning(
            f"Cartesia accepts at most {_MAX_KEYTERMS} keyterms totaling "
            f"{_MAX_KEYTERM_CHARS} characters; dropping {len(terms) - len(prepared)} keyterm(s)"
        )

    return prepared


@dataclass
class CartesiaSTTSettings(STTSettings):
    """Settings for CartesiaSTTService.

    Parameters:
        keyterm: Key terms or phrases to bias transcription towards, sent as
            repeated ``keyterm`` query parameters on the connection URL. Only
            honored by ink-2 models; keyterms set for any other model are
            ignored with a warning. Cartesia binds keyterms to a connection,
            so updating this setting at runtime triggers a reconnect. See
            https://docs.cartesia.ai/use-the-api/stt/keyterms.
    """

    keyterm: list[str] | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


@deprecated(
    "`CartesiaLiveOptions` is deprecated since 0.0.105 and will be removed in 2.0.0. "
    "Use `CartesiaSTTService.Settings` instead."
)
class CartesiaLiveOptions:
    """Configuration options for Cartesia Live STT service.

    .. deprecated:: 0.0.105
        Use ``settings=CartesiaSTTService.Settings(...)`` for model/language and
        direct ``__init__`` parameters for encoding/sample_rate instead.
        Will be removed in 2.0.0.
    """

    def __init__(
        self,
        *,
        model: str = "ink-whisper",
        language: str = Language.EN.value,
        encoding: str = "pcm_s16le",
        sample_rate: int = 16000,
        **kwargs,
    ):
        """Initialize CartesiaLiveOptions with default or provided parameters.

        Args:
            model: The transcription model to use. Defaults to "ink-whisper".
            language: Target language for transcription. Defaults to English.
            encoding: Audio encoding format. Defaults to "pcm_s16le".
            sample_rate: Audio sample rate in Hz. Defaults to 16000.
            **kwargs: Additional parameters for the transcription service.
        """
        self.model = model
        self.language = language
        self.encoding = encoding
        self.sample_rate = sample_rate
        self.additional_params = kwargs

    def to_dict(self):
        """Convert options to dictionary format.

        Returns:
            Dictionary containing all configuration parameters.
        """
        params = {
            "model": self.model,
            "language": self.language if isinstance(self.language, str) else self.language.value,
            "encoding": self.encoding,
            "sample_rate": str(self.sample_rate),
        }

        return params

    def items(self):
        """Get configuration items as key-value pairs.

        Returns:
            Iterator of (key, value) tuples for all configuration parameters.
        """
        return self.to_dict().items()

    def get(self, key, default=None):
        """Get a configuration value by key.

        Args:
            key: The configuration parameter name to retrieve.
            default: Default value if key is not found.

        Returns:
            The configuration value or default if not found.
        """
        if hasattr(self, key):
            return getattr(self, key)
        return self.additional_params.get(key, default)

    @classmethod
    def from_json(cls, json_str: str) -> "CartesiaLiveOptions":
        """Create options from JSON string.

        Args:
            json_str: JSON string containing configuration parameters.

        Returns:
            New CartesiaLiveOptions instance with parsed parameters.
        """
        return cls(**json.loads(json_str))


class CartesiaSTTService(WebsocketSTTService):
    """Speech-to-text service using Cartesia Live API.

    Provides real-time speech transcription through WebSocket connection
    to Cartesia's Live transcription service. Supports both interim and
    final transcriptions with configurable models and languages.

    Cartesia disconnects WebSocket connections after 3 minutes of inactivity.
    The timeout resets with each message (audio data or text command) sent to
    the server. Silence-based keepalive is enabled by default to prevent this.
    See: https://docs.cartesia.ai/api-reference/stt/stt
    """

    Settings = CartesiaSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "",
        encoding: str = "pcm_s16le",
        sample_rate: int | None = None,
        live_options: CartesiaLiveOptions | None = None,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = CARTESIA_TTFS_P99,
        **kwargs,
    ):
        """Initialize CartesiaSTTService with API key and options.

        Args:
            api_key: Authentication key for Cartesia API.
            base_url: Custom API endpoint URL. If empty, uses default.
            encoding: Audio encoding format. Defaults to "pcm_s16le".
            sample_rate: Audio sample rate in Hz. If None, uses the pipeline
                sample rate.
            live_options: Configuration options for transcription service.

                .. deprecated:: 0.0.105
                    Use ``settings=CartesiaSTTService.Settings(...)`` for model/language
                    and direct init parameters for encoding/sample_rate instead.
                    Will be removed in 2.0.0.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to parent STTService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="ink-whisper",
            language=Language.EN.value,
            keyterm=None,
        )

        # 2. Apply live_options overrides — only if settings not provided
        if live_options is not None:
            self._warn_init_param_moved_to_settings("live_options")
            if not settings:
                if live_options.sample_rate and sample_rate is None:
                    sample_rate = live_options.sample_rate
                if live_options.encoding:
                    encoding = live_options.encoding
                if live_options.model:
                    default_settings.model = live_options.model
                if live_options.language:
                    lang = live_options.language
                    default_settings.language = lang.value if isinstance(lang, Language) else lang

        # 3. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            keepalive_timeout=120,
            keepalive_interval=30,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url or "api.cartesia.ai"
        self._receive_task = None

        # Init-only audio config (not runtime-updatable).
        self._encoding = encoding

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate processing metrics.

        Returns:
            True, indicating metrics are supported.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the STT service and establish connection.

        Args:
            frame: Frame indicating service should start.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the STT service and close connection.

        Args:
            frame: Frame indicating service should stop.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the STT service and close connection.

        Args:
            frame: Frame indicating service should be cancelled.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _start_metrics(self):
        """Start performance metrics collection for transcription processing."""
        await self.start_processing_metrics()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle speech events.

        Args:
            frame: The frame to process.
            direction: Direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self._start_metrics()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # Send finalize command to flush the transcription session
            if self._websocket and self._websocket.state is State.OPEN:
                await self._websocket.send("finalize")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Process audio data for speech-to-text transcription.

        Args:
            audio: Raw audio bytes to transcribe.

        Yields:
            None - transcription results are handled via WebSocket responses.
        """
        # If the connection is not open (closed or closing), reconnect
        if not self._websocket or self._websocket.state is not State.OPEN:
            await self._connect()

        if self._websocket is None:
            logger.warning(f"{self}: websocket unavailable after reconnect, dropping audio")
            yield None
            return

        try:
            await self._websocket.send(audio)
        except Exception as e:
            logger.warning(f"{self}: send failed: {e}")
        yield None

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

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta.

        Args:
            delta: A :class:`STTSettings` (or ``CartesiaSTTService.Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)

        if not changed:
            return changed

        await self._request_reconnect()

        return changed

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            logger.debug("Connecting to Cartesia STT")

            params = [
                ("model", self._settings.model),
                ("language", self._settings.language),
                ("encoding", self._encoding),
                ("sample_rate", str(self.sample_rate)),
            ]
            keyterms = _prepare_keyterms(self._settings.keyterm)
            if keyterms:
                if str(self._settings.model).startswith(_KEYTERM_MODEL_PREFIX):
                    params.extend(("keyterm", term) for term in keyterms)
                else:
                    logger.warning(
                        f"keyterms are only supported on {_KEYTERM_MODEL_PREFIX} models; "
                        f"ignoring keyterms for model {self._settings.model!r}"
                    )
            # Cartesia expects spaces inside a keyterm as %20, which urlencode
            # only emits with quote_via=quote.
            query = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
            ws_url = f"wss://{self._base_url}/stt/websocket?{query}"
            headers = {"Cartesia-Version": "2026-03-01", "X-API-Key": self._api_key}

            self._websocket = await self._websocket_connect(ws_url, additional_headers=headers)
            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            await self.push_error(error_msg=f"Unable to connect to Cartesia: {e}", exception=e)

    async def _disconnect_websocket(self):
        ws = self._websocket
        try:
            if ws and ws.state is State.OPEN:
                logger.debug("Disconnecting from Cartesia STT")
                await ws.send("done")
                await ws.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing websocket: {e}", exception=e)
        finally:
            # Only clear if no concurrent _connect has already replaced it.
            if self._websocket is ws:
                self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        """Process incoming WebSocket messages."""
        async for message in self._get_websocket():
            try:
                data = json.loads(message)
                await self._process_response(data)
            except json.JSONDecodeError:
                logger.warning(f"Received non-JSON message: {message}")
            except Exception as e:
                logger.error(f"Error processing message: {e}")

    async def _process_response(self, data):
        if "type" in data:
            if data["type"] == "transcript":
                await self._on_transcript(data)

            elif data["type"] == "error":
                error_msg = data.get("message", "Unknown error")
                await self.push_error(error_msg=error_msg)

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def _on_transcript(self, data):
        if "text" not in data:
            return

        transcript = data.get("text", "")
        is_final = data.get("is_final", False)
        language = None

        if "language" in data:
            try:
                language = Language(data["language"])
            except (ValueError, KeyError):
                pass

        if len(transcript) > 0:
            if is_final:
                # Report usage before the transcription frame so tracing can
                # attach it to the STT span the frame closes.
                await self.emit_stt_usage_metrics()
                await self.push_frame(
                    TranscriptionFrame(
                        transcript,
                        self._user_id,
                        time_now_iso8601(),
                        language,
                        result=data,
                    )
                )
                await self._handle_transcription(transcript, is_final, language)
                await self.stop_processing_metrics()
            else:
                # For interim transcriptions, just push the frame without tracing
                await self.push_frame(
                    InterimTranscriptionFrame(
                        transcript,
                        self._user_id,
                        time_now_iso8601(),
                        language,
                        result=data,
                    )
                )
