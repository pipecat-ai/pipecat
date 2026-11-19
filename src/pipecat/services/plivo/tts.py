#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Plivo text-to-speech service implementations.

Provides two TTS services against Plivo's text-to-speech API:

- :class:`PlivoTTSService` uses the Realtime WebSocket API at
  ``wss://api.plivo.com/v1/Account/{auth_id}/TTS/Realtime/``.
- :class:`PlivoHttpTTSService` uses the Batch HTTP API at
  ``https://api.plivo.com/v1/Account/{auth_id}/TTS/``.

See https://www.plivo.com/docs/models/text-to-speech/quickstart.
"""

import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from loguru import logger
from websockets import ConnectionClosedOK
from websockets.protocol import State

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService, WebsocketTTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven

PLIVO_API_BASE_URL = "https://api.plivo.com/v1/Account"
PLIVO_WS_BASE_URL = "wss://api.plivo.com/v1/Account"


def language_to_plivo_language(language: Language) -> str:
    """Convert a Language enum to a Plivo TTS language code.

    Plivo accepts regional English codes (``en-US``, ``en-GB``, ``en-IN``,
    ``en-AU``) but rejects a bare ``en``, so generic English maps to ``en-US``.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Plivo language code.
    """
    LANGUAGE_MAP = {
        Language.EN: "en-US",
        Language.EN_US: "en-US",
        Language.EN_GB: "en-GB",
        Language.EN_IN: "en-IN",
        Language.EN_AU: "en-AU",
        Language.HI: "hi",
        Language.HI_IN: "hi",
        Language.BN: "bn",
        Language.BN_IN: "bn",
        Language.TA: "ta",
        Language.TA_IN: "ta",
        Language.TE: "te",
        Language.TE_IN: "te",
        Language.MR: "mr",
        Language.MR_IN: "mr",
        Language.GU: "gu",
        Language.GU_IN: "gu",
        Language.KN: "kn",
        Language.KN_IN: "kn",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


def parse_plivo_audio_frame(message: bytes) -> tuple[str, bytes]:
    """Split a Plivo Realtime binary audio message into context ID and audio.

    Each binary message is laid out as a 2-byte little-endian length of the
    context ID, followed by the context ID bytes, followed by audio bytes.

    Args:
        message: The raw binary WebSocket message.

    Returns:
        A ``(context_id, audio)`` tuple.

    Raises:
        ValueError: If the message is too short to contain its header.
    """
    if len(message) < 2:
        raise ValueError("Plivo audio frame too short")
    id_len = int.from_bytes(message[:2], "little")
    if len(message) < 2 + id_len:
        raise ValueError("Plivo audio frame truncated")
    context_id = message[2 : 2 + id_len].decode("utf-8")
    return context_id, message[2 + id_len :]


def calculate_word_times(chars: list[str], start_ms: list[float]) -> list[tuple[str, float]]:
    """Group Plivo character timestamps into word timestamps.

    Plivo sends one ``timestamps`` event per sentence with per-character start
    times that are cumulative within a reply, so each event can be grouped
    independently.

    Args:
        chars: Characters of the sentence.
        start_ms: Start time of each character in milliseconds.

    Returns:
        A list of ``(word, start_seconds)`` tuples.
    """
    if len(chars) != len(start_ms):
        logger.error(
            f"Plivo timestamps length mismatch: chars={len(chars)}, start_ms={len(start_ms)}"
        )
        return []

    words: list[tuple[str, float]] = []
    current = ""
    current_start = 0.0
    for char, start in zip(chars, start_ms):
        if char.isspace():
            if current:
                words.append((current, current_start))
                current = ""
        else:
            if not current:
                current_start = start / 1000.0
            current += char
    if current:
        words.append((current, current_start))
    return words


@dataclass
class PlivoTTSSettings(TTSSettings):
    """Settings for PlivoTTSService and PlivoHttpTTSService.

    Parameters:
        pace: Speaking pace multiplier, from 0.7 to 1.2. ``None`` uses Plivo's default (1.0).
        volume: Volume multiplier, from 0.5 to 2.0. ``None`` uses Plivo's default (1.0).
        lexicon_id: ID of a saved Plivo pronunciation lexicon.
    """

    pace: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    volume: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    lexicon_id: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


def _default_settings(settings: PlivoTTSSettings | None) -> PlivoTTSSettings:
    default_settings = PlivoTTSSettings(
        model=None,
        voice="3gVUACYdS3ibtu8qPruB",  # Prerna (en-IN, female)
        language=Language.EN_IN,
        pace=None,
        volume=None,
        lexicon_id=None,
    )
    if settings is not None:
        default_settings.apply_update(settings)
    return default_settings


def _build_request_params(settings: PlivoTTSSettings, sample_rate: int) -> dict[str, Any]:
    """Build the per-reply request fields shared by the Batch and Realtime APIs.

    Unset fields are omitted, since Plivo rejects explicit ``null`` values.
    """
    params: dict[str, Any] = {
        "voice_id": settings.voice,
        "output_format": {
            "container": "raw",
            "encoding": "pcm_s16le",
            "sample_rate": sample_rate,
        },
    }
    if settings.language:
        params["language"] = str(settings.language)
    if settings.pace is not None:
        params["pace"] = settings.pace
    if settings.volume is not None:
        params["volume"] = settings.volume
    if settings.lexicon_id is not None:
        params["lexicon_id"] = settings.lexicon_id
    return params


def _basic_auth_header(auth_id: str, auth_token: str) -> str:
    token = base64.b64encode(f"{auth_id}:{auth_token}".encode()).decode()
    return f"Basic {token}"


class PlivoTTSService(WebsocketTTSService):
    """Plivo streaming text-to-speech service.

    Uses Plivo's Realtime WebSocket API. Each Pipecat TTS context maps to a
    Plivo reply context: the first ``text`` message for a context carries the
    voice and output settings, later messages carry only text, and
    ``flush_audio`` closes the reply with ``finish: true``. Interruptions send
    a ``cancel`` for the active reply. Word timestamps are enabled and
    converted from Plivo's per-character timings.

    Example::

        tts = PlivoTTSService(
            auth_id=os.environ["PLIVO_AUTH_ID"],
            auth_token=os.environ["PLIVO_AUTH_TOKEN"],
            settings=PlivoTTSService.Settings(voice="3gVUACYdS3ibtu8qPruB"),
        )
    """

    Settings = PlivoTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        auth_id: str,
        auth_token: str,
        url: str | None = None,
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Plivo WebSocket TTS service.

        Args:
            auth_id: Plivo Auth ID.
            auth_token: Plivo Auth Token.
            url: Realtime WebSocket URL. Defaults to
                ``wss://api.plivo.com/v1/Account/{auth_id}/TTS/Realtime/``.
            sample_rate: Output audio sample rate in Hz. If None, uses the
                pipeline default.
            settings: Runtime-updatable settings. ``voice`` defaults to
                ``"3gVUACYdS3ibtu8qPruB"`` (Prerna, en-IN).
            **kwargs: Additional arguments passed to ``WebsocketTTSService``.
        """
        super().__init__(
            push_text_frames=False,
            push_stop_frames=True,
            push_start_frame=True,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            settings=_default_settings(settings),
            **kwargs,
        )

        self._auth_id = auth_id
        self._auth_token = auth_token
        self._url = url or f"{PLIVO_WS_BASE_URL}/{auth_id}/TTS/Realtime/"

        self._receive_task = None
        # Plivo contexts opened on the current connection (and not yet done).
        self._open_context_ids: set[str] = set()

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Plivo service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to Plivo language format.

        Args:
            language: The language to convert.

        Returns:
            The Plivo-specific language code.
        """
        return language_to_plivo_language(language)

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta.

        Plivo fixes voice and output options when a reply opens, so changes
        apply from the next reply without reconnecting.

        Args:
            delta: A :class:`TTSSettings` (or ``PlivoTTSService.Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)
        if "model" in changed:
            self._warn_unhandled_updated_settings({"model": changed["model"]})
        return changed

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and establish the websocket connection.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service and close the connection.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and close the connection.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        """Establish the websocket connection and start the receive task."""
        await super()._connect()

        if self._websocket is None or self._websocket.state is not State.OPEN:
            if self._receive_task:
                await self.cancel_task(self._receive_task)
                self._receive_task = None

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Close the websocket connection and clean up tasks."""
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Connect to Plivo's Realtime TTS websocket."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug(f"{self}: connecting to Plivo TTS")
            headers = {"Authorization": _basic_auth_header(self._auth_id, self._auth_token)}
            self._websocket = await self._websocket_connect(self._url, additional_headers=headers)
            # Context IDs are scoped to a connection.
            self._open_context_ids.clear()
            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Error connecting to Plivo TTS: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close the websocket connection and reset state."""
        try:
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug(f"{self}: disconnecting from Plivo TTS")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error disconnecting from Plivo TTS: {e}", exception=e)
        finally:
            self._websocket = None
            self._open_context_ids.clear()
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Get the active websocket connection or raise an exception."""
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _send_json(self, msg: dict[str, Any]):
        await self._get_websocket().send(json.dumps(msg))

    async def flush_audio(self, context_id: str | None = None):
        """Finish the current Plivo reply so any held text is spoken.

        Args:
            context_id: The context to finish. Defaults to the active context.
        """
        ctx_id = context_id or self.get_active_audio_context_id()
        if not ctx_id or ctx_id not in self._open_context_ids or not self._websocket:
            return
        try:
            await self._send_json(
                {"type": "text", "context_id": ctx_id, "text": "", "finish": True}
            )
        except ConnectionClosedOK:
            logger.debug(f"{self}: connection closed normally during flush")
        except Exception as e:
            logger.error(f"{self} exception: {e}")

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the interrupted Plivo reply.

        Args:
            context_id: The ID of the interrupted audio context.
        """
        await self.stop_all_metrics()
        if context_id and context_id in self._open_context_ids and self._websocket:
            try:
                await self._send_json({"type": "cancel", "context_id": context_id})
            except Exception as e:
                logger.warning(f"{self}: failed to cancel Plivo context {context_id}: {e}")
        await super().on_audio_context_interrupted(context_id)

    async def _end_context(self, ctx_id: str):
        """Push a stop frame for a context and release it."""
        self._open_context_ids.discard(ctx_id)
        if self.audio_context_available(ctx_id):
            await self.append_to_audio_context(ctx_id, TTSStoppedFrame(context_id=ctx_id))
            await self.remove_audio_context(ctx_id)

    async def _receive_messages(self):
        """Process incoming websocket messages, demultiplexing by context ID."""
        async for message in self._get_websocket():
            if isinstance(message, bytes):
                try:
                    ctx_id, audio = parse_plivo_audio_frame(message)
                except ValueError as e:
                    logger.warning(f"{self}: {e}")
                    continue
                if not audio or not self.audio_context_available(ctx_id):
                    continue
                await self.stop_ttfb_metrics()
                frame = TTSAudioRawFrame(
                    audio=audio,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=ctx_id,
                )
                await self.append_to_audio_context(ctx_id, frame)
                continue

            msg = json.loads(message)
            msg_type = msg.get("type")
            ctx_id = msg.get("context_id")

            if msg_type == "timestamps":
                if ctx_id and self.audio_context_available(ctx_id):
                    words = calculate_word_times(msg.get("chars", []), msg.get("start_ms", []))
                    if words:
                        await self.add_word_timestamps(words, ctx_id)
            elif msg_type == "done":
                logger.trace(f"{self}: context {ctx_id} done ({msg.get('reason')})")
                await self.stop_all_metrics()
                if ctx_id:
                    await self._end_context(ctx_id)
            elif msg_type == "flush_done":
                pass
            elif msg_type == "warning":
                logger.warning(f"{self}: Plivo warning: {msg}")
            elif msg_type == "error":
                await self.stop_all_metrics()
                await self.push_error(
                    error_msg=f"Plivo TTS error ({msg.get('code')}): {msg.get('message')}"
                )
                if ctx_id:
                    await self._end_context(ctx_id)
            else:
                logger.debug(f"{self}: unhandled Plivo message: {msg}")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using Plivo's Realtime API.

        Args:
            text: The text to convert to speech.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: None (audio arrives via the receive task), or error frames.
        """
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                self._websocket = None
                await self._connect()

            try:
                msg: dict[str, Any] = {"type": "text", "context_id": context_id, "text": text}
                if context_id not in self._open_context_ids:
                    msg.update(_build_request_params(self._settings, self.sample_rate))
                    msg["timestamps"] = True
                    self._open_context_ids.add(context_id)
                await self._send_json(msg)
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                yield ErrorFrame(error=f"Error sending text to Plivo TTS: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")


class PlivoHttpTTSService(TTSService):
    """Plivo HTTP text-to-speech service.

    Uses Plivo's Batch API, which returns the complete audio for each request.
    Audio is requested as raw 16-bit PCM so frames need no decoding downstream.

    Example::

        async with aiohttp.ClientSession() as session:
            tts = PlivoHttpTTSService(
                auth_id=os.environ["PLIVO_AUTH_ID"],
                auth_token=os.environ["PLIVO_AUTH_TOKEN"],
                aiohttp_session=session,
                settings=PlivoHttpTTSService.Settings(voice="3gVUACYdS3ibtu8qPruB"),
            )
    """

    Settings = PlivoTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        auth_id: str,
        auth_token: str,
        url: str | None = None,
        sample_rate: int | None = None,
        aiohttp_session: aiohttp.ClientSession | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Plivo HTTP TTS service.

        Args:
            auth_id: Plivo Auth ID.
            auth_token: Plivo Auth Token.
            url: Batch API URL. Defaults to
                ``https://api.plivo.com/v1/Account/{auth_id}/TTS/``.
            sample_rate: Output audio sample rate in Hz. If None, uses the
                pipeline default.
            aiohttp_session: Optional shared aiohttp session. If not provided,
                the service creates and owns one.
            settings: Runtime-updatable settings. ``voice`` defaults to
                ``"3gVUACYdS3ibtu8qPruB"`` (Prerna, en-IN).
            **kwargs: Additional arguments passed to ``TTSService``.
        """
        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=_default_settings(settings),
            **kwargs,
        )

        self._auth = aiohttp.BasicAuth(auth_id, auth_token)
        self._url = url or f"{PLIVO_API_BASE_URL}/{auth_id}/TTS/"
        self._session = aiohttp_session
        self._session_owner = aiohttp_session is None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Plivo service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to Plivo language format.

        Args:
            language: The language to convert.

        Returns:
            The Plivo-specific language code.
        """
        return language_to_plivo_language(language)

    async def start(self, frame: StartFrame):
        """Start the service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._session_owner = True

    async def stop(self, frame: EndFrame):
        """Stop the service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._close_session()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._close_session()

    async def cleanup(self):
        """Release resources at teardown."""
        await super().cleanup()
        await self._close_session()

    async def _close_session(self):
        if self._session_owner and self._session and not self._session.closed:
            await self._session.close()
        if self._session_owner:
            self._session = None

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using Plivo's Batch API.

        Args:
            text: The text to convert to speech.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._session_owner = True

        payload = {"text": text, **_build_request_params(self._settings, self.sample_rate)}

        measuring_ttfb = True
        leftover = b""
        try:
            async with self._session.post(self._url, json=payload, auth=self._auth) as response:
                if response.status != 200:
                    error = await response.text(errors="ignore")
                    yield ErrorFrame(
                        error=f"Plivo TTS error (status: {response.status}, error: {error})"
                    )
                    return

                await self.start_tts_usage_metrics(text)

                async for chunk in response.content.iter_chunked(self.chunk_size):
                    if not chunk:
                        continue
                    if measuring_ttfb:
                        await self.stop_ttfb_metrics()
                        measuring_ttfb = False
                    # Keep 16-bit samples aligned across chunk boundaries.
                    data = leftover + chunk
                    usable = len(data) - (len(data) % 2)
                    leftover = data[usable:]
                    if usable:
                        yield TTSAudioRawFrame(
                            data[:usable], self.sample_rate, 1, context_id=context_id
                        )
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
