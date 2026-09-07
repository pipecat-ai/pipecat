#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Palabra AI text-to-speech service implementation.

Streams sentences to Palabra's Realtime TTS WebSocket API and routes the
returned base64-encoded PCM chunks into Pipecat audio contexts. Each sentence
is one Palabra *generation*, tagged with a client-supplied ``generation_id`` so
its audio can be matched back to the Pipecat context that requested it.

Palabra API reference: https://docs.palabra.ai/docs/streaming_api/realtime_tts
"""

import asyncio
import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, WebsocketTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven

REGION_URLS = {
    "eu": "wss://stream.palabra.ai/tts-api/v1/text-to-speech/stream",
    "us": "wss://stream.us.palabra.ai/tts-api/v1/text-to-speech/stream",
}

# Palabra accepts output sample rates in this range for raw PCM.
MIN_SAMPLE_RATE = 8000
MAX_SAMPLE_RATE = 48000

# Palabra rejects text messages longer than this.
MAX_TEXT_LENGTH = 1024

# Error codes after which Palabra keeps the session open, so no reconnect is needed.
_SESSION_KEPT_ERRORS = {"BAD_REQUEST", "VALIDATION_ERROR", "RATE_LIMIT_EXCEEDED"}


def language_to_palabra_tts_language(language: Language) -> str:
    """Convert a Pipecat Language to a Palabra language code.

    For the list of supported languages, see:
    https://docs.palabra.ai/docs/streaming_api/realtime_tts
    """
    return language.value.lower()


@dataclass
class PalabraTTSSettings(TTSSettings):
    """Settings for PalabraTTSService.

    ``language`` and ``model`` are fixed by the session's ``init`` message, so
    changing either reconnects. ``voice``, ``speed`` and ``deaccent_strength``
    are sent with every text message and take effect on the next sentence.

    Parameters:
        voice: Voice identifier. ``"default_low"`` or ``"default_high"`` select
            the language's default voice; a custom voice id from the Voices API
            also works.
        speed: Speech rate multiplier in the range 0.0-2.0. ``None`` uses the
            Palabra default (1.0).
        deaccent_strength: How strongly a foreign accent is reduced, in the
            range 0.0-1.0. ``None`` uses the Palabra default (1.0).
    """

    speed: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    deaccent_strength: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class PalabraTTSService(WebsocketTTSService):
    """Text-to-speech service using Palabra's Realtime TTS WebSocket API.

    Every sentence handed to :meth:`run_tts` becomes one Palabra generation,
    sent with ``is_eos: true``. Audio chunks are routed to the Pipecat audio
    context that issued the generation; a context is closed once its last
    generation reports ``last_chunk`` and no more text is pending for it. An
    interruption sends Palabra's ``cancel`` command instead of reconnecting,
    since Palabra limits new connections per minute.

    Palabra has no end-of-sentence signal other than ``is_eos`` on a text
    message, so :attr:`TextAggregationMode.SENTENCE` (the default) is the
    intended mode; in token mode every token is synthesized as its own
    generation.

    For complete API documentation, see:
    https://docs.palabra.ai/docs/streaming_api/realtime_tts
    """

    Settings = PalabraTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        region: str = "eu",
        url: str | None = None,
        sample_rate: int | None = None,
        settings: Settings | None = None,
        text_aggregation_mode: TextAggregationMode | None = None,
        **kwargs,
    ):
        """Initialize the Palabra TTS service.

        Args:
            api_key: Palabra API key. Create one at https://platform.palabra.ai/api-keys.
            region: Palabra region hosting the TTS endpoint, ``"eu"`` or ``"us"``.
            url: WebSocket URL of the TTS endpoint. Overrides ``region``.
            sample_rate: Output sample rate in Hz, between 8000 and 48000. If
                ``None``, inherits from the pipeline.
            settings: Runtime-updatable settings. Defaults to English with the
                ``default_low`` voice and the ``auto`` model.
            text_aggregation_mode: How to aggregate incoming text before
                synthesis. Defaults to :attr:`TextAggregationMode.SENTENCE`.
            **kwargs: Additional arguments passed to :class:`WebsocketTTSService`.
        """
        default_settings = self.Settings(
            model="auto",
            voice="default_low",
            language=Language.EN,
            speed=None,
            deaccent_strength=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            text_aggregation_mode=text_aggregation_mode,
            # Palabra reports no per-word timing, so the base class pushes each
            # sentence's text up front.
            push_text_frames=True,
            # TTSStoppedFrame is pushed once the context's last generation ends.
            push_stop_frames=False,
            push_start_frame=True,
            pause_frame_processing=False,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        if url is None:
            if region not in REGION_URLS:
                raise ValueError(
                    f"Unknown Palabra region {region!r}; expected one of {sorted(REGION_URLS)}"
                )
            url = REGION_URLS[region]

        self._api_key = api_key
        self._url = url

        # generation_id -> context_id for generations whose audio is still arriving.
        self._generations: dict[str, str] = {}
        # Contexts whose text is complete: close them once their generations finish.
        self._flushed_contexts: set[str] = set()
        self._generation_counter = 0

        self._receive_task: asyncio.Task | None = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Palabra TTS supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to a Palabra TTS language code.

        Args:
            language: The language to convert.

        Returns:
            The Palabra-specific language code.
        """
        return language_to_palabra_tts_language(language)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        if not MIN_SAMPLE_RATE <= self.sample_rate <= MAX_SAMPLE_RATE:
            logger.warning(
                f"{self}: sample_rate={self.sample_rate} is outside Palabra's supported range "
                f"{MIN_SAMPLE_RATE}-{MAX_SAMPLE_RATE}; the server may reject the session."
            )
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Palabra TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Palabra TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def flush_audio(self, context_id: str | None = None):
        """Mark a context's text as complete.

        Every sentence is already sent with ``is_eos: true``, so nothing goes
        over the wire. The context is closed as soon as its outstanding
        generations finish, or right away if none are pending.

        Args:
            context_id: The context to flush. If ``None``, falls back to the
                currently active context.
        """
        flush_id = context_id or self.get_active_audio_context_id()
        if not flush_id:
            return
        self._flushed_contexts.add(flush_id)
        await self._maybe_close_context(flush_id)

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the current synthesis when the bot is interrupted.

        Palabra's ``cancel`` stops the generation in progress and keeps the
        session open, so no reconnect is needed. Audio of generations that
        were already queued is dropped by the base class because their context
        no longer exists.
        """
        await self.stop_all_metrics()
        await self._send_cancel()
        self._forget_context(context_id)
        await super().on_audio_context_interrupted(context_id)

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta, reconnecting if the session ``init`` changed.

        Args:
            delta: A TTS settings delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)
        if changed.keys() & {"language", "model"}:
            await self._disconnect()
            await self._connect()
        return changed

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
            logger.debug("Connecting to Palabra TTS")
            self._websocket = await self._websocket_connect(f"{self._url}?token={self._api_key}")
            await self._get_websocket().send(json.dumps(self._build_init_msg()))
            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            await self.push_error(error_msg=f"Unable to connect to Palabra TTS: {e}", exception=e)
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug("Disconnecting from Palabra TTS")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing Palabra websocket: {e}", exception=e)
        finally:
            await self.remove_active_audio_context()
            self._generations.clear()
            self._flushed_contexts.clear()
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    def _voice_options(self) -> dict[str, Any]:
        s = self._settings
        options: dict[str, Any] = {"voice_id": s.voice}
        if s.speed is not None:
            options["speed"] = s.speed
        if s.deaccent_strength is not None:
            options["deaccent_strength"] = s.deaccent_strength
        return options

    def _build_init_msg(self) -> dict[str, Any]:
        """Build the ``init`` message that opens a Palabra TTS session."""
        return {
            "type": "init",
            "language": self._settings.language,
            "model": self._settings.model,
            "voice_options": self._voice_options(),
            "output": {"format": "pcm", "sample_rate": self.sample_rate},
        }

    def _next_generation_id(self, context_id: str) -> str:
        self._generation_counter += 1
        generation_id = f"{context_id}-{self._generation_counter}"
        self._generations[generation_id] = context_id
        return generation_id

    def _forget_context(self, context_id: str):
        self._flushed_contexts.discard(context_id)
        for generation_id in [g for g, c in self._generations.items() if c == context_id]:
            del self._generations[generation_id]

    def _has_pending_generations(self, context_id: str) -> bool:
        return any(c == context_id for c in self._generations.values())

    async def _maybe_close_context(self, context_id: str):
        """Close ``context_id`` once its text is complete and its audio has all arrived."""
        if context_id not in self._flushed_contexts or self._has_pending_generations(context_id):
            return
        self._flushed_contexts.discard(context_id)
        if self.audio_context_available(context_id):
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)

    async def _send_cancel(self):
        if self._websocket and self._websocket.state is State.OPEN:
            try:
                await self._websocket.send(json.dumps({"type": "cancel"}))
            except Exception as e:
                logger.warning(f"{self}: failed to cancel synthesis: {e}")

    async def _receive_messages(self):
        """Handle incoming WebSocket messages from Palabra."""
        async for message in self._get_websocket():
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"{self}: received non-JSON Palabra message: {message!r}")
                continue

            message_type = msg.get("message_type")
            data = msg.get("data") or {}

            if message_type == "audio_chunk":
                await self._handle_audio_chunk(data)
            elif message_type == "error":
                await self._handle_error(data)
            else:
                logger.trace(f"{self}: ignoring Palabra message {message_type}")

    async def _handle_audio_chunk(self, data: dict[str, Any]):
        generation_id = str(data.get("generation_id", ""))
        context_id = self._generations.get(generation_id)
        if context_id is None:
            # A generation that was cancelled or belongs to an interrupted context.
            return

        audio_b64 = data.get("audio")
        if audio_b64 and self.audio_context_available(context_id):
            await self.stop_ttfb_metrics()
            frame = TTSAudioRawFrame(
                audio=base64.b64decode(audio_b64),
                sample_rate=self.sample_rate,
                num_channels=1,
                context_id=context_id,
            )
            await self.append_to_audio_context(context_id, frame)

        if data.get("last_chunk"):
            self._generations.pop(generation_id, None)
            await self._maybe_close_context(context_id)

    async def _handle_error(self, data: dict[str, Any]):
        code = data.get("code", "UNKNOWN_ERROR")
        desc = data.get("desc", "")
        await self.push_error(error_msg=f"Palabra TTS error {code}: {desc}")
        if code in _SESSION_KEPT_ERRORS:
            return
        # Any other error ends the synthesis in flight: close open contexts so
        # the bot does not wait for audio that will never come.
        for context_id in self.get_audio_contexts():
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)
        self._generations.clear()
        self._flushed_contexts.clear()

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Send one sentence to Palabra as its own generation.

        Audio arrives on the receive task and is appended to the matching
        audio context.

        Args:
            text: The text to synthesize.
            context_id: The audio context the resulting audio belongs to.

        Yields:
            ``None``; audio frames are delivered out of band.
        """
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            if len(text) > MAX_TEXT_LENGTH:
                logger.warning(
                    f"{self}: text of {len(text)} characters exceeds Palabra's limit of "
                    f"{MAX_TEXT_LENGTH}; truncating"
                )
                text = text[:MAX_TEXT_LENGTH]

            # Text for a context that was already flushed reopens it (for
            # example a TTSSpeakFrame reusing the turn context).
            self._flushed_contexts.discard(context_id)

            try:
                msg = {
                    "type": "text",
                    "text": text,
                    "is_eos": True,
                    "generation_id": self._next_generation_id(context_id),
                    "voice_options": self._voice_options(),
                }
                await self._get_websocket().send(json.dumps(msg))
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
