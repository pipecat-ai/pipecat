#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Soniox text-to-speech service implementation.

This module provides a WebSocket-based TTS service using the Soniox real-time
Text-to-Speech API. It streams text to the server incrementally and receives
audio back as base64-encoded chunks, multiplexed across multiple concurrent
streams by ``stream_id``.

Soniox API reference: https://soniox.com/docs/tts/api-reference/websocket-api
"""

import asyncio
import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, WebsocketTTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    import websockets
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Soniox, you need to `pip install pipecat-ai[soniox]`.")
    raise Exception(f"Missing module: {e}")


# Soniox idle timeout is 20-30s; keepalive cadence must stay well inside it.
KEEPALIVE_INTERVAL_SECONDS = 20

# Soniox-supported sample rates for raw PCM formats
VALID_SAMPLE_RATES = {8000, 16000, 24000, 44100, 48000}


def language_to_soniox_tts_language(language: Language) -> str | None:
    """Convert a Pipecat Language to a Soniox TTS language code.

    For the full list of supported languages, see:
    https://soniox.com/docs/tts/concepts/languages
    """
    LANGUAGE_MAP = {
        Language.AF: "af",
        Language.AR: "ar",
        Language.AZ: "az",
        Language.BE: "be",
        Language.BG: "bg",
        Language.BN: "bn",
        Language.BS: "bs",
        Language.CA: "ca",
        Language.CS: "cs",
        Language.CY: "cy",
        Language.DA: "da",
        Language.DE: "de",
        Language.EL: "el",
        Language.EN: "en",
        Language.ES: "es",
        Language.ET: "et",
        Language.EU: "eu",
        Language.FA: "fa",
        Language.FI: "fi",
        Language.FR: "fr",
        Language.GL: "gl",
        Language.GU: "gu",
        Language.HE: "he",
        Language.HI: "hi",
        Language.HR: "hr",
        Language.HU: "hu",
        Language.ID: "id",
        Language.IT: "it",
        Language.JA: "ja",
        Language.KK: "kk",
        Language.KN: "kn",
        Language.KO: "ko",
        Language.LT: "lt",
        Language.LV: "lv",
        Language.MK: "mk",
        Language.ML: "ml",
        Language.MR: "mr",
        Language.MS: "ms",
        Language.NL: "nl",
        Language.NO: "no",
        Language.PA: "pa",
        Language.PL: "pl",
        Language.PT: "pt",
        Language.RO: "ro",
        Language.RU: "ru",
        Language.SK: "sk",
        Language.SL: "sl",
        Language.SQ: "sq",
        Language.SR: "sr",
        Language.SV: "sv",
        Language.SW: "sw",
        Language.TA: "ta",
        Language.TE: "te",
        Language.TH: "th",
        Language.TL: "tl",
        Language.TR: "tr",
        Language.UK: "uk",
        Language.UR: "ur",
        Language.VI: "vi",
        Language.ZH: "zh",
    }
    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


@dataclass
class SonioxTTSSettings(TTSSettings):
    """Settings for SonioxTTSService.

    ``voice``, ``model``, and ``language`` travel in the per-stream
    config message, so changing any of them does not require reconnecting the
    WebSocket. The current context is flushed so the next stream opens with the
    new values.
    """

    pass


class SonioxTTSService(WebsocketTTSService):
    """Soniox WebSocket TTS service with streaming text-in, streaming audio-out.

    Streams text incrementally to Soniox's real-time TTS endpoint and routes the
    returned base64-encoded audio back as :class:`TTSAudioRawFrame` frames.
    Multiple concurrent streams are multiplexed over a single WebSocket
    connection via Pipecat's audio-context mechanism (mapped to Soniox's
    ``stream_id``). Supports up to 5 concurrent streams per connection.

    For complete API documentation, see:
    https://soniox.com/docs/tts/api-reference/websocket-api
    """

    Settings = SonioxTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "wss://tts-rt.soniox.com/tts-websocket",
        sample_rate: int | None = None,
        audio_format: str = "pcm_s16le",
        settings: Settings | None = None,
        text_aggregation_mode: TextAggregationMode | None = None,
        **kwargs,
    ):
        """Initialize the Soniox TTS service.

        Args:
            api_key: Soniox API key for authentication. Create API keys at
                https://console.soniox.com.
            url: WebSocket URL for the Soniox TTS endpoint.
            sample_rate: Output sample rate in Hz. Must be one of
                ``{8000, 16000, 24000, 44100, 48000}`` when using a raw PCM
                audio format. If ``None``, inherits from the pipeline.
            audio_format: Output audio format. Defaults to ``"pcm_s16le"``,
                which matches Pipecat's downstream audio pipeline.
            settings: Runtime-updatable settings. When provided alongside
                deprecated parameters, ``settings`` values take precedence.
            text_aggregation_mode: How to aggregate incoming text before
                synthesis. Defaults to ``TextAggregationMode.SENTENCE``.
            **kwargs: Additional arguments passed to the parent service.
        """
        # Initialize default_settings
        default_settings = self.Settings(
            model="tts-rt-v1-preview",
            voice="Adrian",
            language=Language.EN,
        )

        # Settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            text_aggregation_mode=text_aggregation_mode,
            # Soniox doesn't expose alignment data, so TTSTextFrames can be
            # pushed immediately by the base class.
            push_text_frames=True,
            # We push TTSStoppedFrame ourselves when Soniox sends `terminated`.
            push_stop_frames=False,
            # Let the base class create audio contexts and emit TTSStartedFrame.
            push_start_frame=True,
            pause_frame_processing=False,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._url = url

        # Init-only audio format (not runtime-updatable).
        self._audio_format = audio_format

        # Tracks which context_ids have had their per-stream config sent.
        # Soniox rejects duplicate config for the same stream_id.
        self._configured_contexts: set[str] = set()

        self._receive_task: asyncio.Task | None = None
        self._keepalive_task: asyncio.Task | None = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Soniox TTS supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to a Soniox TTS language code.

        Args:
            language: The language to convert.

        Returns:
            The Soniox-specific language code, or None if not supported.
        """
        return language_to_soniox_tts_language(language)

    async def start(self, frame: StartFrame):
        """Start the Soniox TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        if self._audio_format.startswith("pcm_") and self.sample_rate not in VALID_SAMPLE_RATES:
            logger.warning(
                f"{self}: sample_rate={self.sample_rate} is not in Soniox supported rates "
                f"{sorted(VALID_SAMPLE_RATES)}; the server may reject the stream."
            )
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the Soniox TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Soniox TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def flush_audio(self, context_id: str | None = None):
        """Flush any pending audio and finalize the current stream.

        Args:
            context_id: The specific context to flush. If ``None``, falls back
                to the currently active context.
        """
        flush_id = context_id or self.get_active_audio_context_id()
        if not flush_id or not self._websocket:
            return
        logger.trace(f"{self}: flushing audio for stream {flush_id}")
        msg = {"text": "", "text_end": True, "stream_id": flush_id}
        await self._websocket.send(json.dumps(msg))

    async def _close_stream(self, context_id: str):
        """Cancel a Soniox stream and forget local state.

        Mirrors Inworld's ``_close_context``. ``cancel:true`` works on any
        currently-open stream (Soniox replies with ``terminated``). Gated on
        ``_configured_contexts`` because ``cancel`` on a stream_id Soniox
        never saw would error. Do not call after ``text_end:true`` — that
        already terminates the stream.
        """
        if context_id in self._configured_contexts:
            if self._websocket and self._websocket.state is State.OPEN:
                try:
                    msg = {"stream_id": context_id, "cancel": True}
                    await self._websocket.send(json.dumps(msg))
                except Exception as e:
                    logger.warning(f"{self}: failed to cancel stream {context_id}: {e}")
        self._configured_contexts.discard(context_id)

    async def on_turn_context_created(self, context_id: str):
        """Eagerly open the Soniox stream when a new turn context is created.

        Overlaps Soniox-side stream creation with sentence aggregation so the
        stream is ready by the time text reaches ``run_tts``.
        """
        try:
            await self._send_config(context_id)
        except Exception as e:
            logger.warning(f"{self}: failed to pre-open Soniox stream {context_id}: {e}")

    async def on_turn_context_completed(self):
        """Cancel any eagerly-opened Soniox stream that never received text.

        The base class sends ``text_end:true`` (via ``flush_audio``) for
        streams that received text — that already terminates the stream. For
        an empty turn (e.g., the LLM produced only tool calls), no text
        reaches ``run_tts`` and the eager-opened stream would otherwise sit
        until Soniox's per-stream idle timer fires. Cancel it here.
        """
        ctx_id = self._turn_context_id
        was_active = ctx_id is not None and self.audio_context_available(ctx_id)
        await super().on_turn_context_completed()
        if ctx_id is not None and not was_active:
            await self._close_stream(ctx_id)

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the active Soniox stream when the bot is interrupted."""
        await self.stop_all_metrics()
        await self._close_stream(context_id)
        await super().on_audio_context_interrupted(context_id)

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta, flushing the active stream if needed.

        All Soniox config fields live in the per-stream config message, so
        changes take effect on the next stream. The current stream is flushed
        so subsequent sentences in this turn open a fresh stream with the
        updated values.

        Args:
            delta: A TTS settings delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)
        if not changed:
            return changed

        if changed.keys() & {"voice", "model", "language"}:
            if self._turn_context_id and self.audio_context_available(self._turn_context_id):
                await self.flush_audio(context_id=self._turn_context_id)
            # Assign a new turn context ID so subsequent sentences in this turn
            # open a new Soniox stream with the updated settings.
            if self._turn_context_id:
                self._turn_context_id = None
                self._turn_context_id = self.create_context_id()

        return changed

    async def _connect(self):
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

        if self._websocket and not self._keepalive_task:
            self._keepalive_task = self.create_task(self._keepalive_task_handler())

    async def _disconnect(self):
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        if self._keepalive_task:
            await self.cancel_task(self._keepalive_task)
            self._keepalive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            logger.debug("Connecting to Soniox TTS")
            # Soniox expects the api_key in the per-stream config message, not
            # as a header or query param, so the connect call is bare.
            self._websocket = await websocket_connect(self._url)
            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            await self.push_error(error_msg=f"Unable to connect to Soniox TTS: {e}", exception=e)
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug("Disconnecting from Soniox TTS")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing Soniox websocket: {e}", exception=e)
        finally:
            await self.remove_active_audio_context()
            self._configured_contexts.clear()
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    def _build_config_msg(self, context_id: str) -> dict[str, Any]:
        """Build the per-stream configuration message for a new stream_id."""
        s = self._settings
        config: dict[str, Any] = {
            "api_key": self._api_key,
            "stream_id": context_id,
            "model": s.model,
            "voice": s.voice,
            "audio_format": self._audio_format,
        }
        if s.language is not None:
            config["language"] = s.language
        if self._audio_format.startswith("pcm_"):
            config["sample_rate"] = self.sample_rate
        return config

    async def _send_config(self, context_id: str):
        """Send the per-stream config for ``context_id``, idempotently.

        Soniox rejects duplicate config for the same stream_id, so the set of
        already-configured contexts gates the send. Mirrors Inworld's
        ``_send_context``.
        """
        if context_id in self._configured_contexts:
            return
        config = self._build_config_msg(context_id)
        await self._get_websocket().send(json.dumps(config))
        self._configured_contexts.add(context_id)
        logger.trace(f"{self}: opened Soniox stream {context_id}")

    async def _keepalive_task_handler(self):
        """Send periodic keepalive messages to prevent Soniox's idle timeout.

        Soniox closes idle connections after 20-30s; sending ``{"keep_alive": true}``
        resets the timer without triggering synthesis.
        """
        while True:
            await asyncio.sleep(KEEPALIVE_INTERVAL_SECONDS)
            try:
                if self._websocket and self._websocket.state is State.OPEN:
                    await self._websocket.send(json.dumps({"keep_alive": True}))
                    logger.trace(f"{self}: sent Soniox keepalive")
            except websockets.ConnectionClosed as e:
                logger.warning(f"{self} keepalive error: {e}")
                break
            except Exception as e:
                logger.warning(f"{self}: unexpected keepalive error: {e}")
                break

    async def _receive_messages(self):
        """Handle incoming WebSocket messages from Soniox.

        Routes audio, error, and terminal events to the appropriate audio
        contexts. A failed stream does not close the WebSocket; other active
        streams continue uninterrupted.
        """
        async for message in self._get_websocket():
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"{self}: received non-JSON Soniox message: {message!r}")
                continue

            stream_id = msg.get("stream_id")

            error_code = msg.get("error_code")
            if error_code is not None:
                error_message = msg.get("error_message", "")
                await self.push_error(
                    error_msg=f"Soniox TTS error {error_code} (stream {stream_id}): {error_message}"
                )
                if stream_id and self.audio_context_available(stream_id):
                    await self.append_to_audio_context(
                        stream_id, TTSStoppedFrame(context_id=stream_id)
                    )
                    await self.remove_audio_context(stream_id)
                self._configured_contexts.discard(stream_id)
                continue

            if msg.get("terminated"):
                if stream_id and self.audio_context_available(stream_id):
                    await self.append_to_audio_context(
                        stream_id, TTSStoppedFrame(context_id=stream_id)
                    )
                    await self.remove_audio_context(stream_id)
                self._configured_contexts.discard(stream_id)
                continue

            audio_b64 = msg.get("audio")
            if audio_b64 and stream_id and self.audio_context_available(stream_id):
                await self.stop_ttfb_metrics()
                audio = base64.b64decode(audio_b64)
                frame = TTSAudioRawFrame(audio, self.sample_rate, 1, context_id=stream_id)
                await self.append_to_audio_context(stream_id, frame)

            # audio_end is informational; the real end-of-stream signal is
            # `terminated`, handled above.

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Stream text to Soniox and deliver synthesized audio asynchronously.

        The first ``run_tts`` call for a given ``context_id`` sends the
        per-stream config message; subsequent calls within the same stream
        send only text chunks. Audio arrives via the receive loop and is
        appended to the matching audio context.

        Args:
            text: The text to synthesize.
            context_id: The audio context (maps to Soniox ``stream_id``).

        Yields:
            ``None`` — audio frames are delivered out of band via the receive
            task and the audio-context queue.
        """
        if self._is_streaming_tokens:
            logger.trace(f"{self}: Generating TTS [{text}]")
        else:
            logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                text_msg = {"text": text, "text_end": False, "stream_id": context_id}
                await self._get_websocket().send(json.dumps(text_msg))
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
