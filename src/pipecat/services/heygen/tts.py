#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""HeyGen Voice text-to-speech service implementation.

Provides two TTS services against HeyGen Voice:

- :class:`HeyGenTTSService` streams text in and audio out over the
  ``/v3/models/audio/tts/multi-stream-input`` WebSocket, with word timestamps.
- :class:`HeyGenHttpTTSService` streams raw PCM audio from the
  ``POST /v3/models/audio/tts/stream`` endpoint, one request per sentence.

See https://developers.heygen.com/docs/voices/heygen-voice-speech.
"""

import asyncio
import base64
import json
import time
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlencode

import aiohttp
from loguru import logger
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.protocol import State

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame, TTSStoppedFrame
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService, WebsocketTTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given

# Nadine, a HeyGen public voice. All HeyGen Voice models serve it.
_DEFAULT_VOICE_ID = "83548de556df48ba8c09c42a57c51d85"

_HTTP_SAMPLE_RATES = (8000, 16000, 22050, 24000, 44100, 48000)

_WEBSOCKET_SAMPLE_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)

# HeyGen synthesizes at this rate when it does not offer the pipeline rate. The
# service then resamples the audio.
_FALLBACK_SAMPLE_RATE = 24000

_KEEPALIVE_INTERVAL_S = 10

# The service never opens this context. Idle keepalives close it.
_KEEPALIVE_CONTEXT_ID = "pipecat-keepalive"

# HeyGen can queue a context for 45 seconds at full capacity. The base class ends
# a quiet context and drops its late audio, which HeyGen still bills.
_AUDIO_CONTEXT_TIMEOUT_S = 50.0

_ERROR_CATEGORIES: dict[str, ErrorCategory] = {
    "rate_limit_exceeded": ErrorCategory.RATE_LIMIT,
    "usage_limit_reached": ErrorCategory.QUOTA,
    "insufficient_credit": ErrorCategory.QUOTA,
    "plan_upgrade_required": ErrorCategory.QUOTA,
}

_LANGUAGE_MAP = {
    Language.AR: "ar",
    Language.BE: "be",
    Language.BG: "bg",
    Language.CA: "ca",
    Language.CS: "cs",
    Language.DA: "da",
    Language.DE: "de",
    Language.EL: "el",
    Language.EN: "en",
    Language.ES: "es",
    Language.FA: "fa",
    Language.FI: "fi",
    Language.FR: "fr",
    Language.HE: "he",
    Language.HI: "hi",
    Language.HR: "hr",
    Language.HU: "hu",
    Language.ID: "id",
    Language.IT: "it",
    Language.JA: "ja",
    Language.KO: "ko",
    Language.MK: "mk",
    Language.MS: "ms",
    Language.NL: "nl",
    Language.PL: "pl",
    Language.PT: "pt",
    Language.RO: "ro",
    Language.RU: "ru",
    Language.SK: "sk",
    Language.SL: "sl",
    Language.SR: "sr",
    Language.SV: "sv",
    Language.TA: "ta",
    Language.TH: "th",
    Language.TL: "tl",
    Language.TR: "tr",
    Language.UK: "uk",
    Language.VI: "vi",
    Language.ZH: "zh",
}

_LANGUAGES = frozenset(_LANGUAGE_MAP.values())


def language_to_heygen_language(language: Language) -> str:
    """Convert a Language enum to a HeyGen Voice language code.

    HeyGen Voice takes lowercase ISO 639-1 codes and has no regional accents, so
    regional variants (e.g. ``en-US``) resolve to their base code (``en``).

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding HeyGen language code.
    """
    return resolve_language(language, _LANGUAGE_MAP, use_base_code=True)


def _synthesis_rate(sample_rate: int, supported: tuple[int, ...]) -> int:
    """Return the PCM rate to request from HeyGen for the pipeline's rate."""
    return sample_rate if sample_rate in supported else _FALLBACK_SAMPLE_RATE


def _error_detail(body: str | bytes) -> str:
    """Unwrap HeyGen's ``{"error": {"code", "message", "request_id"}}`` envelope.

    Args:
        body: A response body or WebSocket message.

    Returns:
        The error's code, message and request ID, or the raw body if it is not
        an envelope.
    """
    try:
        error = json.loads(body)["error"]
        return f"{error['code']}: {error['message']} (request_id: {error.get('request_id')})"
    except (ValueError, KeyError, TypeError):
        return str(body[:500])


@dataclass
class HeyGenTTSSettings(TTSSettings):
    """Settings for HeyGenHttpTTSService.

    Parameters:
        model: HeyGen Voice model: ``heygen-voice-1``, ``heygen-voice-1-turbo``,
            or ``heygen-voice-1-turbo-free``.
        voice: HeyGen voice ID: a public voice, or an instant or professional
            clone in your workspace.
        language: Language of the text. If ``None``, HeyGen detects it.
    """

    pass


class HeyGenHttpTTSService(TTSService):
    """HeyGen Voice HTTP text-to-speech service.

    Sends each sentence to HeyGen's streaming speech endpoint and yields the raw
    PCM audio as it arrives.

    Defaults to ``heygen-voice-1-turbo``, the lowest-latency model, and the
    public voice Nadine.
    """

    Settings = HeyGenTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        base_url: str = "https://api.heygen.com",
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the HeyGen TTS service.

        Args:
            api_key: HeyGen API key.
            aiohttp_session: aiohttp ClientSession for HTTP requests.
            base_url: HeyGen API base URL.
            sample_rate: Output audio sample rate in Hz. If None, uses the
                pipeline default.
            settings: Runtime-updatable settings.
            **kwargs: Additional keyword arguments passed to ``TTSService``.
        """
        default_settings = self.Settings(
            model="heygen-voice-1-turbo",
            voice=_DEFAULT_VOICE_ID,
            language=None,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._session = aiohttp_session
        self._url = f"{base_url.rstrip('/')}/v3/models/audio/tts/stream"

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as HeyGen TTS supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to a HeyGen Voice language code.

        Args:
            language: The language to convert.

        Returns:
            The HeyGen language code.
        """
        return language_to_heygen_language(language)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using HeyGen's streaming speech endpoint.

        Args:
            text: Text to synthesize.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        synthesis_sample_rate = _synthesis_rate(self.sample_rate, _HTTP_SAMPLE_RATES)
        headers = {"x-api-key": self._api_key, "Content-Type": "application/json"}
        payload: dict[str, Any] = {
            "text": text,
            "voice_id": assert_given(self._settings.voice),
            "model": assert_given(self._settings.model),
            "output_format": {
                "encoding": "pcm",
                "sample_rate_hertz": synthesis_sample_rate,
            },
        }
        # HeyGen refuses a language it does not speak, and detects one left out.
        language = assert_given(self._settings.language)
        if language in _LANGUAGES:
            payload["language"] = language

        try:
            async with self._session.post(self._url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error = _error_detail(await response.text(errors="ignore"))
                    yield ErrorFrame(error=f"HeyGen TTS error (status: {response.status}): {error}")
                    return

                await self.start_tts_usage_metrics(text)

                # A chunk can split a sample. The helper keeps frames sample-aligned
                # and resamples them.
                async for frame in self._stream_audio_frames_from_iterator(
                    response.content.iter_chunked(self.chunk_size),
                    in_sample_rate=synthesis_sample_rate,
                    context_id=context_id,
                ):
                    await self.stop_ttfb_metrics()
                    yield frame
        except Exception as e:
            yield ErrorFrame(error=f"HeyGen TTS error: {e}")


@dataclass
class HeyGenWebsocketTTSSettings(TTSSettings):
    """Settings for HeyGenTTSService.

    Parameters:
        model: HeyGen Voice model: ``heygen-voice-1``, ``heygen-voice-1-turbo``
            or ``heygen-voice-1-turbo-free``.
        voice: HeyGen voice ID: a public voice or an instant clone in your workspace.
        language: Language of the text. If ``None``, HeyGen detects it.
        apply_text_normalization: Text normalization mode (``"auto"``, ``"on"`` or
            ``"off"``). If ``None``, HeyGen uses ``"auto"``.
        expressiveness_boost: Voice expressiveness from 0.0 to 1.0. If ``None``,
            HeyGen chooses it.
    """

    apply_text_normalization: Literal["auto", "on", "off"] | None | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )
    expressiveness_boost: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


@dataclass
class _WordClock:
    """Word-timing state of one context.

    Parameters:
        offset_s: Seconds of the context's audio before the current message. Each
            alignment's times start at zero, and some audio has no alignment.
        partial_word: A word whose end has not arrived yet.
        partial_word_start_s: Start time of ``partial_word``.
    """

    offset_s: float = 0.0
    partial_word: str = ""
    partial_word_start_s: float = 0.0

    def add(self, alignment: Mapping[str, Any], per_character: bool) -> list[tuple[str, float]]:
        """Take one audio message's character alignment.

        Args:
            alignment: The message's ``chars`` and ``charStartTimesMs``.
            per_character: Whether each character is a word, as in Chinese and
                Japanese, which have no spaces between words.

        Returns:
            The (word, start seconds) pairs this alignment completes.
        """
        words = []
        for char, start_ms in zip(alignment["chars"], alignment["charStartTimesMs"]):
            if char != " ":
                if not self.partial_word:
                    self.partial_word_start_s = self.offset_s + start_ms / 1000
                self.partial_word += char
            if self.partial_word and (char == " " or per_character):
                words.append((self.partial_word, self.partial_word_start_s))
                self.partial_word = ""
        return words


class HeyGenTTSService(WebsocketTTSService):
    """HeyGen Voice WebSocket text-to-speech service.

    Streams text to HeyGen's ``multi-stream-input`` WebSocket and plays the audio
    as it arrives. One connection carries a context per turn, and an interruption
    closes the turn's context without reconnecting. HeyGen's per-character
    alignment drives word-level ``TTSTextFrame`` objects, so an interruption
    commits only the words actually spoken.

    Defaults to ``heygen-voice-1-turbo`` and the public voice Nadine.
    Professional voice clones are served only by :class:`HeyGenHttpTTSService`.
    """

    Settings = HeyGenWebsocketTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "wss://api.heygen.com",
        sample_rate: int | None = None,
        enable_ssml_parsing: bool | None = None,
        enable_logging: bool | None = None,
        end_user_id: str | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the HeyGen WebSocket TTS service.

        Args:
            api_key: HeyGen API key.
            url: HeyGen WebSocket base URL.
            sample_rate: Output audio sample rate in Hz. If None, uses the
                pipeline default.
            enable_ssml_parsing: Whether ``<break time="1s"/>`` tags become pauses.
                Without it, HeyGen closes the WebSocket on a break tag.
            enable_logging: Whether HeyGen stores a copy of the audio and text.
                If None, HeyGen stores them.
            end_user_id: Your identifier for the end user, recorded with HeyGen's
                usage for attribution.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to ``WebsocketTTSService``.
        """
        default_settings = self.Settings(
            model="heygen-voice-1-turbo",
            voice=_DEFAULT_VOICE_ID,
            language=None,
            apply_text_normalization=None,
            expressiveness_boost=None,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_text_frames=False,
            push_start_frame=True,
            stop_frame_timeout_s=_AUDIO_CONTEXT_TIMEOUT_S,
            # HeyGen closes the socket at more than 5 running contexts. So a turn
            # plays its audio before the next turn opens a context.
            pause_frame_processing=True,
            # HeyGen joins text messages without a separator.
            append_trailing_space=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._url = url.rstrip("/")
        self._enable_ssml_parsing = enable_ssml_parsing
        self._enable_logging = enable_logging
        self._end_user_id = end_user_id

        self._receive_task = None
        # HeyGen's synthesis rate. The URL builder sets it.
        self._synthesis_sample_rate = _FALLBACK_SAMPLE_RATE
        # The monotonic time before which a refused handshake asked for no other.
        self._connect_not_before = 0.0
        # Open contexts. A message that names any other context opens it at HeyGen.
        self._open_contexts: set[str] = set()
        self._word_clocks: dict[str, _WordClock] = {}

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as HeyGen TTS supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to a HeyGen Voice language code.

        Args:
            language: The language to convert.

        Returns:
            The HeyGen language code.
        """
        return language_to_heygen_language(language)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def _connect(self):
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

        if self._websocket:
            self._create_keepalive_task(_KEEPALIVE_INTERVAL_S)

    async def _disconnect(self):
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._cancel_keepalive_task()

        await self._disconnect_websocket()

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta. Every setting is in the URL, so a change reconnects."""
        changed = await super()._update_settings(delta)

        if changed:
            await self._disconnect()
            await self._connect()

        return changed

    def _build_websocket_url(self) -> str:
        self._synthesis_sample_rate = _synthesis_rate(self.sample_rate, _WEBSOCKET_SAMPLE_RATES)
        params = {
            "voice_id": assert_given(self._settings.voice),
            "model": assert_given(self._settings.model),
            "output_format": f"pcm_{self._synthesis_sample_rate}",
            # Without auto mode, HeyGen waits for 120 characters of text.
            "auto_mode": "true",
        }
        language = assert_given(self._settings.language)
        if language in _LANGUAGES:
            params["language"] = language
        if self._settings.apply_text_normalization is not None:
            params["apply_text_normalization"] = self._settings.apply_text_normalization
        if self._settings.expressiveness_boost is not None:
            params["expressiveness_boost"] = str(self._settings.expressiveness_boost)
        if self._enable_ssml_parsing is not None:
            params["enable_ssml_parsing"] = str(self._enable_ssml_parsing).lower()
        if self._enable_logging is not None:
            params["enable_logging"] = str(self._enable_logging).lower()
        if self._end_user_id:
            params["end_user_id"] = self._end_user_id
        return f"{self._url}/v3/models/audio/tts/multi-stream-input?{urlencode(params)}"

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to HeyGen")

            self._websocket = await self._websocket_connect(
                self._build_websocket_url(),
                # An audio message can carry many seconds of base64 PCM.
                max_size=16 * 1024 * 1024,
                additional_headers={"x-api-key": self._api_key},
            )

            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            error = f"{e}"
            if isinstance(e, InvalidStatus):
                # The response holds the reason for the refusal.
                if e.response.body:
                    error += f": {_error_detail(e.response.body.decode(errors='replace'))}"
                retry_after = e.response.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    self._connect_not_before = time.monotonic() + int(retry_after)
            await self.push_error(error_msg=f"Error connecting to HeyGen: {error}", exception=e)
            await self._call_event_handler("on_connection_error", error)

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from HeyGen")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error disconnecting from HeyGen: {e}", exception=e)
        finally:
            self._websocket = None
            self._open_contexts.clear()
            await self._call_event_handler("on_disconnected")

    async def _reconnect_websocket(self, attempt_number: int) -> bool:
        """Reconnect, first waiting as long as the last refused handshake asked.

        Every handshake spends HeyGen's rate limit, so one made before that time
        is refused and only delays recovery.
        """
        delay = self._connect_not_before - time.monotonic()
        if delay > 0:
            await self._close_connection_contexts()
            logger.debug(f"{self}: waiting {delay} s before reconnecting, as HeyGen asked")
            await asyncio.sleep(delay)
        return await super()._reconnect_websocket(attempt_number)

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    @property
    def _is_connected(self) -> bool:
        return self._websocket is not None and self._websocket.state is State.OPEN

    async def _send(self, context_id: str, **message: Any):
        if not self._websocket:
            return
        try:
            await self._websocket.send(json.dumps({"context_id": context_id, **message}))
        except ConnectionClosed as e:
            # The receive loop handles a lost connection and its contexts.
            logger.debug(f"{self}: message to context {context_id} not sent: {e}")

    async def _close_context(self, context_id: str | None):
        if context_id in self._open_contexts:
            self._open_contexts.discard(context_id)
            await self._send(context_id, close_context=True)

    def _forget_context(self, context_id: str):
        self._open_contexts.discard(context_id)
        self._word_clocks.pop(context_id, None)

    async def flush_audio(self, context_id: str | None = None):
        """Ask HeyGen to generate any text still buffered for a context.

        Args:
            context_id: The context to flush. If None, the active context.
        """
        flush_id = context_id or self.get_active_audio_context_id()
        if flush_id in self._open_contexts:
            await self._send(flush_id, flush=True)

    async def on_turn_context_completed(self):
        """Close the turn's context so its final message arrives."""
        context_id = self._turn_context_id
        # The parent flushes first. A close without a flush discards unspoken text.
        await super().on_turn_context_completed()
        await self._close_context(context_id)

    async def on_audio_context_interrupted(self, context_id: str):
        """Close the context when the bot is interrupted, keeping the connection."""
        await self.stop_all_metrics()
        await self._close_context(context_id)
        self._forget_context(context_id)
        await super().on_audio_context_interrupted(context_id)

    async def on_audio_context_completed(self, context_id: str):
        """Forget the context once its audio has played."""
        self._forget_context(context_id)
        await super().on_audio_context_completed(context_id)

    async def _send_keepalive(self):
        # A keepalive that names a closed context opens it at HeyGen. With no
        # context_id, it opens "default". So a keepalive names only an open context.
        # With no open context, the service closes an unknown context. This resets
        # HeyGen's inactivity timer and changes nothing else.
        context_id = self.get_active_audio_context_id()
        if context_id in self._open_contexts:
            await self._send(context_id, text="")
        else:
            await self._send(_KEEPALIVE_CONTEXT_ID, close_context=True)

    async def _receive_messages(self):
        async for message in self._get_websocket():
            await self._handle_message(message)

    async def _handle_message(self, message: str | bytes):
        msg = json.loads(message)

        # HeyGen sends an error message just before it closes the socket.
        if msg.get("error"):
            await self.push_error(
                error_msg=f"HeyGen TTS error: {_error_detail(message)}",
                category=_ERROR_CATEGORIES.get(msg["error"].get("code")),
            )
            return

        context_id = msg.get("contextId")

        if msg.get("isFinal") is True:
            await self._finish_context(context_id)
            return

        # An interrupted context's audio and alignment can still arrive.
        if not msg.get("audio") or not self.audio_context_available(context_id):
            return

        audio = base64.b64decode(msg["audio"])
        seconds = len(audio) / 2 / self._synthesis_sample_rate
        if self._synthesis_sample_rate != self.sample_rate:
            audio = await self._resampler.resample(
                audio, self._synthesis_sample_rate, self.sample_rate
            )
        frame = TTSAudioRawFrame(audio, self.sample_rate, 1, context_id=context_id)
        await self.append_to_audio_context(context_id, frame)

        clock = self._word_clocks.setdefault(context_id, _WordClock())
        if msg.get("alignment"):
            await self._add_words(clock.add(msg["alignment"], self._is_cjk), context_id)
        clock.offset_s += seconds

    async def _finish_context(self, context_id: str):
        """End a context HeyGen reported final, which it can do on its own mid-turn."""
        if self.audio_context_available(context_id):
            clock = self._word_clocks.get(context_id)
            if clock and clock.partial_word:
                await self._add_words(
                    [(clock.partial_word, clock.partial_word_start_s)], context_id
                )
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)
        self._forget_context(context_id)
        # A final context takes no more text. The next sentence needs a new context.
        if context_id and context_id == self._turn_context_id:
            self._turn_context_id = None
            self._turn_context_id = self.create_context_id()

    @property
    def _is_cjk(self) -> bool:
        return assert_given(self._settings.language) in ("ja", "zh")

    async def _add_words(self, words: list[tuple[str, float]], context_id: str):
        await self.add_word_timestamps(
            words,
            context_id,
            includes_inter_frame_spaces=self._is_cjk,
            pre_merge_tokens=self._is_cjk,
        )

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Send text to the context for synthesis.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Nothing directly; audio arrives on the receive task.
        """
        try:
            # A handshake during a reconnect, or before the time HeyGen asked for,
            # spends the rate limit.
            if (
                not self._is_connected
                and not self._reconnect_in_progress
                and time.monotonic() >= self._connect_not_before
            ):
                await self._connect()
            if not self._is_connected:
                # No audio can arrive. An open context holds frames until it times out.
                await self.remove_audio_context(context_id)
                yield ErrorFrame(error="HeyGen TTS WebSocket is not connected")
                return

            self._open_contexts.add(context_id)
            await self._send(context_id, text=text)
            await self.start_tts_usage_metrics(text)

            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
