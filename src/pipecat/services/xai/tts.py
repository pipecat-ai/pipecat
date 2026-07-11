#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""xAI text-to-speech service implementation.

Provides two TTS services against xAI's voice API:

- :class:`XAIHttpTTSService` uses the batch HTTP endpoint at
  ``https://api.x.ai/v1/tts``.
- :class:`XAITTSService` uses the streaming WebSocket endpoint at
  ``wss://api.x.ai/v1/tts``.

See https://docs.x.ai/developers/rest-api-reference/inference/voice.
"""

import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import aiohttp
from loguru import logger
from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven, assert_given
from pipecat.services.tts_service import TTSService, WebsocketTTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts


def language_to_xai_language(language: Language) -> str:
    """Convert a Language enum to xAI language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding service language code. If ``language`` is not in
        the verified mapping, falls back to the base language code (e.g.,
        ``en`` from ``en-US``) and logs a warning (via
        ``resolve_language(..., use_base_code=True)``).
    """
    LANGUAGE_MAP = {
        Language.AR: "ar-EG",
        Language.AR_EG: "ar-EG",
        Language.AR_SA: "ar-SA",
        Language.AR_AE: "ar-AE",
        Language.BN: "bn",
        Language.DE: "de",
        Language.EN: "en",
        Language.ES: "es-ES",
        Language.ES_ES: "es-ES",
        Language.ES_MX: "es-MX",
        Language.FR: "fr",
        Language.HI: "hi",
        Language.ID: "id",
        Language.IT: "it",
        Language.JA: "ja",
        Language.KO: "ko",
        Language.PT: "pt-PT",
        Language.PT_BR: "pt-BR",
        Language.PT_PT: "pt-PT",
        Language.RU: "ru",
        Language.TR: "tr",
        Language.VI: "vi",
        Language.ZH: "zh",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


def _xai_word_times(
    graph_chars: list[str],
    graph_times: list[list[float]],
    partial_word: str = "",
    partial_word_start_time: float = 0.0,
) -> tuple[list[tuple[str, float]], str, float]:
    """Convert xAI character timings into ``(word, start_time)`` pairs.

    When ``with_timestamps`` is enabled, each xAI ``audio.delta`` carries
    per-character ``graph_chars`` (including spaces and punctuation) and
    ``graph_times`` as ``[start, end]`` second pairs. The times are absolute
    from the start of the utterance and continue across chunks, so they are
    used as-is. Words are split on spaces, each assigned the start time of its
    first character, and a word that straddles a chunk boundary is carried over
    via ``partial_word``.

    Returns:
        ``(word_times, partial_word, partial_word_start_time)`` where the latter
        two describe an unterminated trailing word for the next chunk to finish.
    """
    if len(graph_chars) != len(graph_times):
        logger.error(
            f"xAI timestamp length mismatch: chars={len(graph_chars)}, times={len(graph_times)}"
        )
        return ([], partial_word, partial_word_start_time)

    words: list[str] = []
    word_start_times: list[float] = []
    current_word = partial_word
    word_start_time: float | None = partial_word_start_time if partial_word else None

    for char, times in zip(graph_chars, graph_times):
        if char == " ":
            if current_word:
                words.append(current_word)
                word_start_times.append(word_start_time or 0.0)
                current_word = ""
                word_start_time = None
        else:
            if word_start_time is None:
                word_start_time = times[0]
            current_word += char

    return (
        list(zip(words, word_start_times)),
        current_word,
        word_start_time if word_start_time is not None else 0.0,
    )


@dataclass
class XAITTSSettings(TTSSettings):
    """Settings for XAIHttpTTSService.

    Parameters:
        speed: Speech speed multiplier from 0.7 to 1.5 (1.0 is normal).
        optimize_streaming_latency: Latency optimization level (0, 1, or 2).
        text_normalization: Whether to normalize text before synthesis.
    """

    speed: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    optimize_streaming_latency: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    text_normalization: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class XAIHttpTTSService(TTSService):
    """xAI HTTP text-to-speech service.

    The service requests raw PCM audio so emitted ``TTSAudioRawFrame`` objects
    match Pipecat's downstream expectations without extra decoding.
    """

    Settings = XAITTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.x.ai/v1/tts",
        sample_rate: int | None = None,
        encoding: str | None = "pcm",
        aiohttp_session: aiohttp.ClientSession | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the xAI TTS service.

        Args:
            api_key: xAI API key for authentication.
            base_url: xAI TTS endpoint. Defaults to ``https://api.x.ai/v1/tts``.
            sample_rate: Audio sample rate. If None, uses default.
            encoding: Output encoding format. Defaults to "pcm".
            aiohttp_session: Optional shared aiohttp session.
            settings: Runtime-updatable settings.
            **kwargs: Additional keyword arguments passed to ``TTSService``.
        """
        default_settings = self.Settings(
            model=None,
            voice="eve",
            language=Language.EN,
            speed=None,
            optimize_streaming_latency=None,
            text_normalization=None,
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
        self._base_url = base_url
        self._session = aiohttp_session
        self._session_owner = aiohttp_session is None
        self._encoding = encoding

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to xAI language format.

        Args:
            language: The language to convert.

        Returns:
            The xAI-specific language code, or None if not supported.
        """
        return language_to_xai_language(language)

    async def start(self, frame):
        """Start the xAI TTS service."""
        await super().start(frame)
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._session_owner = True

    async def stop(self, frame):
        """Stop the xAI TTS service."""
        await super().stop(frame)
        await self._close_session()

    async def cancel(self, frame):
        """Cancel the xAI TTS service."""
        await super().cancel(frame)
        await self._close_session()

    async def cleanup(self):
        """Release xAI TTS resources at teardown."""
        await super().cleanup()
        await self._close_session()

    async def _close_session(self):
        if self._session_owner and self._session and not self._session.closed:
            await self._session.close()
        if self._session_owner:
            self._session = None

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using xAI's TTS API."""
        logger.debug(f"{self}: Generating TTS [{text}]")

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._session_owner = True

        payload: dict[str, Any] = {
            "text": text,
            "voice_id": self._settings.voice,
            "output_format": {
                "codec": self._encoding,
                "sample_rate": self.sample_rate,
            },
        }
        payload["language"] = str(self._settings.language) if self._settings.language else "auto"

        if assert_given(self._settings.speed) is not None:
            payload["speed"] = self._settings.speed
        if assert_given(self._settings.optimize_streaming_latency) is not None:
            payload["optimize_streaming_latency"] = self._settings.optimize_streaming_latency
        if assert_given(self._settings.text_normalization) is not None:
            payload["text_normalization"] = self._settings.text_normalization

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        measuring_ttfb = True
        try:
            async with self._session.post(
                self._base_url, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    error = await response.text(errors="ignore")
                    yield ErrorFrame(
                        error=f"Error getting audio (status: {response.status}, error: {error})"
                    )
                    return

                await self.start_tts_usage_metrics(text)

                async for chunk in response.content.iter_chunked(self.chunk_size):
                    if not chunk:
                        continue
                    if measuring_ttfb:
                        await self.stop_ttfb_metrics()
                        measuring_ttfb = False
                    yield TTSAudioRawFrame(
                        chunk,
                        self.sample_rate,
                        1,
                        context_id=context_id,
                    )
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")


@dataclass
class XAIWebsocketTTSSettings(TTSSettings):
    """Settings for XAITTSService (WebSocket streaming).

    Parameters:
        speed: Speech speed multiplier from 0.7 to 1.5 (1.0 is normal).
        optimize_streaming_latency: Latency optimization level (0, 1, or 2).
        text_normalization: Whether to normalize text before synthesis.
        with_timestamps: Whether to request character timings. When enabled, the
            service converts them into per-word ``TTSTextFrame`` objects.
    """

    speed: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    optimize_streaming_latency: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    text_normalization: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    with_timestamps: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class XAITTSService(WebsocketTTSService):
    """xAI streaming text-to-speech service.

    Connects to xAI's WebSocket TTS endpoint and streams audio chunks back as
    they are synthesized. Text can be sent incrementally via ``text.delta``
    messages and each utterance is terminated with ``text.done``. The server
    responds with ``audio.delta`` chunks followed by an ``audio.done`` message.

    Audio parameters (voice, language, codec, sample rate) are passed as query
    string parameters on the WebSocket URL; changing any of them at runtime
    reconnects the WebSocket. With ``with_timestamps`` enabled, xAI's
    per-character timings are converted into per-word ``TTSTextFrame`` objects.

    Note that xAI delivers timestamps in batches that are decoupled from the
    audio stream (a batch can cover several seconds of speech and arrive in one
    message), so the word ``TTSTextFrame`` objects are pushed in bursts rather
    than evenly spread across playback. Each frame still carries an accurate
    ``pts``, so consumers should schedule off ``pts`` rather than arrival time.
    """

    Settings = XAIWebsocketTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "wss://api.x.ai/v1/tts",
        sample_rate: int | None = None,
        codec: str = "pcm",
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the xAI WebSocket TTS service.

        Args:
            api_key: xAI API key for authentication.
            base_url: xAI TTS WebSocket endpoint. Defaults to
                ``wss://api.x.ai/v1/tts``.
            sample_rate: Output audio sample rate in Hz. If None, uses the
                pipeline default.
            codec: Output audio codec. One of ``pcm``, ``wav``, ``mulaw``,
                ``alaw``. Defaults to ``pcm`` so emitted ``TTSAudioRawFrame``
                objects need no decoding downstream.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to parent
                ``WebsocketTTSService``.
        """
        default_settings = self.Settings(
            model=None,
            voice="eve",
            language=Language.EN,
            speed=None,
            optimize_streaming_latency=None,
            text_normalization=None,
            with_timestamps=True,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        # With word timestamps enabled, per-word TTSTextFrames drive the text
        # output, so suppress the base class's aggregated text frame. Without
        # them, fall back to the normal aggregated-text behavior.
        with_timestamps = bool(assert_given(default_settings.with_timestamps))

        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=not with_timestamps,
            append_trailing_space=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url
        self._codec = codec
        self._receive_task = None

        self._timestamp_context_id: str | None = None
        self._partial_word: str = ""
        self._partial_word_start_time: float = 0.0

    def _reset_timestamp_state(self):
        self._timestamp_context_id = None
        self._partial_word = ""
        self._partial_word_start_time = 0.0

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to xAI language format."""
        return language_to_xai_language(language)

    async def start(self, frame: StartFrame):
        """Start the xAI WebSocket TTS service."""
        await super().start(frame)
        await self._connect()

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

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta. Reconnects if any URL-baked field changes."""
        changed = await super()._update_settings(delta)

        if changed:
            await self._disconnect()
            await self._connect()

        return changed

    def _build_url(self) -> str:
        language = self._settings.language
        if isinstance(language, Language):
            language_value = language_to_xai_language(language) or language.value
        else:
            language_value = str(language) if language is not None else "auto"

        params: dict[str, Any] = {
            "voice": self._settings.voice,
            "language": language_value,
            "codec": self._codec,
            "sample_rate": self.sample_rate,
        }
        if assert_given(self._settings.speed) is not None:
            params["speed"] = self._settings.speed
        if assert_given(self._settings.optimize_streaming_latency) is not None:
            params["optimize_streaming_latency"] = self._settings.optimize_streaming_latency
        # urlencode stringifies bools as "True"/"False"; xAI expects "true"/"false".
        if assert_given(self._settings.text_normalization) is not None:
            params["text_normalization"] = str(self._settings.text_normalization).lower()
        if assert_given(self._settings.with_timestamps) is not None:
            params["with_timestamps"] = str(self._settings.with_timestamps).lower()
        return f"{self._base_url}?{urlencode(params)}"

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to xAI TTS")

            url = self._build_url()
            headers = {"Authorization": f"Bearer {self._api_key}"}
            self._websocket = await websocket_connect(url, additional_headers=headers)

            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from xAI TTS")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error disconnecting from xAI TTS: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def flush_audio(self, context_id: str | None = None):
        """Signal end-of-utterance so xAI begins synthesizing what it has buffered."""
        if not self._websocket or self._websocket.state is State.CLOSED:
            return
        await self._get_websocket().send(json.dumps({"type": "text.done"}))

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the current xAI utterance on barge-in without reconnecting."""
        await self.stop_all_metrics()
        if self._websocket and self._websocket.state is State.OPEN:
            await self._get_websocket().send(json.dumps({"type": "text.clear"}))
        self._reset_timestamp_state()
        await super().on_audio_context_interrupted(context_id)

    async def _close_audio_context(self, context_id: str):
        """Mark a context finished: emit its stop frame and drop it."""
        await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
        await self.remove_audio_context(context_id)

    async def _handle_word_timestamps(self, msg: dict, context_id: str):
        """Emit word-timestamp frames from an ``audio.delta`` timing payload."""
        timestamps = msg.get("audio_timestamps")
        if not timestamps:
            return
        graph_chars = timestamps.get("graph_chars")
        graph_times = timestamps.get("graph_times")
        if not graph_chars or not graph_times:
            return

        # A new utterance clears any word carried over from the previous one.
        if context_id != self._timestamp_context_id:
            self._reset_timestamp_state()
            self._timestamp_context_id = context_id

        word_times, self._partial_word, self._partial_word_start_time = _xai_word_times(
            graph_chars,
            graph_times,
            self._partial_word,
            self._partial_word_start_time,
        )

        if word_times:
            await self.add_word_timestamps(word_times, context_id)

    async def _receive_messages(self):
        async for message in self._get_websocket():
            if isinstance(message, bytes):
                logger.warning(f"{self}: unexpected binary frame from xAI TTS")
                continue
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                logger.error(f"{self}: invalid JSON message: {message}")
                continue

            msg_type = msg.get("type")
            context_id = self.get_active_audio_context_id()

            if msg_type == "audio.delta":
                audio_b64 = msg.get("delta")
                if audio_b64:
                    await self.stop_ttfb_metrics()
                if context_id:
                    if audio_b64:
                        frame = TTSAudioRawFrame(
                            audio=base64.b64decode(audio_b64),
                            sample_rate=self.sample_rate,
                            num_channels=1,
                            context_id=context_id,
                        )
                        await self.append_to_audio_context(context_id, frame)
                    # Timestamps arrive in their own (sometimes audio-less) deltas.
                    await self._handle_word_timestamps(msg, context_id)
            elif msg_type == "audio.done":
                await self.stop_all_metrics()
                if context_id:
                    # Flush a trailing word that had no terminating space.
                    if self._timestamp_context_id == context_id and self._partial_word:
                        await self.add_word_timestamps(
                            [(self._partial_word, self._partial_word_start_time)], context_id
                        )
                    self._reset_timestamp_state()
                    await self._close_audio_context(context_id)
            elif msg_type == "error":
                await self.stop_all_metrics()
                self._reset_timestamp_state()
                error_detail = msg.get("message") or msg.get("error") or str(msg)
                if context_id:
                    await self._close_audio_context(context_id)
                await self.push_error(error_msg=f"xAI TTS error: {error_detail}")
            elif msg_type == "audio.clear":
                logger.trace(f"{self}: xAI acknowledged audio clear")
            else:
                logger.debug(f"{self}: unhandled xAI message type: {msg_type}")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate TTS audio from text using xAI's streaming WebSocket API."""
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                await self._get_websocket().send(json.dumps({"type": "text.delta", "delta": text}))
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
