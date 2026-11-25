#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Plivo text-to-speech service implementations.

See https://www.plivo.com/docs/models/text-to-speech/overview for more details on Koyal,
Plivo's TTS model.
"""

import json
import re
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from loguru import logger
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.protocol import State

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, TTSService, WebsocketTTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.text.phonemes import normalize_ipa
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven


def language_to_plivo_language(language: Language) -> str:
    """Convert a Language enum to Plivo's language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Plivo language code. The four English regions (``en-IN``, ``en-US``,
        ``en-GB``, ``en-AU``) are kept. Any other regional code falls back to its base code
        (e.g. ``hi`` from ``Language.HI_IN``) and logs a warning.
    """
    LANGUAGE_MAP = {
        Language.BN: "bn",
        Language.EN_AU: "en-AU",
        Language.EN_GB: "en-GB",
        Language.EN_IN: "en-IN",
        Language.EN_US: "en-US",
        Language.GU: "gu",
        Language.HI: "hi",
        Language.KN: "kn",
        Language.MR: "mr",
        Language.TA: "ta",
        Language.TE: "te",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


def format_plivo_pronunciation(word: str, ipa: str) -> str | None:
    """Render a pronunciation as Plivo phoneme spans.

    The TTS model reads a phoneme span between braces, in IPA or ARPAbet: one span for each
    word of the IPA.

    Args:
        word: The word being pronounced (unused: the spans replace it).
        ipa: The pronunciation, in IPA.

    Returns:
        The phoneme spans, e.g. ``{ˈpliːvoʊ}``, or None for an empty pronunciation.
    """
    return " ".join("{" + word_ipa + "}" for word_ipa in normalize_ipa(ipa).split()) or None


@dataclass
class PlivoTTSSettings(TTSSettings):
    """Settings for PlivoTTSService and PlivoHttpTTSService.

    Parameters:
        model: TTS model to use. Defaults to ``"koyal-v1"``.
        voice: ID of the voice. Defaults to ``"lC7qzMoRE7SSjaNCVeff"`` (Hailey, US English).
        language: Language of the text, such as ``Language.HI`` or ``Language.EN_IN``. For
            English, the region (``Language.EN_IN``, ``Language.EN_US``, ``Language.EN_GB`` or
            ``Language.EN_AU``) sets how numbers and dates are normalized; the voice keeps its
            accent. By default, the TTS model finds the language of each sentence among the
            voice's languages.
        speed: Speaking rate, from 0.7 to 1.2. Defaults to ``1.0``.
        volume: Volume of the speech, from 0.5 to 2.0. Defaults to ``1.0``.
        pronunciation_dictionary_id: ID of a pronunciation dictionary. Its rules can be used
            to set the pronunciation of a word in IPA, in ARPAbet or as a respelling.

            Another way to guide pronunciation is to pass
            :func:`~pipecat.utils.text.transforms.replace_text` find-and-replace rules
            through the ``text_transforms`` parameter. Pipecat applies them before the
            text reaches the provider, so the same rules can work across TTS services
            and depend less on what each provider supports.
        seed: Seed of the speech, from 0 to 2**53 - 1. Defaults to the voice's default seed.
        continuation: Whether the speech of a context continues from the previous context's,
            for continuity between them. Realtime API only. Defaults to ``True``.
        text_normalization: Whether the TTS model normalizes the text before speech: it spells
            out numbers, dates and similar text. Defaults to ``True``.
    """

    speed: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    volume: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    pronunciation_dictionary_id: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    seed: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    continuation: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    text_normalization: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


def _default_settings(settings: PlivoTTSSettings | None) -> PlivoTTSSettings:
    """Return the default settings, updated with those given."""
    defaults = PlivoTTSSettings(
        model="koyal-v1",
        voice="lC7qzMoRE7SSjaNCVeff",
        language=None,
        speed=None,
        volume=None,
        pronunciation_dictionary_id=None,
        seed=None,
        continuation=None,
        text_normalization=None,
    )
    if settings is not None:
        defaults.apply_update(settings)
    return defaults


def _headers(auth_id: str, auth_token: str, client_ref: str | None) -> dict[str, str]:
    """Return the headers of a request."""
    headers = {"Authorization": aiohttp.encode_basic_auth(auth_id, auth_token)}
    if client_ref:
        headers["client-ref"] = client_ref
    return headers


def _request_fields(
    settings: PlivoTTSSettings, sample_rate: int, streaming: bool
) -> dict[str, Any]:
    """Return the settings as the fields of a request, without those that are None."""
    fields = {
        "model_id": settings.model,
        "voice_id": settings.voice,
        "language": settings.language,
        "pace": settings.speed,
        "volume": settings.volume,
        "pronunciation_dictionary_id": settings.pronunciation_dictionary_id,
        "seed": settings.seed,
        "continuation": settings.continuation,
        "text_normalization": settings.text_normalization,
        "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": sample_rate},
        "timestamps": True if streaming else None,
    }
    return {name: value for name, value in fields.items() if value is not None}


class PlivoTTSService(WebsocketTTSService):
    """Plivo TTS service for the Realtime API, with word timestamps.

    Synthesizes streamed text with the Realtime API. Each turn is one context on a WebSocket
    connection that the turns of a service share. A new model, voice or language inside a turn
    gives the rest of the turn a new context.
    """

    Settings = PlivoTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        auth_id: str,
        auth_token: str,
        url: str = "wss://api.plivo.com",
        client_ref: str | None = None,
        sample_rate: int | None = None,
        settings: Settings | None = None,
        text_aggregation_mode: TextAggregationMode = TextAggregationMode.TOKEN,
        **kwargs,
    ):
        """Initialize the Plivo TTS service for the Realtime API.

        Args:
            auth_id: Plivo auth ID.
            auth_token: Plivo auth token.
            url: Base URL of the Realtime API.
            client_ref: Label of 1 to 64 letters, digits, ``_`` or ``-``. The API ID of each
                request starts with it.
            sample_rate: Sample rate of the audio in Hz: 8000, 16000, 22050, 24000, 32000,
                44100 or 48000. If None, uses the pipeline default.
            settings: Runtime-updatable settings. During a turn, a new model, voice or
                language applies to the rest of the turn, and any other setting applies from
                the next context.
            text_aggregation_mode: How to aggregate incoming text before synthesis. With
                ``TextAggregationMode.TOKEN``, the default, the text is sent as it arrives and
                the TTS model cuts the sentences: a ``text_transforms`` rule sees one token at a
                time, and the TTFB metric starts at the turn's first token. With
                ``TextAggregationMode.SENTENCE``, Pipecat cuts them and the TTS model speaks each
                one as it arrives. In an Indian language, Pipecat also ends a sentence after an
                abbreviation such as ``डॉ.``, which is then spoken as a sentence of its own.
            **kwargs: Additional arguments passed to the parent service.
        """
        super().__init__(
            text_aggregation_mode=text_aggregation_mode,
            # The text frames come from the per-character timestamps.
            push_text_frames=False,
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            # The text messages of a context are joined without a separator.
            append_trailing_space=True,
            settings=_default_settings(settings),
            **kwargs,
        )

        self._url = f"{url}/v1/Account/{auth_id}/TTS/Realtime"
        self._headers = _headers(auth_id, auth_token, client_ref)

        # The contexts opened on this connection. A context takes its settings from the text that
        # opens it, so only that first text carries them.
        self._opened_contexts: set[str] = set()
        # A refused handshake's Retry-After: a text starts no handshake before this monotonic time.
        self._connect_not_before = 0.0
        self._receive_task = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate metrics.

        Returns:
            True, as Plivo TTS supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to Plivo's language code.

        Args:
            language: The language to convert.

        Returns:
            Plivo's language code.
        """
        return language_to_plivo_language(language)

    @classmethod
    def format_pronunciation(cls, word: str, ipa: str) -> str | None:
        """Render a pronunciation as Plivo phoneme spans.

        See :func:`format_plivo_pronunciation`.

        Args:
            word: The word being pronounced.
            ipa: The pronunciation, in IPA.

        Returns:
            The phoneme spans, or None when the pronunciation cannot be used.
        """
        return format_plivo_pronunciation(word, ipa)

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect to the Realtime API.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def _connect(self):
        """Connect to the Realtime API and start the receive task."""
        await super()._connect()

        await self._connect_websocket()

        # A receive loop that gave up reconnecting has ended: a new connection starts another.
        if self._websocket and (not self._receive_task or self._receive_task.done()):
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Stop the receive task and close the connection."""
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Open the WebSocket connection and log its API ID."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            logger.debug("Connecting to Plivo TTS")
            self._websocket = await self._websocket_connect(
                self._url, additional_headers=self._headers
            )
            self._opened_contexts.clear()
            if response := self._websocket.response:
                logger.debug(f"{self}: Plivo API ID {response.headers.get('api-id')}")
            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            error = f"{e}"
            if isinstance(e, InvalidStatus):
                # A refused handshake carries its reason in its body.
                if e.response.body:
                    error += f": {e.response.body.decode(errors='replace')}"
                retry_after = e.response.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    self._connect_not_before = time.monotonic() + int(retry_after)
            await self.push_error(error_msg=f"Error connecting to Plivo: {error}", exception=e)
            await self._call_event_handler("on_connection_error", error)

    async def _disconnect_websocket(self):
        """Close the WebSocket connection, which ends every context opened on it."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug(f"{self}: Disconnecting from Plivo")
                await self._websocket.close()
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Return the WebSocket connection, or raise ConnectionError if there is none."""
        if self._websocket:
            return self._websocket
        raise ConnectionError("Plivo is not connected")

    async def _send(self, kind: str, context_id: str, **fields):
        """Send a control message for a context.

        A lost connection is not reported: the receive loop reconnects, and the contexts of
        the old connection end with it.
        """
        message = {"type": kind, "context_id": context_id, **fields}
        try:
            await self._get_websocket().send(json.dumps(message))
        except (ConnectionError, ConnectionClosed):
            pass

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the context when the bot is interrupted, keeping the connection.

        Args:
            context_id: The ID of the audio context that was interrupted.
        """
        await self._send("cancel", context_id)
        await super().on_audio_context_interrupted(context_id)

    async def flush_audio(self, context_id: str | None = None):
        """Finish the context, so that the TTS model speaks the text it still buffers.

        Args:
            context_id: The context to finish. If None, finishes the active context.
        """
        flush_id = context_id or self.get_active_audio_context_id()
        if flush_id in self._opened_contexts:
            await self._send("text", flush_id, finish=True)

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta. A new model, voice or language gives the turn a new context."""
        changed = await super()._update_settings(delta)
        if changed.keys() & {"voice", "model", "language"} and self._turn_context_id:
            context_id = self._turn_context_id
            # End the sentence in progress with the old context, so that the words already
            # spoken stay in the transcript.
            sentence = await self._aggregated_frame_sequencer.finalize(context_id)
            await self._push_sequencer_frames(sentence, context_id)
            if self.audio_context_available(context_id):
                await self.flush_audio(context_id)
            # The rest of the turn opens a new context with the new settings.
            if self._turn_context_id:
                self._turn_context_id = None
                self._turn_context_id = self.create_context_id()
        return changed

    async def _receive_messages(self):
        """Route each message to its context."""
        async for message in self._get_websocket():
            if isinstance(message, bytes):
                # [2-byte little-endian length of the context ID][context ID][audio]
                audio_start = 2 + int.from_bytes(message[:2], "little")
                context_id = message[2:audio_start].decode()
                # Audio for a context whose audio context has ended, by an interruption or after
                # stop_frame_timeout_s, is dropped, so that the parent does not recreate it.
                if self.audio_context_available(context_id):
                    audio = message[audio_start:]
                    frame = TTSAudioRawFrame(audio, self.sample_rate, 1, context_id=context_id)
                    await self.append_to_audio_context(context_id, frame)
                continue

            event = json.loads(message)
            context_id = event.get("context_id")

            if event["type"] == "warning":
                logger.warning(f"{self}: Plivo warning {event['code']}: {event['message']}")
                continue
            # An interrupted context, or one that went quiet, has no audio context left. Its late
            # events are skipped, so that its error is not reported. An error with no context is
            # the connection's own, such as an account with no credit: it is reported.
            if context_id and not self.audio_context_available(context_id):
                continue

            if event["type"] == "timestamps":
                # Join the event's timed characters into words: a word runs from the start of
                # its first character to the end of its last.
                word_times = []
                for word in re.finditer(r"\S+", "".join(event["chars"])):
                    last = word.end() - 1
                    start = event["start_ms"][word.start()] / 1000
                    end = (event["start_ms"][last] + event["duration_ms"][last]) / 1000
                    word_times.append((word.group(), start, end))
                await self.add_word_timestamps(word_times, context_id)
            elif event["type"] == "done":
                await self.remove_audio_context(context_id)
            elif event["type"] == "error":
                await self.stop_all_metrics()
                error = f"Plivo error {event['code']}: {event['message']}"
                await self.push_error(error_msg=f"{error} (API ID {event['api_id']})")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using the Realtime API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: None once the text is sent; audio arrives on the receive task.
        """
        # A refused handshake's Retry-After asks for no other handshake before its time.
        closed = not self._websocket or self._websocket.state is State.CLOSED
        if closed and time.monotonic() >= self._connect_not_before:
            await self._connect()

        # Text outside a turn is a context of its own, so its message finishes it.
        finish = context_id != self._turn_context_id
        message = {"type": "text", "context_id": context_id, "text": text, "finish": finish}
        if context_id not in self._opened_contexts:
            message |= _request_fields(self._settings, self.sample_rate, streaming=True)

        try:
            await self._get_websocket().send(json.dumps(message))
            self._opened_contexts.add(context_id)
            # In SENTENCE mode Pipecat has cut the sentence, so a flush has it spoken now.
            # A finish already does that.
            if not finish and not self._is_streaming_tokens:
                flush = {"type": "flush", "context_id": context_id}
                await self._get_websocket().send(json.dumps(flush))
        except (ConnectionError, ConnectionClosed):
            # Pushed rather than yielded: a reconnect during the send ends the text's audio
            # context, which a yielded error goes through.
            await self.push_error(error_msg="Plivo connection unavailable: message not sent")
            return

        await self.start_tts_usage_metrics(text)
        yield None


class PlivoHttpTTSService(TTSService):
    """Plivo TTS service for the Batch API.

    Synthesizes each sentence with one request to the Batch API.
    """

    Settings = PlivoTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        auth_id: str,
        auth_token: str,
        aiohttp_session: aiohttp.ClientSession,
        base_url: str = "https://api.plivo.com",
        client_ref: str | None = None,
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Plivo TTS service for the Batch API.

        Args:
            auth_id: Plivo auth ID.
            auth_token: Plivo auth token.
            aiohttp_session: aiohttp ClientSession for the requests. Its timeout applies to
                each request.
            base_url: Base URL of the Batch API.
            client_ref: Label of 1 to 64 letters, digits, ``_`` or ``-``. The API ID of each
                request starts with it.
            sample_rate: Sample rate of the audio in Hz: 8000, 16000, 22050, 24000, 32000,
                44100 or 48000. If None, uses the pipeline default.
            settings: Runtime-updatable settings. An update applies from the next request.
            **kwargs: Additional arguments passed to the parent service.
        """
        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=_default_settings(settings),
            **kwargs,
        )

        self._url = f"{base_url}/v1/Account/{auth_id}/TTS"
        self._headers = _headers(auth_id, auth_token, client_ref)
        self._session = aiohttp_session

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate metrics.

        Returns:
            True, as Plivo TTS supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to Plivo's language code.

        Args:
            language: The language to convert.

        Returns:
            Plivo's language code.
        """
        return language_to_plivo_language(language)

    @classmethod
    def format_pronunciation(cls, word: str, ipa: str) -> str | None:
        """Render a pronunciation as Plivo phoneme spans.

        See :func:`format_plivo_pronunciation`.

        Args:
            word: The word being pronounced.
            ipa: The pronunciation, in IPA.

        Returns:
            The phoneme spans, or None when the pronunciation cannot be used.
        """
        return format_plivo_pronunciation(word, ipa)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using the Batch API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames as they stream back, or an ErrorFrame if the request is refused
            or the stream fails.
        """
        try:
            body = {"text": text} | _request_fields(
                self._settings, self.sample_rate, streaming=False
            )
            async with self._session.post(self._url, json=body, headers=self._headers) as response:
                logger.debug(f"{self}: Plivo API ID {response.headers.get('api-id')}")

                if response.status != 200:
                    refusal = await response.text(errors="replace")
                    yield ErrorFrame(error=f"Plivo error (status {response.status}): {refusal}")
                    return

                await self.start_tts_usage_metrics(text)

                # A chunk can end mid-sample, so the iterator helper keeps frames sample-aligned.
                async for frame in self._stream_audio_frames_from_iterator(
                    response.content.iter_chunked(self.chunk_size), context_id=context_id
                ):
                    yield frame
        except (aiohttp.ClientError, TimeoutError) as e:
            yield ErrorFrame(error=f"Plivo request or audio stream failed: {e!r}")
