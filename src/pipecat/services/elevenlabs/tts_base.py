#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Shared foundation for the ElevenLabs multi-context WebSocket TTS services.

Holds what the text-to-speech and Text-to-Dialogue services have in common: the
connection and task lifecycle, settings-change dispatch, the word-timestamp
clock, and the shape of a synthesis request. Each service supplies its own
endpoint URL and wire protocol.
"""

import asyncio
import json
from abc import abstractmethod
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import websockets
from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import WebsocketTTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts

#: Languages the Flash and Turbo v2.5 models accept as a ``language_code``.
ELEVENLABS_V2_5_LANGUAGES: frozenset[str] = frozenset(
    {
        "ar",
        "bg",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "fi",
        "fil",
        "fr",
        "hi",
        "hr",
        "hu",
        "id",
        "it",
        "ja",
        "ko",
        "ms",
        "nl",
        "no",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sv",
        "ta",
        "tr",
        "uk",
        "vi",
        "zh",
    }
)

#: Languages the Eleven v3 models accept as a ``language_code``. A superset of
#: :data:`ELEVENLABS_V2_5_LANGUAGES`.
ELEVENLABS_V3_LANGUAGES: frozenset[str] = frozenset(
    {
        "af",
        "ar",
        "as",
        "az",
        "be",
        "bg",
        "bn",
        "bs",
        "ca",
        "ceb",
        "cs",
        "cy",
        "da",
        "de",
        "el",
        "en",
        "es",
        "et",
        "fa",
        "fi",
        "fil",
        "fr",
        "ga",
        "gl",
        "gu",
        "ha",
        "he",
        "hi",
        "hr",
        "hu",
        "hy",
        "id",
        "is",
        "it",
        "ja",
        "jv",
        "ka",
        "kk",
        "kn",
        "ko",
        "ky",
        "lb",
        "ln",
        "lt",
        "lv",
        "mk",
        "ml",
        "mr",
        "ms",
        "ne",
        "nl",
        "no",
        "ny",
        "pa",
        "pl",
        "ps",
        "pt",
        "ro",
        "ru",
        "sd",
        "sk",
        "sl",
        "so",
        "sr",
        "sv",
        "sw",
        "ta",
        "te",
        "th",
        "tr",
        "uk",
        "ur",
        "vi",
        "zh",
    }
)

#: Models that accept a ``language_code``, and the languages each one takes.
#: Sending a language a model doesn't cover is rejected with a 400, so the
#: language is checked against this mapping rather than only the model id.
#:
#: Models absent from the mapping take no language code at all. For
#: ``eleven_multilingual_v2`` that is deliberate: ElevenLabs documents
#: ``language_code`` as unsupported for it, and omitting the key is how its
#: auto-detection is meant to be used.
ELEVENLABS_MODEL_LANGUAGES: dict[str, frozenset[str]] = {
    "eleven_flash_v2_5": ELEVENLABS_V2_5_LANGUAGES,
    "eleven_turbo_v2_5": ELEVENLABS_V2_5_LANGUAGES,
    "eleven_v3": ELEVENLABS_V3_LANGUAGES,
    "eleven_v3_conversational": ELEVENLABS_V3_LANGUAGES,
}


def elevenlabs_language_code(model: str | None, language: str | None) -> str | None:
    """Resolve the ``language_code`` to send for a model.

    Args:
        model: The ElevenLabs model the request will use.
        language: An ElevenLabs language code, or None to send none.

    Returns:
        The language code to send, or None if it can't be used - either because
        the model takes no language code or because it doesn't cover this
        language. Both cases are logged.
    """
    if not language:
        return None

    supported = ELEVENLABS_MODEL_LANGUAGES.get(model or "")
    if supported is None:
        logger.warning(
            f"Language code [{language}] not applied. Language codes can only be used with: "
            f"{', '.join(sorted(ELEVENLABS_MODEL_LANGUAGES))}"
        )
        return None

    if language not in supported:
        logger.warning(
            f"Language code [{language}] not applied. {model} supports "
            f"{len(supported)} languages, which don't include [{language}]."
        )
        return None

    logger.debug(f"Using language code: {language}")
    return language


def language_to_elevenlabs_language(language: Language) -> str:
    """Convert a Language enum to ElevenLabs language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding service language code. If ``language`` is not in
        the verified mapping, falls back to the base language code (e.g.,
        ``en`` from ``en-US``) and logs a warning (via
        ``resolve_language(..., use_base_code=True)``).
    """
    LANGUAGE_MAP = {
        Language.AF: "af",
        Language.AR: "ar",
        Language.AS: "as",
        Language.AZ: "az",
        Language.BE: "be",
        Language.BG: "bg",
        Language.BN: "bn",
        Language.BS: "bs",
        Language.CA: "ca",
        Language.CEB: "ceb",
        Language.CS: "cs",
        Language.CY: "cy",
        Language.DA: "da",
        Language.DE: "de",
        Language.EL: "el",
        Language.EN: "en",
        Language.ES: "es",
        Language.ET: "et",
        Language.FA: "fa",
        Language.FI: "fi",
        Language.FIL: "fil",
        Language.FR: "fr",
        Language.GA: "ga",
        Language.GL: "gl",
        Language.GU: "gu",
        Language.HA: "ha",
        Language.HE: "he",
        Language.HI: "hi",
        Language.HR: "hr",
        Language.HU: "hu",
        Language.HY: "hy",
        Language.ID: "id",
        Language.IS: "is",
        Language.IT: "it",
        Language.JA: "ja",
        Language.JV: "jv",
        Language.KA: "ka",
        Language.KK: "kk",
        Language.KN: "kn",
        Language.KO: "ko",
        Language.KY: "ky",
        Language.LB: "lb",
        Language.LN: "ln",
        Language.LT: "lt",
        Language.LV: "lv",
        Language.MK: "mk",
        Language.ML: "ml",
        Language.MR: "mr",
        Language.MS: "ms",
        Language.NE: "ne",
        Language.NL: "nl",
        Language.NO: "no",
        Language.NY: "ny",
        Language.PA: "pa",
        Language.PL: "pl",
        Language.PS: "ps",
        Language.PT: "pt",
        Language.RO: "ro",
        Language.RU: "ru",
        Language.SD: "sd",
        Language.SK: "sk",
        Language.SL: "sl",
        Language.SO: "so",
        Language.SR: "sr",
        Language.SV: "sv",
        Language.SW: "sw",
        Language.TA: "ta",
        Language.TE: "te",
        Language.TH: "th",
        Language.TR: "tr",
        Language.UK: "uk",
        Language.UR: "ur",
        Language.VI: "vi",
        Language.ZH: "zh",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


def output_format_from_sample_rate(sample_rate: int) -> str:
    """Get the appropriate output format string for a given sample rate.

    Args:
        sample_rate: The audio sample rate in Hz.

    Returns:
        The ElevenLabs output format string.
    """
    match sample_rate:
        case 8000:
            return "pcm_8000"
        case 16000:
            return "pcm_16000"
        case 22050:
            return "pcm_22050"
        case 24000:
            return "pcm_24000"
        case 32000:
            return "pcm_32000"
        case 44100:
            return "pcm_44100"
        case 48000:
            return "pcm_48000"
    logger.warning(
        f"ElevenLabsTTSService: No output format available for {sample_rate} sample rate"
    )
    return "pcm_24000"


def _is_chinese_or_japanese_language(language: str) -> bool:
    """Check if the given language is Chinese or Japanese."""
    base_lang = language.split("-")[0].lower()
    return base_lang in {"zh", "ja"}


def _word_timestamps_include_inter_frame_spaces(language: str | None) -> bool:
    """Whether timestamp text should be treated as carrying its own spacing."""
    return bool(language and _is_chinese_or_japanese_language(language))


def _select_alignment(
    msg: Mapping[str, Any],
    *,
    normalized_key: str,
    alignment_key: str,
    prefer_normalized: bool,
) -> Mapping[str, Any] | None:
    """Pick the alignment field to use from a TTS message, with fallback.

    ElevenLabs returns two alignment fields per chunk:

    - ``normalized_key`` (``normalizedAlignment`` for WebSocket,
      ``normalized_alignment`` for HTTP): the post-normalized form of what was
      spoken - pronunciation-dictionary substitutions, text normalization, or
      romanization of non-Latin scripts (e.g., Chinese rendered as pinyin).
    - ``alignment_key`` (``alignment``): the original input characters.

    Prefer ``normalized`` only when a pronunciation dictionary is configured -
    that's the case where ``alignment`` has overlapping restarts that produce
    duplicated/garbled words (issue #4316). Otherwise prefer ``alignment`` so
    the LLM context preserves the original input rather than the normalized
    form. Fall back to the other field if the preferred one is missing or
    null - the API schema marks both as nullable.

    Args:
        msg: TTS response message from ElevenLabs.
        normalized_key: Key for the normalized-alignment field on this transport.
        alignment_key: Key for the original-alignment field on this transport.
        prefer_normalized: True iff the caller is using pronunciation dictionaries.

    Returns:
        The chosen alignment dict, or ``None`` if both fields are absent/null.
    """
    if prefer_normalized:
        return msg.get(normalized_key) or msg.get(alignment_key)
    return msg.get(alignment_key) or msg.get(normalized_key)


def _strip_utterance_leading_spaces(
    alignment: Mapping[str, Any], keys: tuple[str, str, str], should_strip: bool
) -> Mapping[str, Any]:
    """Return alignment with utterance-leading space chars removed, if requested.

    ElevenLabs Flash normalized alignment chunks can begin with a leading space
    at the start of an utterance. Strip only utterance-leading spaces so bot
    turn text does not start with whitespace. On subsequent chunks, however, a
    leading space can be a real inter-word separator (Flash models commonly
    split sentences this way), so it must be preserved for
    ``calculate_word_times`` to flush any partial word carried over from the
    previous chunk.

    Args:
        alignment: Alignment dict from the API.
        keys: Tuple of (chars_key, start_times_key, durations_or_end_times_key)
            naming the three parallel arrays - these differ between the
            WebSocket and HTTP response schemas.
        should_strip: Whether this is still utterance-leading alignment data.
    """
    chars_key, starts_key, tail_key = keys
    chars = alignment.get(chars_key) or []
    if should_strip and chars and chars[0] == " ":
        strip_count = 0
        while strip_count < len(chars) and chars[strip_count] == " ":
            strip_count += 1

        stripped = dict(alignment)
        stripped[chars_key] = chars[strip_count:]
        stripped[starts_key] = alignment.get(starts_key, [])[strip_count:]
        stripped[tail_key] = alignment.get(tail_key, [])[strip_count:]
        return stripped
    return alignment


def calculate_word_times(
    alignment_info: Mapping[str, Any],
    cumulative_time: float,
    partial_word: str = "",
    partial_word_start_time: float = 0.0,
) -> tuple[list[tuple[str, float]], str, float]:
    """Calculate word timestamps from character alignment information.

    Args:
        alignment_info: Character alignment data from ElevenLabs API.
        cumulative_time: Base time offset for this chunk.
        partial_word: Partial word carried over from previous chunk.
        partial_word_start_time: Start time of the partial word.

    Returns:
        Tuple of (word_times, new_partial_word, new_partial_word_start_time):
        - word_times: List of (word, timestamp) tuples for complete words
        - new_partial_word: Incomplete word at end of chunk (empty if chunk ends with space)
        - new_partial_word_start_time: Start time of the incomplete word
    """
    chars = alignment_info["chars"]
    char_start_times_ms = alignment_info["charStartTimesMs"]

    if len(chars) != len(char_start_times_ms):
        logger.error(
            f"calculate_word_times: length mismatch - chars={len(chars)}, times={len(char_start_times_ms)}"
        )
        return ([], partial_word, partial_word_start_time)

    # Build words and track their start positions
    words = []
    word_start_times = []
    current_word = partial_word  # Start with any partial word from previous chunk
    word_start_time = partial_word_start_time if partial_word else None

    for i, char in enumerate(chars):
        if char == " ":
            # End of current word
            if current_word:  # Only add non-empty words
                words.append(current_word)
                word_start_times.append(word_start_time)
                current_word = ""
                word_start_time = None
        else:
            # Building a word
            if word_start_time is None:  # First character of new word
                # Convert from milliseconds to seconds and add cumulative offset
                word_start_time = cumulative_time + (char_start_times_ms[i] / 1000.0)
            current_word += char

    # Build result for complete words
    word_times = list(zip(words, word_start_times))

    # Return any incomplete word at the end of this chunk
    new_partial_word = current_word if current_word else ""
    new_partial_word_start_time = word_start_time if word_start_time is not None else 0.0

    return (word_times, new_partial_word, new_partial_word_start_time)


@dataclass
class ElevenLabsTTSSettingsBase(TTSSettings):
    """Settings shared by the ElevenLabs WebSocket TTS services.

    Carries the two declarative sets that
    :meth:`ElevenLabsTTSBase._update_settings` dispatches on. A service names
    the settings it can only change by reconnecting, and the settings it can
    change by opening a fresh context.
    """

    #: Fields in the WS URL — changing any of these requires a reconnect.
    URL_FIELDS: ClassVar[frozenset[str]] = frozenset()

    #: Fields carried in a context's opening message — changing these requires
    #: closing the current audio context so the next one picks them up.
    VOICE_SETTINGS_FIELDS: ClassVar[frozenset[str]] = frozenset()


class ElevenLabsTTSBase(WebsocketTTSService):
    """Shared behavior for the ElevenLabs multi-context WebSocket TTS services.

    Owns the WebSocket connection and its receive and keepalive tasks, the
    settings-change dispatch, the word-timestamp clock, and the shape of a
    synthesis request. Subclasses supply the endpoint URL and the wire protocol:
    how text, keepalives, flushes and context closes are framed, and how server
    messages are read.
    """

    Settings = ElevenLabsTTSSettingsBase
    _settings: Settings

    #: Name of the endpoint, used in connection log lines.
    CONNECTION_NAME: ClassVar[str] = "ElevenLabs"

    def __init__(
        self,
        *,
        api_key: str,
        url: str,
        enable_logging: bool | None = None,
        **kwargs,
    ):
        """Initialize the ElevenLabs WebSocket TTS service.

        Args:
            api_key: ElevenLabs API key for authentication.
            url: Base WebSocket URL for the ElevenLabs API.
            enable_logging: Whether to enable ElevenLabs server-side logging.
            **kwargs: Additional arguments passed to the parent service.
        """
        super().__init__(**kwargs)

        self._api_key = api_key
        self._url = url
        self._enable_logging = enable_logging

        self._output_format = ""  # initialized in setup()
        self._voice_settings = self._set_voice_settings()

        self._cumulative_time = 0.0
        # Track partial words that span across alignment chunks
        self._partial_word = ""
        self._partial_word_start_time = 0.0

        self._receive_task = None
        self._keepalive_task = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as ElevenLabs service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to ElevenLabs language format.

        Args:
            language: The language to convert.

        Returns:
            The ElevenLabs-specific language code, or None if not supported.
        """
        return language_to_elevenlabs_language(language)

    @abstractmethod
    def _set_voice_settings(self) -> dict[str, Any] | None:
        """Build the voice settings sent when a context opens."""

    @abstractmethod
    def _build_websocket_url(self) -> str:
        """Build the endpoint URL, including query parameters, for a new connection."""

    @abstractmethod
    async def _send_context_init(self, context_id: str):
        """Send the message that opens a server-side context."""

    @abstractmethod
    async def _send_text(self, text: str, context_id: str):
        """Send text to the WebSocket for synthesis."""

    @abstractmethod
    async def _send_keepalive(self):
        """Reset the server's inactivity timer on the connection."""

    @abstractmethod
    async def _close_context(self, context_id: str):
        """Close a server-side context to free its slot."""

    async def _on_websocket_connected(self):
        """Run any protocol-specific setup once the connection is open."""

    def _clear_connection_state(self):
        """Drop per-connection context bookkeeping after the socket closes."""

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta, reconnecting as needed.

        Uses the declarative ``URL_FIELDS`` and ``VOICE_SETTINGS_FIELDS`` sets on
        the service's ``Settings`` to decide whether to reconnect the WebSocket
        or close the current audio context.

        Args:
            delta: A :class:`TTSSettings` (or service-specific ``Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)

        if not changed:
            return changed

        # Rebuild voice settings for next context
        self._voice_settings = self._set_voice_settings()

        url_changed = bool(changed.keys() & self.Settings.URL_FIELDS)
        voice_settings_changed = bool(changed.keys() & self.Settings.VOICE_SETTINGS_FIELDS)

        if url_changed:
            logger.debug(
                f"URL-level setting changed ({changed.keys() & self.Settings.URL_FIELDS}), "
                f"reconnecting WebSocket"
            )
            await self._disconnect()
            await self._connect()
        elif voice_settings_changed:
            logger.debug(
                f"Voice settings changed ({changed.keys() & self.Settings.VOICE_SETTINGS_FIELDS}), "
                f"closing current context to apply changes"
            )
            for ctx_id in self.get_audio_contexts():
                await self._close_context(ctx_id)
                self._reset_alignment_state(ctx_id)

        if not url_changed:
            # Reconnect applies all settings; only warn about fields not handled
            # by voice settings or URL changes.
            handled = self.Settings.URL_FIELDS | self.Settings.VOICE_SETTINGS_FIELDS
            self._warn_unhandled_updated_settings(changed.keys() - handled)

        return changed

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._output_format = output_format_from_sample_rate(self.sample_rate)
        await self._connect()

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

            logger.debug(f"Connecting to {self.CONNECTION_NAME}")

            # Set max websocket message size to 16MB for large audio responses
            self._websocket = await self._websocket_connect(
                self._build_websocket_url(),
                max_size=16 * 1024 * 1024,
                additional_headers={"xi-api-key": self._api_key},
            )

            await self._on_websocket_connected()

            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            websocket = self._websocket
            if websocket:
                logger.debug(f"Disconnecting from {self.CONNECTION_NAME}")
                # The multi-stream protocol tears down in two steps: we ask
                # ElevenLabs to close, then it closes. Wait for its close before
                # forcing ours, so we don't race the closing handshake (which
                # otherwise ends a notable fraction of sessions in a 1006 close).
                # The timeout is only a fallback ceiling; the clean close
                # normally arrives well within it.
                await websocket.send(json.dumps({"close_socket": True}))
                try:
                    await asyncio.wait_for(websocket.wait_closed(), timeout=2.0)
                except TimeoutError:
                    logger.debug(
                        "ElevenLabs did not close the WebSocket within 2.0s; closing from our side"
                    )
                await websocket.close()
                logger.debug(f"Disconnected from {self.CONNECTION_NAME}")
        except websockets.ConnectionClosed as e:
            # The server closed the connection first — normal during teardown, or a race
            # on the closing handshake. The connection is gone either way; this is not a
            # pipeline error, so don't push an ErrorFrame (which would e.g. trigger a
            # spurious ServiceSwitcherStrategyFailover switch during shutdown).
            logger.debug(f"{self} websocket already closed during disconnect: {e}")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            await self.remove_active_audio_context()
            self._websocket = None
            self._clear_connection_state()
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    def _reset_word_clock(self):
        """Reset the running word-timestamp clock for a new context."""
        self._cumulative_time = 0.0
        self._partial_word = ""
        self._partial_word_start_time = 0.0

    def _reset_alignment_state(self, context_id: str):
        self._reset_word_clock()

    async def on_audio_context_interrupted(self, context_id: str):
        """Close the ElevenLabs context when the bot is interrupted."""
        await self._close_context(context_id)
        self._reset_alignment_state(context_id)
        await super().on_audio_context_interrupted(context_id)

    async def on_audio_context_completed(self, context_id: str):
        """Reset alignment state after all audio for the context has played."""
        self._reset_alignment_state(context_id)
        await super().on_audio_context_completed(context_id)

    async def _keepalive_task_handler(self):
        """Send periodic keepalive messages to maintain WebSocket connection."""
        KEEPALIVE_SLEEP = 10
        while True:
            await asyncio.sleep(KEEPALIVE_SLEEP)
            try:
                await self._send_keepalive()
            except websockets.ConnectionClosed as e:
                logger.warning(f"{self} keepalive error: {e}")
                break

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using ElevenLabs' streaming WebSocket API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            if self._websocket is None:
                logger.warning(f"{self}: websocket unavailable after reconnect, skipping TTS")
                yield ErrorFrame(error="websocket unavailable")
                return

            try:
                if not self.audio_context_available(context_id):
                    await self.create_audio_context(context_id)
                    await self.start_ttfb_metrics()
                    yield TTSStartedFrame(context_id=context_id)
                    self._reset_word_clock()

                    await self._send_context_init(context_id)
                    logger.trace(f"Created new context {context_id}")

                await self._send_text(text, context_id)
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                yield TTSStoppedFrame(context_id=context_id)
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
                return
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
