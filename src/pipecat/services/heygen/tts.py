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

import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal
from urllib.parse import urlencode

import aiohttp
from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame, TTSStoppedFrame
from pipecat.services.elevenlabs.tts_base import (
    ElevenLabsTTSBase,
    ElevenLabsTTSSettingsBase,
    _strip_utterance_leading_spaces,
    _word_timestamps_include_inter_frame_spaces,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, TTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given

# Nadine, a HeyGen public voice. Works with every HeyGen Voice model, no clone needed.
HEYGEN_DEFAULT_VOICE_ID = "83548de556df48ba8c09c42a57c51d85"

# PCM sample rates HeyGen can synthesize over HTTP.
HEYGEN_PCM_SAMPLE_RATES = (8000, 16000, 22050, 24000, 44100, 48000)

# PCM sample rates HeyGen can synthesize over the WebSocket.
HEYGEN_WEBSOCKET_PCM_SAMPLE_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)

# Rate requested when the pipeline runs at a rate HeyGen does not offer; the
# audio is then resampled to the pipeline rate.
HEYGEN_FALLBACK_SAMPLE_RATE = 24000

# A context ID the service never opens; idle keepalives close it.
HEYGEN_KEEPALIVE_CONTEXT_ID = "pipecat-keepalive"


HEYGEN_LANGUAGE_MAP = {
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

# Languages HeyGen Voice speaks, as the codes it accepts.
HEYGEN_LANGUAGES = frozenset(HEYGEN_LANGUAGE_MAP.values())


def language_to_heygen_language(language: Language) -> str:
    """Convert a Language enum to a HeyGen Voice language code.

    HeyGen Voice takes lowercase ISO 639-1 codes and has no regional accents, so
    regional variants (e.g. ``en-US``) resolve to their base code (``en``).

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding HeyGen language code.
    """
    code = resolve_language(language, HEYGEN_LANGUAGE_MAP, use_base_code=True)
    if code not in HEYGEN_LANGUAGES:
        logger.warning(f"HeyGen Voice does not speak {language}; HeyGen will detect the language.")
    return code


def _request_language(language: Any) -> str | None:
    """The language to send to HeyGen, or None to let HeyGen detect it.

    HeyGen refuses a language it does not speak, so those are left out.
    """
    if not language:
        return None
    code = str(language)
    return code if code in HEYGEN_LANGUAGES else None


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
            voice=HEYGEN_DEFAULT_VOICE_ID,
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

    def _synthesis_sample_rate(self) -> int:
        """Return the PCM rate to request from HeyGen for the pipeline's rate."""
        if self.sample_rate in HEYGEN_PCM_SAMPLE_RATES:
            return self.sample_rate
        return HEYGEN_FALLBACK_SAMPLE_RATE

    def _build_payload(self, text: str, synthesis_sample_rate: int) -> dict[str, Any]:
        """Build the request body for one piece of text.

        Args:
            text: Text to synthesize.
            synthesis_sample_rate: PCM sample rate to request.

        Returns:
            The JSON request body.
        """
        payload: dict[str, Any] = {
            "text": text,
            "voice_id": assert_given(self._settings.voice),
            "model": assert_given(self._settings.model),
            "output_format": {
                "encoding": "pcm",
                "sample_rate_hertz": synthesis_sample_rate,
            },
        }
        language = _request_language(assert_given(self._settings.language))
        if language:
            payload["language"] = language
        return payload

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using HeyGen's streaming speech endpoint.

        Args:
            text: Text to synthesize.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        synthesis_sample_rate = self._synthesis_sample_rate()
        headers = {"x-api-key": self._api_key, "Content-Type": "application/json"}

        try:
            async with self._session.post(
                self._url,
                json=self._build_payload(text, synthesis_sample_rate),
                headers=headers,
            ) as response:
                request_id = response.headers.get("X-Request-Id")
                if response.status != 200:
                    error = await response.text(errors="ignore")
                    yield ErrorFrame(
                        error=f"HeyGen TTS error (status: {response.status}, "
                        f"request_id: {request_id}, error: {error})"
                    )
                    return

                logger.debug(f"{self}: HeyGen request {request_id}")
                await self.start_tts_usage_metrics(text)

                # Raw PCM16 arrives at arbitrary byte boundaries; the helper keeps
                # frames sample-aligned and resamples when the rates differ.
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
class HeyGenWebsocketTTSSettings(ElevenLabsTTSSettingsBase):
    """Settings for HeyGenTTSService.

    Parameters:
        model: HeyGen Voice model: ``heygen-voice-1``, ``heygen-voice-1-turbo``
            or ``heygen-voice-1-turbo-free``.
        voice: HeyGen voice ID: a public voice or an instant clone in your workspace.
        language: Language of the text. If ``None``, HeyGen detects it.
        apply_text_normalization: Text normalization mode (``"auto"``, ``"on"`` or
            ``"off"``). If ``None``, HeyGen uses ``"auto"``.
        expressiveness_boost: Voice expressiveness from 0.0 to 1.0. If ``None``,
            HeyGen uses 1.0.
    """

    apply_text_normalization: Literal["auto", "on", "off"] | None | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )
    expressiveness_boost: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)

    #: All settings are WebSocket URL parameters; changing one reconnects.
    URL_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"model", "voice", "language", "apply_text_normalization", "expressiveness_boost"}
    )


class HeyGenTTSService(ElevenLabsTTSBase):
    """HeyGen Voice WebSocket text-to-speech service.

    Streams text to HeyGen's ``multi-stream-input`` WebSocket and plays the audio
    as it arrives. The socket speaks ElevenLabs' multi-context messages, so this
    service shares :class:`ElevenLabsTTSBase` and differs only in its endpoint,
    authentication and parameter names. HeyGen's per-character alignment drives
    word-level ``TTSTextFrame`` objects, so an interruption commits only the
    words actually spoken.

    Defaults to ``heygen-voice-1-turbo`` and the public voice Nadine.
    Professional voice clones are served only by :class:`HeyGenHttpTTSService`.
    """

    Settings = HeyGenWebsocketTTSSettings
    _settings: Settings

    CONNECTION_NAME: ClassVar[str] = "HeyGen"

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "wss://api.heygen.com",
        sample_rate: int | None = None,
        auto_mode: bool | None = None,
        text_aggregation_mode: TextAggregationMode | None = None,
        pause_frame_processing: bool = True,
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
            auto_mode: Whether HeyGen starts generating at each sentence end
                instead of waiting for more text. Defaults to True unless
                ``text_aggregation_mode`` is ``TOKEN``, where HeyGen's own
                chunk schedule gathers enough text for natural speech.
            text_aggregation_mode: How to aggregate incoming text before synthesis.
            pause_frame_processing: Whether to hold incoming frames after a turn
                until its audio has played. With False, turns synthesize
                concurrently on the one socket, which HeyGen limits to 5
                running contexts.
            enable_ssml_parsing: Whether ``<break time="1s"/>`` tags become pauses.
                Without it, HeyGen closes the WebSocket on a break tag.
            enable_logging: Whether HeyGen stores a copy of the audio and text.
                If None, HeyGen stores them.
            end_user_id: Your identifier for the end user, recorded with HeyGen's
                usage for attribution.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to the parent service.
        """
        default_settings = self.Settings(
            model="heygen-voice-1-turbo",
            voice=HEYGEN_DEFAULT_VOICE_ID,
            language=None,
            apply_text_normalization=None,
            expressiveness_boost=None,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            api_key=api_key,
            url=url,
            text_aggregation_mode=text_aggregation_mode,
            push_text_frames=False,
            push_stop_frames=False,
            # HeyGen joins text messages as sent, so sentences need a separating space.
            append_trailing_space=True,
            pause_frame_processing=pause_frame_processing,
            enable_logging=enable_logging,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._enable_ssml_parsing = enable_ssml_parsing
        self._end_user_id = end_user_id

        # The rate HeyGen synthesizes at; set when the URL is built.
        self._audio_sample_rate = HEYGEN_FALLBACK_SAMPLE_RATE

        if auto_mode is None:
            auto_mode = self._text_aggregation_mode != TextAggregationMode.TOKEN
        self._auto_mode = auto_mode

        # Contexts opened and not yet closed, so keepalives may target them.
        self._context_init_sent: set[str] = set()
        # Contexts whose first alignment has arrived; only the first can carry
        # utterance-leading spaces.
        self._alignment_started_context_ids: set[str] = set()

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to a HeyGen Voice language code.

        Args:
            language: The language to convert.

        Returns:
            The HeyGen language code.
        """
        return language_to_heygen_language(language)

    def _websocket_headers(self) -> dict[str, str]:
        return {"x-api-key": self._api_key}

    def _build_websocket_url(self) -> str:
        # At a rate HeyGen does not offer, synthesize at the fallback rate and
        # resample each audio message to the pipeline rate.
        if self.sample_rate in HEYGEN_WEBSOCKET_PCM_SAMPLE_RATES:
            self._audio_sample_rate = self.sample_rate
        else:
            self._audio_sample_rate = HEYGEN_FALLBACK_SAMPLE_RATE
        params = {
            "voice_id": assert_given(self._settings.voice),
            "model": assert_given(self._settings.model),
            "output_format": f"pcm_{self._audio_sample_rate}",
            "auto_mode": str(self._auto_mode).lower(),
        }
        language = _request_language(assert_given(self._settings.language))
        if language:
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

    def _set_voice_settings(self) -> dict[str, Any] | None:
        return None

    def _clear_connection_state(self):
        self._context_init_sent.clear()

    def _reset_alignment_state(self, context_id: str):
        super()._reset_alignment_state(context_id)
        self._alignment_started_context_ids.discard(context_id)
        self._context_init_sent.discard(context_id)

    async def flush_audio(self, context_id: str | None = None):
        """Ask HeyGen to generate any text still buffered for a context.

        Args:
            context_id: The context to flush. If None, the active context.
        """
        flush_id = context_id or self.get_active_audio_context_id()
        if not flush_id or not self._websocket:
            return
        await self._websocket.send(json.dumps({"context_id": flush_id, "flush": True}))

    async def on_turn_context_completed(self):
        """Close the server-side context at the end of a turn so its final message arrives."""
        context_id = self._turn_context_id
        # The parent flushes the turn's context if it was opened; a close without
        # a flush discards text HeyGen has not spoken yet.
        await super().on_turn_context_completed()
        if context_id in self._context_init_sent:
            await self._close_context(context_id)

    async def _close_context(self, context_id: str):
        self._context_init_sent.discard(context_id)
        if context_id and self._websocket:
            try:
                await self._websocket.send(
                    json.dumps({"context_id": context_id, "close_context": True})
                )
            except Exception as e:
                await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)

    async def _send_context_init(self, context_id: str):
        self._context_init_sent.add(context_id)
        await self._get_websocket().send(json.dumps({"text": " ", "context_id": context_id}))

    async def _send_text(self, text: str, context_id: str):
        if self._websocket and context_id:
            await self._websocket.send(json.dumps({"text": text, "context_id": context_id}))

    async def _send_keepalive(self):
        if not self._websocket or self._websocket.state is not State.OPEN:
            return
        # HeyGen opens a context for a keepalive naming one that is not open
        # (with no context_id, one named "default"), so a keepalive names only an
        # open context. Without one, closing a context that does not exist resets
        # HeyGen's inactivity timer and changes nothing else.
        context_id = self.get_active_audio_context_id()
        if context_id and context_id in self._context_init_sent:
            message = {"text": "", "context_id": context_id}
        else:
            message = {"context_id": HEYGEN_KEEPALIVE_CONTEXT_ID, "close_context": True}
        await self._websocket.send(json.dumps(message))

    async def _receive_messages(self):
        """Handle audio, alignment, final and error messages from HeyGen."""
        async for message in self._get_websocket():
            msg = json.loads(message)

            # HeyGen sends an error message just before it closes the socket.
            if msg.get("error"):
                error = msg["error"] if isinstance(msg["error"], dict) else {}
                await self.push_error(
                    error_msg=(
                        f"HeyGen TTS error: {msg.get('message') or error.get('message')} "
                        f"(code: {error.get('code')}, request_id: {error.get('request_id')})"
                    )
                )
                continue

            context_id = msg.get("contextId")

            if msg.get("isFinal") is True:
                if self.audio_context_available(context_id):
                    await self.append_to_audio_context(
                        context_id, TTSStoppedFrame(context_id=context_id)
                    )
                    await self.remove_audio_context(context_id)
                continue

            # An interrupted context's audio and alignment can still arrive.
            if not self.audio_context_available(context_id):
                continue

            if msg.get("audio"):
                audio = base64.b64decode(msg["audio"])
                if self._audio_sample_rate != self.sample_rate:
                    audio = await self._resampler.resample(
                        audio, self._audio_sample_rate, self.sample_rate
                    )
                frame = TTSAudioRawFrame(audio, self.sample_rate, 1, context_id=context_id)
                await self.append_to_audio_context(context_id, frame)

            alignment = msg.get("alignment")
            if alignment:
                alignment = _strip_utterance_leading_spaces(
                    alignment,
                    ("chars", "charStartTimesMs", "charDurationsMs"),
                    context_id not in self._alignment_started_context_ids,
                )
                self._alignment_started_context_ids.add(context_id)
                word_times = self._word_times_from_alignment(context_id, alignment)
                if word_times:
                    language = assert_given(self._settings.language)
                    await self.add_word_timestamps(
                        word_times,
                        context_id,
                        includes_inter_frame_spaces=(
                            True if _word_timestamps_include_inter_frame_spaces(language) else None
                        ),
                    )
