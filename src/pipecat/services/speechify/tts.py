#
# Copyright (c) 2024-2026, Daily
# Copyright (c) 2026, Speechify
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Speechify text-to-speech service for Pipecat."""

from __future__ import annotations

import base64
import os
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, InterruptionFrame, TTSStoppedFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven, assert_given
from pipecat.services.tts_service import TextAggregationMode, TTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from speechify.client import AsyncSpeechify
    from speechify.core.api_error import ApiError
    from speechify.types.get_speech_options_request import GetSpeechOptionsRequest
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        'In order to use Speechify, you need to `pip install "pipecat-ai[speechify]"`.'
    )
    raise ImportError(f"Missing module: {e}") from e

# Speechify synthesizes PCM at a fixed 24 kHz mono, little-endian 16-bit. Audio
# is resampled to the pipeline's sample rate by the base class when they differ.
SPEECHIFY_SAMPLE_RATE = 24000

SpeechifyTTSModel = Literal[
    "simba-english",
    "simba-multilingual",
    "simba-3.0",
    "simba-3.2",
]

DEFAULT_VOICE_ID = "dominic_32"
DEFAULT_MODEL: SpeechifyTTSModel = "simba-3.2"


class _SpeechRequestKwargs(TypedDict, total=False):
    """Keyword arguments for ``AsyncSpeechify.audio.speech``."""

    audio_format: Literal["pcm"]
    input: str
    voice_id: str
    model: str
    language: str
    options: GetSpeechOptionsRequest


def language_to_speechify_language(language: Language) -> str:
    """Convert a Language enum to a Speechify BCP-47 language code.

    Base languages map to Speechify's default locale for that language;
    regional variants Speechify supports (e.g. ``en-GB``, ``pt-PT``) resolve to
    their own BCP-47 value. Unsupported languages fall back to their BCP-47
    string with a warning. The supported set comes from the Speechify
    ``/v1/voices`` endpoint.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Speechify BCP-47 language code (e.g. ``"en-US"``).
    """
    language_map = {
        Language.AR: "ar-AE",
        Language.BN: "bn-IN",
        Language.DA: "da-DK",
        Language.DE: "de-DE",
        Language.EL: "el-GR",
        Language.EN: "en-US",
        Language.ES: "es-MX",
        Language.ET: "et-EE",
        Language.FI: "fi-FI",
        Language.FR: "fr-FR",
        Language.GU: "gu-IN",
        Language.HE: "he-IL",
        Language.HI: "hi-IN",
        Language.IT: "it-IT",
        Language.JA: "ja-JP",
        Language.KO: "ko-KR",
        Language.MR: "mr-IN",
        Language.NB: "nb-NO",
        Language.NL: "nl-NL",
        Language.PL: "pl-PL",
        Language.PT: "pt-BR",
        Language.RU: "ru-RU",
        Language.SV: "sv-SE",
        Language.TA: "ta-IN",
        Language.TE: "te-IN",
        Language.TR: "tr-TR",
        Language.UK: "uk-UA",
        Language.UR: "ur-IN",
        Language.VI: "vi-VN",
        Language.YUE: "yue-CN",
    }
    return resolve_language(language, language_map, use_base_code=False)


@dataclass
class SpeechifyTTSSettings(TTSSettings):
    """Runtime-updatable settings for :class:`SpeechifyTTSService`.

    Parameters:
        loudness_normalization: Normalize output loudness to a standard level.
            Adds a small amount of latency when enabled.
        text_normalization: Expand numbers, dates and similar tokens into words
            before synthesis. Adds a small amount of latency when enabled.
    """

    # Default to NOT_GIVEN (not None) so a partial settings delta passed to
    # update_settings leaves unset fields unchanged; the _NotGiven arm of the
    # union mirrors the base TTSSettings field typing.
    loudness_normalization: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    text_normalization: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class SpeechifyTTSService(TTSService):
    """Speechify text-to-speech service.

    Synthesizes speech through the Speechify ``POST /v1/audio/speech`` endpoint,
    which returns base64-encoded PCM audio (24 kHz mono) together with
    word-level speech marks in a single JSON response. Pipecat's base
    ``TTSService`` aggregates incoming text into sentences by default, so
    ``run_tts`` is invoked once per sentence, giving near-streaming
    time-to-first-audio while preserving Speechify's word timestamps.

    Args:
        api_key: Speechify API key. Falls back to the ``SPEECHIFY_API_KEY``
            environment variable when omitted.
        voice_id: Id of the voice to synthesize with (see the ``/v1/voices``
            endpoint). Defaults to ``dominic_32``.
        model: Synthesis model. One of ``simba-english``, ``simba-multilingual``,
            ``simba-3.0`` or ``simba-3.2``. Defaults to ``simba-3.2``.
        language: Input language, as a Pipecat ``Language`` enum or a Speechify
            BCP-47 code (e.g. ``en-US``). Optional; Speechify auto-detects when
            omitted.
        loudness_normalization: Normalize output loudness to a standard level.
            Adds a small amount of latency when enabled.
        text_normalization: Expand numbers, dates and similar tokens into words
            before synthesis. Adds a small amount of latency when enabled.
        base_url: Override the Speechify API base URL.
        sample_rate: Output sample rate. Speechify synthesizes at 24 kHz; audio
            is resampled to this rate when it differs. Defaults to the pipeline
            rate when ``None``.
        client: A preconfigured ``AsyncSpeechify`` client. When provided,
            ``api_key`` and ``base_url`` are ignored.
        settings: Runtime-updatable settings. Values set here take precedence
            over the equivalent constructor arguments.
        text_aggregation_mode: How to aggregate incoming text before synthesis.
        aggregate_sentences: Whether to aggregate sentences within the TTSService.

            .. deprecated:: 0.0.104
                Use ``text_aggregation_mode`` instead.
                Will be removed in 2.0.0.

        **kwargs: Additional arguments passed to the base ``TTSService``.
    """

    Settings = SpeechifyTTSSettings
    _settings: SpeechifyTTSSettings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        voice_id: str = DEFAULT_VOICE_ID,
        model: SpeechifyTTSModel = DEFAULT_MODEL,
        language: Language | str | None = None,
        loudness_normalization: bool | None = None,
        text_normalization: bool | None = None,
        base_url: str | None = None,
        sample_rate: int | None = None,
        client: AsyncSpeechify | None = None,
        settings: SpeechifyTTSSettings | None = None,
        text_aggregation_mode: TextAggregationMode | None = None,
        aggregate_sentences: bool | None = None,
        **kwargs: Any,
    ):
        """Initialize the Speechify TTS service."""
        # Materialize every settings field to a concrete value (never NOT_GIVEN);
        # NOT_GIVEN is only valid in a delta passed to ``update_settings``.
        default_settings = self.Settings(
            model=model,
            voice=voice_id,
            language=language,
            loudness_normalization=loudness_normalization,
            text_normalization=text_normalization,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            text_aggregation_mode=text_aggregation_mode,
            aggregate_sentences=aggregate_sentences,
            push_start_frame=True,
            push_stop_frames=True,
            # Aligned per-word text frames come from add_word_timestamps, so
            # suppress the base class's full-sentence text frame to avoid
            # emitting the spoken text into the LLM context twice.
            push_text_frames=False,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        if client is not None:
            self._client = client
        else:
            token = api_key or os.environ.get("SPEECHIFY_API_KEY")
            if not token:
                raise ValueError(
                    "Speechify API key is required, either as the api_key argument "
                    "or via the SPEECHIFY_API_KEY environment variable."
                )
            self._client = AsyncSpeechify(token=token, base_url=base_url)

        # Speech marks reset to zero on every request. Track a running offset so
        # word timestamps stay monotonic across the sentences of a single turn;
        # it is reset when the turn ends (see ``push_frame``).
        self._cumulative_time = 0.0

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Speechify supports TTFB and usage metrics.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to a Speechify BCP-47 language code.

        Called by the base class to normalize ``Language`` enums to strings at
        construction time and on ``update_settings``.

        Args:
            language: The language to convert.

        Returns:
            The Speechify BCP-47 language code, or None if not supported.
        """
        return language_to_speechify_language(language)

    async def push_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        """Push a frame and reset word-timestamp state on interruption or stop.

        Args:
            frame: The frame to push.
            direction: The direction to push the frame.
        """
        await super().push_frame(frame, direction)
        if isinstance(frame, (InterruptionFrame, TTSStoppedFrame)):
            self._cumulative_time = 0.0

    def _speech_request_kwargs(self, text: str) -> _SpeechRequestKwargs:
        """Build keyword arguments for ``client.audio.speech``.

        Args:
            text: Text to convert to speech.

        Returns:
            Keyword arguments for the Speechify SDK call. The ``options`` object
            is omitted unless at least one Speechify normalization setting is
            configured.
        """
        options_kwargs: dict[str, bool] = {}
        loudness = assert_given(self._settings.loudness_normalization)
        if loudness is not None:
            options_kwargs["loudness_normalization"] = loudness
        text_norm = assert_given(self._settings.text_normalization)
        if text_norm is not None:
            options_kwargs["text_normalization"] = text_norm

        voice_id = assert_given(self._settings.voice)
        if not voice_id:
            raise ValueError("Speechify voice_id is required.")

        kwargs: _SpeechRequestKwargs = {
            "audio_format": "pcm",
            "input": text,
            "voice_id": voice_id,
        }
        model = assert_given(self._settings.model)
        if model:
            kwargs["model"] = model
        language = assert_given(self._settings.language)
        if language:
            kwargs["language"] = language
        if options_kwargs:
            kwargs["options"] = GetSpeechOptionsRequest(**options_kwargs)
        return kwargs

    def _word_times(self, speech_marks: object) -> tuple[list[tuple[str, float]], float]:
        """Flatten Speechify speech marks into ``(word, start_seconds)`` tuples.

        Speech-mark times are in milliseconds and reset per request, so each
        word start is offset by the running cumulative time. Returns the word
        tuples and this utterance's end time (in seconds, not cumulative) so the
        caller can advance the offset.

        Args:
            speech_marks: Speechify SDK speech marks object with a ``chunks``
                attribute, where each chunk exposes ``value``, ``start_time``
                and ``end_time``.

        Returns:
            A pair containing word timestamps and the utterance end time in
            seconds relative to this request.
        """
        chunks = getattr(speech_marks, "chunks", None) or []
        word_times: list[tuple[str, float]] = []
        utterance_end = 0.0
        for chunk in chunks:
            value = getattr(chunk, "value", None)
            start = getattr(chunk, "start_time", None)
            if not value or start is None:
                continue
            word_times.append((value, self._cumulative_time + start / 1000.0))
            end = getattr(chunk, "end_time", None)
            if end is not None:
                utterance_end = max(utterance_end, end / 1000.0)
        return word_times, utterance_end

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Synthesize ``text`` and yield audio frames with word timestamps.

        Args:
            text: The text to synthesize into speech.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: ``TTSAudioRawFrame`` audio frames plus any ``ErrorFrame`` on
            failure. ``TTSStartedFrame`` and ``TTSStoppedFrame`` are pushed by
            the base class.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics()
            response = await self._client.audio.speech(**self._speech_request_kwargs(text))
            await self.start_tts_usage_metrics(text)

            word_times, utterance_end = self._word_times(response.speech_marks)
            if word_times:
                await self.add_word_timestamps(word_times, context_id)
                self._cumulative_time += utterance_end

            audio = base64.b64decode(response.audio_data)

            async def audio_iterator() -> AsyncIterator[bytes]:
                yield audio

            async for frame in self._stream_audio_frames_from_iterator(
                audio_iterator(),
                in_sample_rate=SPEECHIFY_SAMPLE_RATE,
                context_id=context_id,
            ):
                await self.stop_ttfb_metrics()
                yield frame
        except ApiError as e:
            message = str(e.body) if e.body is not None else "Speechify API error"
            logger.error(f"{self}: Speechify API error (status={e.status_code}): {message}")
            yield ErrorFrame(error=f"Speechify API error: {message}")
        except Exception as e:
            logger.error(f"{self}: exception: {e}")
            yield ErrorFrame(error=f"Speechify TTS error: {e}")
        finally:
            await self.stop_ttfb_metrics()
            logger.debug(f"{self}: Finished TTS [{text}]")
