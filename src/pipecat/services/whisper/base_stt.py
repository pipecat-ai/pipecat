#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Base class for Whisper-based speech-to-text services.

This module provides common functionality for services implementing the Whisper API
interface, including language mapping, metrics generation, and error handling.
"""

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

from loguru import logger
from openai import AsyncOpenAI
from openai.types.audio import Transcription

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven
from pipecat.services.stt_latency import WHISPER_TTFS_P99
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt


@dataclass
class BaseWhisperSTTSettings(STTSettings):
    """Settings for BaseWhisperSTTService.

    Parameters:
        prompt: Optional text to guide the model's style or continue
            a previous segment.
        temperature: Sampling temperature between 0 and 1.
    """

    prompt: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    temperature: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


def language_to_whisper_language(language: Language) -> str:
    """Maps pipecat Language enum to Whisper API language codes.

    Language support for Whisper API.
    Docs: https://platform.openai.com/docs/guides/speech-to-text#supported-languages

    Args:
        language: A Language enum value representing the input language.

    Returns:
        The corresponding service language code. If ``language`` is not in
        the verified mapping, falls back to the base language code (e.g.,
        ``en`` from ``en-US``) and logs a warning (via
        ``resolve_language(..., use_base_code=True)``).
    """
    LANGUAGE_MAP = {
        Language.AF: "af",
        Language.AR: "ar",
        Language.HY: "hy",
        Language.AZ: "az",
        Language.BE: "be",
        Language.BS: "bs",
        Language.BG: "bg",
        Language.CA: "ca",
        Language.ZH: "zh",
        Language.HR: "hr",
        Language.CS: "cs",
        Language.DA: "da",
        Language.NL: "nl",
        Language.EN: "en",
        Language.ET: "et",
        Language.FI: "fi",
        Language.FR: "fr",
        Language.GL: "gl",
        Language.DE: "de",
        Language.EL: "el",
        Language.HE: "he",
        Language.HI: "hi",
        Language.HU: "hu",
        Language.IS: "is",
        Language.ID: "id",
        Language.IT: "it",
        Language.JA: "ja",
        Language.KN: "kn",
        Language.KK: "kk",
        Language.KO: "ko",
        Language.LV: "lv",
        Language.LT: "lt",
        Language.MK: "mk",
        Language.MS: "ms",
        Language.MR: "mr",
        Language.MI: "mi",
        Language.NE: "ne",
        Language.NO: "no",
        Language.FA: "fa",
        Language.PL: "pl",
        Language.PT: "pt",
        Language.RO: "ro",
        Language.RU: "ru",
        Language.SR: "sr",
        Language.SK: "sk",
        Language.SL: "sl",
        Language.ES: "es",
        Language.SW: "sw",
        Language.SV: "sv",
        Language.TL: "tl",
        Language.TA: "ta",
        Language.TH: "th",
        Language.TR: "tr",
        Language.UK: "uk",
        Language.UR: "ur",
        Language.VI: "vi",
        Language.CY: "cy",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


class BaseWhisperSTTService(SegmentedSTTService):
    """Base class for Whisper-based speech-to-text services.

    Provides common functionality for services implementing the Whisper API interface,
    including metrics generation and error handling.
    """

    Settings = BaseWhisperSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        language: Language | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        include_prob_metrics: bool = False,
        push_empty_transcripts: bool = False,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = WHISPER_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Whisper STT service.

        Args:
            model: Name of the Whisper model to use.

                .. deprecated:: 0.0.105
                    Use ``settings=BaseWhisperSTTService.Settings(model=...)`` instead.

            api_key: Service API key. Defaults to None.
            base_url: Service API base URL. Defaults to None.
            language: Language of the audio input.

                .. deprecated:: 0.0.105
                    Use ``settings=BaseWhisperSTTService.Settings(language=...)`` instead.

            prompt: Optional text to guide the model's style or continue a previous segment.

                .. deprecated:: 0.0.105
                    Use ``settings=BaseWhisperSTTService.Settings(prompt=...)`` instead.

            temperature: Sampling temperature between 0 and 1.

                .. deprecated:: 0.0.105
                    Use ``settings=BaseWhisperSTTService.Settings(temperature=...)`` instead.

            include_prob_metrics: If True, enables probability metrics in API response.
                Each service implements this differently (see child classes).
                Defaults to False.
            push_empty_transcripts: If true, allow empty `TranscriptionFrame` frames to be
                pushed downstream instead of discarding them. This is intended for situations
                where VAD fires even though the user did not speak. In these cases, it is
                useful to know that nothing was transcribed so that the agent can resume
                speaking, instead of waiting longer for a transcription.
                Defaults to False.
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to SegmentedSTTService.
        """
        # --- 1. Hardcoded defaults ---
        default_settings = self.Settings(
            model=None,
            language=None,
            prompt=None,
            temperature=None,
        )

        # --- 2. Deprecated direct-arg overrides ---
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model
        if language is not None:
            self._warn_init_param_moved_to_settings("language", "language")
            default_settings.language = language
        if prompt is not None:
            self._warn_init_param_moved_to_settings("prompt", "prompt")
            default_settings.prompt = prompt
        if temperature is not None:
            self._warn_init_param_moved_to_settings("temperature", "temperature")
            default_settings.temperature = temperature

        # --- 3. (no params object for this service) ---

        # --- 4. Settings delta (canonical API, always wins) ---
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            ttfs_p99_latency=ttfs_p99_latency,
            settings=default_settings,
            **kwargs,
        )
        self._client = self._create_client(api_key, base_url)
        self._include_prob_metrics = include_prob_metrics
        self._push_empty_transcripts = push_empty_transcripts

    def _create_client(self, api_key: str | None, base_url: str | None):
        return AsyncOpenAI(api_key=api_key, base_url=base_url)

    def can_generate_metrics(self) -> bool:
        """Whether this service can generate processing metrics.

        Returns:
            bool: True, as this service supports metric generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert from pipecat Language to service language code.

        Args:
            language: The Language enum value to convert.

        Returns:
            The corresponding service language code, or None if not supported.
        """
        return language_to_whisper_language(language)

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing."""
        pass

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe audio data to text.

        Args:
            audio: Raw audio data to transcribe.

        Yields:
            Frame: Either a TranscriptionFrame containing the transcribed text
                  or an ErrorFrame if transcription fails.
        """
        try:
            await self.start_processing_metrics()

            response = await self._transcribe(audio)

            await self.stop_processing_metrics()

            text = response.text.strip()

            if not text:
                logger.warning("Received empty transcription from API")

            if text or self._push_empty_transcripts:
                await self._handle_transcription(text, True, self._settings.language)
                logger.debug(f"Transcription: [{text}]")
                yield TranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    result=response,
                )

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")

    async def _transcribe(self, audio: bytes) -> Transcription:
        """Transcribe audio data to text.

        Args:
            audio: Raw audio data in WAV format.

        Returns:
            Transcription: Object containing the transcribed text.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
