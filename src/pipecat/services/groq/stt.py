#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Groq speech-to-text service implementation using Whisper models."""

from typing import Optional

from pipecat.services.settings import _warn_deprecated_param
from pipecat.services.stt_latency import GROQ_TTFS_P99
from pipecat.services.whisper.base_stt import (
    BaseWhisperSTTService,
    BaseWhisperSTTSettings,
    Transcription,
)
from pipecat.transcriptions.language import Language


class GroqSTTService(BaseWhisperSTTService):
    """Groq Whisper speech-to-text service.

    Uses Groq's Whisper API to convert audio to text. Requires a Groq API key
    set via the api_key parameter or GROQ_API_KEY environment variable.
    """

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai/v1",
        language: Optional[Language] = None,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        settings: Optional[BaseWhisperSTTSettings] = None,
        ttfs_p99_latency: Optional[float] = GROQ_TTFS_P99,
        **kwargs,
    ):
        """Initialize Groq STT service.

        Args:
            model: Whisper model to use.

                .. deprecated:: 1.0.0
                    Use ``settings=BaseWhisperSTTSettings(model=...)`` instead.

            api_key: Groq API key. Defaults to None.
            base_url: API base URL. Defaults to "https://api.groq.com/openai/v1".
            language: Language of the audio input.

                .. deprecated:: 1.0.0
                    Use ``settings=BaseWhisperSTTSettings(language=...)`` instead.

            prompt: Optional text to guide the model's style or continue a previous segment.

                .. deprecated:: 1.0.0
                    Use ``settings=BaseWhisperSTTSettings(prompt=...)`` instead.

            temperature: Optional sampling temperature between 0 and 1.

                .. deprecated:: 1.0.0
                    Use ``settings=BaseWhisperSTTSettings(temperature=...)`` instead.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to BaseWhisperSTTService.
        """
        # --- 1. Hardcoded defaults ---
        default_settings = BaseWhisperSTTSettings(
            model="whisper-large-v3-turbo",
            language=self.language_to_service_language(Language.EN),
            base_url=base_url,
        )

        # --- 2. Deprecated direct-arg overrides ---
        if model is not None:
            _warn_deprecated_param("model", BaseWhisperSTTSettings, "model")
            default_settings.model = model
        if language is not None:
            _warn_deprecated_param("language", BaseWhisperSTTSettings, "language")
            default_settings.language = self.language_to_service_language(language)
        if prompt is not None:
            _warn_deprecated_param("prompt", BaseWhisperSTTSettings, "prompt")
            default_settings.prompt = prompt
        if temperature is not None:
            _warn_deprecated_param("temperature", BaseWhisperSTTSettings, "temperature")
            default_settings.temperature = temperature

        # --- 3. (no params object for this service) ---

        # --- 4. Settings delta (canonical API, always wins) ---
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            settings=default_settings,
            ttfs_p99_latency=ttfs_p99_latency,
            **kwargs,
        )

    async def _transcribe(self, audio: bytes) -> Transcription:
        assert self._language is not None  # Assigned in the BaseWhisperSTTService class

        # Build kwargs dict with only set parameters
        kwargs = {
            "file": ("audio.wav", audio, "audio/wav"),
            "model": self._settings.model,
            # Use verbose_json to get probability metrics
            "response_format": "verbose_json" if self._include_prob_metrics else "json",
            "language": self._language,
        }

        if self._prompt is not None:
            kwargs["prompt"] = self._prompt

        if self._temperature is not None:
            kwargs["temperature"] = self._temperature

        return await self._client.audio.transcriptions.create(**kwargs)
