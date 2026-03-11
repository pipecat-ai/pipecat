#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""SambaNova's Speech-to-Text service implementation for real-time transcription."""

from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger

from pipecat.services.settings import _warn_deprecated_param
from pipecat.services.stt_latency import SAMBANOVA_TTFS_P99
from pipecat.services.whisper.base_stt import (
    BaseWhisperSTTService,
    Transcription,
)
from pipecat.transcriptions.language import Language


@dataclass
class SambaNovaSTTSettings(BaseWhisperSTTService.Settings):
    """Settings for the SambaNova STT service."""

    pass


class SambaNovaSTTService(BaseWhisperSTTService):  # type: ignore
    """SambaNova Whisper speech-to-text service.

    Uses SambaNova's Whisper API to convert audio to text.
    Requires a SambaNova API key set via the api_key parameter or SAMBANOVA_API_KEY environment variable.
    """

    Settings = SambaNovaSTTSettings

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: str = "https://api.sambanova.ai/v1",
        language: Optional[Language] = None,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        settings: Optional[Settings] = None,
        ttfs_p99_latency: Optional[float] = SAMBANOVA_TTFS_P99,
        **kwargs: Any,
    ) -> None:
        """Initialize SambaNova STT service.

        Args:
            model: Whisper model to use.

                .. deprecated:: 0.0.105
                    Use ``settings=SambaNovaSTTService.Settings(model=...)`` instead.

            api_key: SambaNova API key. Defaults to None.
            base_url: API base URL. Defaults to "https://api.sambanova.ai/v1".
            language: Language of the audio input.

                .. deprecated:: 0.0.105
                    Use ``settings=SambaNovaSTTService.Settings(language=...)`` instead.

            prompt: Optional text to guide the model's style or continue a previous segment.

                .. deprecated:: 0.0.105
                    Use ``settings=SambaNovaSTTService.Settings(prompt=...)`` instead.

            temperature: Optional sampling temperature between 0 and 1.

                .. deprecated:: 0.0.105
                    Use ``settings=SambaNovaSTTService.Settings(temperature=...)`` instead.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to `pipecat.services.whisper.base_stt.BaseWhisperSTTService`.
        """
        # --- 1. Hardcoded defaults ---
        default_settings = self.Settings(
            model="Whisper-Large-v3",
            language=self.language_to_service_language(Language.EN),
            prompt=None,
            temperature=None,
        )

        # --- 2. Deprecated direct-arg overrides ---
        if model is not None:
            _warn_deprecated_param("model", self.Settings, "model")
            default_settings.model = model
        if language is not None:
            _warn_deprecated_param("language", self.Settings, "language")
            default_settings.language = self.language_to_service_language(language)
        if prompt is not None:
            _warn_deprecated_param("prompt", self.Settings, "prompt")
            default_settings.prompt = prompt
        if temperature is not None:
            _warn_deprecated_param("temperature", self.Settings, "temperature")
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
        assert self._settings.language is not None

        if self._include_prob_metrics:
            # https://docs.sambanova.ai/docs/en/features/audio#request-parameters
            logger.warning(
                "SambaNova STT does not support probability metrics "
                "(include_prob_metrics parameter has no effect). "
                "Check their docs: https://docs.sambanova.ai/docs/en/features/audio#request-parameters for more details."
            )

        # Build kwargs dict with only set parameters
        kwargs = {
            "file": ("audio.wav", audio, "audio/wav"),
            "model": self._settings.model,
            "response_format": "json",
            "language": self._settings.language,
        }

        if self._settings.prompt is not None:
            kwargs["prompt"] = self._settings.prompt

        if self._settings.temperature is not None:
            kwargs["temperature"] = self._settings.temperature

        return await self._client.audio.transcriptions.create(**kwargs)
