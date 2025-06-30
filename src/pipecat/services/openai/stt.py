#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Speech-to-Text service implementation using OpenAI's transcription API."""

from typing import Optional

from pipecat.services.whisper.base_stt import BaseWhisperSTTService, Transcription
from pipecat.transcriptions.language import Language


class OpenAISTTService(BaseWhisperSTTService):
    """OpenAI Speech-to-Text service that generates text from audio.

    Uses OpenAI's transcription API to convert audio to text. Requires an OpenAI API key
    set via the api_key parameter or OPENAI_API_KEY environment variable.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o-transcribe",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        language: Optional[Language] = Language.EN,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ):
        """Initialize OpenAI STT service.

        Args:
            model: Model to use — either gpt-4o or Whisper. Defaults to "gpt-4o-transcribe".
            api_key: OpenAI API key. Defaults to None.
            base_url: API base URL. Defaults to None.
            language: Language of the audio input. Defaults to English.
            prompt: Optional text to guide the model's style or continue a previous segment.
            temperature: Optional sampling temperature between 0 and 1. Defaults to 0.0.
            **kwargs: Additional arguments passed to BaseWhisperSTTService.
        """
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            language=language,
            prompt=prompt,
            temperature=temperature,
            **kwargs,
        )

    async def _transcribe(self, audio: bytes) -> Transcription:
        assert self._language is not None  # Assigned in the BaseWhisperSTTService class

        # Build kwargs dict with only set parameters
        kwargs = {
            "file": ("audio.wav", audio, "audio/wav"),
            "model": self.model_name,
            "language": self._language,
        }

        if self._prompt is not None:
            kwargs["prompt"] = self._prompt

        if self._temperature is not None:
            kwargs["temperature"] = self._temperature

        return await self._client.audio.transcriptions.create(**kwargs)
