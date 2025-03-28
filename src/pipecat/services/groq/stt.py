#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import Optional

from pipecat.services.whisper.base_stt import BaseWhisperSTTService, Transcription
from pipecat.transcriptions.language import Language


class GroqSTTService(BaseWhisperSTTService):
    """Groq Whisper speech-to-text service.

    Uses Groq's Whisper API to convert audio to text. Requires a Groq API key
    set via the api_key parameter or GROQ_API_KEY environment variable.

    Args:
        model: Whisper model to use. Defaults to "whisper-large-v3-turbo".
        api_key: Groq API key. Defaults to None.
        base_url: API base URL. Defaults to "https://api.groq.com/openai/v1".
        language: Language of the audio input. Defaults to English.
        prompt: Optional text to guide the model's style or continue a previous segment.
        temperature: Optional sampling temperature between 0 and 1. Defaults to 0.0.
        **kwargs: Additional arguments passed to BaseWhisperSTTService.
    """

    def __init__(
        self,
        *,
        model: str = "whisper-large-v3-turbo",
        api_key: Optional[str] = None,
        base_url: str = "https://api.groq.com/openai/v1",
        language: Optional[Language] = Language.EN,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ):
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
            "response_format": "json",
            "language": self._language,
        }

        if self._prompt is not None:
            kwargs["prompt"] = self._prompt

        if self._temperature is not None:
            kwargs["temperature"] = self._temperature

        return await self._client.audio.transcriptions.create(**kwargs)
