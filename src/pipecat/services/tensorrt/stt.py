#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Simplismart Speech-to-Text service implementation using Simplismart's transcription API."""

from typing import Optional
import base64
from fastapi import requests
from pipecat.services.whisper.base_stt import BaseWhisperSTTService, Transcription
from pipecat.transcriptions.language import Language
import httpx
import time

class SimplismartSTTService(BaseWhisperSTTService):
    """OpenAI Speech-to-Text service that generates text from audio.

    Uses Simplismart's transcription API to convert audio to text. Requires an Simplismart API key
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

        self._base_url = base_url + "/predict"

    async def _transcribe(self, audio: bytes) -> Transcription:
        assert self._language is not None  # Assigned in the BaseWhisperSTTService class

        # Build kwargs dict with only set parameters
        audio_b64 = base64.b64encode(audio).decode("utf-8")
        payload = {
            "audio_file": audio_b64,
            "language": self._language
        }
        if self._prompt is not None:
            payload["initial_prompt"] = self._prompt

        if self._temperature is not None:
            payload["temperature"] = self._temperature

        response = httpx.post(self._base_url, json=payload)
        text = response.json()["transcription"]
        text = "".join([i["text"] for i in text])

        return Transcription(text=text)
