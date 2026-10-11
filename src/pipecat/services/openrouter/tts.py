#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenRouter text-to-speech service implementation.

This module provides integration with OpenRouter's text-to-speech API for
generating high-quality synthetic speech from text input. The API is
OpenAI-compatible and supports multiple providers (OpenAI, Google, Mistral)
and their respective voice sets.
"""

from collections.abc import AsyncGenerator
from dataclasses import dataclass

from loguru import logger
from openai import AsyncOpenAI, BadRequestError

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
)
from pipecat.services.openai.tts import OpenAITTSService, OpenAITTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.services.settings import assert_given
from pipecat.utils.tracing.service_decorators import traced_tts

OPENROUTER_TTS_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_TTS_MODEL = "openai/gpt-4o-mini-tts-2025-12-15"
OPENROUTER_DEFAULT_TTS_VOICE = "alloy"


@dataclass
class OpenRouterTTSSettings(OpenAITTSSettings):
    """Settings for OpenRouterTTSService.

    Inherits all OpenAI TTS settings. Voice values are provider-namespaced:
    OpenAI voices use short names (``alloy``, ``nova``), Voxtral encodes
    language/persona/emotion (e.g. ``en_paul_happy``), and Kokoro prefixes
    with language/gender (e.g. ``af_bella``).
    """

    pass


class OpenRouterTTSService(OpenAITTSService):
    """OpenRouter Text-to-Speech service that generates audio from text.

    This service routes TTS requests through OpenRouter's OpenAI-compatible
    endpoint (``/api/v1/audio/speech``), supporting multiple underlying
    providers (OpenAI, Google, Mistral) and their respective voice sets.

    PCM audio is returned at 24kHz. Voice identifiers vary by model — check
    each model's page on openrouter.ai for the supported voices.

    Example usage::

        tts = OpenRouterTTSService(
            api_key=os.environ["OPENROUTER_API_KEY"],
            settings=OpenRouterTTSService.Settings(
                model="openai/gpt-4o-mini-tts-2025-12-15",
                voice="alloy",
            ),
        )
    """

    Settings = OpenRouterTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = OPENROUTER_TTS_BASE_URL,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize OpenRouter TTS service.

        Args:
            api_key: OpenRouter API key. If None, reads from the ``OPENROUTER_API_KEY``
                environment variable.
            base_url: OpenRouter API base URL. Defaults to ``https://openrouter.ai/api/v1``.
            settings: Model, voice, and synthesis options. When omitted, defaults to
                model ``openai/gpt-4o-mini-tts-2025-12-15`` and voice ``alloy``.
            **kwargs: Additional keyword arguments passed to TTSService (e.g.
                ``sample_rate``, ``push_start_frame``).
        """
        default_settings = self.Settings(
            model=OPENROUTER_DEFAULT_TTS_MODEL,
            voice=OPENROUTER_DEFAULT_TTS_VOICE,
            language=None,
            instructions=None,
            speed=None,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        TTSService.__init__(
            self,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using OpenRouter's TTS API.

        Unlike the OpenAI implementation, voice validation is skipped because
        OpenRouter supports multiple provider voice sets with different naming
        conventions (e.g. ``alloy``, ``en_paul_happy``, ``af_bella``).

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech data.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        voice = assert_given(self._settings.voice)
        if voice is None:
            yield ErrorFrame(error="OpenRouter TTS voice must be specified")
            return

        try:
            create_params = {
                "input": text,
                "model": self._settings.model,
                "voice": voice,
                "response_format": "pcm",
            }

            if self._settings.instructions:
                create_params["instructions"] = self._settings.instructions

            if self._settings.speed:
                create_params["speed"] = self._settings.speed

            async with self._client.audio.speech.with_streaming_response.create(
                **create_params
            ) as r:
                if r.status_code != 200:
                    error = await r.text()
                    logger.error(
                        f"{self} error getting audio (status: {r.status_code}, error: {error})"
                    )
                    yield ErrorFrame(
                        error=f"Error getting audio (status: {r.status_code}, error: {error})"
                    )
                    return

                await self.start_tts_usage_metrics(text)

                CHUNK_SIZE = self.chunk_size

                async for chunk in r.iter_bytes(CHUNK_SIZE):
                    if len(chunk) > 0:
                        await self.stop_ttfb_metrics()
                        frame = TTSAudioRawFrame(chunk, self.sample_rate, 1, context_id=context_id)
                        yield frame
        except BadRequestError as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
