"""Speaches TTS Service — uses the OpenAI-compatible /v1/audio/speech endpoint."""

from dataclasses import dataclass

from loguru import logger
from openai import BadRequestError

from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame
from pipecat.services.openai.tts import OpenAITTSService, OpenAITTSSettings
from pipecat.utils.tracing.service_decorators import traced_tts


@dataclass
class SpeachesTTSSettings(OpenAITTSSettings):
    """Settings for Speaches TTS service."""

    pass


class SpeachesTTSService(OpenAITTSService):
    """Speaches TTS service using the OpenAI-compatible audio speech endpoint."""

    Settings = SpeachesTTSSettings

    def __init__(
        self,
        *,
        api_key: str = "none",
        base_url: str = "http://localhost:8000/v1",
        sample_rate: int = 24000,
        settings: SpeachesTTSSettings | None = None,
        **kwargs,
    ):
        """Initialize the Speaches TTS service.

        Args:
            api_key: API key for authentication.
            base_url: Base URL of the Speaches-compatible endpoint.
            sample_rate: Audio sample rate in Hz.
            settings: Optional service settings.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            sample_rate=sample_rate,
            settings=settings,
            **kwargs,
        )

    @traced_tts
    async def run_tts(self, text: str, context_id: str):
        """Generate speech using the configured voice string as-is.

        Speaches exposes an OpenAI-compatible API surface, but unlike OpenAI it
        accepts provider-specific voice identifiers such as ``fettah``. The
        upstream OpenAI service adapter maps voices through a fixed whitelist,
        which breaks custom Speaches voices with a ``KeyError``.
        """
        try:
            create_params = {
                "input": text,
                "model": self._settings.model,
                "voice": self._settings.voice,
                "response_format": "pcm",
            }

            if self._settings.instructions:
                create_params["instructions"] = self._settings.instructions

            if self._settings.speed:
                create_params["speed"] = self._settings.speed

            async with self._client.audio.speech.with_streaming_response.create(
                **create_params
            ) as response:
                if response.status_code != 200:
                    error = await response.text()
                    logger.error(
                        f"{self} error getting audio (status: {response.status_code}, error: {error})"
                    )
                    yield ErrorFrame(
                        error=f"Error getting audio (status: {response.status_code}, error: {error})"
                    )
                    return

                await self.start_tts_usage_metrics(text)

                async for chunk in response.iter_bytes(self.chunk_size):
                    if len(chunk) > 0:
                        await self.stop_ttfb_metrics()
                        yield TTSAudioRawFrame(
                            chunk,
                            self.sample_rate,
                            1,
                            context_id=context_id,
                        )
        except BadRequestError as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
