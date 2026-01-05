#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""[Hathora-hosted](https://models.hathora.dev) text-to-speech services."""

import io
import os
import wave
from typing import AsyncGenerator, Optional, Tuple

import aiohttp

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

from .utils import ConfigOption


def _decode_audio_payload(
    audio_bytes: bytes,
    *,
    fallback_sample_rate: int = 24000,
    fallback_channels: int = 1,
) -> Tuple[bytes, int, int]:
    """Convert a WAV/PCM payload into raw PCM samples for TTSAudioRawFrame."""
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wav_reader:
            channels = wav_reader.getnchannels()
            sample_rate = wav_reader.getframerate()
            frames = wav_reader.readframes(wav_reader.getnframes())
            return frames, sample_rate, channels
    except (wave.Error, EOFError):
        # If the payload is already raw PCM, just pass it through.
        return audio_bytes, fallback_sample_rate, fallback_channels


class HathoraTTSService(TTSService):
    """This service supports several different text-to-speech models hosted by Hathora.

    [Documentation](https://models.hathora.dev)
    """

    def __init__(
        self,
        *,
        model: str,
        voice_id: Optional[str] = None,
        speed: Optional[float] = None,
        model_config: Optional[list[ConfigOption]] = None,
        base_url: str = "https://api.models.hathora.dev/inference/v1/tts",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the Hathora TTS service.

        Args:
            model: Model to use; find available models
                [here](https://models.hathora.dev).
            voice: Voice to use for synthesis (if supported by model).
            speed: Speech speed multiplier (if supported by model).
            model_config: Some models support additional config, refer to
                [docs](https://models.hathora.dev) for each model to see
                what is supported.
            base_url: Base API URL for the Hathora TTS service.
            api_key: API key for authentication with the Hathora service;
                provision one [here](https://models.hathora.dev/tokens).
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(
            **kwargs,
        )
        self._model = model
        self._voice = voice
        self._speed = speed
        self._model_config = model_config
        self._base_url = base_url
        self._api_key = api_key or os.getenv("HATHORA_API_KEY")

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Run text-to-speech synthesis on the provided text.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        try:
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()

            url = f"{self._base_url}"

            payload = {"model": self._model, "text": text}

            if self._voice is not None:
                payload["voice"] = self._voice
            if self._speed is not None:
                payload["speed"] = self._speed
            if self._model_config is not None:
                payload["model_config"] = [
                    {"name": option.name, "value": option.value} for option in self._model_config
                ]

            yield TTSStartedFrame()

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                ) as resp:
                    audio_data = await resp.read()

            pcm_audio, sample_rate, num_channels = _decode_audio_payload(
                audio_data,
                fallback_sample_rate=self.sample_rate,
            )

            frame = TTSAudioRawFrame(
                audio=pcm_audio,
                sample_rate=sample_rate,
                num_channels=num_channels,
            )

            yield frame

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        finally:
            await self.stop_ttfb_metrics()
            await self.stop_processing_metrics()
            yield TTSStoppedFrame()
