#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""[Hathora-hosted](https://models.hathora.dev) text-to-speech services."""

import io
import os
import wave
from typing import Optional, Tuple

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

def _decode_audio_payload(
    audio_bytes: bytes,
    *,
    fallback_sample_rate: int,
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

class KokoroTTSService(TTSService):
    """Kokoro is an open-weight TTS model with 82 million parameters.

    This service uses the Hathora-hosted Kokoro model via the HTTP API.

    [Documentation](https://models.hathora.dev/model/hexgrad-kokoro-82m)
    """

    def __init__(
        self,
        *,
        base_url = None,
        api_key = None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
        **kwargs,
    ):
        """Initialize the Hathora-hosted Kokoro TTS service.

        Args:
            base_url: Base URL for the Hathora Kokoro TTS API, .
            api_key: API key for authentication with the Hathora service;
                provisiion one [here](https://models.hathora.dev/tokens).
            voice: Voice to use for synthesis (see the
                [Hathora docs](https://models.hathora.dev/model/hexgrad-kokoro-82m)
                for the default value; [list of voices](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)).
            speed: Speech speed multiplier (0.5 = half speed, 2.0 = double speed, default: 1).
        """
        super().__init__(
            **kwargs,
        )
        self._base_url = base_url
        self._api_key = api_key
        self._voice = voice
        self._speed = speed

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str):
        try:
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()

            url = f"{self._base_url}"

            api_key = self._api_key or os.getenv("HATHORA_API_KEY")

            payload = {
                "text": text
            }

            if self._voice is not None:
                payload["voice"] = self._voice
            if self._speed is not None:
                payload["speed"] = self._speed

            yield TTSStartedFrame()

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}", "Accept": "application/octet-stream"},
                    json=payload,
                ) as resp:
                    audio_data = await resp.read()

            pcm_audio, sample_rate, num_channels = _decode_audio_payload(
                audio_data,
                fallback_sample_rate=self.sample_rate or self._init_sample_rate or 24000,
            )

            await self.stop_ttfb_metrics()

            frame = TTSAudioRawFrame(
                audio=pcm_audio,
                sample_rate=sample_rate,
                num_channels=num_channels,
            )

            yield frame

        except Exception as e:
            logger.error(f"Hathora error: {e}")
            yield ErrorFrame(f"Hathora error: {str(e)}")
        finally:
            await self.stop_ttfb_metrics()
            await self.stop_processing_metrics()
            yield TTSStoppedFrame()

class ChatterboxTTSService(TTSService):
    """Chatterbox is a public text-to-speech model optimized for natural and expressive voice synthesis.

    This service uses the Hathora-hosted Chatterbox model via the HTTP API.

    [Documentation](https://models.hathora.dev/model/resemble-ai-chatterbox)
    """

    def __init__(
        self,
        *,
        base_url = None,
        api_key = None,
        exaggeration: Optional[float] = None,
        audio_prompt: Optional[bytes] = None,
        cfg_weight: Optional[float] = None,
        **kwargs,
    ):
        """Initialize the Hathora-hosted Chatterbox TTS service.

        Args:
            base_url: Base URL for the Hathora Chatterbox TTS API.
            api_key: API key for authentication with the Hathora service;
                provisiion one [here](https://models.hathora.dev/tokens).
            exaggeration: Controls emotional intensity (default: 0.5).
            audio_prompt: Reference audio file for voice cloning.
            cfg_weight: Controls adherence to reference voice (default: 0.5).
        """

        super().__init__(
            **kwargs,
        )
        self._base_url = base_url
        self._api_key = api_key
        self._exaggeration = exaggeration
        self._audio_prompt = audio_prompt
        self._cfg_weight = cfg_weight

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str):
        try:
            await self.start_ttfb_metrics()

            url = f"{self._base_url}"

            url_query_params = []
            if self._exaggeration is not None:
                url_query_params.append(f"exaggeration={self._exaggeration}")
            if self._cfg_weight is not None:
                url_query_params.append(f"cfg_weight={self._cfg_weight}")

            if len(url_query_params) > 0:
                url += "?" + "&".join(url_query_params)

            api_key = self._api_key or os.getenv("HATHORA_API_KEY")

            form_data = aiohttp.FormData()
            form_data.add_field("text", text)

            if self._audio_prompt is not None:
                form_data.add_field("audio_prompt", self._audio_prompt, filename="audio.wav", content_type="application/octet-stream")

            yield TTSStartedFrame()

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                    data=form_data,
                ) as resp:
                    audio_data = await resp.read()

            await self.start_tts_usage_metrics(text)

            pcm_audio, sample_rate, num_channels = _decode_audio_payload(
                audio_data,
                fallback_sample_rate=self.sample_rate or self._init_sample_rate or 24000,
            )

            await self.stop_ttfb_metrics()

            frame = TTSAudioRawFrame(
                audio=pcm_audio,
                sample_rate=sample_rate,
                num_channels=num_channels,
            )

            yield frame

        except Exception as e:
            logger.error(f"Hathora error: {e}")
            yield ErrorFrame(f"Hathora error: {str(e)}")
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()
