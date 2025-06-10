#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts


def language_to_sarvam_language(language: Language) -> Optional[str]:
    """Convert Pipecat Language enum to Sarvam AI language codes."""
    LANGUAGE_MAP = {
        Language.BN: "bn-IN",  # Bengali
        Language.EN: "en-IN",  # English (India)
        Language.GU: "gu-IN",  # Gujarati
        Language.HI: "hi-IN",  # Hindi
        Language.KN: "kn-IN",  # Kannada
        Language.ML: "ml-IN",  # Malayalam
        Language.MR: "mr-IN",  # Marathi
        Language.OR: "od-IN",  # Odia
        Language.PA: "pa-IN",  # Punjabi
        Language.TA: "ta-IN",  # Tamil
        Language.TE: "te-IN",  # Telugu
    }

    return LANGUAGE_MAP.get(language)


class SarvamTTSService(TTSService):
    """Text-to-Speech service using Sarvam AI's API.

    Converts text to speech using Sarvam AI's TTS models with support for multiple
    Indian languages. Provides control over voice characteristics like pitch, pace,
    and loudness.

    Args:
        api_key: Sarvam AI API subscription key.
        voice_id: Speaker voice ID (e.g., "anushka", "meera").
        model: TTS model to use ("bulbul:v1" or "bulbul:v2").
        aiohttp_session: Shared aiohttp session for making requests.
        base_url: Sarvam AI API base URL.
        sample_rate: Audio sample rate in Hz (8000, 16000, 22050, 24000).
        params: Additional voice and preprocessing parameters.

    Example:
        ```python
        tts = SarvamTTSService(
            api_key="your-api-key",
            voice_id="anushka",
            model="bulbul:v2",
            aiohttp_session=session,
            params=SarvamTTSService.InputParams(
                language=Language.HI,
                pitch=0.1,
                pace=1.2
            )
        )
        ```
    """

    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN
        pitch: Optional[float] = Field(default=0.0, ge=-0.75, le=0.75)
        pace: Optional[float] = Field(default=1.0, ge=0.3, le=3.0)
        loudness: Optional[float] = Field(default=1.0, ge=0.1, le=3.0)
        enable_preprocessing: Optional[bool] = False

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str = "anushka",
        model: str = "bulbul:v2",
        aiohttp_session: aiohttp.ClientSession,
        base_url: str = "https://api.sarvam.ai",
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or SarvamTTSService.InputParams()

        self._api_key = api_key
        self._base_url = base_url
        self._session = aiohttp_session

        self._settings = {
            "language": self.language_to_service_language(params.language)
            if params.language
            else "en-IN",
            "pitch": params.pitch,
            "pace": params.pace,
            "loudness": params.loudness,
            "enable_preprocessing": params.enable_preprocessing,
        }

        self.set_model_name(model)
        self.set_voice(voice_id)

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        return language_to_sarvam_language(language)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._settings["sample_rate"] = self.sample_rate

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics()

            payload = {
                "text": text,
                "target_language_code": self._settings["language"],
                "speaker": self._voice_id,
                "pitch": self._settings["pitch"],
                "pace": self._settings["pace"],
                "loudness": self._settings["loudness"],
                "speech_sample_rate": self.sample_rate,
                "enable_preprocessing": self._settings["enable_preprocessing"],
                "model": self._model_name,
            }

            headers = {
                "api-subscription-key": self._api_key,
                "Content-Type": "application/json",
            }

            url = f"{self._base_url}/text-to-speech"

            yield TTSStartedFrame()

            async with self._session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Sarvam API error: {error_text}")
                    await self.push_error(ErrorFrame(f"Sarvam API error: {error_text}"))
                    return

                response_data = await response.json()

            await self.start_tts_usage_metrics(text)

            # Decode base64 audio data
            if "audios" not in response_data or not response_data["audios"]:
                logger.error("No audio data received from Sarvam API")
                await self.push_error(ErrorFrame("No audio data received"))
                return

            # Get the first audio (there should be only one for single text input)
            base64_audio = response_data["audios"][0]
            audio_data = base64.b64decode(base64_audio)

            # Strip WAV header (first 44 bytes) if present
            if audio_data.startswith(b"RIFF"):
                logger.debug("Stripping WAV header from Sarvam audio data")
                audio_data = audio_data[44:]

            frame = TTSAudioRawFrame(
                audio=audio_data,
                sample_rate=self.sample_rate,
                num_channels=1,
            )

            yield frame

        except Exception as e:
            logger.error(f"{self} exception: {e}")
            await self.push_error(ErrorFrame(f"Error generating TTS: {e}"))
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()
