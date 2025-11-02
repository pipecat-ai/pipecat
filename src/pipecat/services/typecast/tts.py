#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Typecast text-to-speech service implementation."""

from __future__ import annotations

from typing import AsyncGenerator, Dict, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts

AUTHORIZATION_HEADER = "X-API-KEY"
DEFAULT_BASE_URL = "https://api.typecast.ai/v1/text-to-speech"
DEFAULT_MODEL = "ssfm-v21"
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_VOICE_ID = "tc_62a8975e695ad26f7fb514d1"

ISO2_TO_ISO3_LANGUAGE_MAP: Dict[str, str] = {
    "en": "eng",
    "ko": "kor",
    "ja": "jpn",
    "zh": "zho",
    "es": "spa",
    "de": "deu",
    "fr": "fra",
    "it": "ita",
    "ru": "rus",
    "ar": "ara",
    "pt": "por",
    "nl": "nld",
    "pl": "pol",
    "sv": "swe",
    "tr": "tur",
    "hi": "hin",
    "th": "tha",
    "vi": "vie",
    "id": "ind",
}


def language_to_typecast_language(language: Language) -> Optional[str]:
    """Convert Pipecat language enum values to Typecast ISO-639-3 codes."""

    if not language:
        return None

    base_code = str(language.value).split("-")[0].lower()
    return ISO2_TO_ISO3_LANGUAGE_MAP.get(base_code)


class PromptOptions(BaseModel):
    """Emotion control options for Typecast synthesis."""

    emotion_preset: str = Field(default="normal")
    emotion_intensity: float = Field(default=1.0, ge=0.0, le=2.0)


class OutputOptions(BaseModel):
    """Audio output configuration supported by Typecast."""

    volume: int = Field(default=100, ge=0, le=200)
    audio_pitch: int = Field(default=0, ge=-12, le=12)
    audio_tempo: float = Field(default=1.0, ge=0.5, le=2.0)
    audio_format: str = Field(default="wav")

    model_config = {"validate_assignment": True}


class TypecastInputParams(BaseModel):
    """Input parameters for Typecast TTS configuration."""

    language: Optional[Language] = Language.EN
    seed: Optional[int] = Field(default=None, ge=0)
    prompt_options: PromptOptions = Field(default_factory=PromptOptions)
    output_options: OutputOptions = Field(default_factory=OutputOptions)


class TypecastTTSService(TTSService):
    """HTTP-based Typecast TTS service."""

    InputParams = TypecastInputParams

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        voice_id: str = DEFAULT_VOICE_ID,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        sample_rate: Optional[int] = DEFAULT_SAMPLE_RATE,
        params: Optional[TypecastInputParams] = None,
        **kwargs,
    ):
        """Initialize the Typecast TTS service."""
        super().__init__(sample_rate=sample_rate, **kwargs)

        if not api_key:
            raise ValueError("Typecast API key is required.")

        params = params or TypecastTTSService.InputParams()

        self._api_key = api_key
        self._session = aiohttp_session
        self._base_url = base_url.rstrip("/")

        language_code = (
            self.language_to_service_language(params.language) if params.language else None
        )
        prompt_config = params.prompt_options.model_dump(exclude_none=True)
        output_config = params.output_options.model_dump(exclude_none=True)

        self._settings = {
            "base_url": self._base_url,
            "model": model,
            "voice_id": voice_id,
            "language": language_code,
            "seed": params.seed,
            "prompt": prompt_config,
            "output": output_config,
        }

        self.set_model_name(model)
        self.set_voice(voice_id)

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        return language_to_typecast_language(language)

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech audio using the Typecast REST API."""
        logger.debug(f"{self}: Generating TTS [{text}]")

        headers = {
            "Content-Type": "application/json",
            AUTHORIZATION_HEADER: self._api_key,
        }

        prompt_options = dict(self._settings.get("prompt") or {})
        output_options = dict(self._settings.get("output") or {})

        audio_format = output_options.get("audio_format", "wav")
        if audio_format != "wav":
            error_message = (
                f"TypecastTTSService only supports 'wav' audio_format, not '{audio_format}'."
            )
            logger.error(error_message)
            yield ErrorFrame(error_message)
            return

        output_options["audio_format"] = "wav"

        payload: Dict[str, object] = {
            "text": text,
            "model": self._settings.get("model", DEFAULT_MODEL),
            "voice_id": self._settings.get("voice_id", DEFAULT_VOICE_ID),
            "prompt": prompt_options,
            "output": output_options,
        }

        language = self._settings.get("language")
        if language:
            payload["language"] = language

        seed = self._settings.get("seed")
        if seed is not None:
            payload["seed"] = seed

        base_url = self._settings.get("base_url", self._base_url)
        ttfb_stopped = False

        try:
            await self.start_ttfb_metrics()

            async with self._session.post(base_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_detail: str
                    try:
                        error_body = await response.json()
                        error_detail = error_body.get("message", str(error_body))
                    except Exception:
                        error_detail = await response.text()

                    logger.error(
                        f"{self}: Typecast API error (status {response.status}): {error_detail}"
                    )
                    yield ErrorFrame(
                        f"Error from Typecast API: status {response.status}, detail: {error_detail}"
                    )
                    return

                await self.start_tts_usage_metrics(text)
                yield TTSStartedFrame()

                chunk_iterator = response.content.iter_chunked(self.chunk_size)
                first_frame = True

                async for frame in self._stream_audio_frames_from_iterator(
                    chunk_iterator, strip_wav_header=True
                ):
                    if first_frame:
                        await self.stop_ttfb_metrics()
                        ttfb_stopped = True
                        first_frame = False
                    yield frame

                if first_frame and not ttfb_stopped:
                    await self.stop_ttfb_metrics()
                    ttfb_stopped = True

        except Exception as exc:
            logger.exception(f"{self}: Error generating audio: {exc}")
            yield ErrorFrame(str(exc))
        finally:
            if not ttfb_stopped:
                await self.stop_ttfb_metrics()
            logger.debug(f"{self}: Finished TTS [{text}]")
            yield TTSStoppedFrame()
