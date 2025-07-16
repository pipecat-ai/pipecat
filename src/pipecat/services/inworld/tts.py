#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Inworld's text-to-speech service implementations."""

import base64
import json
import uuid
import warnings
from typing import AsyncGenerator, List, Optional, Union

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field
import io, json, base64

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.tts_service import AudioContextWordTTSService, TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.asyncio.watchdog_async_iterator import WatchdogAsyncIterator
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator
from pipecat.utils.text.skip_tags_aggregator import SkipTagsAggregator
from pipecat.utils.tracing.service_decorators import traced_tts


def language_to_inworld_language(language: Language) -> Optional[str]:
    """Convert Pipecat's Language enum to Inworld's language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Inworld language code, or None if not supported.
    """
    BASE_LANGUAGES = {
        Language.EN: "en",
        Language.ES: "es",
        Language.FR: "fr",
        Language.KO: "ko",
        Language.NL: "nl",
        Language.ZH: "zh",
    }

    result = BASE_LANGUAGES.get(language)

    # If not found in base languages, try to find the base language from a variant
    if not result:
        # Convert enum value to string and get the base language part (e.g. es-ES -> es)
        lang_str = str(language.value)
        base_code = lang_str.split("-")[0].lower()
        # Look up the base code in our supported languages
        result = base_code if base_code in BASE_LANGUAGES.values() else None

    return result


class InworldHttpTTSService(TTSService):
    """Inworld HTTP-based TTS service.

    Provides text-to-speech using Inworld's HTTP API for simpler, non-streaming
    synthesis. Suitable for use cases where streaming is not required and simpler
    integration is preferred.
    """

    class InputParams(BaseModel):
        """Input parameters for Inworld HTTP TTS configuration.

        Parameters:
            language: Language to use for synthesis.
            speed: Voice speed control (string or float).
            emotion: List of emotion controls.

                .. deprecated:: 0.0.68
                        The `emotion` parameter is deprecated and will be removed in a future version.
        """

        language: Optional[Language] = Language.EN
        voice_id: str = "Ashley"

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        model: str = "inworld-tts-1",
        base_url: str = "https://api.inworld.ai/tts/v1/voice:stream",
        sample_rate: Optional[int] = 48000,
        encoding: str = "LINEAR16",
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Inworld HTTP TTS service.

        Args:
            api_key: Inworld API key for authentication.
            aiohttp_session: Shared aiohttp session for HTTP requests.
            voice_id: ID of the voice to use for synthesis.
            model: TTS model to use (e.g., "sonic-2").
            endpoint_url: Base URL for Inworld HTTP API.
            sample_rate: Audio sample rate. If None, uses default.
            encoding: Audio encoding format.
            params: Additional input parameters for voice customization.
            **kwargs: Additional arguments passed to the parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or InworldTTSService.InputParams()

        self._api_key = api_key
        self._session = aiohttp_session
        self._base_url = base_url
        self._settings = {
            "voiceId": params.voice_id,
            "modelId": model,
            "audio_config": {
                "audio_encoding": encoding,
                "sample_rate_hertz": sample_rate,
            },
            "language": self.language_to_service_language(params.language)
            if params.language
            else "en",
        }
        self.set_voice(params.voice_id)
        self.set_model_name(model)


    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Inworld HTTP service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to Inworld language format.

        Args:
            language: The language to convert.

        Returns:
            The Inworld-specific language code, or None if not supported.
        """
        return language_to_inworld_language(language)

    async def start(self, frame: StartFrame):
        """Start the Inworld HTTP TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._settings["audio_config"]["sample_rate_hertz"] = self.sample_rate

    async def stop(self, frame: EndFrame):
        """Stop the Inworld HTTP TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel the Inworld HTTP TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Inworld's HTTP API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        payload = {
            "text": text,
            "voiceId": self._settings["voiceId"],
            "modelId": self._settings["modelId"],
            "audio_config": self._settings["audio_config"],
            "language": self._settings["language"],
        }

        headers = {
            "Authorization": f"Basic {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            await self.start_ttfb_metrics()

            yield TTSStartedFrame()

            async with self._session.post(self._base_url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Inworld API error: {error_text}")
                    await self.push_error(ErrorFrame(f"Inworld API error: {error_text}"))
                    return

                raw_audio_data = io.BytesIO()

                async for line in response.content.iter_lines():
                    line_str = line.decode('utf-8').strip()
                    if not line_str:
                        continue
                    
                    try:
                        chunk = json.loads(line_str)
                        if "result" in chunk and "audioContent" in chunk["result"]:
                            audio_chunk = base64.b64decode(chunk["result"]["audioContent"])
                            # Skip WAV header if present (first 44 bytes)
                            if len(audio_chunk) > 44 and audio_chunk.startswith(b"RIFF"):
                                audio_data = audio_chunk[44:]
                            else:
                                audio_data = audio_chunk
                            raw_audio_data.write(audio_data)
                    except json.JSONDecodeError:
                        continue

            await self.start_tts_usage_metrics(text)

            audio_bytes = raw_audio_data.getvalue()
            if not audio_bytes:
                logger.error("No audio data received from Inworld API")
                await self.push_error(ErrorFrame("No audio data received"))
                return

            frame = TTSAudioRawFrame(
                audio=audio_bytes,
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
