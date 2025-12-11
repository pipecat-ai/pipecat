#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MiniMax text-to-speech service implementation.

This module provides integration with MiniMax's T2A (Text-to-Audio) API
for streaming text-to-speech synthesis.
"""

import json
from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts


def language_to_minimax_language(language: Language) -> Optional[str]:
    """Convert a Language enum to MiniMax language format.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding MiniMax language name, or None if not supported.
    """
    LANGUAGE_MAP = {
        Language.AF: "Afrikaans",
        Language.AR: "Arabic",
        Language.BG: "Bulgarian",
        Language.CA: "Catalan",
        Language.CS: "Czech",
        Language.DA: "Danish",
        Language.DE: "German",
        Language.EL: "Greek",
        Language.EN: "English",
        Language.ES: "Spanish",
        Language.FA: "Persian",  # ⚠️ Only supported by speech-2.6-* models
        Language.FI: "Finnish",
        Language.FIL: "Filipino",  # ⚠️ Only supported by speech-2.6-* models
        Language.FR: "French",
        Language.HE: "Hebrew",
        Language.HI: "Hindi",
        Language.HR: "Croatian",
        Language.HU: "Hungarian",
        Language.ID: "Indonesian",
        Language.IT: "Italian",
        Language.JA: "Japanese",
        Language.KO: "Korean",
        Language.MS: "Malay",
        Language.NB: "Norwegian",
        Language.NN: "Nynorsk",
        Language.NL: "Dutch",
        Language.PL: "Polish",
        Language.PT: "Portuguese",
        Language.RO: "Romanian",
        Language.RU: "Russian",
        Language.SK: "Slovak",
        Language.SL: "Slovenian",
        Language.SV: "Swedish",
        Language.TA: "Tamil",  # ⚠️ Only supported by speech-2.6-* models
        Language.TH: "Thai",
        Language.TR: "Turkish",
        Language.UK: "Ukrainian",
        Language.VI: "Vietnamese",
        Language.YUE: "Chinese,Yue",
        Language.ZH: "Chinese",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=False)


class MiniMaxHttpTTSService(TTSService):
    """Text-to-speech service using MiniMax's T2A (Text-to-Audio) API.

    Provides streaming text-to-speech synthesis using MiniMax's HTTP API
    with support for various voice settings, emotions, and audio configurations.
    Supports real-time audio streaming with configurable voice parameters.

    Platform documentation:
    https://www.minimax.io/platform/document/T2A%20V2?key=66719005a427f0c8a5701643
    """

    class InputParams(BaseModel):
        """Configuration parameters for MiniMax TTS.

        Parameters:
            language: Language for TTS generation. Supports 40 languages.
                Note: Filipino, Tamil, and Persian require speech-2.6-* models.
            speed: Speech speed (range: 0.5 to 2.0).
            volume: Speech volume (range: 0 to 10).
            pitch: Pitch adjustment (range: -12 to 12).
            emotion: Emotional tone (options: "happy", "sad", "angry", "fearful",
                "disgusted", "surprised", "calm", "fluent").
            english_normalization: Deprecated; use `text_normalization` instead

                .. deprecated:: 0.0.96
                    The `english_normalization` parameter is deprecated and will be removed in a future version.
                    Use the `text_normalization` parameter instead.

            text_normalization: Enable text normalization (Chinese/English).
            latex_read: Enable LaTeX formula reading.
            exclude_aggregated_audio: Whether to exclude aggregated audio in final chunk.
        """

        language: Optional[Language] = Language.EN
        speed: Optional[float] = 1.0
        volume: Optional[float] = 1.0
        pitch: Optional[int] = 0
        emotion: Optional[str] = None
        english_normalization: Optional[bool] = None  # Deprecated
        text_normalization: Optional[bool] = None
        latex_read: Optional[bool] = None
        exclude_aggregated_audio: Optional[bool] = None

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.minimax.io/v1/t2a_v2",
        group_id: str,
        model: str = "speech-02-turbo",
        voice_id: str = "Calm_Woman",
        aiohttp_session: aiohttp.ClientSession,
        sample_rate: Optional[int] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the MiniMax TTS service.

        Args:
            api_key: MiniMax API key for authentication.
            base_url: API base URL, defaults to MiniMax's T2A endpoint.
                Global: https://api.minimax.io/v1/t2a_v2
                Mainland China: https://api.minimaxi.chat/v1/t2a_v2
                Western United States: https://api-uw.minimax.io/v1/t2a_v2
            group_id: MiniMax Group ID to identify project.
            model: TTS model name. Defaults to "speech-02-turbo". Options include:
                "speech-2.6-hd", "speech-2.6-turbo" (latest, supports Filipino/Tamil/Persian),
                "speech-02-hd", "speech-02-turbo",
                "speech-01-hd", "speech-01-turbo".
            voice_id: Voice identifier. Defaults to "Calm_Woman".
            aiohttp_session: aiohttp.ClientSession for API communication.
            sample_rate: Output audio sample rate in Hz. If None, uses pipeline default.
            params: Additional configuration parameters.
            **kwargs: Additional arguments passed to parent TTSService.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)

        params = params or MiniMaxHttpTTSService.InputParams()

        self._api_key = api_key
        self._group_id = group_id
        self._base_url = f"{base_url}?GroupId={group_id}"
        self._session = aiohttp_session
        self._model_name = model
        self._voice_id = voice_id

        # Create voice settings
        self._settings = {
            "stream": True,
            "voice_setting": {
                "speed": params.speed,
                "vol": params.volume,
                "pitch": params.pitch,
            },
            "audio_setting": {
                "bitrate": 128000,
                "format": "pcm",
                "channel": 1,
            },
        }

        # Set voice and model
        self.set_voice(voice_id)
        self.set_model_name(model)

        # Add language boost if provided
        if params.language:
            service_lang = self.language_to_service_language(params.language)
            if service_lang:
                self._settings["language_boost"] = service_lang

        # Add optional emotion if provided
        if params.emotion:
            # Validate emotion is in the supported list
            supported_emotions = [
                "happy",
                "sad",
                "angry",
                "fearful",
                "disgusted",
                "surprised",
                "neutral",
                "fluent",
            ]
            if params.emotion in supported_emotions:
                self._settings["voice_setting"]["emotion"] = params.emotion
            else:
                logger.warning(
                    f"Unsupported emotion: {params.emotion}. Supported emotions: {supported_emotions}"
                )

        # If `english_normalization`, add `text_normalization` and print warning
        if params.english_normalization is not None:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "Parameter `english_normalization` is deprecated and will be removed in a future version. Use `text_normalization` instead.",
                    DeprecationWarning,
                )
            self._settings["voice_setting"]["text_normalization"] = params.english_normalization

        # Add text_normalization if provided (corrected parameter name)
        if params.text_normalization is not None:
            self._settings["voice_setting"]["text_normalization"] = params.text_normalization

        # Add latex_read if provided
        if params.latex_read is not None:
            self._settings["voice_setting"]["latex_read"] = params.latex_read

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as MiniMax service supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Convert a Language enum to MiniMax service language format.

        Args:
            language: The language to convert.

        Returns:
            The MiniMax-specific language name, or None if not supported.
        """
        return language_to_minimax_language(language)

    def set_model_name(self, model: str):
        """Set the TTS model to use.

        Args:
            model: The model name to use for synthesis.
        """
        self._model_name = model

    def set_voice(self, voice: str):
        """Set the voice to use.

        Args:
            voice: The voice identifier to use for synthesis.
        """
        self._voice_id = voice
        if "voice_setting" in self._settings:
            self._settings["voice_setting"]["voice_id"] = voice

    async def start(self, frame: StartFrame):
        """Start the MiniMax TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._settings["audio_setting"]["sample_rate"] = self.sample_rate
        logger.debug(f"MiniMax TTS initialized with sample_rate: {self.sample_rate}")

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate TTS audio from text using MiniMax's streaming API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        headers = {
            "accept": "application/json, text/plain, */*",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        # Create payload from settings
        payload = self._settings.copy()
        payload["model"] = self._model_name
        payload["text"] = text

        try:
            await self.start_ttfb_metrics()

            async with self._session.post(
                self._base_url, headers=headers, json=payload
            ) as response:
                if response.status != 200:
                    error_message = f"MiniMax TTS error: HTTP {response.status}"
                    yield ErrorFrame(error=error_message)
                    return

                await self.start_tts_usage_metrics(text)
                yield TTSStartedFrame()

                # Process the streaming response
                buffer = bytearray()

                CHUNK_SIZE = self.chunk_size

                async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                    if not chunk:
                        continue

                    buffer.extend(chunk)

                    # Find complete data blocks
                    while b"data:" in buffer:
                        start = buffer.find(b"data:")
                        next_start = buffer.find(b"data:", start + 5)

                        if next_start == -1:
                            # No next data block found, keep current data for next iteration
                            if start > 0:
                                buffer = buffer[start:]
                            break

                        # Extract a complete data block
                        data_block = buffer[start:next_start]
                        buffer = buffer[next_start:]

                        try:
                            data = json.loads(data_block[5:].decode("utf-8"))
                            # Skip data blocks containing extra_info
                            if "extra_info" in data:
                                logger.debug("Received final chunk with extra info")
                                continue

                            chunk_data = data.get("data", {})
                            if not chunk_data:
                                continue

                            audio_data = chunk_data.get("audio")
                            if not audio_data:
                                continue

                            # Process audio data in chunks
                            for i in range(0, len(audio_data), CHUNK_SIZE * 2):  # *2 for hex string
                                # Split hex string
                                hex_chunk = audio_data[i : i + CHUNK_SIZE * 2]
                                if not hex_chunk:
                                    continue

                                try:
                                    # Convert this chunk of data
                                    audio_chunk = bytes.fromhex(hex_chunk)
                                    if audio_chunk:
                                        await self.stop_ttfb_metrics()
                                        yield TTSAudioRawFrame(
                                            audio=audio_chunk,
                                            sample_rate=self.sample_rate,
                                            num_channels=1,
                                        )
                                except ValueError as e:
                                    logger.error(
                                        f"Error converting hex to binary: {e}",
                                    )
                                    continue

                        except json.JSONDecodeError as e:
                            logger.error(
                                f"Error decoding JSON: {e}, data: {data_block[:100]}",
                            )
                            continue

        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}", exception=e)
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()
