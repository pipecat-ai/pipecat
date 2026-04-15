#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""MiniMax text-to-speech service implementation.

This module provides integration with MiniMax's T2A (Text-to-Audio) API
for streaming text-to-speech synthesis.
"""

import json
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, ClassVar, Dict, Mapping, Optional

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
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven
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


@dataclass
class MiniMaxTTSSettings(TTSSettings):
    """Settings for MiniMax TTS service.

    Parameters:
        stream: Whether to use streaming mode.
        speed: Speech speed (range: 0.5 to 2.0).
        volume: Speech volume (range: 0 to 10).
        pitch: Pitch adjustment (range: -12 to 12).
        emotion: Emotional tone (options: "happy", "sad", "angry", "fearful",
            "disgusted", "surprised", "calm", "fluent").
        text_normalization: Enable text normalization (Chinese/English).
        latex_read: Enable LaTeX formula reading.
        audio_bitrate: Audio bitrate in bps.
        audio_format: Audio output format.
        audio_channel: Number of audio channels.
        audio_sample_rate: Audio sample rate in Hz.
        language_boost: Language boost string for multilingual support.
    """

    stream: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    speed: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    volume: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    pitch: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    emotion: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    text_normalization: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    latex_read: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    audio_bitrate: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    audio_format: str | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    audio_channel: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    audio_sample_rate: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    language_boost: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)

    _aliases: ClassVar[Dict[str, str]] = {"voice_id": "voice"}

    @classmethod
    def from_mapping(cls, settings: Mapping[str, Any]) -> "MiniMaxTTSSettings":
        """Construct settings from a plain dict, destructuring legacy nested dicts.

        Handles ``voice_setting`` (with ``vol`` → ``volume`` rename) and
        ``audio_setting`` (with prefixed field mapping).
        """
        flat = dict(settings)

        voice = flat.pop("voice_setting", None)
        if isinstance(voice, dict):
            flat.setdefault("speed", voice.get("speed"))
            flat.setdefault("volume", voice.get("vol"))
            flat.setdefault("pitch", voice.get("pitch"))
            flat.setdefault("emotion", voice.get("emotion"))
            flat.setdefault("text_normalization", voice.get("text_normalization"))
            flat.setdefault("latex_read", voice.get("latex_read"))

        audio = flat.pop("audio_setting", None)
        if isinstance(audio, dict):
            flat.setdefault("audio_bitrate", audio.get("bitrate"))
            flat.setdefault("audio_format", audio.get("format"))
            flat.setdefault("audio_channel", audio.get("channel"))
            flat.setdefault("audio_sample_rate", audio.get("sample_rate"))

        return super().from_mapping(flat)


class MiniMaxHttpTTSService(TTSService):
    """Text-to-speech service using MiniMax's T2A (Text-to-Audio) API.

    Provides streaming text-to-speech synthesis using MiniMax's HTTP API
    with support for various voice settings, emotions, and audio configurations.
    Supports real-time audio streaming with configurable voice parameters.

    Platform documentation:
    https://www.minimax.io/platform/document/T2A%20V2?key=66719005a427f0c8a5701643
    """

    _settings: MiniMaxTTSSettings

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
        params = params or MiniMaxHttpTTSService.InputParams()

        super().__init__(
            sample_rate=sample_rate,
            settings=MiniMaxTTSSettings(
                model=model,
                voice=voice_id,
                language=None,
                stream=True,
                speed=params.speed,
                volume=params.volume,
                pitch=params.pitch,
                language_boost=None,
                emotion=None,
                text_normalization=None,
                latex_read=None,
                audio_bitrate=128000,
                audio_format="pcm",
                audio_channel=1,
                audio_sample_rate=0,
            ),
            **kwargs,
        )

        self._api_key = api_key
        self._group_id = group_id
        self._base_url = f"{base_url}?GroupId={group_id}"
        self._session = aiohttp_session

        # Add language boost if provided
        if params.language:
            service_lang = self.language_to_service_language(params.language)
            if service_lang:
                self._settings.language_boost = service_lang

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
                self._settings.emotion = params.emotion
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
            self._settings.text_normalization = params.english_normalization

        # Add text_normalization if provided (corrected parameter name)
        if params.text_normalization is not None:
            self._settings.text_normalization = params.text_normalization

        # Add latex_read if provided
        if params.latex_read is not None:
            self._settings.latex_read = params.latex_read

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

    async def start(self, frame: StartFrame):
        """Start the MiniMax TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        self._settings.audio_sample_rate = self.sample_rate
        logger.debug(f"MiniMax TTS initialized with sample_rate: {self.sample_rate}")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate TTS audio from text using MiniMax's streaming API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        headers = {
            "accept": "application/json, text/plain, */*",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        # Build voice_setting dict for API
        voice_setting = {
            "voice_id": self._settings.voice,
            "speed": self._settings.speed,
            "vol": self._settings.volume,
            "pitch": self._settings.pitch,
        }
        if self._settings.emotion is not None:
            voice_setting["emotion"] = self._settings.emotion
        if self._settings.text_normalization is not None:
            voice_setting["text_normalization"] = self._settings.text_normalization
        if self._settings.latex_read is not None:
            voice_setting["latex_read"] = self._settings.latex_read

        # Build audio_setting dict for API
        audio_setting = {
            "bitrate": self._settings.audio_bitrate,
            "format": self._settings.audio_format,
            "channel": self._settings.audio_channel,
            "sample_rate": self._settings.audio_sample_rate,
        }

        # Create payload from settings
        payload = {
            "stream": self._settings.stream,
            "voice_setting": voice_setting,
            "audio_setting": audio_setting,
            "model": self._settings.model,
            "text": text,
        }
        if self._settings.language_boost is not None:
            payload["language_boost"] = self._settings.language_boost

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
                yield TTSStartedFrame(context_id=context_id)

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
                                            context_id=context_id,
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
            yield TTSStoppedFrame(context_id=context_id)
