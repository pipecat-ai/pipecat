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
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts


def language_to_minimax_language(language: Language) -> Optional[str]:
    """Convert a Language enum to MiniMax language format.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding MiniMax language name, or None if not supported.
    """
    BASE_LANGUAGES = {
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
    
    # Languages that require speech-2.6-* models
    V26_ONLY_LANGUAGES = {Language.FA, Language.FIL, Language.TA}

    result = BASE_LANGUAGES.get(language)

    # If not found in base languages, try to find the base language from a variant
    if not result:
        # Convert enum value to string and get the base language part (e.g. es-ES -> es)
        lang_str = str(language.value)
        base_code = lang_str.split("-")[0].lower()
        # Find matching language
        for code, name in BASE_LANGUAGES.items():
            if str(code.value).lower().startswith(base_code):
                result = name
                break

    return result


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
            text_normalization: Enable text normalization (Chinese/English).
            latex_read: Enable LaTeX formula reading.
            force_cbr: Enable Constant Bitrate (CBR) for audio encoding (MP3 only).
            exclude_aggregated_audio: Whether to exclude aggregated audio in final chunk.
            subtitle_enable: Enable subtitle generation (non-streaming only).
            subtitle_type: Subtitle timestamp granularity (options: "word", "sentence").
                Only effective when subtitle_enable is True. Defaults to "sentence".
        """

        language: Optional[Language] = Language.EN
        speed: Optional[float] = 1.0
        volume: Optional[float] = 1.0
        pitch: Optional[int] = 0
        emotion: Optional[str] = None
        text_normalization: Optional[bool] = None
        latex_read: Optional[bool] = None
        force_cbr: Optional[bool] = None
        exclude_aggregated_audio: Optional[bool] = None
        subtitle_enable: Optional[bool] = None
        subtitle_type: Optional[str] = "sentence"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.minimax.io/v1/t2a_v2", 
        # https://api-uw.minimax.io/v1/t2a_v2
        # support west of unite state 
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
        self._current_trace_id: Optional[str] = None

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

        # Add stream_options if exclude_aggregated_audio is set
        if params.exclude_aggregated_audio is not None:
            self._settings["stream_options"] = {
                "exclude_aggregated_audio": params.exclude_aggregated_audio
            }

        # Set voice and model
        self.set_voice(voice_id)
        self.set_model_name(model)

        # Add language boost if provided
        if params.language:
            service_lang = self.language_to_service_language(params.language)
            if service_lang:
                self._settings["language_boost"] = service_lang
                
                # Validate language-model compatibility
                # Filipino, Tamil, Persian only supported by speech-2.6-* models
                if params.language in {Language.FA, Language.FIL, Language.TA}:
                    if not model.startswith("speech-2.6"):
                        logger.warning(
                            f"Language {params.language.value} ({service_lang}) is only supported by "
                            f"speech-2.6-hd and speech-2.6-turbo models. "
                            f"Current model '{model}' may not support this language. "
                            f"Consider using 'speech-2.6-turbo' or 'speech-2.6-hd'."
                        )

        # Add optional emotion if provided
        if params.emotion:
            # Validate emotion is in the supported list (updated per official docs)
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

        # Add text_normalization if provided (corrected parameter name)
        if params.text_normalization is not None:
            self._settings["voice_setting"]["text_normalization"] = params.text_normalization

        # Add latex_read if provided
        if params.latex_read is not None:
            self._settings["voice_setting"]["latex_read"] = params.latex_read

        # Add force_cbr if provided (for MP3 format only)
        if params.force_cbr is not None:
            self._settings["audio_setting"]["force_cbr"] = params.force_cbr

        # Add subtitle settings if provided
        if params.subtitle_enable is not None:
            # Note: subtitle_enable only works in non-streaming mode (stream=false)
            # Current implementation uses streaming mode (stream=true)
            if params.subtitle_enable:
                logger.warning(
                    "subtitle_enable is set to True, but this service uses streaming mode. "
                    "Subtitle generation (subtitle_enable) only works in non-streaming mode. "
                    "Subtitles may not be generated. "
                    "For subtitle support, consider implementing a non-streaming TTS service."
                )
            
            self._settings["subtitle_enable"] = params.subtitle_enable
            
            # Add subtitle_type only when subtitle_enable is True
            if params.subtitle_enable and params.subtitle_type:
                # Validate subtitle_type
                if params.subtitle_type not in ["word", "sentence"]:
                    logger.warning(
                        f"Invalid subtitle_type: {params.subtitle_type}. "
                        f"Must be 'word' or 'sentence'. Using default 'sentence'."
                    )
                    self._settings["subtitle_type"] = "sentence"
                else:
                    self._settings["subtitle_type"] = params.subtitle_type
            elif not params.subtitle_enable and params.subtitle_type != "sentence":
                # Warn if subtitle_type is set but subtitle_enable is False
                logger.debug(
                    f"subtitle_type='{params.subtitle_type}' will be ignored because "
                    f"subtitle_enable is False."
                )

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
        logger.debug(f"MiniMax TTS initialized with sample_rate={self.sample_rate}")

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate TTS audio from text using MiniMax's streaming API.

        Args:
            text: The text to synthesize into speech.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        # Reset trace_id for new request
        self._current_trace_id = None

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
                # Extract trace_id from response header (available in all responses)
                trace_id = response.headers.get("Trace-Id") or response.headers.get("trace-id") or response.headers.get("X-Trace-Id") or "unknown"
                self._current_trace_id = trace_id
                
                # Log trace_id for all requests
                logger.info(f"MiniMax TTS request trace_id={trace_id}, status={response.status}")
                
                if response.status != 200:
                    # Try to read error response body
                    try:
                        error_body = await response.text()
                        error_data = json.loads(error_body)

                        # Extract MiniMax error details from body
                        base_resp = error_data.get("base_resp", {})
                        status_code = base_resp.get("status_code", response.status)
                        status_msg = base_resp.get("status_msg", "Unknown error")

                        error_message = (
                            f"MiniMax TTS error: HTTP {response.status}, "
                            f"status_code={status_code}, status_msg={status_msg}, "
                            f"trace_id={trace_id}"
                        )
                        logger.error(
                            error_message,
                            extra={
                                "trace_id": trace_id,
                                "http_status": response.status,
                                "status_code": status_code,
                                "text_length": len(text),
                            },
                        )
                    except Exception as parse_error:
                        # If parsing fails, use basic error message
                        error_message = f"MiniMax TTS error: HTTP {response.status}, trace_id={trace_id}"
                        logger.error(
                            error_message,
                            extra={"http_status": response.status, "trace_id": trace_id, "parse_error": str(parse_error)},
                        )

                    yield ErrorFrame(error=error_message)
                    return

                await self.start_tts_usage_metrics(text)
                yield TTSStartedFrame()

                # Process the streaming response
                logger.debug(f"Starting to read streaming response, status={response.status}, trace_id={trace_id}")
                buffer = bytearray()

                CHUNK_SIZE = self.chunk_size
                chunk_count = 0

                async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                    chunk_count += 1
                    logger.debug(f"Received chunk #{chunk_count}, size={len(chunk)} bytes")
                    if not chunk:
                        continue

                    buffer.extend(chunk)
                    
                    # Log raw buffer content for debugging
                    if chunk_count == 1:
                        logger.debug(f"Raw buffer content: {buffer[:200]}")  # First 200 bytes
                        
                        # Check if first chunk is a direct JSON error (not streaming format)
                        if not buffer.startswith(b"data:"):
                            try:
                                error_data = json.loads(buffer.decode("utf-8"))
                                base_resp = error_data.get("base_resp", {})
                                status_code = base_resp.get("status_code", 0)
                                
                                if status_code != 0:
                                    # This is a non-streaming error response
                                    # Use trace_id from header (already extracted above)
                                    status_msg = base_resp.get("status_msg", "Unknown error")
                                    
                                    error_message = (
                                        f"MiniMax TTS API error: status_code={status_code}, "
                                        f"status_msg={status_msg}, trace_id={self._current_trace_id}"
                                    )
                                    logger.error(
                                        error_message,
                                        extra={"trace_id": self._current_trace_id, "status_code": status_code},
                                    )
                                    yield ErrorFrame(error=error_message)
                                    return
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                # Not a valid JSON, continue with streaming processing
                                pass

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
                            data_str = data_block[5:].decode("utf-8")
                            logger.debug(f"Parsing data block: {data_str[:200]}...")  # Log first 200 chars
                            data = json.loads(data_str)

                            # Check for business errors in base_resp
                            base_resp = data.get("base_resp", {})
                            status_code = base_resp.get("status_code", 0)

                            if status_code != 0:
                                # API returned business error
                                status_msg = base_resp.get("status_msg", "Unknown error")
                                error_message = (
                                    f"MiniMax TTS API error: status_code={status_code}, "
                                    f"status_msg={status_msg}, trace_id={self._current_trace_id}"
                                )
                                logger.error(
                                    error_message,
                                    extra={"trace_id": self._current_trace_id, "status_code": status_code},
                                )
                                yield ErrorFrame(error=error_message)
                                return

                            # Handle final chunk with extra_info
                            if "extra_info" in data:
                                extra_info = data.get("extra_info", {})
                                logger.info(
                                    f"MiniMax TTS completed successfully, trace_id={self._current_trace_id}",
                                    extra={
                                        "trace_id": self._current_trace_id,
                                        "audio_length": extra_info.get("audio_length"),
                                        "audio_size": extra_info.get("audio_size"),
                                        "usage_characters": extra_info.get("usage_characters"),
                                        "word_count": extra_info.get("word_count"),
                                    },
                                )
                                continue  # No audio data in this block

                            # Extract audio data
                            chunk_data = data.get("data", {})
                            if not chunk_data:
                                continue

                            # Check for subtitle file (if subtitle generation is enabled)
                            subtitle_file = chunk_data.get("subtitle_file")
                            if subtitle_file:
                                logger.info(
                                    f"Subtitle file available: {subtitle_file}",
                                    extra={"trace_id": self._current_trace_id, "subtitle_url": subtitle_file},
                                )

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
                                    # Convert hex to binary
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
                                        extra={"trace_id": self._current_trace_id},
                                    )
                                    continue

                        except json.JSONDecodeError as e:
                            logger.error(
                                f"Error decoding JSON: {e}, data: {data_block[:100]}",
                                extra={"trace_id": self._current_trace_id or "unknown"},
                            )
                            continue

        except aiohttp.ClientError as e:
            error_msg = f"MiniMax TTS network error: {str(e)}"
            logger.exception(error_msg, extra={"trace_id": self._current_trace_id or "unknown"})
            yield ErrorFrame(error=error_msg)
        except Exception as e:
            error_msg = f"MiniMax TTS error: {str(e)}"
            logger.exception(error_msg, extra={"trace_id": self._current_trace_id or "unknown"})
            yield ErrorFrame(error=error_msg)
        finally:
            if self._current_trace_id:
                logger.debug(
                    f"MiniMax TTS request finished, trace_id={self._current_trace_id}, "
                    f"received {chunk_count if 'chunk_count' in locals() else 0} chunks"
                )
            else:
                logger.debug(f"MiniMax TTS request finished with no trace_id (no data received)")
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()
