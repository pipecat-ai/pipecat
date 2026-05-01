#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Groq text-to-speech service implementation."""

import io
import wave
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Literal, cast

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
)
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven, assert_given
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from groq import AsyncGroq
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Groq, you need to `pip install pipecat-ai[groq]`.")
    raise Exception(f"Missing module: {e}")

# Hint set for `output_format`. The values mirror the Literal that
# `groq.resources.audio.speech.AsyncSpeech.create` accepts on its
# `response_format` parameter (also visible as the `response_format` field of
# `groq.types.audio.SpeechCreateParams`). The groq SDK does not export this as
# a named alias, so we redeclare it here.
#
# This alias is used in unions like `GroqAudioFormat | str`, so pyright shows
# these values as completion hints without rejecting other strings. If groq
# adds a new format before this list is updated, callers can still pass it and
# we forward it through (with a cast at the API boundary). Keep in sync on a
# best-effort basis when bumping the groq dep.
GroqAudioFormat = Literal["flac", "mp3", "mulaw", "ogg", "wav"]


@dataclass
class GroqTTSSettings(TTSSettings):
    """Settings for GroqTTSService.

    Parameters:
        speed: Speech speed multiplier. Defaults to 1.0.
    """

    speed: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class GroqTTSService(TTSService):
    """Groq text-to-speech service implementation.

    Provides text-to-speech synthesis using Groq's TTS API. The service
    operates at a fixed 48kHz sample rate and supports various voices
    and output formats.
    """

    Settings = GroqTTSSettings
    _settings: Settings

    class InputParams(BaseModel):
        """Input parameters for Groq TTS configuration.

        .. deprecated:: 0.0.105
            Use ``settings=GroqTTSService.Settings(...)`` instead.

        Parameters:
            language: Language for speech synthesis. Defaults to English.
            speed: Speech speed multiplier. Defaults to 1.0.
        """

        language: Language | None = Language.EN
        speed: float | None = 1.0

    GROQ_SAMPLE_RATE = 48000  # Groq TTS only supports 48kHz sample rate

    def __init__(
        self,
        *,
        api_key: str,
        output_format: GroqAudioFormat | str = "wav",
        params: InputParams | None = None,
        model_name: str | None = None,
        voice_id: str | None = None,
        sample_rate: int | None = GROQ_SAMPLE_RATE,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize Groq TTS service.

        Args:
            api_key: Groq API key for authentication.
            output_format: Audio output format. Defaults to "wav".
            params: Additional input parameters for voice customization.

                .. deprecated:: 0.0.105
                    Use ``settings=GroqTTSService.Settings(...)`` instead.

            model_name: TTS model to use.

                .. deprecated:: 0.0.105
                    Use ``settings=GroqTTSService.Settings(model=...)`` instead.

            voice_id: Voice identifier to use.

                .. deprecated:: 0.0.105
                    Use ``settings=GroqTTSService.Settings(voice=...)`` instead.

            sample_rate: Audio sample rate. Must be 48000 Hz for Groq TTS.
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional arguments passed to parent TTSService class.
        """
        if sample_rate != self.GROQ_SAMPLE_RATE:
            logger.warning(f"Groq TTS only supports {self.GROQ_SAMPLE_RATE}Hz sample rate. ")

        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="canopylabs/orpheus-v1-english",
            voice="autumn",
            language="en",
            speed=1.0,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if model_name is not None:
            self._warn_init_param_moved_to_settings("model_name", "model")
            default_settings.model = model_name
        if voice_id is not None:
            self._warn_init_param_moved_to_settings("voice_id", "voice")
            default_settings.voice = voice_id

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                default_settings.language = str(params.language) if params.language else "en"
                default_settings.speed = params.speed

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            pause_frame_processing=True,
            push_start_frame=True,
            push_stop_frames=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._output_format: str = output_format

        self._client = AsyncGroq(api_key=self._api_key)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Groq TTS service supports metrics generation.
        """
        return True

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Groq's TTS API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech data.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")
        measuring_ttfb = True
        try:
            model = assert_given(self._settings.model)
            voice = assert_given(self._settings.voice)
            speed = assert_given(self._settings.speed)
            if model is None:
                raise ValueError("Groq TTS model must be specified")
            if voice is None:
                raise ValueError("Groq TTS voice must be specified")
            if speed is None:
                raise ValueError("Groq TTS speed must be specified")
            response = await self._client.audio.speech.create(
                model=model,
                voice=voice,
                # Cast satisfies groq's stricter Literal typing while letting
                # callers pass any string (e.g. a newer groq format we haven't
                # yet added to GroqAudioFormat). If the value is unsupported,
                # groq's API will surface a runtime error with a clear message.
                response_format=cast(GroqAudioFormat, self._output_format),
                # Note: as of 2026-02-25, only a speed of 1.0 is supported, but
                # here we pass it for completeness and future-proofing
                speed=speed,
                input=text,
            )

            async for data in response.iter_bytes():
                if measuring_ttfb:
                    await self.stop_ttfb_metrics()
                    measuring_ttfb = False

                with wave.open(io.BytesIO(data)) as w:
                    channels = w.getnchannels()
                    frame_rate = w.getframerate()
                    num_frames = w.getnframes()
                    bytes = w.readframes(num_frames)
                    yield TTSAudioRawFrame(bytes, frame_rate, channels, context_id=context_id)
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
