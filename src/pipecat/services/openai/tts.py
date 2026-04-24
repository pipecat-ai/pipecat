#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI text-to-speech service implementation.

This module provides integration with OpenAI's text-to-speech API for
generating high-quality synthetic speech from text input.
"""

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Literal

from loguru import logger
from openai import AsyncOpenAI, BadRequestError
from pydantic import BaseModel

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
)
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven, assert_given
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

ValidVoice = Literal[
    "alloy",
    "ash",
    "ballad",
    "cedar",
    "coral",
    "echo",
    "fable",
    "marin",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
]

VALID_VOICES: dict[str, ValidVoice] = {
    "alloy": "alloy",
    "ash": "ash",
    "ballad": "ballad",
    "cedar": "cedar",
    "coral": "coral",
    "echo": "echo",
    "fable": "fable",
    "marin": "marin",
    "nova": "nova",
    "onyx": "onyx",
    "sage": "sage",
    "shimmer": "shimmer",
    "verse": "verse",
}


@dataclass
class OpenAITTSSettings(TTSSettings):
    """Settings for OpenAITTSService.

    Parameters:
        instructions: Instructions to guide voice synthesis behavior.
        speed: Voice speed control (0.25 to 4.0, default 1.0).
    """

    instructions: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    speed: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class OpenAITTSService(TTSService):
    """OpenAI Text-to-Speech service that generates audio from text.

    This service uses the OpenAI TTS API to generate PCM-encoded audio at 24kHz.
    Supports multiple voice models and configurable parameters for high-quality
    speech synthesis with streaming audio output.
    """

    Settings = OpenAITTSSettings
    _settings: Settings

    OPENAI_SAMPLE_RATE = 24000  # OpenAI TTS always outputs at 24kHz

    class InputParams(BaseModel):
        """Input parameters for OpenAI TTS configuration.

        .. deprecated:: 0.0.105
            Use ``settings=OpenAITTSService.Settings(...)`` instead.

        Parameters:
            instructions: Instructions to guide voice synthesis behavior.
            speed: Voice speed control (0.25 to 4.0, default 1.0).
        """

        instructions: str | None = None
        speed: float | None = None

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        voice: str | None = None,
        model: str | None = None,
        sample_rate: int | None = None,
        instructions: str | None = None,
        speed: float | None = None,
        params: InputParams | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize OpenAI TTS service.

        Args:
            api_key: OpenAI API key for authentication. If None, uses environment variable.
            base_url: Custom base URL for OpenAI API. If None, uses default.
            voice: Voice ID to use for synthesis. Defaults to "alloy".

                .. deprecated:: 0.0.105
                    Use ``settings=OpenAITTSService.Settings(voice=...)`` instead.

            model: TTS model to use. Defaults to "gpt-4o-mini-tts".

                .. deprecated:: 0.0.105
                    Use ``settings=OpenAITTSService.Settings(model=...)`` instead.

            sample_rate: Output audio sample rate in Hz. If None, uses OpenAI's default 24kHz.
            instructions: Optional instructions to guide voice synthesis behavior.

                .. deprecated:: 0.0.105
                    Use ``settings=OpenAITTSService.Settings(instructions=...)`` instead.

            speed: Voice speed control (0.25 to 4.0, default 1.0).

                .. deprecated:: 0.0.105
                    Use ``settings=OpenAITTSService.Settings(speed=...)`` instead.

            params: Optional synthesis controls (acting instructions, speed, ...).

                .. deprecated:: 0.0.105
                    Use ``settings=OpenAITTSService.Settings(...)`` instead.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional keyword arguments passed to TTSService.
        """
        if sample_rate and sample_rate != self.OPENAI_SAMPLE_RATE:
            logger.warning(
                f"OpenAI TTS only supports {self.OPENAI_SAMPLE_RATE}Hz sample rate. "
                f"Current rate of {sample_rate}Hz may cause issues."
            )

        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="gpt-4o-mini-tts",
            voice="alloy",
            language=None,
            instructions=None,
            speed=None,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if voice is not None:
            self._warn_init_param_moved_to_settings("voice", "voice")
            default_settings.voice = voice
        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model
        if instructions is not None:
            self._warn_init_param_moved_to_settings("instructions", "instructions")
            default_settings.instructions = instructions
        if speed is not None:
            self._warn_init_param_moved_to_settings("speed", "speed")
            default_settings.speed = speed

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                if params.instructions is not None:
                    default_settings.instructions = params.instructions
                if params.speed is not None:
                    default_settings.speed = params.speed

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as OpenAI TTS service supports metrics generation.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the OpenAI TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        if self.sample_rate != self.OPENAI_SAMPLE_RATE:
            logger.warning(
                f"OpenAI TTS requires {self.OPENAI_SAMPLE_RATE}Hz sample rate. "
                f"Current rate of {self.sample_rate}Hz may cause issues."
            )

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using OpenAI's TTS API.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech data.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")
        try:
            # Setup API parameters
            create_params = {
                "input": text,
                "model": self._settings.model,
                "voice": VALID_VOICES[assert_given(self._settings.voice)],
                "response_format": "pcm",
            }

            if self._settings.instructions:
                create_params["instructions"] = self._settings.instructions

            if self._settings.speed:
                create_params["speed"] = self._settings.speed

            async with self._client.audio.speech.with_streaming_response.create(
                **create_params
            ) as r:
                if r.status_code != 200:
                    error = await r.text()
                    logger.error(
                        f"{self} error getting audio (status: {r.status_code}, error: {error})"
                    )
                    yield ErrorFrame(
                        error=f"Error getting audio (status: {r.status_code}, error: {error})"
                    )
                    return

                await self.start_tts_usage_metrics(text)

                CHUNK_SIZE = self.chunk_size

                async for chunk in r.iter_bytes(CHUNK_SIZE):
                    if len(chunk) > 0:
                        await self.stop_ttfb_metrics()
                        frame = TTSAudioRawFrame(chunk, self.sample_rate, 1, context_id=context_id)
                        yield frame
        except BadRequestError as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
