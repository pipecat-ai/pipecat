#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Kokoro TTS service implementation."""

import asyncio
from typing import AsyncGenerator, AsyncIterator, Optional

import numpy as np
from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from kokoro import KPipeline
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Kokoro, you need to `pip install pipecat-ai[kokoro]`.")
    raise Exception(f"Missing module: {e}")


def language_to_kokoro_language(language: Language) -> str:
    """Convert a Language enum to Kokoro language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Kokoro language code, or None if not supported.
    """
    LANGUAGE_MAP = {
        Language.EN: "a",
        Language.EN_US: "a",
        Language.EN_GB: "b",
        Language.ES: "e",
        Language.FR: "f",
        Language.HI: "h",
        Language.IT: "i",
        Language.JA: "j",
        Language.PT: "p",
        Language.ZH: "z",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


class KokoroTTSService(TTSService):
    """Kokoro TTS service implementation.

    Provides local text-to-speech synthesis using the Kokoro-82M model.
    Automatically downloads the model on first use.
    """

    class InputParams(BaseModel):
        """Input parameters for Kokoro TTS configuration.

        Parameters:
            language: Language to use for synthesis.
        """

        language: Language = Language.EN

    def __init__(
        self,
        *,
        voice_id: str,
        repo_id="hexgrad/Kokoro-82M",
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Kokoro TTS service.

        Args:
            voice_id: Voice identifier to use for synthesis.
            repo_id: Hugging Face repository ID for the Kokoro model.
                Defaults to "hexgrad/Kokoro-82M".
            params: Configuration parameters for synthesis.
            **kwargs: Additional arguments passed to parent `TTSService`.
        """
        super().__init__(**kwargs)

        params = params or KokoroTTSService.InputParams()

        self._voice_id = voice_id
        self._lang_code = language_to_kokoro_language(params.language)
        self._pipeline = KPipeline(lang_code=self._lang_code, repo_id=repo_id)

        self._resampler = create_stream_resampler()

    def can_generate_metrics(self) -> bool:
        """Indicate that this service supports TTFB and usage metrics."""
        return True

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Synthesize speech from text using the Kokoro pipeline.

        Runs the Kokoro pipeline in a background thread and streams audio
        frames as they become available.

        Args:
            text: The text to synthesize.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        def async_next(it):
            try:
                return next(it)
            except StopIteration:
                return None

        async def async_iterator(iterator) -> AsyncIterator[bytes]:
            while True:
                item = await asyncio.to_thread(async_next, iterator)
                if item is None:
                    return

                (_, _, audio) = item

                # Kokoro outputs a PyTorch tensor at 24kHz, convert to int16 bytes
                audio_np = np.array(audio)
                audio_int16 = (audio_np * 32767).astype(np.int16).tobytes()

                yield audio_int16

        try:
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)
            yield TTSStartedFrame()

            async for frame in self._stream_audio_frames_from_iterator(
                async_iterator(self._pipeline(text, voice=self._voice_id)),
                in_sample_rate=24000,
            ):
                await self.stop_ttfb_metrics()
                yield frame
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        finally:
            logger.debug(f"{self}: Finished TTS [{text}]")
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()
