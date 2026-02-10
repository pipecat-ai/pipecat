#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Kokoro TTS service implementation using kokoro-onnx."""

import os
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np
from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    import requests
    from kokoro_onnx import Kokoro
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Kokoro, you need to `pip install pipecat-ai[kokoro]`.")
    raise Exception(f"Missing module: {e}")

KOKORO_CACHE_DIR = Path(os.path.expanduser("~/.cache/kokoro-onnx"))
KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
)


def _download_file(url: str, dest: Path):
    """Download a file from a URL to a destination path."""
    logger.debug(f"Downloading {url} to {dest}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.debug(f"Downloaded {dest}")


def _ensure_model_files(model_path: Path, voices_path: Path):
    """Download model files if they don't exist."""
    if not model_path.exists():
        _download_file(KOKORO_MODEL_URL, model_path)
    if not voices_path.exists():
        _download_file(KOKORO_VOICES_URL, voices_path)


def language_to_kokoro_language(language: Language) -> str:
    """Convert a Language enum to kokoro-onnx language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding kokoro-onnx locale code.

    """
    LANGUAGE_MAP = {
        Language.EN: "en-us",
        Language.EN_US: "en-us",
        Language.EN_GB: "en-gb",
        Language.ES: "es",
        Language.FR: "fr",
        Language.HI: "hi",
        Language.IT: "it",
        Language.JA: "ja",
        Language.PT: "pt",
        Language.ZH: "zh",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


class KokoroTTSService(TTSService):
    """Kokoro TTS service implementation.

    Provides local text-to-speech synthesis using kokoro-onnx.
    Automatically downloads model files on first use.
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
        model_path: Optional[str] = None,
        voices_path: Optional[str] = None,
        params: Optional[InputParams] = None,
        **kwargs,
    ):
        """Initialize the Kokoro TTS service.

        Args:
            voice_id: Voice identifier to use for synthesis.
            model_path: Path to the kokoro ONNX model file. Defaults to auto-downloaded file.
            voices_path: Path to the voices binary file. Defaults to auto-downloaded file.
            params: Configuration parameters for synthesis.
            **kwargs: Additional arguments passed to parent `TTSService`.

        """
        super().__init__(**kwargs)

        params = params or KokoroTTSService.InputParams()

        self._voice_id = voice_id
        self._lang_code = language_to_kokoro_language(params.language)

        model = Path(model_path) if model_path else KOKORO_CACHE_DIR / "kokoro-v1.0.onnx"
        voices = Path(voices_path) if voices_path else KOKORO_CACHE_DIR / "voices-v1.0.bin"

        _ensure_model_files(model, voices)

        self._kokoro = Kokoro(str(model), str(voices))

        self._resampler = create_stream_resampler()

    def can_generate_metrics(self) -> bool:
        """Indicate that this service supports TTFB and usage metrics."""
        return True

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Synthesize speech from text using kokoro-onnx.

        Uses the async streaming API to generate audio frames.

        Args:
            text: The text to synthesize.
            context_id: Unique identifier for this TTS context.

        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_ttfb_metrics()
            await self.start_tts_usage_metrics(text)
            yield TTSStartedFrame(context_id=context_id)

            stream = self._kokoro.create_stream(
                text, voice=self._voice_id, lang=self._lang_code, speed=1.0
            )

            async for samples, sample_rate in stream:
                await self.stop_ttfb_metrics()

                audio_int16 = (samples * 32767).astype(np.int16).tobytes()
                audio_data = await self._resampler.resample(
                    audio_int16, sample_rate, self.sample_rate
                )

                yield TTSAudioRawFrame(
                    audio=audio_data,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=context_id,
                )
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame(context_id=context_id)
