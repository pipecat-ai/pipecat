#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Kokoro TTS service implementation using kokoro-onnx."""

import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from pydantic import BaseModel

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
)
from pipecat.services.settings import TTSSettings
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


@dataclass
class KokoroTTSSettings(TTSSettings):
    """Settings for KokoroTTSService."""

    pass


class KokoroTTSService(TTSService):
    """Kokoro TTS service implementation.

    Provides local text-to-speech synthesis using kokoro-onnx.
    Automatically downloads model files on first use.
    """

    Settings = KokoroTTSSettings
    _settings: Settings

    class InputParams(BaseModel):
        """Input parameters for Kokoro TTS configuration.

        .. deprecated:: 0.0.105
            Use ``KokoroTTSService.Settings`` directly via the ``settings`` parameter instead.

        Parameters:
            language: Language to use for synthesis.
        """

        language: Language = Language.EN

    def __init__(
        self,
        *,
        voice_id: str | None = None,
        model_path: str | None = None,
        voices_path: str | None = None,
        params: InputParams | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Kokoro TTS service.

        Args:
            voice_id: Voice identifier to use for synthesis.

                .. deprecated:: 0.0.105
                    Use ``settings=KokoroTTSService.Settings(voice=...)`` instead.

            model_path: Path to the kokoro ONNX model file. Defaults to auto-downloaded file.
            voices_path: Path to the voices binary file. Defaults to auto-downloaded file.
            params: Configuration parameters for synthesis.

                .. deprecated:: 0.0.105
                    Use ``settings=KokoroTTSService.Settings(...)`` instead.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional arguments passed to parent `TTSService`.

        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model=None,
            voice=None,
            language=Language.EN,
        )

        # 2. Apply direct init arg overrides (deprecated)
        if voice_id is not None:
            self._warn_init_param_moved_to_settings("voice_id", "voice")
            default_settings.voice = voice_id

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                default_settings.language = params.language

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        model_file = Path(model_path) if model_path else KOKORO_CACHE_DIR / "kokoro-v1.0.onnx"
        voices = Path(voices_path) if voices_path else KOKORO_CACHE_DIR / "voices-v1.0.bin"

        _ensure_model_files(model_file, voices)

        self._kokoro = Kokoro(str(model_file), str(voices))

        self._resampler = create_stream_resampler()

    def can_generate_metrics(self) -> bool:
        """Indicate that this service supports TTFB and usage metrics."""
        return True

    def language_to_service_language(self, language: Language) -> str:
        """Convert a Language enum to kokoro-onnx language format.

        Args:
            language: The language to convert.

        Returns:
            The kokoro-onnx language code.
        """
        return language_to_kokoro_language(language)

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
            await self.start_tts_usage_metrics(text)

            stream = self._kokoro.create_stream(
                text, voice=self._settings.voice, lang=self._settings.language, speed=1.0
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
