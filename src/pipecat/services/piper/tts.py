#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Piper TTS service implementation."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSStoppedFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from piper import PiperVoice
    from piper.download_voices import download_voice
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Piper, you need to `pip install pipecat-ai[piper]`.")
    raise Exception(f"Missing module: {e}")


@dataclass
class PiperTTSSettings(TTSSettings):
    """Settings for PiperTTSService."""

    pass


class PiperTTSService(TTSService):
    """Piper TTS service implementation.

    Provides local text-to-speech synthesis using Piper voice models. Automatically
    downloads voice models if not already present and resamples audio output to
    match the configured sample rate.
    """

    Settings = PiperTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        voice_id: str | None = None,
        download_dir: Path | None = None,
        force_redownload: bool = False,
        use_cuda: bool = False,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Piper TTS service.

        Args:
            voice_id: Piper voice model identifier (e.g. `en_US-ryan-high`).

                .. deprecated:: 0.0.105
                    Use ``settings=PiperTTSService.Settings(voice=...)`` instead.

            download_dir: Directory for storing voice model files. Defaults to
                the current working directory.
            force_redownload: Re-download the voice model even if it already exists.
            use_cuda: Use CUDA for GPU-accelerated inference.
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional arguments passed to the parent `TTSService`.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(model=None, voice=None, language=None)

        # 2. Apply direct init arg overrides (deprecated)
        if voice_id is not None:
            self._warn_init_param_moved_to_settings("voice_id", "voice")
            default_settings.voice = voice_id

        # 3. (No step 3, as there's no params object to apply)

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        download_dir = download_dir or Path.cwd()

        _voice = self._settings.voice
        model_file = f"{_voice}.onnx"
        model_path_resolved = Path(download_dir) / model_file

        if not model_path_resolved.exists():
            logger.debug(f"Downloading Piper '{_voice}' model")
            download_voice(_voice, download_dir, force_redownload=force_redownload)

        logger.debug(f"Loading Piper '{_voice}' model from {model_path_resolved}")

        self._voice = PiperVoice.load(model_path_resolved, use_cuda=use_cuda)

        logger.debug(f"Loaded Piper '{_voice}' model")

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Piper service supports metrics generation.
        """
        return True

    async def _update_settings(self, delta: Settings) -> dict[str, Any]:
        """Apply a settings delta.

        Settings are stored but not applied to the active connection.
        """
        changed = await super()._update_settings(delta)
        if not changed:
            return changed
        # TODO: voice changes would require re-downloading and loading the model.
        self._warn_unhandled_updated_settings(changed)
        return changed

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Piper.

        Args:
            text: The text to convert to speech.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: Audio frames containing the synthesized speech and status frames.
        """

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
                yield item.audio_int16_bytes

        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_tts_usage_metrics(text)

            async for frame in self._stream_audio_frames_from_iterator(
                async_iterator(self._voice.synthesize(text)),
                in_sample_rate=self._voice.config.sample_rate,
                context_id=context_id,
            ):
                await self.stop_ttfb_metrics()
                yield frame
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        finally:
            logger.debug(f"{self}: Finished TTS [{text}]")
            await self.stop_ttfb_metrics()


# This assumes a running TTS service running:
# https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_HTTP.md
#
# Usage:
#
#  $ uv pip install "piper-tts[http]"
#  $ uv run python -m piper.http_server -m en_US-ryan-high
#
@dataclass
class PiperHttpTTSSettings(TTSSettings):
    """Settings for PiperHttpTTSService."""

    pass


class PiperHttpTTSService(TTSService):
    """Piper HTTP TTS service implementation.

    Provides integration with Piper's HTTP TTS server for text-to-speech
    synthesis. Supports streaming audio generation with configurable sample
    rates and automatic WAV header removal.
    """

    Settings = PiperHttpTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        base_url: str,
        aiohttp_session: aiohttp.ClientSession,
        voice_id: str | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Piper TTS service.

        Args:
            base_url: Base URL for the Piper TTS HTTP server.
            aiohttp_session: aiohttp ClientSession for making HTTP requests.
            voice_id: Piper voice model identifier (e.g. `en_US-ryan-high`).

                .. deprecated:: 0.0.105
                    Use ``settings=PiperHttpTTSService.Settings(voice=...)`` instead.

            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            **kwargs: Additional arguments passed to the parent TTSService.
        """
        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(model=None, voice=None, language=None)

        # 2. Apply direct init arg overrides (deprecated)
        if voice_id is not None:
            self._warn_init_param_moved_to_settings("voice_id", "voice")
            default_settings.voice = voice_id

        # 3. (No step 3, as there's no params object to apply)

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        if base_url.endswith("/"):
            logger.warning("Base URL ends with a slash, this is not allowed.")
            base_url = base_url[:-1]

        self._base_url = base_url
        self._session = aiohttp_session

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Piper service supports metrics generation.
        """
        return True

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Piper's HTTP API.

        Args:
            text: The text to convert to speech.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: Audio frames containing the synthesized speech and status frames.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")
        headers = {
            "Content-Type": "application/json",
        }
        try:
            data = {
                "text": text,
                "voice": self._settings.voice,
            }

            async with self._session.post(self._base_url, json=data, headers=headers) as response:
                if response.status != 200:
                    error = await response.text()
                    yield ErrorFrame(
                        error=f"Error getting audio (status: {response.status}, error: {error})"
                    )
                    yield TTSStoppedFrame(context_id=context_id)
                    return

                await self.start_tts_usage_metrics(text)

                CHUNK_SIZE = self.chunk_size

                async for frame in self._stream_audio_frames_from_iterator(
                    response.content.iter_chunked(CHUNK_SIZE),
                    strip_wav_header=True,
                    context_id=context_id,
                ):
                    await self.stop_ttfb_metrics()
                    yield frame
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        finally:
            await self.stop_ttfb_metrics()
