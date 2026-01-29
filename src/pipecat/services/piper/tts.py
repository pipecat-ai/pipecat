#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Piper TTS service implementation."""

import asyncio
from pathlib import Path
from typing import AsyncGenerator, AsyncIterator, Optional

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from piper import PiperVoice
    from piper.download_voices import download_voice
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Piper, you need to `pip install pipecat-ai[piper]`.")
    raise Exception(f"Missing module: {e}")


class PiperTTSService(TTSService):
    """Piper TTS service implementation.

    Provides local text-to-speech synthesis using Piper voice models. Automatically
    downloads voice models if not already present and resamples audio output to
    match the configured sample rate.
    """

    def __init__(
        self,
        *,
        voice_id: str,
        download_dir: Optional[Path] = None,
        force_redownload: bool = False,
        use_cuda: bool = False,
        **kwargs,
    ):
        """Initialize the Piper TTS service.

        Args:
            voice_id: Piper voice model identifier (e.g. `en_US-ryan-high`).
            download_dir: Directory for storing voice model files. Defaults to
                the current working directory.
            force_redownload: Re-download the voice model even if it already exists.
            use_cuda: Use CUDA for GPU-accelerated inference.
            **kwargs: Additional arguments passed to the parent `TTSService`.
        """
        super().__init__(**kwargs)

        self._voice_id = voice_id

        download_dir = download_dir or Path.cwd()

        model_file = f"{voice_id}.onnx"
        model_path = Path(download_dir) / model_file

        if not model_path.exists():
            logger.debug(f"Downloading Piper '{voice_id}' model")
            download_voice(voice_id, download_dir, force_redownload=force_redownload)

        logger.debug(f"Loading Piper '{voice_id}' model from {model_path}")

        self._voice = PiperVoice.load(model_path, use_cuda=use_cuda)

        logger.debug(f"Loaded Piper '{voice_id}' model")

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Piper service supports metrics generation.
        """
        return True

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Piper.

        Args:
            text: The text to convert to speech.

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
            await self.start_ttfb_metrics()

            await self.start_tts_usage_metrics(text)

            yield TTSStartedFrame()

            async for frame in self._stream_audio_frames_from_iterator(
                async_iterator(self._voice.synthesize(text)),
                in_sample_rate=self._voice.config.sample_rate,
            ):
                await self.stop_ttfb_metrics()
                yield frame
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        finally:
            logger.debug(f"{self}: Finished TTS [{text}]")
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()


# This assumes a running TTS service running:
# https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_HTTP.md
#
# Usage:
#
#  $ uv pip install "piper-tts[http]"
#  $ uv run python -m piper.http_server -m en_US-ryan-high
#
class PiperHttpTTSService(TTSService):
    """Piper HTTP TTS service implementation.

    Provides integration with Piper's HTTP TTS server for text-to-speech
    synthesis. Supports streaming audio generation with configurable sample
    rates and automatic WAV header removal.
    """

    def __init__(
        self,
        *,
        base_url: str,
        aiohttp_session: aiohttp.ClientSession,
        voice_id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the Piper TTS service.

        Args:
            base_url: Base URL for the Piper TTS HTTP server.
            aiohttp_session: aiohttp ClientSession for making HTTP requests.
            voice_id: Piper voice model identifier (e.g. `en_US-ryan-high`).
            **kwargs: Additional arguments passed to the parent TTSService.
        """
        super().__init__(**kwargs)

        if base_url.endswith("/"):
            logger.warning("Base URL ends with a slash, this is not allowed.")
            base_url = base_url[:-1]

        self._base_url = base_url
        self._session = aiohttp_session
        self._model_id = voice_id

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as Piper service supports metrics generation.
        """
        return True

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Piper's HTTP API.

        Args:
            text: The text to convert to speech.

        Yields:
            Frame: Audio frames containing the synthesized speech and status frames.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")
        headers = {
            "Content-Type": "application/json",
        }
        try:
            await self.start_ttfb_metrics()

            data = {
                "text": text,
                "voice": self._model_id,
            }

            async with self._session.post(self._base_url, json=data, headers=headers) as response:
                if response.status != 200:
                    error = await response.text()
                    yield ErrorFrame(
                        error=f"Error getting audio (status: {response.status}, error: {error})"
                    )
                    return

                await self.start_tts_usage_metrics(text)

                yield TTSStartedFrame()

                CHUNK_SIZE = self.chunk_size

                async for frame in self._stream_audio_frames_from_iterator(
                    response.content.iter_chunked(CHUNK_SIZE), strip_wav_header=True
                ):
                    await self.stop_ttfb_metrics()
                    yield frame
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        finally:
            logger.debug(f"{self}: Finished TTS [{text}]")
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()
