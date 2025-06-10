#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts


# This assumes a running TTS service running: https://github.com/rhasspy/piper/blob/master/src/python_run/README_http.md
class PiperTTSService(TTSService):
    """Piper TTS service implementation.

    Provides integration with Piper's TTS server.

    Args:
        base_url: API base URL
        aiohttp_session: aiohttp ClientSession
        sample_rate: Output sample rate
    """

    def __init__(
        self,
        *,
        base_url: str,
        aiohttp_session: aiohttp.ClientSession,
        # When using Piper, the sample rate of the generated audio depends on the
        # voice model being used.
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        if base_url.endswith("/"):
            logger.warning("Base URL ends with a slash, this is not allowed.")
            base_url = base_url[:-1]

        self._base_url = base_url
        self._session = aiohttp_session
        self._settings = {"base_url": base_url}

    def can_generate_metrics(self) -> bool:
        return True

    @traced_tts
    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Piper API.

        Args:
            text: The text to convert to speech

        Yields:
            Frames containing audio data and status information
        """
        logger.debug(f"{self}: Generating TTS [{text}]")
        headers = {
            "Content-Type": "text/plain",
        }
        try:
            await self.start_ttfb_metrics()

            async with self._session.post(self._base_url, data=text, headers=headers) as response:
                if response.status != 200:
                    error = await response.text()
                    logger.error(
                        f"{self} error getting audio (status: {response.status}, error: {error})"
                    )
                    yield ErrorFrame(
                        f"Error getting audio (status: {response.status}, error: {error})"
                    )
                    return

                await self.start_tts_usage_metrics(text)

                CHUNK_SIZE = self.chunk_size

                yield TTSStartedFrame()
                async for chunk in response.content.iter_chunked(CHUNK_SIZE):
                    # remove wav header if present
                    if chunk.startswith(b"RIFF"):
                        chunk = chunk[44:]
                    if len(chunk) > 0:
                        await self.stop_ttfb_metrics()
                        yield TTSAudioRawFrame(chunk, self.sample_rate, 1)
        except Exception as e:
            logger.error(f"Error in run_tts: {e}")
            yield ErrorFrame(error=str(e))
        finally:
            logger.debug(f"{self}: Finished TTS [{text}]")
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()
