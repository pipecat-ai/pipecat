#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from typing import AsyncGenerator

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.ai_services import TTSService

# This assumes a running TTS service running: https://github.com/rhasspy/piper/blob/master/src/python_run/README_http.md


class PiperTTSService(TTSService):
    """Piper TTS service implementation.

    Provides integration with Piper's TTS server.
    """

    def __init__(
        self,
        *,
        base_url: str,
        aiohttp_session: aiohttp.ClientSession | None = None,
        sample_rate: int = 24000,
        **kwargs,
    ):
        """Initialize the PiperTTSService class instance.

        Args:
            base_url (str): Base URL of the Piper TTS server (should not end with a slash).
            aiohttp_session (aiohttp.ClientSession, optional): Optional aiohttp session to use for requests. Defaults to None.
            sample_rate (int, optional): Sample rate in Hz. Defaults to 24000.
            **kwargs (dict): Additional keyword arguments.
        """
        super().__init__(sample_rate=sample_rate, **kwargs)
        if not aiohttp_session:
            aiohttp_session = aiohttp.ClientSession()

        if base_url.endswith("/"):
            logger.warning("Base URL ends with a slash, this is not allowed.")
            base_url = base_url[:-1]

        self._settings = {"base_url": base_url}
        self.set_voice("voice_id")
        self._aiohttp_session = aiohttp_session

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"Generating TTS: [{text}]")

        url = self._settings["base_url"] + "/?text=" + text.replace(".", "").replace("*", "")

        await self.start_ttfb_metrics()

        async with self._aiohttp_session.get(url) as r:
            if r.status != 200:
                text = await r.text()
                logger.error(f"{self} error getting audio (status: {r.status}, error: {text})")
                yield ErrorFrame(f"Error getting audio (status: {r.status}, error: {text})")
                return

            await self.start_tts_usage_metrics(text)

            yield TTSStartedFrame()

            buffer = bytearray()
            async for chunk in r.content.iter_chunked(1024):
                if len(chunk) > 0:
                    await self.stop_ttfb_metrics()
                    # Append new chunk to the buffer.
                    buffer.extend(chunk)

                    # Check if buffer has enough data for processing.
                    while (
                        len(buffer) >= 48000
                    ):  # Assuming at least 0.5 seconds of audio data at 24000 Hz
                        # Process the buffer up to a safe size for resampling.
                        process_data = buffer[:48000]
                        # Remove processed data from buffer.
                        buffer = buffer[48000:]

                        frame = TTSAudioRawFrame(process_data, self._sample_rate, 1)
                        yield frame

            # Process any remaining data in the buffer.
            if len(buffer) > 0:
                frame = TTSAudioRawFrame(buffer, self._sample_rate, 1)
                yield frame

            yield TTSStoppedFrame()
