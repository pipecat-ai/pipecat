#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gandr text-to-speech service implementation.

This module provides integration with the Gandr TTS API for generating
speech from text input. The service sends a POST request with a JSON body
of ``model``, ``input``, ``voice``, and ``response_format``.
"""

from collections.abc import AsyncGenerator
from dataclasses import dataclass

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

GANDR_INPUT_CHAR_LIMIT = 2000
GANDR_SAMPLE_RATE = 24000


@dataclass
class GandrTTSSettings(TTSSettings):
    """Settings for GandrTTSService."""

    pass


class GandrTTSService(TTSService):
    """Gandr text-to-speech service.

    The service sends a plain HTTP POST to the Gandr speech endpoint at
    ``https://tts.gandr.ai/v1/audio/speech`` with an
    ``Authorization: Bearer`` header. It requests the ``pcm`` response
    format, which is headerless 16-bit little-endian mono audio at 24000 Hz,
    so emitted ``TTSAudioRawFrame`` objects need no decoding. Audio is
    resampled to the pipeline sample rate when it differs.
    """

    Settings = GandrTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://tts.gandr.ai/v1/audio/speech",
        aiohttp_session: aiohttp.ClientSession | None = None,
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Gandr TTS service.

        Args:
            api_key: Gandr API key for authentication (a ``gnd_`` key from
                gandr.ai).
            base_url: Gandr speech endpoint. Defaults to
                ``https://tts.gandr.ai/v1/audio/speech``.
            aiohttp_session: Optional shared aiohttp session.
            sample_rate: Output audio sample rate in Hz. If None, uses the
                pipeline default. Gandr PCM audio is produced at 24000 Hz and
                is resampled when the pipeline rate differs.
            settings: Runtime-updatable settings.
            **kwargs: Additional keyword arguments passed to ``TTSService``.
        """
        default_settings = self.Settings(
            model="gandr-1",
            voice="gandr-mia",
            language=None,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url
        self._session = aiohttp_session
        self._session_owner = aiohttp_session is None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as the Gandr TTS service supports metrics generation.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the Gandr TTS service.

        Args:
            frame: The start frame containing initialization parameters.
        """
        await super().start(frame)
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._session_owner = True

    async def stop(self, frame: EndFrame):
        """Stop the Gandr TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._close_session()

    async def cancel(self, frame: CancelFrame):
        """Cancel the Gandr TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._close_session()

    async def cleanup(self):
        """Release Gandr TTS resources at teardown."""
        await super().cleanup()
        await self._close_session()

    async def _close_session(self):
        if self._session_owner and self._session and not self._session.closed:
            await self._session.close()
        if self._session_owner:
            self._session = None

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using the Gandr TTS API.

        Args:
            text: The text to synthesize into speech (max 2000 characters).
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        if len(text) > GANDR_INPUT_CHAR_LIMIT:
            logger.warning(
                f"Text too long for Gandr TTS (max {GANDR_INPUT_CHAR_LIMIT} chars), truncating"
            )
            text = text[:GANDR_INPUT_CHAR_LIMIT]

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._session_owner = True

        payload = {
            "model": self._settings.model,
            "input": text,
            "voice": self._settings.voice,
            "response_format": "pcm",
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with self._session.post(
                self._base_url, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    error = await response.text(errors="ignore")
                    yield ErrorFrame(
                        error=f"Error getting audio (status: {response.status}, error: {error})"
                    )
                    return

                await self.start_tts_usage_metrics(text)

                # Gandr PCM audio is headerless 16-bit mono at 24000 Hz, so
                # the iterator helper keeps frames sample-aligned and
                # resamples to the pipeline sample rate when needed.
                async for frame in self._stream_audio_frames_from_iterator(
                    response.content.iter_chunked(self.chunk_size),
                    in_sample_rate=GANDR_SAMPLE_RATE,
                    context_id=context_id,
                ):
                    await self.stop_ttfb_metrics()
                    yield frame
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
