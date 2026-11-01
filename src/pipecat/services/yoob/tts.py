#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Yoob text-to-speech (Overtone) for Pipecat.

Each sentence Pipecat hands over is rendered with one streaming request to ``POST /v1/audio/speech``
(raw 48 kHz PCM), and the audio is resampled to the pipeline's output rate as it arrives.
"""

import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

import aiohttp
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven

YOOB_BASE_URL = "https://tts-api.yoob.com/v1"
YOOB_TTS_SAMPLE_RATE = 48000


@dataclass
class YoobTTSSettings(TTSSettings):
    """Settings for YoobTTSService.

    Parameters:
        quality: ``best`` (default) or ``fast`` (quicker first sound).
        instructions: Free-text delivery, e.g. ``"warm voice, softly spoken"``.
        speed: 0.5 to 2.0.
        seed: Makes renders repeatable.
    """

    quality: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    instructions: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    speed: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    seed: int | str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class YoobTTSService(TTSService):
    """Yoob expressive speech.

    Text may carry inline tags such as ``[laughing]``, ``[sigh]`` or ``[whisper] ... [/whisper]``.
    """

    Settings = YoobTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        aiohttp_session: aiohttp.ClientSession | None = None,
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Yoob TTS service.

        Args:
            api_key: Defaults to ``YOOB_API_KEY``. Keys come from https://app.yoob.com.
            base_url: Defaults to ``YOOB_BASE_URL`` or https://tts-api.yoob.com/v1.
            aiohttp_session: A session to reuse; one is created (and closed) otherwise.
            sample_rate: Output rate. Defaults to the pipeline's; audio is resampled from 48 kHz.
            settings: ``YoobTTSService.Settings(voice=..., quality=..., instructions=...)``.
            **kwargs: Passed to ``TTSService``. ``stop_frame_timeout_s`` defaults to 10 s here (3 s in Pipecat) so
                a slow first sentence is not taken for a context with no audio.
        """
        kwargs.setdefault("stop_frame_timeout_s", 10.0)
        default_settings = self.Settings(
            model="yoob-tts-1",
            voice="w1243",
            language=None,
            quality="best",
            instructions=None,
            speed=None,
            seed=None,
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
        self._api_key = api_key or os.environ.get("YOOB_API_KEY") or ""
        if not self._api_key:
            raise ValueError("Yoob API key is required: pass api_key or set YOOB_API_KEY")
        self._base_url = (base_url or os.environ.get("YOOB_BASE_URL") or YOOB_BASE_URL).rstrip("/")
        self._session = aiohttp_session
        self._own_session = aiohttp_session is None

    def can_generate_metrics(self) -> bool:
        """Yoob TTS reports TTFB and usage metrics."""
        return True

    def _body(self, text: str) -> dict:
        s = self._settings
        body = {
            "model": s.model,
            "voice": s.voice,
            "input": text,
            "response_format": "pcm",
            "stream": True,
            "quality": s.quality,
            "instructions": s.instructions,
            "speed": s.speed,
            "seed": s.seed,
        }
        return {k: v for k, v in body.items() if v is not None}

    async def cleanup(self):
        """Close the HTTP session if this service created it."""
        await super().cleanup()
        if self._own_session and self._session is not None:
            await self._session.close()
            self._session = None

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Render one piece of text and stream its audio.

        Args:
            text: The text to speak.
            context_id: The TTS context the audio belongs to.

        Yields:
            Frame: audio frames, or an ErrorFrame.
        """
        if self._session is None:
            self._session = aiohttp.ClientSession()
        try:
            async with self._session.post(
                f"{self._base_url}/audio/speech",
                json=self._body(text),
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=30),
            ) as r:
                if r.status != 200:
                    error = await r.text()
                    logger.error(f"{self} error getting audio (status: {r.status}, error: {error})")
                    yield ErrorFrame(error=f"Yoob TTS error (status: {r.status}, error: {error})")
                    return

                await self.start_tts_usage_metrics(text)
                async for frame in self._stream_audio_frames_from_iterator(
                    r.content.iter_chunked(self.chunk_size),
                    in_sample_rate=YOOB_TTS_SAMPLE_RATE,
                    context_id=context_id,
                ):
                    await self.stop_ttfb_metrics()
                    yield frame
        except aiohttp.ClientError as e:
            yield ErrorFrame(error=f"Yoob TTS request failed: {e}")
