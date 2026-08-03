#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bland text-to-speech service implementation.

See https://docs.bland.ai/api-v2/post/tts.
"""

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from pipecat.frames.frames import ErrorFrame, Frame
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven, assert_given
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

# Rates Bland renders directly; asking for one of these avoids a resample.
_SAMPLE_RATES = (8000, 16000, 24000, 44100, 48000)

# Used when the pipeline rate is not one Bland renders.
_DEFAULT_SAMPLE_RATE = 48000

_DEFAULT_VOICE_ID = "f04af0e5-1a80-48a9-b02d-52f30d417cfa"


@dataclass
class BlandTTSSettings(TTSSettings):
    """Settings for BlandTTSService.

    Parameters:
        expressiveness: 0.0-1.0. Higher values produce more varied intonation.
        stability: 0.0-1.0. Higher values produce more consistent delivery.
    """

    expressiveness: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    stability: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class BlandTTSService(TTSService):
    """Bland HTTP text-to-speech service.

    Generates speech with Bland's ``/v2/tts`` endpoint. The voice sets the model;
    ``expressiveness`` and ``stability`` are calibrated for ``BTTS_V3`` and newer.
    """

    Settings = BlandTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.bland.ai/v2",
        sample_rate: int | None = None,
        aiohttp_session: aiohttp.ClientSession | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Bland TTS service.

        Args:
            api_key: Bland API key for authentication.
            base_url: Base URL for the Bland API. Defaults to
                ``https://api.bland.ai/v2``.
            sample_rate: Output sample rate in Hz. If None, uses the pipeline
                default.
            aiohttp_session: Optional shared aiohttp session. When omitted, the
                service creates and owns one.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to ``TTSService``.
        """
        default_settings = self.Settings(
            model=None,
            voice=_DEFAULT_VOICE_ID,
            language=None,
            expressiveness=None,
            stability=None,
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
            True, as the Bland service supports metrics generation.
        """
        return True

    async def start(self, frame):
        """Start the service, creating an aiohttp session if one was not provided."""
        await super().start(frame)
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._session_owner = True

    async def stop(self, frame):
        """Stop the service and release an owned aiohttp session."""
        await super().stop(frame)
        await self._close_session()

    async def cancel(self, frame):
        """Cancel the service and release an owned aiohttp session."""
        await super().cancel(frame)
        await self._close_session()

    async def cleanup(self):
        """Release Bland TTS resources at teardown."""
        await super().cleanup()
        await self._close_session()

    async def _close_session(self):
        if self._session_owner and self._session and not self._session.closed:
            await self._session.close()
        if self._session_owner:
            self._session = None

    def _bland_sample_rate(self) -> int:
        return self.sample_rate if self.sample_rate in _SAMPLE_RATES else _DEFAULT_SAMPLE_RATE

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using Bland's ``/v2/tts`` endpoint.

        Args:
            text: The text to synthesize.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._session_owner = True

        bland_sample_rate = self._bland_sample_rate()

        payload: dict[str, Any] = {
            "text": text,
            "voice": self._settings.voice,
            "audio": {"encoding": "pcm_s16le", "sample_rate": bland_sample_rate},
        }

        controls: dict[str, float] = {}
        expressiveness = assert_given(self._settings.expressiveness)
        if expressiveness is not None:
            controls["expressiveness"] = expressiveness
        stability = assert_given(self._settings.stability)
        if stability is not None:
            controls["stability"] = stability
        if controls:
            payload["controls"] = controls

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with self._session.post(
                f"{self._base_url}/tts", json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    yield ErrorFrame(error=await _error_message(response))
                    return

                await self.start_tts_usage_metrics(text)

                async for frame in self._stream_audio_frames_from_iterator(
                    response.content.iter_chunked(self.chunk_size),
                    in_sample_rate=bland_sample_rate,
                    context_id=context_id,
                ):
                    await self.stop_ttfb_metrics()
                    yield frame
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        finally:
            await self.stop_ttfb_metrics()


async def _error_message(response: aiohttp.ClientResponse) -> str:
    """Unwrap the v2 ``{"error": {"code", "message"}}`` envelope, falling back to the raw body."""
    try:
        payload = await response.json()
    except Exception:
        body = await response.text(errors="ignore")
        return f"Error getting audio (status: {response.status}, error: {body})"

    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        detail = ": ".join(str(v) for v in (error.get("code"), error.get("message")) if v)
        if detail:
            return f"Error getting audio (status: {response.status}, error: {detail})"
    return f"Error getting audio (status: {response.status}, error: {payload})"
