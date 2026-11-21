#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Zoom text-to-speech service.

:class:`ZoomAIHttpTTSService` synthesizes speech with the Zoom AI Services Text-to-Speech
API in Fast mode: each sentence is one HTTPS request, and the audio streams back in
the response, so playback starts before synthesis finishes.

For voices, limits and errors, see https://developers.zoom.us/docs/ai-services/tts/
and https://www.zoom.ai/
"""

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from loguru import logger

from pipecat.frames.frames import Frame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.services.zoom_ai.utils import check_url, new_tracking_id, request_headers
from pipecat.utils.errors import ErrorCategory, classify_http_status_code
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven, is_given

# Zoom TTS returns signed 16-bit little-endian mono PCM at 44.1 kHz only.
ZOOM_TTS_SAMPLE_RATE = 44100
# The longest input one request accepts, in Unicode code points.
MAX_INPUT_CHARS = 4000


def _split_text(text: str, limit: int = MAX_INPUT_CHARS) -> list[str]:
    """Split ``text`` into pieces of at most ``limit`` code points, at whitespace if possible."""
    pieces = []
    while len(text) > limit:
        cut = text.rfind(" ", 0, limit + 1)
        if cut <= 0:
            cut = limit
        pieces.append(text[:cut])
        text = text[cut:].lstrip()
    if text:
        pieces.append(text)
    return pieces


async def _describe_error(response: aiohttp.ClientResponse) -> str:
    """Describe an error response, with Zoom's reason and message when its body has them."""
    body = await response.text()
    try:
        error = json.loads(body)
        parts = [error.get("reason"), error.get("message")]
    except (TypeError, ValueError, AttributeError):  # no body, or not a JSON object
        parts = [body.strip()]
    return " ".join([f"HTTP {response.status}", *(str(p) for p in parts if p)])


@dataclass
class ZoomAITTSSettings(TTSSettings):
    """Settings for :class:`ZoomAIHttpTTSService`.

    ``voice`` is a Zoom TTS voice ID, such as ``jenna`` (the default) or ``ava``; see
    https://developers.zoom.us/docs/ai-services/tts/#language-and-voices for the list.
    The API has no language or model field, so ``language`` and ``model`` are
    ``None``.

    Parameters:
        speed: Speaking rate multiplier, from 0.5 to 2.0. ``None`` uses the voice's
            normal rate.
        volume: Volume multiplier, greater than 0 and up to 10. ``None`` uses the
            voice's normal volume.
    """

    speed: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    volume: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class ZoomAIHttpTTSService(TTSService):
    """Zoom AI Services text-to-speech service.

    Sends each sentence to the Zoom TTS API in its own request and streams the
    audio into the pipeline as it arrives. Zoom TTS returns 44.1 kHz audio, which
    is resampled to the pipeline's output rate. An interruption closes the request,
    which stops synthesis on the server.

    Example::

        async with aiohttp.ClientSession() as session:
            tts = ZoomAIHttpTTSService(
                api_key=os.getenv("ZOOM_AI_API_KEY"),
                aiohttp_session=session,
                settings=ZoomAIHttpTTSService.Settings(voice="ava"),
            )
    """

    Settings = ZoomAITTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        base_url: str = "https://api.zoom.us/v2/aiservices/tts/synthesize",
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Zoom TTS service.

        Args:
            api_key: Zoom AI Services API key or JWT, from https://www.zoom.ai/
            aiohttp_session: Shared aiohttp session for HTTP requests.
            base_url: Text-to-Speech synthesize endpoint URL. A plaintext ``http://``
                URL to a host other than localhost logs a warning.
            sample_rate: Output sample rate in Hz. If None, uses the pipeline's
                output rate.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to the parent TTSService.
        """
        default_settings = self.Settings(
            model=None, voice="jenna", language=None, speed=None, volume=None
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
        self._session = aiohttp_session
        self._base_url = check_url(base_url, service="Zoom TTS", data="text")

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as this service supports metrics.
        """
        return True

    def _payload(self, text: str) -> dict[str, Any]:
        payload: dict[str, Any] = {"input": text, "response_format": "pcm_44100"}
        for name in ("voice", "speed", "volume"):
            value = getattr(self._settings, name)
            if is_given(value) and value is not None:
                payload[name] = value
        return payload

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Synthesize ``text`` with Zoom TTS and stream its audio.

        Text longer than one request accepts is sent in several requests, split at
        whitespace.

        Args:
            text: The text to synthesize.
            context_id: The context ID for the audio frames.

        Yields:
            Frame: Audio frames with the synthesized speech.
        """
        if not text.strip():
            return
        try:
            for piece in _split_text(text):
                async for frame in self._synthesize(piece, context_id):
                    yield frame
        finally:
            await self.stop_all_metrics()

    async def _synthesize(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        tracking_id = new_tracking_id()
        headers = {
            **request_headers(self._api_key, tracking_id),
            "Content-Type": "application/json",
        }
        # Non-ASCII text stays UTF-8 rather than \u escapes, which keeps a request
        # of the longest input within the API's 32 KiB body limit.
        body = json.dumps(self._payload(text), ensure_ascii=False).encode("utf-8")
        try:
            # Leaving this block mid-stream, as an interruption or a failed stream does,
            # closes the connection, which stops synthesis on the server.
            async with self._session.post(self._base_url, data=body, headers=headers) as response:
                if response.status != 200:
                    detail = await _describe_error(response)
                    await self.push_error(
                        error_msg=f"Zoom TTS error: {detail} (x-zm-trackingid {tracking_id})",
                        category=classify_http_status_code(response.status),
                    )
                    return
                logger.debug(f"{self}: synthesizing (x-zm-trackingid {tracking_id})")
                await self.start_tts_usage_metrics(text)
                async for frame in self._stream_audio_frames_from_iterator(
                    response.content.iter_chunked(self.chunk_size),
                    in_sample_rate=ZOOM_TTS_SAMPLE_RATE,
                    context_id=context_id,
                ):
                    await self.stop_ttfb_metrics()
                    yield frame
        except aiohttp.ClientPayloadError as e:
            # Once audio is streaming the status can't change, so a synthesis failure
            # ends the response early instead.
            await self.push_error(
                error_msg=f"Zoom TTS stream ended early (x-zm-trackingid {tracking_id}): {e}",
                exception=e,
                category=ErrorCategory.SERVER,
            )
        except Exception as e:
            await self.push_error(
                error_msg=f"Zoom TTS request failed (x-zm-trackingid {tracking_id}): {e}",
                exception=e,
            )
