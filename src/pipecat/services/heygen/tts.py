#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""HeyGen Voice text-to-speech service implementation.

Contains :class:`HeyGenHttpTTSService`, which streams raw PCM audio from HeyGen's
``POST /v3/models/audio/tts/stream`` endpoint.

See https://developers.heygen.com/docs/voices/heygen-voice-speech.
"""

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import aiohttp
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import assert_given

# Nadine, a HeyGen public voice. Works with every HeyGen Voice model, no clone needed.
HEYGEN_DEFAULT_VOICE_ID = "83548de556df48ba8c09c42a57c51d85"

# PCM sample rates HeyGen can synthesize.
HEYGEN_PCM_SAMPLE_RATES = (8000, 16000, 22050, 24000, 44100, 48000)

# Rate requested when the pipeline runs at a rate HeyGen does not offer; the
# audio is then resampled to the pipeline rate.
HEYGEN_FALLBACK_SAMPLE_RATE = 24000


def language_to_heygen_language(language: Language) -> str:
    """Convert a Language enum to a HeyGen Voice language code.

    HeyGen Voice takes lowercase ISO 639-1 codes and has no regional accents, so
    regional variants (e.g. ``en-US``) resolve to their base code (``en``).

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding HeyGen language code.
    """
    LANGUAGE_MAP = {
        Language.AR: "ar",
        Language.BE: "be",
        Language.BG: "bg",
        Language.CA: "ca",
        Language.CS: "cs",
        Language.DA: "da",
        Language.DE: "de",
        Language.EL: "el",
        Language.EN: "en",
        Language.ES: "es",
        Language.FA: "fa",
        Language.FI: "fi",
        Language.FR: "fr",
        Language.HE: "he",
        Language.HI: "hi",
        Language.HR: "hr",
        Language.HU: "hu",
        Language.ID: "id",
        Language.IT: "it",
        Language.JA: "ja",
        Language.KO: "ko",
        Language.MK: "mk",
        Language.MS: "ms",
        Language.NL: "nl",
        Language.PL: "pl",
        Language.PT: "pt",
        Language.RO: "ro",
        Language.RU: "ru",
        Language.SK: "sk",
        Language.SL: "sl",
        Language.SR: "sr",
        Language.SV: "sv",
        Language.TA: "ta",
        Language.TH: "th",
        Language.TL: "tl",
        Language.TR: "tr",
        Language.UK: "uk",
        Language.VI: "vi",
        Language.ZH: "zh",
    }
    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


@dataclass
class HeyGenTTSSettings(TTSSettings):
    """Settings for HeyGenHttpTTSService.

    Parameters:
        model: HeyGen Voice model: ``heygen-voice-1``, ``heygen-voice-1-turbo``,
            or ``heygen-voice-1-turbo-free``.
        voice: HeyGen voice ID: a public voice, or an instant or professional
            clone in your workspace.
        language: Language of the text. If ``None``, HeyGen detects it.
    """

    pass


class HeyGenHttpTTSService(TTSService):
    """HeyGen Voice HTTP text-to-speech service.

    Sends each sentence to HeyGen's streaming speech endpoint and yields the raw
    PCM audio as it arrives.

    Defaults to ``heygen-voice-1-turbo``, the lowest-latency model, and the
    public voice Nadine.
    """

    Settings = HeyGenTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        base_url: str = "https://api.heygen.com",
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the HeyGen TTS service.

        Args:
            api_key: HeyGen API key.
            aiohttp_session: aiohttp ClientSession for HTTP requests.
            base_url: HeyGen API base URL.
            sample_rate: Output audio sample rate in Hz. If None, uses the
                pipeline default.
            settings: Runtime-updatable settings.
            **kwargs: Additional keyword arguments passed to ``TTSService``.
        """
        default_settings = self.Settings(
            model="heygen-voice-1-turbo",
            voice=HEYGEN_DEFAULT_VOICE_ID,
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
        self._session = aiohttp_session
        self._url = f"{base_url.rstrip('/')}/v3/models/audio/tts/stream"

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as HeyGen TTS supports metrics generation.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to a HeyGen Voice language code.

        Args:
            language: The language to convert.

        Returns:
            The HeyGen language code.
        """
        return language_to_heygen_language(language)

    def _synthesis_sample_rate(self) -> int:
        """Return the PCM rate to request from HeyGen for the pipeline's rate."""
        if self.sample_rate in HEYGEN_PCM_SAMPLE_RATES:
            return self.sample_rate
        return HEYGEN_FALLBACK_SAMPLE_RATE

    def _build_payload(self, text: str, synthesis_sample_rate: int) -> dict[str, Any]:
        """Build the request body for one piece of text.

        Args:
            text: Text to synthesize.
            synthesis_sample_rate: PCM sample rate to request.

        Returns:
            The JSON request body.
        """
        payload: dict[str, Any] = {
            "text": text,
            "voice_id": assert_given(self._settings.voice),
            "model": assert_given(self._settings.model),
            "output_format": {
                "encoding": "pcm",
                "sample_rate_hertz": synthesis_sample_rate,
            },
        }
        language = assert_given(self._settings.language)
        if language:
            payload["language"] = str(language)
        return payload

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using HeyGen's streaming speech endpoint.

        Args:
            text: Text to synthesize.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: Audio frames containing the synthesized speech.
        """
        synthesis_sample_rate = self._synthesis_sample_rate()
        headers = {"x-api-key": self._api_key, "Content-Type": "application/json"}

        try:
            async with self._session.post(
                self._url,
                json=self._build_payload(text, synthesis_sample_rate),
                headers=headers,
            ) as response:
                request_id = response.headers.get("X-Request-Id")
                if response.status != 200:
                    error = await response.text(errors="ignore")
                    yield ErrorFrame(
                        error=f"HeyGen TTS error (status: {response.status}, "
                        f"request_id: {request_id}, error: {error})"
                    )
                    return

                logger.debug(f"{self}: HeyGen request {request_id}")
                await self.start_tts_usage_metrics(text)

                # Raw PCM16 arrives at arbitrary byte boundaries; the helper keeps
                # frames sample-aligned and resamples when the rates differ.
                async for frame in self._stream_audio_frames_from_iterator(
                    response.content.iter_chunked(self.chunk_size),
                    in_sample_rate=synthesis_sample_rate,
                    context_id=context_id,
                ):
                    await self.stop_ttfb_metrics()
                    yield frame
        except Exception as e:
            yield ErrorFrame(error=f"HeyGen TTS error: {e}")
