#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""ConvoZen Ragini text-to-speech service implementation."""

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import ClassVar

import aiohttp
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven

CONVOZEN_BASE_URL = "https://voice.convozen.ai/sdk-models/developer-api/api"

# Native output rate per model. Ragini resamples on request, but asking for the
# model's own rate avoids a resampling pass on the server.
MODEL_SAMPLE_RATES: dict[str, int] = {
    "ragini-v1": 24000,
    "ragini-lite": 22050,
}

LANGUAGES: set[str] = {"bn", "en", "gu", "hi", "kn", "ml", "mr", "ta", "te"}

_MAX_ERROR_BODY = 512


def language_to_convozen_language(language: Language) -> str | None:
    """Convert a Language enum to a ConvoZen language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        The two-letter code Ragini accepts, or None if unsupported.
    """
    base = str(language.value).split("-")[0].lower()
    return base if base in LANGUAGES else None


@dataclass
class ConvozenHttpTTSSettings(TTSSettings):
    """Settings for ConvozenHttpTTSService.

    Parameters:
        speed: Speaking rate multiplier; 1.0 is the natural pace.
    """

    speed: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class ConvozenHttpTTSService(TTSService):
    """ConvoZen Ragini text-to-speech service.

    Synthesizes speech through Ragini's REST API, covering nine Indian
    languages. Audio streams back as a chunked WAV as it is generated, which
    keeps time-to-first-audio down.

    Example::

        tts = ConvozenHttpTTSService(
            api_key=os.getenv("CONVOZEN_API_KEY"),
            aiohttp_session=session,
            settings=ConvozenHttpTTSService.Settings(
                voice="roohi",
                language=Language.HI,
            ),
        )
    """

    Settings: ClassVar[type[ConvozenHttpTTSSettings]] = ConvozenHttpTTSSettings
    _settings: ConvozenHttpTTSSettings

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        model: str = "ragini-v1",
        base_url: str = CONVOZEN_BASE_URL,
        sample_rate: int | None = None,
        settings: ConvozenHttpTTSSettings | None = None,
        **kwargs,
    ):
        """Initialize the ConvoZen TTS service.

        Args:
            api_key: ConvoZen API key for authentication.
            aiohttp_session: Shared aiohttp session for HTTP requests.
            model: Ragini model, ``"ragini-v1"`` (default) or ``"ragini-lite"``.
            base_url: API base URL, for self-hosted deployments.
            sample_rate: Output sample rate. Defaults to the model's native rate.
            settings: Runtime-updatable settings (voice, language, speed).
            **kwargs: Additional arguments passed to TTSService.
        """
        default_settings = self.Settings(
            model=model,
            voice="roohi",
            language=Language.EN,
            speed=1.0,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(sample_rate=sample_rate, settings=default_settings, **kwargs)

        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._session = aiohttp_session

    def can_generate_metrics(self) -> bool:
        """Return whether this service can generate usage metrics.

        Returns:
            True — TTFB and usage metrics are reported for each request.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to a ConvoZen language code.

        Args:
            language: The Language enum value to convert.

        Returns:
            The ConvoZen language code, or None if unsupported.
        """
        return language_to_convozen_language(language)

    def _request_sample_rate(self) -> int:
        """Rate to ask Ragini for, defaulting to the model's native rate."""
        return self.sample_rate or MODEL_SAMPLE_RATES.get(str(self._settings.model), 24000)

    def _build_body(self, text: str) -> dict[str, str]:
        """Build the urlencoded body for the synthesis endpoint.

        The wire name for the voice is ``speaker``. There is no format field:
        the endpoint always returns WAV.
        """
        language = self._settings.language
        speed = self._settings.speed
        return {
            "text": text,
            "language": "en" if isinstance(language, NotGiven) else str(language),
            "speaker": str(self._settings.voice),
            "model": str(self._settings.model),
            "sample_rate": str(self._request_sample_rate()),
            "speed": "1.0" if isinstance(speed, NotGiven) else str(speed),
            "stream": "true",
        }

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Synthesize speech through the Ragini REST API.

        Args:
            text: The text to synthesize.
            context_id: Pipeline context identifier for this utterance.

        Yields:
            TTSAudioRawFrame on success, ErrorFrame on API or network failure.
        """
        try:
            await self.start_ttfb_metrics()

            async with self._session.post(
                f"{self._base_url}/v1/ragini/tts",
                data=self._build_body(text),
                headers={"x-api-key": self._api_key},
            ) as response:
                if response.status != 200:
                    body = (await response.text())[:_MAX_ERROR_BODY]
                    msg = f"ConvoZen Ragini error ({response.status}): {body}"
                    await self.push_error(error_msg=msg)
                    yield ErrorFrame(error=msg)
                    return

                await self.start_tts_usage_metrics(text)

                # Ragini streams a WAV container whose RIFF length fields are
                # placeholders. strip_wav_header reads only the sample rate and
                # skips the 44-byte header, so the placeholders never matter.
                async for frame in self._stream_audio_frames_from_iterator(
                    response.content.iter_chunked(self.chunk_size),
                    strip_wav_header=True,
                    context_id=context_id,
                ):
                    await self.stop_ttfb_metrics()
                    yield frame
        except Exception as e:
            # The exception is not interpolated into the message: an aiohttp error
            # carries RequestInfo, including the API key header, in its detail.
            msg = f"ConvoZen Ragini request failed: {type(e).__name__}"
            await self.push_error(error_msg=msg, exception=e)
            yield ErrorFrame(error=msg)
        finally:
            await self.stop_ttfb_metrics()
            logger.debug(f"{self}: Finished TTS [{text}]")
