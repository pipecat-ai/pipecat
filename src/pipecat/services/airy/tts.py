#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Text-to-speech synthesis through Airy's HTTP streaming API."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field

import aiohttp

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.errors import ErrorCategory, classify_http_exception, classify_http_status_code
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven


@dataclass
class AiryTTSSettings(TTSSettings):
    """Runtime settings for Airy text-to-speech.

    Parameters:
        style: Speaking style: ``normal``, ``bright``, ``calm``, or ``whisper``.
    """

    style: str | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class AiryHttpTTSService(TTSService):
    """Stream Korean or English speech from Airy.

    Airy returns 24 kHz, signed 16-bit little-endian mono PCM. Audio is
    resampled to the pipeline's output rate when necessary.
    """

    Settings = AiryTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        base_url: str = "https://api.airy.so",
        sample_rate: int | None = None,
        request_timeout: float = 30.0,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Airy HTTP TTS service.

        Args:
            api_key: Airy API key.
            aiohttp_session: HTTP session owned and closed by the caller.
            base_url: Airy API origin, without the ``/v1`` path.
            sample_rate: Output sample rate in Hz. Defaults to the pipeline rate.
            request_timeout: Maximum duration of a synthesis request in seconds.
            settings: Runtime settings. Defaults to model ``airy-tts-v1``, voice
                ``a597bb7a98fc9ec1`` (Silvia), English, and style ``normal``.
            **kwargs: Additional arguments passed to :class:`TTSService`.
        """
        default_settings = self.Settings(
            model="airy-tts-v1",
            voice="a597bb7a98fc9ec1",
            language=Language.EN,
            style="normal",
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )
        self._api_key = api_key
        self._session = aiohttp_session
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=request_timeout)
        # Network gaps must not discard audio held in the resampling filter.
        self._resampler = create_stream_resampler(clear_after_secs=None)

    def can_generate_metrics(self) -> bool:
        """Report support for processing, first-byte, and usage metrics.

        Returns:
            True.
        """
        return True

    def language_to_service_language(self, language: Language) -> str:
        """Convert a language or locale to its base language code.

        Args:
            language: Language used for synthesis.

        Returns:
            Base language code, such as ``ko`` for ``ko-KR``.
        """
        return language.value.split("-")[0]

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Synthesize text and yield PCM audio frames.

        Args:
            text: Text to synthesize, up to 1,280 characters.
            context_id: Identifier of the current TTS context.

        Yields:
            Audio frames for the current context, or an error frame on failure.
        """
        keepalive_task = None
        try:
            if not 1 <= len(text) <= 1280:
                yield ErrorFrame(
                    error="Airy TTS input must contain between 1 and 1280 characters",
                    category=ErrorCategory.INVALID_REQUEST,
                )
                return

            keepalive_task = self.create_task(
                self._keep_audio_context_alive(context_id), name=f"{self} HTTP keepalive"
            )
            payload = {
                "input": text,
                "model": self._settings.model,
                "voice": self._settings.voice,
                "language": self._settings.language,
                "style": self._settings.style,
            }
            async with self._session.post(
                f"{self._base_url}/v1/audio/speech/stream",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
                allow_redirects=False,
            ) as response:
                if response.status != 200:
                    yield await self._response_error(response)
                    return
                if response.content_type != "audio/pcm":
                    raise ValueError(f"Expected audio/pcm, received {response.content_type}")

                await self.start_tts_usage_metrics(text)
                async for frame in self._stream_audio_frames_from_iterator(
                    self._audio_chunks(response),
                    in_sample_rate=24000,
                    context_id=context_id,
                ):
                    yield frame

                # The resampler can retain an entire short utterance in its filter.
                tail = await self._resampler.flush()
                if tail:
                    yield TTSAudioRawFrame(tail, self.sample_rate, 1, context_id=context_id)
        except TimeoutError:
            yield ErrorFrame(
                error="Airy TTS request timed out", category=ErrorCategory.CONNECTIVITY
            )
        except aiohttp.ClientError as e:
            yield ErrorFrame(
                error=f"Airy TTS request failed: {e}", category=classify_http_exception(e)
            )
        except ValueError as e:
            yield ErrorFrame(error=f"Airy TTS response error: {e}", category=ErrorCategory.SERVER)
        finally:
            if keepalive_task:
                await self.cancel_task(keepalive_task)
            await self.stop_ttfb_metrics()
            await self._resampler.reset()

    async def _keep_audio_context_alive(self, context_id: str):
        # The HTTP request timeout owns the lifetime of an in-flight synthesis.
        while True:
            self._refresh_audio_context(context_id)
            await asyncio.sleep(self._stop_frame_timeout_s / 2)

    async def _audio_chunks(self, response: aiohttp.ClientResponse) -> AsyncIterator[bytes]:
        size = 0
        async for chunk in response.content.iter_chunked(8192):
            if chunk:
                await self.stop_ttfb_metrics()
                size += len(chunk)
                yield chunk
        if not size:
            raise ValueError("Empty audio response")
        if size % 2:
            raise ValueError("Audio response ends with an incomplete PCM sample")

    async def _response_error(self, response: aiohttp.ClientResponse) -> ErrorFrame:
        message = f"Airy TTS request failed (HTTP {response.status})"
        try:
            body = await response.json()
        except (ValueError, aiohttp.ContentTypeError):
            body = None
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                for key in ("code", "message", "param"):
                    if error.get(key) is not None:
                        message += f"; {key}={error[key]}"
            if body.get("request_id"):
                message += f"; request_id={body['request_id']}"
        if self._api_key:
            message = message.replace(self._api_key, "[redacted]")
        return ErrorFrame(error=message, category=classify_http_status_code(response.status))
