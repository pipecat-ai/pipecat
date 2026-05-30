#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Qwen TTS service implementation using DashScope HTTP SSE streaming."""

import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts

# DashScope TTS endpoints
QWEN_TTS_URL_INTL = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text2audio/generation"
QWEN_TTS_URL_CN = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2audio/generation"


def language_to_qwen_language(language: Language) -> str | None:
    """Convert a Language enum to a Qwen TTS language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding Qwen TTS language code, or None if not supported.
    """
    LANGUAGE_MAP = {
        Language.ZH: "zh",
        Language.EN: "en",
        Language.JA: "ja",
        Language.KO: "ko",
    }
    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


@dataclass
class QwenTTSSettings(TTSSettings):
    """Settings for QwenTTSService.

    Parameters:
        model: DashScope TTS model identifier.
        voice: Voice identifier (e.g. ``"Chelsie"``, ``"Ethan"``, ``"Serena"``).
        language: Language for synthesis. Qwen TTS uses the voice to infer
            language, so this setting controls only the ``language_to_service_language``
            conversion when a voice-level language override is needed.
    """

    pass


class QwenTTSService(TTSService):
    """Alibaba Cloud Qwen (DashScope) text-to-speech service.

    Provides streaming TTS synthesis using the DashScope HTTP SSE API.
    Outputs raw 16-bit PCM audio frames at the configured sample rate,
    suitable for direct use in a Pipecat voice pipeline.

    Requires a DashScope API key. International users should use the default
    endpoint; users in mainland China should pass ``base_url=QWEN_TTS_URL_CN``.

    Supported voices include ``"Chelsie"`` and ``"Serena"`` (English female),
    ``"Ethan"`` (English male), ``"Cherry"`` and ``"Aria"`` (Chinese female),
    ``"Rocky"`` (Chinese male), and others. See the DashScope documentation for
    the full voice list.

    Example::

        import aiohttp

        async with aiohttp.ClientSession() as session:
            tts = QwenTTSService(
                api_key="your-dashscope-api-key",
                aiohttp_session=session,
                settings=QwenTTSService.Settings(
                    model="qwen3-tts-flash",
                    voice="Chelsie",
                ),
            )
    """

    Settings = QwenTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        voice: str | None = None,
        model: str | None = None,
        base_url: str = QWEN_TTS_URL_INTL,
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Qwen TTS service.

        Args:
            api_key: DashScope API key for authentication.
            aiohttp_session: Shared aiohttp session for HTTP requests.
            voice: Voice identifier (e.g. ``"Chelsie"``).

                .. deprecated:: 0.0.106
                    Use ``settings=QwenTTSService.Settings(voice=...)`` instead.

            model: TTS model identifier (e.g. ``"qwen3-tts-flash"``).

                .. deprecated:: 0.0.106
                    Use ``settings=QwenTTSService.Settings(model=...)`` instead.

            base_url: DashScope TTS endpoint URL. Defaults to the international
                endpoint. Pass ``QWEN_TTS_URL_CN`` for mainland China.
            sample_rate: Audio sample rate in Hz. If ``None``, uses 22050 Hz,
                which is the native output rate of most Qwen TTS voices.
            settings: Runtime-updatable settings object. Takes precedence over
                deprecated positional arguments when both are provided.
            **kwargs: Additional keyword arguments forwarded to TTSService.
        """
        # Set language=None: Qwen TTS infers language from voice selection,
        # so there is no service-level language parameter to pass.
        default_settings = self.Settings(model="qwen3-tts-flash", voice="Chelsie", language=None)

        if model is not None:
            self._warn_init_param_moved_to_settings("model", "model")
            default_settings.model = model

        if voice is not None:
            self._warn_init_param_moved_to_settings("voice", "voice")
            default_settings.voice = voice

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_text_frames=False,
            push_stop_frames=True,
            push_start_frame=True,
            sample_rate=sample_rate or 22050,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._session = aiohttp_session
        self._base_url = base_url

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to a Qwen TTS language code.

        Args:
            language: The language to convert.

        Returns:
            A Qwen TTS language code, or ``None`` if the language is not supported.
        """
        return language_to_qwen_language(language)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using the Qwen TTS HTTP SSE streaming API.

        Sends a request to the DashScope TTS endpoint with the
        ``X-DashScope-SSE: enable`` header, then yields :class:`TTSAudioRawFrame`
        objects as base64-encoded PCM chunks arrive in the SSE stream.

        Args:
            text: Text to synthesize.
            context_id: Identifier for the current TTS context.

        Yields:
            Frame: :class:`TTSAudioRawFrame` for each audio chunk, or an
            :class:`ErrorFrame` if the request fails.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "X-DashScope-SSE": "enable",
        }

        payload = {
            "model": self._settings.model,
            "input": {"text": text},
            "parameters": {
                "voice": self._settings.voice,
                "format": "pcm",
                "sample_rate": self.sample_rate,
            },
        }

        try:
            async with self._session.post(
                self._base_url, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(
                        f"{self}: DashScope TTS error (HTTP {response.status}): {error_text}"
                    )
                    yield ErrorFrame(error=f"Qwen TTS API error: {error_text}")
                    return

                await self.start_tts_usage_metrics(text)

                async for raw_line in response.content:
                    line = raw_line.decode("utf-8").strip()

                    # SSE data lines start with "data:"
                    if not line.startswith("data:"):
                        continue

                    data_str = line[len("data:") :].strip()
                    if not data_str:
                        continue

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError as exc:
                        logger.warning(f"{self}: Failed to parse SSE JSON: {exc}")
                        continue

                    output = data.get("output", {})
                    audio_info = output.get("audio")

                    if isinstance(audio_info, dict):
                        audio_b64 = audio_info.get("data")
                        if audio_b64:
                            await self.stop_ttfb_metrics()
                            audio = base64.b64decode(audio_b64)
                            yield TTSAudioRawFrame(
                                audio=audio,
                                sample_rate=self.sample_rate,
                                num_channels=1,
                                context_id=context_id,
                            )

                    # DashScope signals completion via finish_reason == "stop"
                    finish_reason = output.get("finish_reason")
                    if finish_reason == "stop":
                        break

        except Exception as exc:
            logger.error(f"{self}: Unexpected error during TTS generation: {exc}")
            yield ErrorFrame(error=f"Qwen TTS unexpected error: {exc}")
        finally:
            await self.stop_ttfb_metrics()
