#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""xAI text-to-speech service implementation.

Provides two TTS services against xAI's voice API:

- :class:`XAIHttpTTSService` uses the batch HTTP endpoint at
  ``https://api.x.ai/v1/tts``.
- :class:`XAITTSService` uses the streaming WebSocket endpoint at
  ``wss://api.x.ai/v1/tts``.

See https://docs.x.ai/developers/rest-api-reference/inference/voice.
"""

import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import aiohttp
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import InterruptibleTTSService, TTSService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use XAITTSService, you need to `pip install pipecat-ai[xai]`.")
    raise Exception(f"Missing module: {e}")


def language_to_xai_language(language: Language) -> str:
    """Convert a Language enum to xAI language code.

    Args:
        language: The Language enum value to convert.

    Returns:
        The corresponding service language code. If ``language`` is not in
        the verified mapping, falls back to the base language code (e.g.,
        ``en`` from ``en-US``) and logs a warning (via
        ``resolve_language(..., use_base_code=True)``).
    """
    LANGUAGE_MAP = {
        Language.AR: "ar-EG",
        Language.AR_EG: "ar-EG",
        Language.AR_SA: "ar-SA",
        Language.AR_AE: "ar-AE",
        Language.BN: "bn",
        Language.DE: "de",
        Language.EN: "en",
        Language.ES: "es-ES",
        Language.ES_ES: "es-ES",
        Language.ES_MX: "es-MX",
        Language.FR: "fr",
        Language.HI: "hi",
        Language.ID: "id",
        Language.IT: "it",
        Language.JA: "ja",
        Language.KO: "ko",
        Language.PT: "pt-PT",
        Language.PT_BR: "pt-BR",
        Language.PT_PT: "pt-PT",
        Language.RU: "ru",
        Language.TR: "tr",
        Language.VI: "vi",
        Language.ZH: "zh",
    }

    return resolve_language(language, LANGUAGE_MAP, use_base_code=True)


@dataclass
class XAITTSSettings(TTSSettings):
    """Settings for XAIHttpTTSService."""

    pass


class XAIHttpTTSService(TTSService):
    """xAI HTTP text-to-speech service.

    The service requests raw PCM audio so emitted ``TTSAudioRawFrame`` objects
    match Pipecat's downstream expectations without extra decoding.
    """

    Settings = XAITTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.x.ai/v1/tts",
        sample_rate: int | None = None,
        encoding: str | None = "pcm",
        aiohttp_session: aiohttp.ClientSession | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the xAI TTS service.

        Args:
            api_key: xAI API key for authentication.
            base_url: xAI TTS endpoint. Defaults to ``https://api.x.ai/v1/tts``.
            sample_rate: Audio sample rate. If None, uses default.
            encoding: Output encoding format. Defaults to "pcm".
            aiohttp_session: Optional shared aiohttp session.
            settings: Runtime-updatable settings.
            **kwargs: Additional keyword arguments passed to ``TTSService``.
        """
        default_settings = self.Settings(
            model=None,
            voice="eve",
            language=Language.EN,
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
        self._encoding = encoding

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to xAI language format.

        Args:
            language: The language to convert.

        Returns:
            The xAI-specific language code, or None if not supported.
        """
        return language_to_xai_language(language)

    async def start(self, frame):
        """Start the xAI TTS service."""
        await super().start(frame)
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._session_owner = True

    async def stop(self, frame):
        """Stop the xAI TTS service."""
        await super().stop(frame)
        await self._close_session()

    async def cancel(self, frame):
        """Cancel the xAI TTS service."""
        await super().cancel(frame)
        await self._close_session()

    async def _close_session(self):
        if self._session_owner and self._session and not self._session.closed:
            await self._session.close()
        if self._session_owner:
            self._session = None

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using xAI's TTS API."""
        logger.debug(f"{self}: Generating TTS [{text}]")

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._session_owner = True

        payload = {
            "text": text,
            "voice_id": self._settings.voice,
            "output_format": {
                "codec": self._encoding,
                "sample_rate": self.sample_rate,
            },
        }
        if self._settings.language:
            payload["language"] = str(self._settings.language)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        measuring_ttfb = True
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

                async for chunk in response.content.iter_chunked(self.chunk_size):
                    if not chunk:
                        continue
                    if measuring_ttfb:
                        await self.stop_ttfb_metrics()
                        measuring_ttfb = False
                    yield TTSAudioRawFrame(
                        chunk,
                        self.sample_rate,
                        1,
                        context_id=context_id,
                    )
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")


@dataclass
class XAIWebsocketTTSSettings(TTSSettings):
    """Settings for XAITTSService (WebSocket streaming)."""

    pass


class XAITTSService(InterruptibleTTSService):
    """xAI streaming text-to-speech service.

    Connects to xAI's WebSocket TTS endpoint and streams audio chunks back as
    they are synthesized. Text can be sent incrementally via ``text.delta``
    messages and each utterance is terminated with ``text.done``. The server
    responds with ``audio.delta`` chunks followed by an ``audio.done`` message.

    Audio parameters (voice, language, codec, sample rate, bit rate) are passed
    as query string parameters on the WebSocket URL; changing any of them at
    runtime reconnects the WebSocket.
    """

    Settings = XAIWebsocketTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "wss://api.x.ai/v1/tts",
        sample_rate: int | None = None,
        codec: str = "pcm",
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the xAI WebSocket TTS service.

        Args:
            api_key: xAI API key for authentication.
            base_url: xAI TTS WebSocket endpoint. Defaults to
                ``wss://api.x.ai/v1/tts``.
            sample_rate: Output audio sample rate in Hz. If None, uses the
                pipeline default.
            codec: Output audio codec. One of ``pcm``, ``wav``, ``mulaw``,
                ``alaw``. Defaults to ``pcm`` so emitted ``TTSAudioRawFrame``
                objects need no decoding downstream.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to parent
                ``InterruptibleTTSService``.
        """
        default_settings = self.Settings(
            model=None,
            voice="eve",
            language=Language.EN,
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
        self._base_url = base_url
        self._codec = codec
        self._receive_task = None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics."""
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """Convert a Language enum to xAI language format."""
        return language_to_xai_language(language)

    async def start(self, frame: StartFrame):
        """Start the xAI WebSocket TTS service."""
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the xAI WebSocket TTS service."""
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the xAI WebSocket TTS service."""
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta. Reconnects if any URL-baked field changes."""
        changed = await super()._update_settings(delta)

        if changed:
            await self._disconnect()
            await self._connect()

        return changed

    def _build_url(self) -> str:
        language = self._settings.language
        if isinstance(language, Language):
            language_value = language_to_xai_language(language) or language.value
        else:
            language_value = str(language) if language is not None else "auto"

        params: dict[str, Any] = {
            "voice": self._settings.voice,
            "language": language_value,
            "codec": self._codec,
            "sample_rate": self.sample_rate,
        }
        return f"{self._base_url}?{urlencode(params)}"

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to xAI TTS")

            url = self._build_url()
            headers = {"Authorization": f"Bearer {self._api_key}"}
            self._websocket = await websocket_connect(url, additional_headers=headers)

            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from xAI TTS")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error disconnecting from xAI TTS: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def flush_audio(self, context_id: str | None = None):
        """Signal end-of-utterance so xAI begins synthesizing what it has buffered."""
        if not self._websocket or self._websocket.state is State.CLOSED:
            return
        await self._get_websocket().send(json.dumps({"type": "text.done"}))

    async def _receive_messages(self):
        async for message in self._get_websocket():
            if isinstance(message, bytes):
                logger.warning(f"{self}: unexpected binary frame from xAI TTS")
                continue
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                logger.error(f"{self}: invalid JSON message: {message}")
                continue

            msg_type = msg.get("type")
            context_id = self.get_active_audio_context_id()

            if msg_type == "audio.delta":
                audio_b64 = msg.get("delta")
                if not audio_b64:
                    continue
                audio = base64.b64decode(audio_b64)
                await self.stop_ttfb_metrics()
                if context_id:
                    frame = TTSAudioRawFrame(
                        audio=audio,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        context_id=context_id,
                    )
                    await self.append_to_audio_context(context_id, frame)
            elif msg_type == "audio.done":
                await self.stop_all_metrics()
                if context_id:
                    await self.append_to_audio_context(
                        context_id, TTSStoppedFrame(context_id=context_id)
                    )
                    await self.remove_audio_context(context_id)
            elif msg_type == "error":
                await self.stop_all_metrics()
                error_detail = msg.get("message") or msg.get("error") or str(msg)
                if context_id:
                    await self.append_to_audio_context(
                        context_id, TTSStoppedFrame(context_id=context_id)
                    )
                    await self.remove_audio_context(context_id)
                await self.push_error(error_msg=f"xAI TTS error: {error_detail}")
            else:
                logger.debug(f"{self}: unhandled xAI message type: {msg_type}")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate TTS audio from text using xAI's streaming WebSocket API."""
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                await self._get_websocket().send(json.dumps({"type": "text.delta", "delta": text}))
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                yield ErrorFrame(error=f"Unknown error occurred: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
