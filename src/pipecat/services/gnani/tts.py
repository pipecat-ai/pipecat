#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gnani Vachana text-to-speech service implementations.

**Voices:** Karan (default), Simran, Nara, Riya, Viraj, Raju

Services:
- GnaniHttpTTSService: REST-based single-request synthesis
- GnaniSSETTSService: SSE streaming synthesis (lower latency than REST)
- GnaniTTSService: WebSocket streaming synthesis with interruption handling

For API docs see: https://docs.inya.ai/vachana/TTS/tts-inference
"""

import asyncio
import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

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
from pipecat.services.gnani._common import (
    GNANI_TTS_REST_URL,
    GNANI_TTS_SSE_URL,
    GNANI_TTS_WS_URL,
    SUPPORTED_VOICES,
    TTS_SUPPORTED_SAMPLE_RATES,
    tts_language_to_gnani,
)
from pipecat.services.gnani._sdk import sdk_headers
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven
from pipecat.services.tts_service import InterruptibleTTSService, TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from websockets.asyncio.client import connect as websocket_connect
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Gnani Vachana TTS, you need to "
        "`pip install pipecat-ai[gnani]` or `pip install websockets aiohttp`."
    )
    raise ImportError(f"Missing module: {e}") from e


def _validate_voice(voice: str | None) -> None:
    if voice and voice not in SUPPORTED_VOICES:
        raise ValueError(
            f"Voice '{voice}' not supported. Choose from: {sorted(SUPPORTED_VOICES)}"
        )


def _build_audio_config(settings, sample_rate: int) -> dict:
    return {
        "sample_rate": sample_rate,
        "encoding": settings.encoding or "linear_pcm",
        "num_channels": 1,
        "sample_width": settings.sample_width or 2,
        "container": settings.container or "wav",
    }


def _strip_wav_header(audio_bytes: bytes) -> bytes:
    if len(audio_bytes) > 44 and audio_bytes.startswith(b"RIFF"):
        return audio_bytes[44:]
    return audio_bytes


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class GnaniHttpTTSSettings(TTSSettings):
    """Settings for GnaniHttpTTSService and GnaniSSETTSService.

    Parameters:
        encoding: Audio encoding (linear_pcm or oggopus).
        container: Audio container (raw, mp3, wav, mulaw, ogg).
        sample_width: Sample width in bytes (1-4). Defaults to 2.
    """

    encoding: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    container: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    sample_width: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


@dataclass
class GnaniSSETTSSettings(GnaniHttpTTSSettings):
    """Settings for GnaniSSETTSService (SSE streaming)."""

    pass


@dataclass
class GnaniTTSSettings(GnaniHttpTTSSettings):
    """Settings for GnaniTTSService (WebSocket streaming)."""

    pass


# ---------------------------------------------------------------------------
# REST TTS
# ---------------------------------------------------------------------------


class GnaniHttpTTSService(TTSService):
    """REST-based text-to-speech service using Gnani Vachana API.

    Returns complete audio in a single response via POST /api/v1/tts/inference.
    Suitable for non-streaming use cases where latency is less critical.

    Example::

        tts = GnaniHttpTTSService(
            api_key="your-api-key",
            aiohttp_session=session,
            settings=GnaniHttpTTSService.Settings(
                voice="Karan",
                language=Language.HI,
            ),
        )
    """

    Settings = GnaniHttpTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        voice_id: str | None = None,
        model: str = "vachana-voice-v3",
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        resolved_rate = sample_rate or 16000
        if resolved_rate not in TTS_SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate must be one of {TTS_SUPPORTED_SAMPLE_RATES}, got {resolved_rate}"
            )

        default_settings = self.Settings(
            model=model,
            voice=voice_id or "Karan",
            language="en-IN",
            encoding="linear_pcm",
            container="wav",
            sample_width=2,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        _validate_voice(default_settings.voice)

        super().__init__(
            sample_rate=resolved_rate,
            push_stop_frames=True,
            push_start_frame=True,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._session = aiohttp_session

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        return tts_language_to_gnani(language)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            payload = {
                "text": text,
                "voice": self._settings.voice or "Karan",
                "model": self._settings.model or "vachana-voice-v3",
                "audio_config": _build_audio_config(self._settings, self.sample_rate),
            }
            headers = {
                "X-API-Key-ID": self._api_key,
                "Content-Type": "application/json",
                **sdk_headers(),
            }

            async with self._session.post(
                GNANI_TTS_REST_URL, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield ErrorFrame(error=f"Gnani TTS API error: {error_text}")
                    return
                audio_data = await response.read()

            await self.start_tts_usage_metrics(text)
            audio_data = _strip_wav_header(audio_data)

            yield TTSAudioRawFrame(
                audio=audio_data,
                sample_rate=self.sample_rate,
                num_channels=1,
                context_id=context_id,
            )

        except Exception as e:
            yield ErrorFrame(error=f"Error generating TTS: {e}", exception=e)
        finally:
            await self.stop_ttfb_metrics()


# ---------------------------------------------------------------------------
# SSE Streaming TTS
# ---------------------------------------------------------------------------


class GnaniSSETTSService(TTSService):
    """SSE streaming text-to-speech using Gnani Vachana API.

    Streams audio chunks via Server-Sent Events as they are generated
    via POST /api/v1/tts/sse. Lower latency than the REST service --
    playback can start before the full response is ready.

    Example::

        tts = GnaniSSETTSService(
            api_key="your-api-key",
            aiohttp_session=session,
            settings=GnaniSSETTSService.Settings(
                voice="Karan",
                language=Language.HI,
            ),
        )
    """

    Settings = GnaniSSETTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        voice_id: str | None = None,
        model: str = "vachana-voice-v3",
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        resolved_rate = sample_rate or 16000
        if resolved_rate not in TTS_SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate must be one of {TTS_SUPPORTED_SAMPLE_RATES}, got {resolved_rate}"
            )

        default_settings = self.Settings(
            model=model,
            voice=voice_id or "Karan",
            language="en-IN",
            encoding="linear_pcm",
            container="wav",
            sample_width=2,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        _validate_voice(default_settings.voice)

        super().__init__(
            sample_rate=resolved_rate,
            push_stop_frames=True,
            push_start_frame=True,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._session = aiohttp_session

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        return tts_language_to_gnani(language)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Streaming SSE TTS [{text}]")

        try:
            payload = {
                "text": text,
                "voice": self._settings.voice or "Karan",
                "model": self._settings.model or "vachana-voice-v3",
                "audio_config": _build_audio_config(self._settings, self.sample_rate),
            }
            headers = {
                "X-API-Key-ID": self._api_key,
                "Content-Type": "application/json",
                **sdk_headers(),
            }

            async with self._session.post(
                GNANI_TTS_SSE_URL, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield ErrorFrame(error=f"Gnani TTS SSE error: {error_text}")
                    return

                await self.start_tts_usage_metrics(text)

                buf = ""
                async for raw_line in response.content:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    if line.startswith("event:"):
                        continue
                    if line.startswith("data:"):
                        line = line[len("data:"):].strip()

                    buf += line
                    try:
                        data = json.loads(buf)
                    except json.JSONDecodeError:
                        continue
                    buf = ""

                    if "error" in data or data.get("status") == "error":
                        yield ErrorFrame(
                            error=f"Gnani TTS SSE: {data.get('message', data.get('error', ''))}"
                        )
                        return

                    if data.get("status") == "streaming_started":
                        continue

                    audio_b64 = data.get("audio", "")
                    if not audio_b64:
                        continue

                    audio_bytes = _strip_wav_header(base64.b64decode(audio_b64))
                    if audio_bytes:
                        await self.stop_ttfb_metrics()
                        yield TTSAudioRawFrame(
                            audio=audio_bytes,
                            sample_rate=self.sample_rate,
                            num_channels=1,
                            context_id=context_id,
                        )

                    if data.get("is_final", False):
                        return

        except asyncio.CancelledError:
            raise
        except Exception as e:
            yield ErrorFrame(error=f"Error in SSE TTS: {e}", exception=e)
        finally:
            await self.stop_ttfb_metrics()


# ---------------------------------------------------------------------------
# WebSocket Streaming TTS
# ---------------------------------------------------------------------------


class GnaniTTSService(InterruptibleTTSService):
    """WebSocket streaming text-to-speech using Gnani Vachana.

    Lowest latency option with real-time audio generation and interruption
    handling via a persistent WebSocket connection to
    wss://api.vachana.ai/api/v1/tts.

    Example::

        tts = GnaniTTSService(
            api_key="your-api-key",
            settings=GnaniTTSService.Settings(
                voice="Karan",
                language=Language.HI,
            ),
        )
    """

    Settings = GnaniTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str | None = None,
        model: str = "vachana-voice-v3",
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        resolved_rate = sample_rate or 16000
        if resolved_rate not in TTS_SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate must be one of {TTS_SUPPORTED_SAMPLE_RATES}, got {resolved_rate}"
            )

        default_settings = self.Settings(
            model=model,
            voice=voice_id or "Karan",
            language="IND-IN",
            encoding="linear_pcm",
            container="wav",
            sample_width=2,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        _validate_voice(default_settings.voice)

        super().__init__(
            sample_rate=resolved_rate,
            push_stop_frames=True,
            push_start_frame=True,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._ws = None
        self._receive_task = None
        self._bot_speaking = False

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        return tts_language_to_gnani(language)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect_websocket()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect_websocket()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect_websocket()

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Streaming TTS [{text}]")

        if not self._ws:
            await self._connect_websocket()

        if not self._ws:
            yield ErrorFrame(error="Gnani TTS WebSocket not connected")
            return

        try:
            request_body = {
                "text": text,
                "voice": self._settings.voice or "Karan",
                "model": self._settings.model or "vachana-voice-v3",
                "language": self._settings.language or "IND-IN",
                "audio_config": _build_audio_config(self._settings, self.sample_rate),
            }

            self._bot_speaking = True
            await self._ws.send(json.dumps(request_body))
            await self.start_tts_usage_metrics(text)

        except Exception as e:
            yield ErrorFrame(error=f"Error sending TTS request: {e}", exception=e)
        finally:
            await self.stop_ttfb_metrics()

        yield None

    async def _connect_websocket(self):
        try:
            if self._ws:
                return

            logger.debug("Connecting to Gnani Vachana TTS WebSocket")
            headers = {
                "Content-Type": "application/json",
                "X-API-Key-ID": self._api_key,
                **sdk_headers(),
            }

            self._ws = await websocket_connect(
                GNANI_TTS_WS_URL,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
            )

            self._receive_task = asyncio.create_task(
                self._receive_messages(), name="gnani-tts-recv"
            )

            await self._call_event_handler("on_connected")

        except Exception as e:
            logger.error(f"Failed to connect to Gnani TTS: {e}")
            self._ws = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._ws:
            try:
                logger.debug("Disconnecting from Gnani Vachana TTS WebSocket")
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def _receive_messages(self):
        try:
            async for msg in self._ws:
                if isinstance(msg, bytes):
                    await self._handle_audio_chunk(msg)
                    continue

                data = json.loads(msg)
                msg_type = data.get("type", "")

                if msg_type == "audio":
                    audio_b64 = data.get("audio", "")
                    if audio_b64:
                        await self._handle_audio_chunk(base64.b64decode(audio_b64))

                elif msg_type == "complete":
                    audio_b64 = data.get("audio", "")
                    if audio_b64:
                        await self._handle_audio_chunk(base64.b64decode(audio_b64))
                    self._bot_speaking = False
                    await self.push_frame(TTSStoppedFrame())

                elif msg_type == "error":
                    error_msg = data.get("message", "Unknown error")
                    logger.error(f"Gnani TTS stream error: {error_msg}")
                    self._bot_speaking = False
                    await self.push_frame(ErrorFrame(error=f"Gnani TTS: {error_msg}"))

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Gnani TTS receive error: {e}")
            self._bot_speaking = False
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _handle_audio_chunk(self, audio_bytes: bytes):
        audio_bytes = _strip_wav_header(audio_bytes)
        if audio_bytes:
            await self.push_frame(
                TTSAudioRawFrame(
                    audio=audio_bytes,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                )
            )
