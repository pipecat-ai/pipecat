#
# Copyright (c) 2024–2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gnani Vachana Speech-to-Text service implementations.

Services:
- GnaniHttpSTTService: REST-based file transcription (requires VAD in pipeline)
- GnaniSTTService: WebSocket streaming transcription with real-time VAD

Supported languages: as-IN, bn-IN, en-IN, gu-IN, hi-IN, kn-IN,
ml-IN, mr-IN, or-IN, pa-IN, ta-IN, te-IN.

For API docs see: https://docs.gnani.ai/api/STT/speech-to-text
"""

import asyncio
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
    TranscriptionFrame,
)
from pipecat.services.gnani._common import (
    GNANI_STT_REST_URL,
    GNANI_STT_WS_URL,
    STT_SUPPORTED_FORMATS,
    STT_SUPPORTED_SAMPLE_RATES,
    get_language_string,
    stt_language_to_gnani,
)
from pipecat.services.gnani._sdk import sdk_headers
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven
from pipecat.services.stt_service import SegmentedSTTService, STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from websockets.asyncio.client import connect as websocket_connect
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Gnani Vachana STT, you need to "
        "`pip install pipecat-ai[gnani]` or `pip install websockets aiohttp`."
    )
    raise ImportError(f"Missing module: {e}") from e


# ---------------------------------------------------------------------------
# REST (HTTP) STT
# ---------------------------------------------------------------------------


@dataclass
class GnaniHttpSTTSettings(STTSettings):
    """Settings for GnaniHttpSTTService (REST)."""

    pass


class GnaniHttpSTTService(SegmentedSTTService):
    """REST-based speech-to-text service using Gnani Vachana API.

    Transcribes complete audio segments via HTTP POST to /stt/v3. Requires
    VAD to be enabled in the pipeline so that speech segments are buffered
    and sent as whole utterances.

    Example::

        stt = GnaniHttpSTTService(
            api_key="your-api-key",
            aiohttp_session=session,
            settings=GnaniHttpSTTService.Settings(
                language=Language.HI_IN,
            ),
        )
    """

    Settings = GnaniHttpSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        aiohttp_session: aiohttp.ClientSession,
        base_url: str = GNANI_STT_REST_URL,
        settings: Settings | None = None,
        **kwargs,
    ):
        default_settings = self.Settings(language=Language.EN_IN, model=None)

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(settings=default_settings, **kwargs)

        self._api_key = api_key
        self._session = aiohttp_session
        self._base_url = base_url

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> str:
        return stt_language_to_gnani(language)

    @traced_stt
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribe a complete audio segment via Gnani REST API."""
        try:
            await self.start_processing_metrics()

            lang = get_language_string(self._settings, stt_language_to_gnani)

            headers = {"X-API-Key-ID": self._api_key, **sdk_headers()}

            form = aiohttp.FormData()
            form.add_field(
                "audio_file", audio, filename="audio.wav", content_type="audio/wav"
            )
            form.add_field("language_code", lang or "en-IN")

            async with self._session.post(
                self._base_url, data=form, headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield ErrorFrame(error=f"Gnani STT API error: {error_text}")
                    return
                result = await response.json()

            await self.stop_processing_metrics()

            text = result.get("transcript", "").strip()
            if text:
                yield TranscriptionFrame(
                    text=text,
                    user_id=self._user_id if hasattr(self, "_user_id") else "",
                    timestamp=time_now_iso8601(),
                    language=lang,
                )

        except Exception as e:
            yield ErrorFrame(error=f"Error transcribing audio: {e}", exception=e)


# ---------------------------------------------------------------------------
# WebSocket streaming STT
# ---------------------------------------------------------------------------


@dataclass
class GnaniSTTSettings(STTSettings):
    """Settings for GnaniSTTService (WebSocket streaming).

    Parameters:
        sample_rate: Audio sample rate (8000, 16000, 44100, or 48000 Hz).
        format: 'verbatim' for raw output, 'transcribe' for ITN-normalized output.
        itn_native_numerals: When format='transcribe', render digits in native script.
    """

    sample_rate: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    format: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    itn_native_numerals: bool | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class GnaniSTTService(STTService):
    """WebSocket streaming speech-to-text using Gnani Vachana.

    Provides real-time speech recognition for Indian languages via a
    persistent WebSocket connection to wss://api.vachana.ai/stt/v3/stream.

    Event handlers (in addition to STTService events):

    - on_connected(service): Connected to Gnani WebSocket
    - on_disconnected(service): Disconnected from Gnani WebSocket
    - on_connection_error(service, error): Connection error occurred

    Example::

        stt = GnaniSTTService(
            api_key="your-api-key",
            settings=GnaniSTTService.Settings(
                language=Language.HI_IN,
            ),
        )
    """

    Settings = GnaniSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        sample_rate: int | None = None,
        settings: Settings | None = None,
        keepalive_timeout: float | None = None,
        keepalive_interval: float = 5.0,
        **kwargs,
    ):
        resolved_rate = sample_rate or 16000
        if resolved_rate not in STT_SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"sample_rate must be one of {STT_SUPPORTED_SAMPLE_RATES}, got {resolved_rate}"
            )

        default_settings = self.Settings(language=Language.EN_IN, model=None)

        if settings is not None:
            default_settings.apply_update(settings)

        if (
            hasattr(default_settings, "format")
            and default_settings.format
            and default_settings.format not in (NOT_GIVEN, None)
            and default_settings.format not in STT_SUPPORTED_FORMATS
        ):
            raise ValueError(
                f"format must be one of {STT_SUPPORTED_FORMATS}, "
                f"got '{default_settings.format}'"
            )

        super().__init__(
            sample_rate=resolved_rate,
            keepalive_timeout=keepalive_timeout,
            keepalive_interval=keepalive_interval,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._ws = None
        self._receive_task = None

    def language_to_service_language(self, language: Language) -> str:
        return stt_language_to_gnani(language)

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if not self._ws:
            await self._connect()
            if not self._ws:
                yield None
                return

        try:
            await self._ws.send(audio)
        except Exception as e:
            logger.warning(f"Gnani STT send failed, reconnecting: {e}")
            self._ws = None
            await self._connect()
            if self._ws:
                try:
                    await self._ws.send(audio)
                except Exception as e2:
                    yield ErrorFrame(error=f"Error sending audio to Gnani: {e2}", exception=e2)

        yield None

    async def _connect(self):
        logger.debug("Connecting to Gnani Vachana STT")

        try:
            lang = get_language_string(self._settings, stt_language_to_gnani)
            headers = {
                "x-api-key-id": self._api_key,
                "lang_code": lang or "en-IN",
                "x-sample-rate": str(self.sample_rate),
                **sdk_headers(),
            }

            fmt = getattr(self._settings, "format", None)
            if fmt and fmt not in (NOT_GIVEN, None):
                headers["x-format"] = fmt

            itn = getattr(self._settings, "itn_native_numerals", None)
            if itn and itn not in (NOT_GIVEN, None):
                headers["itn_native_numerals"] = str(itn).lower()

            self._ws = await websocket_connect(
                GNANI_STT_WS_URL,
                additional_headers=headers,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=10,
            )

            connected_msg = await asyncio.wait_for(self._ws.recv(), timeout=10)
            connected_data = json.loads(connected_msg)
            if connected_data.get("type") == "connected":
                logger.info(f"Gnani STT connected: {connected_data.get('message', '')}")
            else:
                logger.warning(f"Unexpected first message from Gnani STT: {connected_data}")

            self._receive_task = asyncio.create_task(
                self._receive_messages(), name="gnani-stt-recv"
            )

            await self._call_event_handler("on_connected")

        except Exception as e:
            logger.error(f"Failed to connect to Gnani STT: {e}")
            self._ws = None
            await self._call_event_handler("on_connection_error", str(e))

    async def _disconnect(self):
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        await self._call_event_handler("on_disconnected")

    async def _receive_messages(self):
        try:
            async for msg in self._ws:
                if isinstance(msg, bytes):
                    continue

                data = json.loads(msg)
                msg_type = data.get("type", "")

                if msg_type == "transcript":
                    text = data.get("text", "")
                    if text:
                        lang = get_language_string(self._settings, stt_language_to_gnani)
                        await self.push_frame(
                            TranscriptionFrame(
                                text=text,
                                user_id=self._user_id if hasattr(self, "_user_id") else "",
                                timestamp=time_now_iso8601(),
                                language=lang,
                            )
                        )

                elif msg_type in (
                    "processing", "speech_start", "vad_start", "speech_end", "vad_end",
                ):
                    pass

                elif msg_type == "error":
                    error_msg = data.get("message", "Unknown error")
                    logger.error(f"Gnani STT stream error: {error_msg}")
                    self._ws = None
                    await self.push_frame(ErrorFrame(error=f"Gnani STT: {error_msg}"))

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Gnani STT receive error: {e}")
            self._ws = None
            await self._call_event_handler("on_connection_error", str(e))
