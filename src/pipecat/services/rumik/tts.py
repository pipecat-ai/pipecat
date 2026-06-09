#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Rumik Voice API text-to-speech service implementations.

This module provides TTS services using Rumik's Voice API. Requests
authenticate with a Bearer token in the ``Authorization`` header.

**Service Variants:**

- **RumikTTSService**: WebSocket TTS service for interactive voice agents.
    - Mints a one-time streaming token with ``POST /v1/tts/ws-connect``.
    - Connects to ``/ws/tts`` using the returned ``ws_url`` and token.
    - Sends synthesis parameters over one persistent
      WebSocket connection.
    - Receives raw PCM int16 audio chunks at 24 kHz, mono.
    - Handles ``queued``, ``done``, ``timeout``, and ``error`` control frames.
    - Serializes synthesis requests and reconnects on interruption because the
      WebSocket protocol does not currently echo per-message context IDs.

- **RumikHttpTTSService**: HTTP TTS service for request/response synthesis.
    - Sends synthesis requests with ``POST /v1/tts``.
    - Receives WAV audio containing PCM int16 at 24 kHz, mono.
    - Converts WAV responses to raw PCM frames for Pipecat playback.

**Models and Settings:**

- ``muga``: Conversational speech model.
- ``mulberry``: Higher-quality TTS model when enabled by the Rumik deployment.
- ``voice`` is sent to Rumik as ``speaker`` for preset Mulberry voices.
- Synthesis settings include ``description``, ``f0_up_key``, ``temperature``,
  ``top_p``, ``top_k``, ``repetition_penalty``, and ``max_new_tokens``.

Rumik currently returns 24 kHz mono PCM audio. The ``sample_rate`` parameter
must therefore be 24000 Hz.
"""

import asyncio
import io
import json
import re
import ssl
import wave
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
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
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven, assert_given
from pipecat.services.tts_service import InterruptibleTTSService, TTSService
from pipecat.utils.text.base_text_aggregator import (
    Aggregation,
    AggregationType,
    BaseTextAggregator,
)
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Rumik, you need to `pip install pipecat-ai[rumik]`.")
    raise ImportError(f"Missing module: {e}") from e


RUMIK_SAMPLE_RATE = 24000
RUMIK_DEFAULT_MODEL = "muga"
RUMIK_DEFAULT_TEMPERATURE = 0.6
RUMIK_DEFAULT_TOP_P = 0.95
RUMIK_DEFAULT_TOP_K = 50
RUMIK_DEFAULT_REPETITION_PENALTY = 1.2
RUMIK_DEFAULT_MAX_NEW_TOKENS = 2048


@dataclass
class RumikTTSSettings(TTSSettings):
    """Settings for Rumik TTS services.

    Parameters:
        description: Natural-language voice description for Mulberry.
        f0_up_key: Pitch shift in semitones for Mulberry preset speakers.
        temperature: Sampling temperature.
        top_p: Nucleus sampling value.
        top_k: Top-k sampling value.
        repetition_penalty: Penalty for repeated tokens.
        max_new_tokens: Maximum generated audio tokens.
    """

    description: str | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    f0_up_key: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    temperature: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    top_p: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    top_k: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    repetition_penalty: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    max_new_tokens: int | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


def _validate_sample_rate(sample_rate: int | None) -> int:
    if sample_rate is None:
        return RUMIK_SAMPLE_RATE
    if sample_rate != RUMIK_SAMPLE_RATE:
        raise ValueError(
            f"Rumik TTS currently returns {RUMIK_SAMPLE_RATE} Hz PCM; "
            f"sample_rate must be {RUMIK_SAMPLE_RATE}."
        )
    return sample_rate


def _default_settings() -> RumikTTSSettings:
    return RumikTTSSettings(
        model=RUMIK_DEFAULT_MODEL,
        voice=None,
        language=None,
        description=None,
        f0_up_key=None,
        temperature=RUMIK_DEFAULT_TEMPERATURE,
        top_p=RUMIK_DEFAULT_TOP_P,
        top_k=RUMIK_DEFAULT_TOP_K,
        repetition_penalty=RUMIK_DEFAULT_REPETITION_PENALTY,
        max_new_tokens=RUMIK_DEFAULT_MAX_NEW_TOKENS,
    )


def _build_synthesis_payload(
    settings: RumikTTSSettings, text: str, *, include_model: bool = True
) -> dict[str, Any]:
    payload: dict[str, Any] = {"text": text}

    model = assert_given(settings.model) or RUMIK_DEFAULT_MODEL
    voice = assert_given(settings.voice)
    description = assert_given(settings.description)
    f0_up_key = assert_given(settings.f0_up_key)
    temperature = assert_given(settings.temperature)
    top_p = assert_given(settings.top_p)
    top_k = assert_given(settings.top_k)
    repetition_penalty = assert_given(settings.repetition_penalty)
    max_new_tokens = assert_given(settings.max_new_tokens)

    if include_model:
        payload["model"] = model
    if description is not None:
        payload["description"] = description
    if voice is not None:
        payload["speaker"] = voice
    if f0_up_key is not None:
        payload["f0_up_key"] = f0_up_key
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if top_k is not None:
        payload["top_k"] = top_k
    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty
    if max_new_tokens is not None:
        payload["max_new_tokens"] = max_new_tokens
    return payload


class _FullResponseTextAggregator(BaseTextAggregator):
    """Buffer a full LLM response and flush it as one TTS request."""

    def __init__(self):
        super().__init__(aggregation_type=AggregationType.SENTENCE)
        self._text = ""

    @property
    def text(self) -> Aggregation:
        return Aggregation(text=self._text.strip(" "), type=AggregationType.SENTENCE)

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        self._text += text
        if False:
            yield Aggregation(text="", type=AggregationType.SENTENCE)

    async def flush(self) -> Aggregation | None:
        if not self._text:
            return None
        result = self._text
        await self.reset()
        return Aggregation(text=result.strip(" "), type=AggregationType.SENTENCE)

    async def handle_interruption(self):
        await self.reset()

    async def reset(self):
        self._text = ""


class RumikTTSService(InterruptibleTTSService):
    """Rumik WebSocket text-to-speech service.

    This service keeps a persistent WebSocket connection open and streams raw
    PCM chunks from Rumik into Pipecat audio contexts. Since Rumik's WebSocket
    protocol does not currently echo a per-request context ID, the service
    serializes synthesis requests and reconnects on interruption.
    """

    Settings = RumikTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        gateway_url: str,
        settings: Settings | None = None,
        sample_rate: int | None = RUMIK_SAMPLE_RATE,
        full_response_aggregation: bool = True,
        **kwargs,
    ):
        """Initialize the Rumik WebSocket TTS service.

        Args:
            api_key: Rumik API key.
            gateway_url: Rumik gateway base URL.
            settings: Runtime-updatable Rumik TTS settings.
            sample_rate: Output audio sample rate. Rumik currently returns 24 kHz PCM.
            full_response_aggregation: When true, buffer a complete LLM response
                before sending text to Rumik to avoid sentence-level TTFB gaps.
            **kwargs: Additional arguments passed to ``InterruptibleTTSService``.
        """
        default_settings = _default_settings()
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_stop_frames=False,
            push_start_frame=True,
            pause_frame_processing=True,
            sample_rate=_validate_sample_rate(sample_rate),
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._gateway_url = gateway_url.rstrip("/")
        self._receive_task: asyncio.Task | None = None
        self._active_context_id: str | None = None
        self._synthesis_lock = asyncio.Lock()
        self._interruption_restart_in_progress = False

        if full_response_aggregation:
            self._text_aggregator = _FullResponseTextAggregator()

    def can_generate_metrics(self) -> bool:
        """Return whether this service can generate Pipecat metrics."""
        return True

    async def start(self, frame: StartFrame):
        """Start the service and connect to Rumik."""
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service and close the Rumik connection."""
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and close the Rumik connection."""
        await super().cancel(frame)
        await self._disconnect()

    async def cleanup(self):
        """Release Rumik resources when the pipeline is torn down abruptly."""
        try:
            await self._disconnect()
            await self._stop_audio_context_task()
        finally:
            await super().cleanup()

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        changed = await super()._update_settings(delta)
        if "model" in changed and self._websocket:
            await self._disconnect()
            await self._connect()
        return changed

    async def _connect(self):
        await super()._connect()
        if self._receive_task and self._receive_task.done():
            self._receive_task = None
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self, *, clear_active_context: bool = True):
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()
        if clear_active_context:
            self._clear_active_context()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug(f"{self}: minting Rumik WebSocket session")
            session = await self._mint_websocket_session()
            request_id = session.get("request_id")

            separator = "&" if "?" in session["ws_url"] else "?"
            ws_url = f"{session['ws_url']}{separator}{urlencode({'token': session['token']})}"
            logger.debug(f"{self}: connecting Rumik WS request_id={request_id}")
            ssl_context = ssl.create_default_context() if ws_url.startswith("wss://") else None
            self._websocket = await websocket_connect(
                ws_url,
                ssl=ssl_context,
                ping_interval=20,
                ping_timeout=30,
            )
            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            await self.push_error_frame(ErrorFrame(error=f"Rumik WS connect failed: {e}"))
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket and self._websocket.state is State.OPEN:
                try:
                    await self._websocket.send(json.dumps({"type": "close"}))
                except Exception:
                    logger.debug(f"{self}: unable to send Rumik close frame")
                await self._websocket.close()
        except Exception as e:
            await self.push_error_frame(ErrorFrame(error=f"Rumik WS disconnect error: {e}"))
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def _mint_websocket_session(self) -> dict[str, Any]:
        mint_url = f"{self._gateway_url}/v1/tts/ws-connect"
        model = assert_given(self._settings.model) or RUMIK_DEFAULT_MODEL
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                mint_url,
                json={"text": "init", "model": model},
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                data = await response.json()

        for key in ("ws_url", "token"):
            if key not in data:
                raise ValueError(f"Rumik ws-connect response missing {key!r}")
        return data

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("WebSocket not connected")

    async def _report_error(self, error: ErrorFrame):
        await self._finish_active_context()
        await super()._report_error(error)

    async def on_audio_context_interrupted(self, context_id: str):
        """Stop any in-flight Rumik request when Pipecat interrupts playback."""
        request_in_flight = context_id == self._active_context_id or self._synthesis_lock.locked()
        if request_in_flight:
            self._interruption_restart_in_progress = True
            self._active_context_id = None
            try:
                await self._disconnect(clear_active_context=False)
                await self._connect()
            finally:
                self._interruption_restart_in_progress = False
                self._clear_active_context()
                self._bot_speaking = False
        await super().on_audio_context_interrupted(context_id)

    async def _receive_messages(self):
        try:
            async for message in self._get_websocket():
                if isinstance(message, bytes):
                    await self.stop_ttfb_metrics()
                    context_id = self._active_context_id
                    if not context_id:
                        logger.debug(f"{self}: received audio without an active context")
                        continue
                    frame = TTSAudioRawFrame(
                        audio=message,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                        context_id=context_id,
                    )
                    await self.append_to_audio_context(context_id, frame)
                    continue

                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    logger.error(f"{self}: invalid JSON from Rumik: {message}")
                    continue

                message_type = data.get("type")
                if message_type == "queued":
                    logger.debug(f"{self}: Rumik queued depth={data.get('queue_depth')}")
                elif message_type == "done":
                    logger.debug(
                        f"{self}: Rumik done audio={data.get('audio_duration')}s "
                        f"rtf={data.get('rtf')}"
                    )
                    await self._finish_active_context()
                elif message_type == "timeout":
                    logger.debug(f"{self}: Rumik idle timeout: {data.get('message')}")
                    self._disconnecting = True
                    await self._disconnect_websocket()
                    break
                elif message_type == "error" or data.get("error"):
                    await self._finish_active_context(error_msg=f"Rumik TTS error: {data}")
                else:
                    logger.debug(f"{self}: unknown Rumik message: {data}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if self._active_context_id:
                await self._finish_active_context(error_msg=f"Rumik WS receive error: {e}")
            raise
        else:
            if self._active_context_id:
                await self._finish_active_context(
                    error_msg="Rumik WebSocket closed before synthesis completed"
                )

    async def _finish_active_context(self, *, error_msg: str | None = None):
        if self._interruption_restart_in_progress:
            return

        context_id = self._active_context_id
        if not context_id:
            self._clear_active_context()
            return

        if error_msg:
            await self.append_to_audio_context(context_id, ErrorFrame(error=error_msg))
        await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
        await self.remove_audio_context(context_id)
        await self.stop_all_metrics()
        self._clear_active_context()

    def _clear_active_context(self):
        self._active_context_id = None
        if self._synthesis_lock.locked():
            self._synthesis_lock.release()

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Send text to Rumik and let the receive loop stream the audio."""
        text = re.sub(r"\s+", " ", text).strip()
        logger.debug(f"{self}: generating Rumik TTS [{text}]")

        if not text:
            yield TTSStoppedFrame(context_id=context_id)
            await self.remove_audio_context(context_id)
            return

        await self._synthesis_lock.acquire()
        self._active_context_id = context_id

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            await self._get_websocket().send(
                json.dumps(_build_synthesis_payload(self._settings, text, include_model=False))
            )
            await self.start_tts_usage_metrics(text)
            yield None
        except Exception as e:
            error_msg = f"Rumik send failed: {e}"
            yield ErrorFrame(error=error_msg)
            yield TTSStoppedFrame(context_id=context_id)
            await self.remove_audio_context(context_id)
            self._clear_active_context()
            await self._disconnect()


class RumikHttpTTSService(TTSService):
    """Rumik HTTP text-to-speech service."""

    Settings = RumikTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        gateway_url: str,
        aiohttp_session: aiohttp.ClientSession,
        settings: Settings | None = None,
        sample_rate: int | None = RUMIK_SAMPLE_RATE,
        **kwargs,
    ):
        """Initialize the Rumik HTTP TTS service.

        Args:
            api_key: Rumik API key.
            gateway_url: Rumik gateway base URL.
            aiohttp_session: Caller-owned HTTP session.
            settings: Runtime-updatable Rumik TTS settings.
            sample_rate: Output audio sample rate. Rumik currently returns 24 kHz PCM.
            **kwargs: Additional arguments passed to ``TTSService``.
        """
        default_settings = _default_settings()
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            sample_rate=_validate_sample_rate(sample_rate),
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._gateway_url = gateway_url.rstrip("/")
        self._session = aiohttp_session

    def can_generate_metrics(self) -> bool:
        """Return whether this service can generate Pipecat metrics."""
        return True

    def _payload(self, text: str) -> dict[str, Any]:
        return _build_synthesis_payload(self._settings, text, include_model=True)

    @staticmethod
    def _wav_to_pcm(wav_audio: bytes) -> tuple[bytes, int, int]:
        with wave.open(io.BytesIO(wav_audio), "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            compression = wav.getcomptype()
            pcm = wav.readframes(wav.getnframes())

        if sample_width != 2:
            raise ValueError(f"Expected 16-bit PCM WAV, got {sample_width * 8}-bit audio")
        if compression != "NONE":
            raise ValueError(f"Expected PCM WAV, got compression type {compression!r}")
        if sample_rate != RUMIK_SAMPLE_RATE:
            raise ValueError(f"Expected {RUMIK_SAMPLE_RATE} Hz WAV, got {sample_rate} Hz")
        if channels != 1:
            raise ValueError(f"Expected mono WAV, got {channels} channels")
        return pcm, sample_rate, channels

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text using Rumik's HTTP TTS API."""
        text = re.sub(r"\s+", " ", text).strip()
        logger.debug(f"{self}: generating Rumik HTTP TTS [{text}]")
        if not text:
            return

        url = f"{self._gateway_url}/v1/tts"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            await self.start_ttfb_metrics()

            async with self._session.post(url, json=self._payload(text), headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    await self.stop_all_metrics()
                    yield ErrorFrame(
                        error=f"Rumik HTTP TTS error: HTTP {response.status}: {error_text}"
                    )
                    return

                wav_audio = await response.read()
                await self.stop_ttfb_metrics()
                await self.start_tts_usage_metrics(text)

            pcm, sample_rate, channels = self._wav_to_pcm(wav_audio)
            chunk_size = self.chunk_size or int(sample_rate * 0.5 * 2)
            for offset in range(0, len(pcm), chunk_size):
                chunk = pcm[offset : offset + chunk_size]
                if chunk:
                    yield TTSAudioRawFrame(
                        audio=chunk,
                        sample_rate=sample_rate,
                        num_channels=channels,
                        context_id=context_id,
                    )
        except Exception as e:
            await self.stop_all_metrics()
            yield ErrorFrame(error=f"Rumik HTTP TTS error: {e}")
