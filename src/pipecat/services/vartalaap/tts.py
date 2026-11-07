#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Vartalaap text-to-speech service implementations.

Vartalaap (https://vartalaap.si) is a Hinglish and Indian-language TTS API. This module
provides a WebSocket service for conversational use and an HTTP service for simpler
pipelines. Both deliver 16-bit PCM, 24 kHz, mono, streamed from the first decoded frame.
"""

import json
import os
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, ClassVar

import aiohttp
from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService, WebsocketTTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven, is_given

VARTALAAP_API_KEY_ENV = "VARTALAAP_API_KEY"
VARTALAAP_BASE_URL = "https://api.vartaloop.ai"
VARTALAAP_WS_URL = "wss://api.vartaloop.ai/v1/tts/stream"
VARTALAAP_SAMPLE_RATE = 24000
VARTALAAP_DEFAULT_MODEL = "narad"
VARTALAAP_DEFAULT_VOICE = "saanvi"


@dataclass
class VartalaapTTSSettings(TTSSettings):
    """Settings for VartalaapTTSService and VartalaapHttpTTSService.

    Every field travels in the body of each utterance, so a runtime update applies
    from the next sentence on without reconnecting.

    Parameters:
        model: The model family; ``narad`` is the current one. Kept for metrics and
            future routing, the server picks the model from the voice.
        voice: Voice id (``saanvi``, ``sahil``, …; ``GET /v1/voices`` lists them). The voice
            decides the language and so the text normaliser.
        language: Not used: the voice decides the language. Always ``None``.
        expression: Delivery of the voice: ``neutral`` (default, sends nothing), ``excited``,
            ``sad`` or ``whisper``. A voice without that expression plays neutral.
        digits: How digits are read in Hinglish text: ``auto`` (default), ``hi`` (Hindi
            number words), ``mixed`` (codes and phones in English digit names, amounts in
            Hindi) or ``en`` (English number words with Indian grouping).
        speed: Speaking pace, 0.5 to 2.0, pitch preserved. ``None`` plays the voice's pace.
        temperature: Sampling noise, 0 to 2 (server default 0.3).
        seed: Fixed sampling seed for reproducible audio.
        postfx: The server's output chain (loudness to -16 LUFS, high-pass, limiter).
            ``None`` keeps the server default (on).
        profile: ``default`` or ``telephony`` (shaped for 8 kHz mu-law delivery).
        raw: Skip the server's text normaliser; the text is already normalised.
    """

    expression: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    digits: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    speed: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    temperature: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    seed: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    postfx: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    profile: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    raw: bool | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)

    _aliases: ClassVar[dict[str, str]] = {
        "voice_id": "voice",
        "style": "expression",
        "pace": "speed",
    }


def _default_settings() -> VartalaapTTSSettings:
    return VartalaapTTSSettings(
        model=VARTALAAP_DEFAULT_MODEL,
        voice=VARTALAAP_DEFAULT_VOICE,
        language=None,
        expression="neutral",
        digits="auto",
        speed=None,
        temperature=None,
        seed=None,
        postfx=None,
        profile=None,
        raw=False,
    )


def _resolve_api_key(api_key: str | None) -> str:
    key = api_key or os.environ.get(VARTALAAP_API_KEY_ENV)
    if not key:
        raise ValueError(
            f"Vartalaap API key is required: pass api_key= or set {VARTALAAP_API_KEY_ENV}"
        )
    return key


def _utterance_body(
    settings: VartalaapTTSSettings, *, text: str, request_id: str
) -> dict[str, Any]:
    """The JSON of one utterance, shared by the WebSocket and the HTTP service.

    Only values that differ from the server's defaults are sent, so the server's own
    default keeps applying to anything left unset.
    """
    body: dict[str, Any] = {
        "text": text,
        "voice": settings.voice,
        "format": "pcm16",
        "request_id": request_id,
    }
    if is_given(settings.expression) and settings.expression not in (None, "neutral"):
        body["style"] = settings.expression
    if is_given(settings.digits) and settings.digits is not None:
        body["digits"] = settings.digits
    if is_given(settings.speed) and settings.speed is not None:
        body["speed"] = settings.speed
    if is_given(settings.temperature) and settings.temperature is not None:
        body["temperature"] = settings.temperature
    if is_given(settings.seed) and settings.seed is not None:
        body["seed"] = settings.seed
    if is_given(settings.postfx) and settings.postfx is not None:
        body["postfx"] = settings.postfx
    if is_given(settings.profile) and settings.profile not in (None, "default"):
        body["profile"] = settings.profile
    if is_given(settings.raw) and settings.raw:
        body["raw"] = True
    return body


class VartalaapTTSService(WebsocketTTSService):
    """Text-to-speech over Vartalaap's WebSocket API (``WS /v1/tts/stream``).

    One connection per session. Every sentence of a turn is sent as its own utterance
    message with a ``request_id``; the server serves them in order on the connection and
    streams each one's audio back as binary frames, with ``start`` and ``end`` events
    around it, so the next sentence is being synthesised while the current one plays.
    Audio frames are tagged with the turn's context id from those events, so an
    interrupted turn's leftover audio is dropped rather than played.

    The server cancels an utterance only when the socket closes, so an interruption
    with utterances still in flight reopens the connection.
    """

    Settings = VartalaapTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        url: str = VARTALAAP_WS_URL,
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Vartalaap TTS service.

        Args:
            api_key: Vartalaap API key (``vt_live_…``). Read from ``VARTALAAP_API_KEY``
                when not given.
            url: The WebSocket endpoint.
            sample_rate: Output sample rate. The API produces 24 kHz; audio is resampled
                on the way out when the pipeline runs at another rate.
            settings: Runtime-updatable settings. Defaults to the ``saanvi`` voice, neutral
                expression and ``auto`` digits.
            **kwargs: Additional arguments passed to the parent class.
        """
        default_settings = _default_settings()
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = _resolve_api_key(api_key)
        self._url = url

        self._receive_task = None
        # request_id -> context_id of every utterance sent and not yet ended, in order
        self._utterances: dict[str, str] = {}
        # the utterance whose binary audio is arriving: (request_id, context_id)
        self._current: tuple[str, str] | None = None
        # contexts whose turn has ended: closed once their last utterance ends
        self._closing: set[str] = set()

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as the service reports time to first byte.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """The voice decides the language; there is no language parameter.

        Args:
            language: The language to convert.

        Returns:
            None.
        """
        return None

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def _connect(self):
        """Establish the websocket connection and start the receive task."""
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Close the websocket connection and clean up tasks."""
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Connect to the Vartalaap websocket with the key on the handshake."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Vartalaap")
            headers = {"Authorization": f"Bearer {self._api_key}"}
            self._websocket = await self._websocket_connect(self._url, additional_headers=headers)
            # a new connection carries none of the old one's utterances
            self._utterances.clear()
            self._current = None
            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Error connecting: {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        """Close the websocket connection and reset state."""
        try:
            await self.stop_all_metrics()
            if self._websocket:
                logger.debug("Disconnecting from Vartalaap")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error disconnecting: {e}", exception=e)
        finally:
            self._websocket = None
            self._utterances.clear()
            self._current = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        """Get the active websocket connection or raise."""
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    def _build_msg(self, text: str, request_id: str) -> dict[str, Any]:
        """Build the JSON message of one utterance from the current settings."""
        return _utterance_body(self._settings, text=text, request_id=request_id)

    async def flush_audio(self, context_id: str | None = None):
        """Mark the turn's context complete: it closes once its last utterance ends.

        Args:
            context_id: The context to close. Falls back to the active context.
        """
        ctx = context_id or self.get_active_audio_context_id()
        if not ctx:
            return
        self._closing.add(ctx)
        await self._maybe_close_context(ctx)

    async def _maybe_close_context(self, context_id: str):
        """Close a completed context once nothing of it is still in flight."""
        if context_id not in self._closing:
            return
        if any(c == context_id for c in self._utterances.values()):
            return
        if self._current and self._current[1] == context_id:
            return
        self._closing.discard(context_id)
        if self.audio_context_available(context_id):
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)

    async def on_audio_context_interrupted(self, context_id: str):
        """Drop the connection when an interruption leaves utterances in flight.

        The server stops an utterance only when its socket closes, and queued ones are
        dropped with it, so this is how the interrupted speech is cancelled. A new
        connection is opened right away for the next turn.
        """
        self._closing.discard(context_id)
        in_flight = context_id in self._utterances.values() or (
            self._current is not None and self._current[1] == context_id
        )
        await super().on_audio_context_interrupted(context_id)
        if in_flight:
            await self._disconnect()
            await self._connect()

    async def _receive_messages(self):
        """Process incoming websocket messages: JSON events and binary PCM frames."""
        async for message in self._get_websocket():
            if isinstance(message, bytes):
                if not self._current:
                    # audio of an utterance that is not ours (left over after a reconnect)
                    continue
                _, context_id = self._current
                if not self.audio_context_available(context_id):
                    continue
                audio = message
                if self.sample_rate != VARTALAAP_SAMPLE_RATE:
                    audio = await self._resampler.resample(
                        audio, VARTALAAP_SAMPLE_RATE, self.sample_rate
                    )
                frame = TTSAudioRawFrame(
                    audio=audio,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=context_id,
                )
                await self.append_to_audio_context(context_id, frame)
                continue

            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"{self} received a non-JSON text message")
                continue

            msg_type = msg.get("type")
            request_id = msg.get("request_id")

            if msg_type == "error":
                status = msg.get("status")
                error = msg.get("error", "unknown error")
                if request_id in self._utterances:
                    context_id = self._utterances.pop(request_id)
                    if self._current and self._current[0] == request_id:
                        self._current = None
                    await self.stop_all_metrics()
                    await self.push_error(error_msg=f"Vartalaap error ({status}): {error}")
                    await self._maybe_close_context(context_id)
                elif request_id is None:
                    # untagged: the server could not read our message at all
                    await self.push_error(error_msg=f"Vartalaap error ({status}): {error}")
                else:
                    # a late error of an utterance that is not ours (before a reconnect)
                    logger.debug(
                        f"{self} ignoring error of unknown utterance {request_id}: {error}"
                    )
                continue

            if request_id not in self._utterances:
                continue  # an event of an utterance that is not ours

            if msg_type == "start":
                self._current = (request_id, self._utterances[request_id])
            elif msg_type == "first_audio_ms":
                await self.stop_ttfb_metrics()
            elif msg_type == "end":
                context_id = self._utterances.pop(request_id)
                self._current = None
                await self._maybe_close_context(context_id)
            # chunk, chunk_end: progress only

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Send one utterance; its audio arrives on the receive loop.

        Args:
            text: The text to convert to speech.
            context_id: Unique identifier for this TTS context (the turn).

        Yields:
            Frame: TTSStartedFrame on the first utterance of a context, then None.
        """
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            try:
                if not self.audio_context_available(context_id):
                    await self.create_audio_context(context_id)
                    await self.start_ttfb_metrics()
                    yield TTSStartedFrame(context_id=context_id)

                request_id = uuid.uuid4().hex
                msg = self._build_msg(text=text, request_id=request_id)
                self._utterances[request_id] = context_id
                await self._get_websocket().send(json.dumps(msg))
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


class VartalaapHttpTTSService(TTSService):
    """Text-to-speech over Vartalaap's HTTP API (``POST /v1/tts``).

    One request per sentence, the chunked PCM body forwarded as it arrives. Simpler than
    the WebSocket service and fine for pipelines without interruptions; a conversational
    agent should prefer :class:`VartalaapTTSService`.
    """

    Settings = VartalaapTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        aiohttp_session: aiohttp.ClientSession | None = None,
        base_url: str = VARTALAAP_BASE_URL,
        sample_rate: int | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Vartalaap HTTP TTS service.

        Args:
            api_key: Vartalaap API key (``vt_live_…``). Read from ``VARTALAAP_API_KEY``
                when not given.
            aiohttp_session: Shared aiohttp session for HTTP requests. One is created
                when not given.
            base_url: The API base URL.
            sample_rate: Output sample rate; audio is resampled from 24 kHz when needed.
            settings: Runtime-updatable settings.
            **kwargs: Additional arguments passed to the parent class.
        """
        default_settings = _default_settings()
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = _resolve_api_key(api_key)
        self._base_url = base_url.rstrip("/")
        self._session = aiohttp_session
        self._owns_session = aiohttp_session is None

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True, as the service reports time to first byte.
        """
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """The voice decides the language; there is no language parameter.

        Args:
            language: The language to convert.

        Returns:
            None.
        """
        return None

    async def cleanup(self):
        """Release the HTTP session the service created."""
        await super().cleanup()
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using the HTTP API.

        Args:
            text: The text to convert to speech.
            context_id: Unique identifier for this TTS context.

        Yields:
            Frame: Audio frames as the chunked response arrives.
        """
        request_id = uuid.uuid4().hex
        payload = _utterance_body(self._settings, text=text, request_id=request_id)
        payload["stream"] = True
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            await self.start_ttfb_metrics()

            async with self._ensure_session().post(
                f"{self._base_url}/v1/tts", json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"{self} error {response.status}: {error_text}")
                    yield ErrorFrame(error=f"Vartalaap error ({response.status}): {error_text}")
                    return

                await self.start_tts_usage_metrics(text)

                async for frame in self._stream_audio_frames_from_iterator(
                    response.content.iter_any(),
                    in_sample_rate=VARTALAAP_SAMPLE_RATE,
                    context_id=context_id,
                ):
                    await self.stop_ttfb_metrics()
                    yield frame
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"Vartalaap error: {e}")
        finally:
            await self.stop_all_metrics()
