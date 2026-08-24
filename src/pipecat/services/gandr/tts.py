#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gandr text-to-speech service for Pipecat.

Streams audio from Gandr's WebSocket surface at ``wss://tts.gandr.ai/ws``.

Wire protocol (documented at https://gandr.ai/docs):

* Open a WebSocket with an ``x-api-key`` (or ``Authorization: Bearer``) header.
* Send one JSON object per utterance. Fields: ``text``, ``lang``, ``voice_id``,
  ``output_sample_rate``, and the optional expression controls ``speed``,
  ``volume``, ``temperature``, ``cfg_weight``, ``seed``. A cloned voice is
  registered for the life of the connection by attaching ``voice_wav_b64`` to
  the first utterance.
* The server replies with **binary** frames of raw PCM16LE mono audio, streamed
  as it renders, followed by a JSON object that closes the utterance.
* Errors arrive as JSON of the shape ``{"error": "..."}``. ``need_voice`` means
  the connection has no registered voice yet; ``busy`` is soft backpressure.

The connection is long-lived and carries many utterances, but it renders one at
a time, so this service serialises sends: ``run_tts`` queues an utterance and a
single sender task delivers it, waits for the closing JSON, and only then
delivers the next. That is what keeps ``busy`` from ever being the normal case.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from pydantic import BaseModel, field_validator
from websockets.asyncio.client import ClientConnection
from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.services.gandr._text import MAX_REQUEST_CHARS, split_for_request
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, WebsocketTTSService
from pipecat.utils.tracing.service_decorators import traced_tts

#: Default streaming endpoint.
DEFAULT_WS_URL = "wss://tts.gandr.ai/ws"

#: Output rates the API renders at.
SAMPLE_RATES = (8000, 16000, 22050, 24000)

#: Frame ceiling for the receive side. A full-length utterance at 24 kHz PCM16
#: stays far below this.
MAX_FRAME_BYTES = 16 * 1024 * 1024

#: Stock voice identifiers. A cloned voice is a ``gnd:`` identifier instead.
STOCK_VOICES = (
    "gandr-mia",
    "gandr-ava",
    "gandr-jenny",
    "gandr-dane",
    "gandr-leo",
    "gandr-lewis",
)


@dataclass
class _Utterance:
    """One JSON message on the wire, and the state of its round trip."""

    text: str
    context_id: str
    is_final: bool
    done: asyncio.Event = field(default_factory=asyncio.Event)
    error: str | None = None
    needs_voice: bool = False
    busy: bool = False
    abandoned: bool = False
    ttfb_stopped: bool = False


class GandrTTSService(WebsocketTTSService):
    """Gandr WebSocket text-to-speech service.

    Holds one connection for the life of the pipeline and streams raw PCM16LE
    mono audio back as it renders. Voice, language and expression controls are
    set per utterance, so ``TTSUpdateSettingsFrame`` takes effect on the next
    thing the bot says.
    """

    class InputParams(BaseModel):
        """Input parameters for Gandr TTS configuration.

        Parameters:
            voice_id: Stock voice identifier (see ``STOCK_VOICES``) or a
                ``gnd:`` cloned-voice identifier. Defaults to ``"gandr-mia"``.
            language: ISO language code for the input text. Defaults to ``"en"``.
            sample_rate: Output rate in Hz. One of 8000, 16000, 22050, 24000.
                If set, takes priority over the service constructor's
                ``sample_rate``. Defaults to 24000, the API's default output
                rate; take any downsample locally rather than asking the server
                for a narrowband stream.
            speed: Playback speed, 0.6 to 1.5, pitch preserving.
            volume: Output gain, 0.5 to 2.0, soft-ceiling mastered.
            temperature: Expression control. Omit to let the API choose.
            cfg_weight: Expression control. Omit to send nothing at all.
            seed: Fixes the render for a reproducible result.
            voice_wav_b64: Base64 WAV of reference audio for a cloned voice.
                Sent with the first utterance on each connection, which
                registers the voice for that connection.
        """

        voice_id: str | None = "gandr-mia"
        language: str | None = "en"
        sample_rate: int | None = None
        speed: float | None = None
        volume: float | None = None
        temperature: float | None = None
        cfg_weight: float | None = None
        seed: int | None = None
        voice_wav_b64: str | None = None

        @field_validator("voice_id")
        @classmethod
        def validate_voice_id(cls, v: str | None) -> str | None:
            """Reject a blank voice id.

            An empty string is not the same as omitting the field: omitted
            means "use the service default", blank means somebody built the
            value from a variable that was not set. The door would answer
            bad_voice, but by then it is a runtime error in a live call.
            """
            if v is not None and not v.strip():
                raise ValueError("voice_id cannot be empty or whitespace")
            return v

        @field_validator("language")
        @classmethod
        def validate_language(cls, v: str | None) -> str | None:
            """Reject a blank language tag.

            Same reasoning as voice_id. Note the door takes BARE two-letter
            codes, en, es, fr, de, pt, ar, zh, ja, and ignores a region
            suffix, so "en-GB" silently renders as English rather than
            failing. That is not checked here because the accepted set is
            the service's to define, not this plugin's to freeze.
            """
            if v is not None and not v.strip():
                raise ValueError("language cannot be empty or whitespace")
            return v

        @field_validator("sample_rate")
        @classmethod
        def validate_sample_rate(cls, v: int | None) -> int | None:
            """Restrict to the rates the service actually serves.

            An unsupported rate is worth catching here rather than at the
            socket: a pipeline whose output rate disagrees with the rest of
            its chain produces audio that plays at the wrong pitch, which
            reads as a voice-quality problem rather than a config one.
            """
            if v is not None and v not in SAMPLE_RATES:
                raise ValueError(f"sample_rate must be one of {list(SAMPLE_RATES)}, got {v}")
            return v

        @field_validator("speed")
        @classmethod
        def validate_speed(cls, v: float | None) -> float | None:
            """Bound speed to 0.6-1.5.

            Outside that range the engine still renders, and the result is
            unusable rather than merely fast or slow. Bounding it at
            construction turns a bad-sounding call into an error a
            developer sees while writing the code.
            """
            if v is not None and not (0.6 <= v <= 1.5):
                raise ValueError(f"speed must be between 0.6 and 1.5, got {v}")
            return v

        @field_validator("volume")
        @classmethod
        def validate_volume(cls, v: float | None) -> float | None:
            """Bound volume to 0.5-2.0, for the same reason as speed.

            The upper end clips rather than getting louder, and clipping in
            a phone call is indistinguishable from a broken connection.
            """
            if v is not None and not (0.5 <= v <= 2.0):
                raise ValueError(f"volume must be between 0.5 and 2.0, got {v}")
            return v

    def __init__(
        self,
        *,
        api_key: str,
        url: str = DEFAULT_WS_URL,
        params: InputParams | None = None,
        text_aggregation_mode: TextAggregationMode | None = None,
        utterance_timeout_s: float = 30.0,
        busy_retry_s: float = 0.5,
        max_attempts: int = 3,
        reconnect_on_interruption: bool = True,
        **kwargs,
    ):
        """Initialize the Gandr TTS service.

        Args:
            api_key: Gandr API key (``gnd_…``).
            url: WebSocket endpoint. Leave as the default unless you have been
                given a different one.
            params: Voice, language and expression configuration.
            text_aggregation_mode: How to aggregate incoming text before
                synthesis.
            utterance_timeout_s: How long to wait for an utterance's closing
                frame before treating it as failed.
            busy_retry_s: How long to wait before retrying after the server
                answers ``busy``.
            max_attempts: Total attempts per utterance, including retries for
                ``busy`` and for voice registration.
            reconnect_on_interruption: On barge-in, drop and reopen the
                connection so the next turn's audio is not queued behind the
                audio the listener already interrupted. Turn this off if you
                would rather keep the connection and discard the tail of the
                interrupted render client-side, at the cost of the next turn
                waiting for that render to finish.
            **kwargs: Passed through to ``WebsocketTTSService``.

        Raises:
            ValueError: If ``api_key`` is empty or whitespace.
        """
        params = params or GandrTTSService.InputParams()

        # Sample rate precedence: InputParams.sample_rate > constructor
        # sample_rate > 24000.
        constructor_sample_rate = kwargs.pop("sample_rate", None)
        resolved_sample_rate = (
            params.sample_rate
            if params.sample_rate is not None
            else (constructor_sample_rate if constructor_sample_rate is not None else 24000)
        )

        default_settings = TTSSettings(
            model=None,
            voice=params.voice_id or "gandr-mia",
            language=None,
        )

        super().__init__(
            sample_rate=resolved_sample_rate,
            text_aggregation_mode=text_aggregation_mode,
            push_text_frames=True,
            push_start_frame=True,
            pause_frame_processing=False,
            settings=default_settings,
            **kwargs,
        )

        if not api_key or not api_key.strip():
            raise ValueError("Gandr API key is required and cannot be empty")

        self._api_key = api_key
        self._url = url
        self._utterance_timeout_s = utterance_timeout_s
        self._busy_retry_s = busy_retry_s
        self._max_attempts = max(1, max_attempts)
        self._reconnect_on_interruption = reconnect_on_interruption

        self._gandr_settings: dict[str, Any] = {
            "lang": params.language or "en",
            "output_sample_rate": resolved_sample_rate,
            "speed": params.speed,
            "volume": params.volume,
            "temperature": params.temperature,
            "cfg_weight": params.cfg_weight,
            "seed": params.seed,
        }
        self._voice_wav_b64 = params.voice_wav_b64

        self._websocket: ClientConnection | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._send_task: asyncio.Task[None] | None = None
        self._reconnect_task: asyncio.Task[None] | None = None
        self._outbox: asyncio.Queue[_Utterance] = asyncio.Queue()
        self._inflight: _Utterance | None = None
        self._voice_registered = False
        # run_tts, the send task and the interruption handler can all decide a
        # connection is needed at the same moment. This makes sure that only
        # ever opens one socket.
        self._connect_lock = asyncio.Lock()

    def can_generate_metrics(self) -> bool:
        """Check if this service can generate processing metrics.

        Returns:
            True. Time to first byte is stopped on the first audio byte off the
            wire, and usage metrics are reported per request.
        """
        return True

    # ── lifecycle ────────────────────────────────────────────────────────

    async def start(self, frame: StartFrame) -> None:
        """Start the Gandr TTS service.

        Args:
            frame: The start frame carrying negotiated pipeline settings.
        """
        await super().start(frame)
        self._gandr_settings["output_sample_rate"] = self.sample_rate
        await self._connect()

    async def stop(self, frame: EndFrame) -> None:
        """Stop the Gandr TTS service.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame) -> None:
        """Cancel the Gandr TTS service.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self) -> None:
        """Open the connection and start the send and receive tasks."""
        await super()._connect()

        await self._connect_websocket()

        # Both tasks start even if the connection did not come up: each one
        # reopens the connection itself when it needs it. Gating them on a
        # successful connect is what would leave a queued utterance waiting
        # forever on a sender that was never created.
        if not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))
        if not self._send_task:
            self._send_task = self.create_task(self._send_task_handler())

    async def _disconnect(self) -> None:
        """Stop the tasks and close the connection."""
        await super()._disconnect()

        current = asyncio.current_task()

        if self._send_task and self._send_task is not current:
            await self.cancel_task(self._send_task)
            self._send_task = None

        if self._receive_task and self._receive_task is not current:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        if self._reconnect_task and self._reconnect_task is not current:
            await self.cancel_task(self._reconnect_task)
            self._reconnect_task = None

        await self._disconnect_websocket()

    async def _reconnect(self) -> None:
        """Close the connection and open a fresh one."""
        await self._disconnect()
        await self._connect()

    async def _connect_websocket(self) -> None:
        """Open the Gandr websocket."""
        async with self._connect_lock:
            try:
                if self._websocket and self._websocket.state is State.OPEN:
                    return

                logger.debug("Connecting to Gandr")
                self._websocket = await websocket_connect(
                    self._url,
                    additional_headers={"x-api-key": self._api_key},
                    max_size=MAX_FRAME_BYTES,
                )
                # A cloned voice is registered per connection, so a fresh
                # connection has to register it again.
                self._voice_registered = False
                logger.debug("Connected to Gandr")

            except Exception as e:
                logger.error(f"{self} initialization error: {e}")
                self._websocket = None
                await self.push_error(error_msg=f"{self} connection error: {e}", exception=e)

    async def _disconnect_websocket(self) -> None:
        """Close the Gandr websocket."""
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Gandr")
                await self._websocket.close()
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")
        finally:
            await self.remove_active_audio_context()
            self._websocket = None
            self._inflight = None
            self._voice_registered = False

    def _get_websocket(self) -> ClientConnection:
        """Get the active connection.

        Returns:
            The open websocket.

        Raises:
            Exception: If the websocket is not connected.
        """
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _verify_connection(self) -> bool:
        """Verify the websocket is open and answering.

        Returns:
            True if a ping round-tripped, False otherwise.
        """
        try:
            if not self._websocket:
                return False
            await self._websocket.ping()
            return True
        except Exception as e:
            logger.error(f"{self} connection verification failed: {e}")
            return False

    # ── receiving ────────────────────────────────────────────────────────

    async def _receive_messages(self) -> None:
        """Receive from Gandr, reopening the connection if it ends."""
        while True:
            if self._websocket is None or self._websocket.state is not State.OPEN:
                await self._connect_websocket()
            if self._websocket is None:
                # The connection could not be opened. _connect_websocket has
                # already reported it; back off rather than spin.
                await asyncio.sleep(self._busy_retry_s)
                continue
            await self._process_messages()
            logger.debug(f"{self} websocket connection ended, reconnecting")

    async def _process_messages(self) -> None:
        """Drain frames from the connection until it closes."""
        async for message in self._get_websocket():
            try:
                if isinstance(message, (bytes, bytearray)):
                    await self._process_audio_frame(bytes(message))
                else:
                    await self._process_text_frame(message)
            except Exception as e:
                logger.error(f"{self} error processing message: {e}")
                await self.push_error(
                    error_msg=f"{self} error processing message: {e}", exception=e
                )

    async def _process_audio_frame(self, audio: bytes) -> None:
        """Push one binary frame of PCM16LE audio into its audio context.

        Args:
            audio: Raw PCM16LE mono bytes exactly as they came off the wire.
        """
        if not audio:
            return

        utterance = self._inflight
        if utterance is None:
            logger.warning(
                f"{self} received {len(audio)} audio bytes with no utterance in flight; discarding"
            )
            return

        if not utterance.ttfb_stopped:
            # The first audio byte, which is the number this service reports.
            utterance.ttfb_stopped = True
            await self.stop_ttfb_metrics()

        if utterance.abandoned:
            return
        if not self.audio_context_available(utterance.context_id):
            return

        frame = TTSAudioRawFrame(
            audio=audio,
            sample_rate=self.sample_rate,
            num_channels=1,
            context_id=utterance.context_id,
        )
        await self.append_to_audio_context(utterance.context_id, frame)

    async def _process_text_frame(self, message: str) -> None:
        """Handle a JSON frame: an error, or the close of an utterance.

        Args:
            message: The text frame received from Gandr.
        """
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.warning(f"{self} received a non-JSON text frame; ignoring")
            return
        if not isinstance(data, dict):
            logger.warning(f"{self} received a JSON frame that is not an object")
            return

        utterance = self._inflight

        if "error" in data:
            code = str(data["error"])
            if utterance is None:
                logger.error(f"{self} error with no utterance in flight: {code}")
                return
            if code == "need_voice":
                # The connection has no registered voice. The sender retries
                # with the reference audio attached.
                utterance.needs_voice = True
            elif code == "busy":
                # Soft backpressure. The sender waits and tries again.
                utterance.busy = True
            else:
                utterance.error = f"{self} error: {code}"
            utterance.done.set()
            return

        if "ttfa_ms" in data or "audio_ms" in data:
            # The utterance is complete. The server reports its own timings
            # here; this service does not consume or log them, so the only
            # latency it ever reports is the client-side one above.
            if utterance is None:
                return
            utterance.done.set()
            return

        logger.debug(f"{self} received an unrecognised frame: {sorted(data.keys())}")

    # ── sending ──────────────────────────────────────────────────────────

    def _language_code(self) -> str:
        """Resolve the language code to send, honouring a settings update.

        Returns:
            The ISO language code as a plain string.
        """
        language = getattr(self._settings, "language", None)
        if language is None:
            return str(self._gandr_settings["lang"])
        return str(getattr(language, "value", language))

    def _build_message(self, text: str, *, include_voice: bool) -> dict[str, Any]:
        """Build one utterance message for the Gandr websocket.

        Args:
            text: The transcript for this utterance.
            include_voice: Attach the reference audio that registers a cloned
                voice for this connection.

        Returns:
            The JSON-serialisable message body.
        """
        message: dict[str, Any] = {
            "text": text,
            "lang": self._language_code(),
            "voice_id": self._settings.voice,
            "output_sample_rate": self._gandr_settings["output_sample_rate"],
        }

        for key in ("speed", "volume", "temperature", "cfg_weight", "seed"):
            value = self._gandr_settings.get(key)
            if value is not None:
                message[key] = value

        if include_voice and self._voice_wav_b64:
            message["voice_wav_b64"] = self._voice_wav_b64

        # Deliberately logged without the message body: voice_wav_b64 is large,
        # and nothing here should print a key or a payload.
        logger.debug(
            f"{self} sending utterance: {len(text)} chars, "
            f"voice={message['voice_id']}, lang={message['lang']}, "
            f"reference_audio={'yes' if 'voice_wav_b64' in message else 'no'}"
        )
        return message

    async def _send_task_handler(self) -> None:
        """Deliver queued utterances one at a time, in order."""
        while True:
            utterance = await self._outbox.get()
            try:
                await self._deliver(utterance)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"{self} error delivering utterance: {e}")
                await self._fail(utterance, f"{self} error delivering utterance: {e}")
            finally:
                self._outbox.task_done()

    async def _deliver(self, utterance: _Utterance) -> None:
        """Send one utterance and wait for the server to close it.

        Retries once per attempt budget for ``busy`` backpressure and for voice
        registration, then reports whatever is left as an error.

        Args:
            utterance: The utterance to put on the wire.
        """
        if not self.audio_context_available(utterance.context_id):
            # Interrupted between run_tts and the wire. Nothing to send.
            return

        send_voice = self._voice_wav_b64 is not None and not self._voice_registered
        attempts = 0

        while True:
            attempts += 1

            if self._websocket is None or self._websocket.state is not State.OPEN:
                await self._connect_websocket()
                if self._websocket is None:
                    await self._fail(utterance, f"{self} could not open a connection")
                    return
                send_voice = self._voice_wav_b64 is not None

            utterance.done.clear()
            utterance.error = None
            utterance.needs_voice = False
            utterance.busy = False
            self._inflight = utterance

            try:
                await self._get_websocket().send(
                    json.dumps(self._build_message(utterance.text, include_voice=send_voice))
                )
            except Exception as e:
                self._inflight = None
                if attempts < self._max_attempts:
                    logger.warning(f"{self} send failed, reconnecting: {e}")
                    await self._disconnect_websocket()
                    continue
                await self._fail(utterance, f"{self} error sending utterance: {e}")
                return

            try:
                await asyncio.wait_for(utterance.done.wait(), timeout=self._utterance_timeout_s)
            except TimeoutError:
                utterance.error = (
                    f"{self} received no completion frame within {self._utterance_timeout_s}s"
                )

            self._inflight = None

            if utterance.abandoned:
                return

            if utterance.needs_voice:
                if self._voice_wav_b64 and attempts < self._max_attempts:
                    self._voice_registered = False
                    send_voice = True
                    continue
                await self._fail(
                    utterance,
                    f"{self} voice {self._settings.voice!r} is not registered on "
                    "this connection; pass reference audio as "
                    "InputParams(voice_wav_b64=...) to use a cloned voice",
                )
                return

            if utterance.busy:
                if attempts < self._max_attempts:
                    await asyncio.sleep(self._busy_retry_s)
                    continue
                await self._fail(
                    utterance,
                    f"{self} stayed busy across {attempts} attempts",
                )
                return

            if utterance.error:
                await self._fail(utterance, utterance.error)
                return

            break

        if send_voice and self._voice_wav_b64:
            # The utterance completed with the reference audio attached, so the
            # voice is registered for the rest of this connection.
            self._voice_registered = True

        if utterance.is_final:
            await self.stop_ttfb_metrics()
            await self.append_to_audio_context(
                utterance.context_id, TTSStoppedFrame(context_id=utterance.context_id)
            )
            await self.remove_audio_context(utterance.context_id)

    async def _fail(self, utterance: _Utterance, error_msg: str) -> None:
        """Close out an utterance that could not be rendered.

        Args:
            utterance: The utterance that failed.
            error_msg: What to log and surface upstream.
        """
        if utterance.abandoned:
            return
        logger.error(error_msg)
        await self.push_frame(TTSStoppedFrame(context_id=utterance.context_id))
        await self.stop_all_metrics()
        await self.push_error(error_msg=error_msg)
        self.reset_active_audio_context()
        if self.audio_context_available(utterance.context_id):
            await self.remove_audio_context(utterance.context_id)

    # ── interruption ─────────────────────────────────────────────────────

    def _drain_outbox(self) -> None:
        """Drop every utterance still waiting to go on the wire."""
        dropped = 0
        while True:
            try:
                self._outbox.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._outbox.task_done()
            dropped += 1
        if dropped:
            logger.debug(f"{self} dropped {dropped} queued utterance(s) on interruption")

    async def on_audio_context_interrupted(self, context_id: str) -> None:
        """Abandon the current turn when the bot is interrupted.

        The wire protocol has no cancel message, so audio already rendering is
        discarded client-side. By default the connection is then reopened, so
        the next turn's first audio byte does not queue behind audio nobody is
        listening to any more.

        Args:
            context_id: The audio context that was interrupted.
        """
        await self.stop_all_metrics()
        self._drain_outbox()

        utterance = self._inflight
        if utterance is None:
            return

        utterance.abandoned = True
        utterance.done.set()

        if not self._reconnect_on_interruption:
            return

        logger.debug(f"{self} reopening connection after interruption")
        current = asyncio.current_task()
        if current is self._send_task or current is self._receive_task:
            # Reopening from inside one of our own tasks would cancel the task
            # doing the reopening, so hand it to a fresh one.
            self._reconnect_task = self.create_task(self._reconnect())
        else:
            await self._reconnect()

    # ── synthesis ────────────────────────────────────────────────────────

    async def flush_audio(self, context_id: str | None = None) -> None:
        """No-op.

        Gandr closes each utterance itself with a completion frame, so there is
        no end-of-turn message for this service to send.

        Args:
            context_id: Unused; present to match the base-class signature.
        """
        return

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Generate speech from text over Gandr's streaming WebSocket API.

        Queues the utterance and returns. Audio frames reach the pipeline from
        the receive task, through the audio context named by ``context_id``.

        Args:
            text: The text to synthesize.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: An error frame and a stopped frame if the utterance could
            not be queued at all.
        """
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()

            if self._send_task is None:
                raise RuntimeError("sender task is not running")

            pieces = split_for_request(text)
            if not pieces:
                await self.append_to_audio_context(
                    context_id, TTSStoppedFrame(context_id=context_id)
                )
                await self.remove_audio_context(context_id)
                return

            await self.start_tts_usage_metrics(text)

            last = len(pieces) - 1
            for index, piece in enumerate(pieces):
                await self._outbox.put(
                    _Utterance(text=piece, context_id=context_id, is_final=index == last)
                )
        except Exception as e:
            logger.error(f"{self} error queueing utterance: {e}")
            yield ErrorFrame(error=f"{self} error queueing utterance: {e}")
            yield TTSStoppedFrame(context_id=context_id)
            if self.audio_context_available(context_id):
                await self.remove_audio_context(context_id)
            return
