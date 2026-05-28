#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""SLNG speech-to-text services."""

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import NOT_GIVEN, STTSettings, _NotGiven, is_given
from pipecat.services.stt_service import STTService, WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use SLNG STT, you need to `pip install pipecat-ai[slng]`.")
    raise Exception(f"Missing module: {e}")


@dataclass
class SlngSTTSettings(STTSettings):
    """Settings for SlngSTTService.

    Parameters:
        language: Language for speech recognition.
        enable_vad: Whether to enable server-side VAD.
        enable_partials: Whether to receive partial (interim) transcriptions.
    """

    enable_vad: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    enable_partials: bool | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class SlngSTTService(WebsocketSTTService):
    """Speech-to-text service using the SLNG bridge WebSocket API.

    Provides real-time speech transcription through a WebSocket connection
    to the SLNG STT bridge:

    - Audio is sent as raw binary WebSocket frames (no JSON wrapping).
    - Connection-level config (``sample_rate``, ``encoding``, ``language``,
      ``enable_partials``, ``enable_vad``) is supplied via URL query parameters.
    - Control messages are JSON text frames with a PascalCase ``type`` field:
      ``KeepAlive``, ``Finalize``, ``CloseStream``.
    - Transcription results arrive as ``Results`` messages
      (``channel.alternatives[0].transcript``, ``is_final``, ``from_finalize``).
    """

    Settings = SlngSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "slng/deepgram/nova:3-en",
        base_url: str = "api.slng.ai",
        encoding: str = "linear16",
        sample_rate: int | None = None,
        region_override: str | None = None,
        world_part_override: str | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize SlngSTTService.

        Args:
            api_key: Authentication key for the SLNG API.
            model: The transcription model to use. Defaults to "slng/deepgram/nova:3-en".
            base_url: The API host (without scheme). Defaults to "api.slng.ai".
            encoding: Audio encoding format. Defaults to "linear16".
            sample_rate: Audio sample rate in Hz. If None, uses the pipeline sample rate.
            region_override: Pin requests to a specific datacenter. One of
                ``"ap-southeast-2"``, ``"eu-north-1"``, ``"us-east-1"``. Sets the
                ``X-Region-Override`` header (takes precedence over ``world_part_override``).
            world_part_override: Constrain routing to a broad geographic zone.
                One of ``"ap"``, ``"eu"``, ``"na"``. Sets the ``X-World-Part-Override``
                header.
            settings: Runtime-updatable settings override.
            **kwargs: Additional arguments passed to parent WebsocketSTTService.
        """
        default_settings = self.Settings(
            model=model,
            language=Language.EN,
            enable_vad=True,
            enable_partials=True,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            keepalive_timeout=120,
            keepalive_interval=30,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url
        self._encoding = encoding
        self._receive_task = None
        self._region_override = region_override
        self._world_part_override = world_part_override
        # Signalled by the receive task when the server emits a ``ready`` message
        # acknowledging the init message. Audio sends block on this event so that
        # binary frames cannot race ahead of the init handshake on reconnect.
        self._ready_event = asyncio.Event()
        self._ready_timeout = 5.0

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate processing metrics.

        Returns:
            True, indicating metrics are supported.
        """
        return True

    async def start(self, frame: StartFrame):
        """Start the STT service and establish the WebSocket connection.

        Args:
            frame: Frame indicating service should start.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the STT service and close the WebSocket connection.

        Args:
            frame: Frame indicating service should stop.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the STT service and close the WebSocket connection.

        Args:
            frame: Frame indicating service should be cancelled.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle speech events.

        Args:
            frame: The frame to process.
            direction: Direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self.start_processing_metrics()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._websocket and self._websocket.state is State.OPEN:
                await self._websocket.send(json.dumps({"type": "finalize"}))

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Process audio data for speech-to-text transcription.

        Sends raw PCM audio bytes as a binary WebSocket frame.

        Args:
            audio: Raw PCM audio bytes to transcribe.

        Yields:
            None — transcription results are delivered via WebSocket responses.
        """
        if not self._websocket or self._websocket.state is not State.OPEN:
            await self._connect()

        if self._websocket is None:
            logger.warning(f"{self}: websocket unavailable after reconnect, dropping audio")
            yield None
            return

        # Block until the server has acknowledged the init handshake. Without
        # this gate, audio frames buffered by the pipeline (or replayed by the
        # base class on reconnect) can hit the wire before the init message and
        # the server closes the stream with policy violation 1008.
        if not self._ready_event.is_set():
            try:
                await asyncio.wait_for(self._ready_event.wait(), timeout=self._ready_timeout)
            except TimeoutError:
                logger.warning(f"{self}: init ack timed out, sending audio anyway")

        try:
            await self._websocket.send(audio)
        except Exception as e:
            logger.warning(f"{self}: send failed: {e}")
        yield None

    async def _send_keepalive(self, silence: bytes):
        """Send a ``KeepAlive`` JSON control frame.

        Args:
            silence: Silent PCM bytes (ignored; a ``KeepAlive`` JSON frame is sent instead).
        """
        if self._websocket is None:
            return
        try:
            await self._websocket.send(json.dumps({"type": "keepalive"}))
        except Exception as e:
            logger.warning(f"{self}: keepalive send failed: {e}")

    async def _connect(self):
        await self._connect_websocket()
        await super()._connect()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    def _build_config(self) -> dict[str, Any]:
        """Build the Configure-message body from the current settings."""
        config: dict[str, Any] = {
            "sample_rate": self.sample_rate,
            "encoding": self._encoding,
        }

        if is_given(self._settings.language) and self._settings.language is not None:
            config["language"] = str(self._settings.language)

        if is_given(self._settings.enable_vad):
            config["enable_vad"] = bool(self._settings.enable_vad)

        if is_given(self._settings.enable_partials):
            config["enable_partials"] = bool(self._settings.enable_partials)

        return config

    async def _connect_websocket(self):
        """Establish the WebSocket connection and send the initial ``Configure``.

        The SLNG bridge requires a ``Configure`` text message before any audio
        bytes are accepted; otherwise the server closes with policy violation
        ``1008``.
        """
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            logger.debug(f"Connecting to SLNG STT ({self._settings.model})")

            model = self._settings.model or "slng/deepgram/nova:3-en"
            # Model may contain slashes (e.g. "slng/deepgram/nova:3-en") that are
            # part of the path; encode any other reserved chars with quote(safe="/").
            model_path = quote(model, safe="/:")
            if "://" in self._base_url:
                ws_url = f"{self._base_url}/v1/bridges/unmute/stt/{model_path}"
            else:
                ws_url = f"wss://{self._base_url}/v1/bridges/unmute/stt/{model_path}"

            headers: dict[str, str] = {"Authorization": f"Bearer {self._api_key}"}
            if self._region_override:
                headers["X-Region-Override"] = self._region_override
            if self._world_part_override:
                headers["X-World-Part-Override"] = self._world_part_override

            self._ready_event.clear()
            self._websocket = await websocket_connect(ws_url, additional_headers=headers)

            config = self._build_config()
            # Init message uses a distinct ``init`` tag (separate from the
            # post-init variant enum CloseStream/Configure/Sync/KeepAlive/
            # Finalize). Server emits a ``ready`` message once it has been
            # accepted; ``run_stt`` blocks on ``_ready_event`` until then.
            await self._websocket.send(json.dumps({"type": "init", "config": config}))

            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            await self.push_error(error_msg=f"Unable to connect to SLNG STT: {e}", exception=e)

    async def _disconnect_websocket(self):
        """Send a ``CloseStream`` message and shut down the WebSocket."""
        ws = self._websocket
        try:
            if ws and ws.state is State.OPEN:
                logger.debug("Disconnecting from SLNG STT")
                await ws.send(json.dumps({"type": "close"}))
                await ws.close()
        except Exception as e:
            await self.push_error(error_msg=f"Error closing SLNG STT websocket: {e}", exception=e)
        finally:
            if self._websocket is ws:
                self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("SLNG STT websocket not connected")

    async def _receive_messages(self):
        """Receive and dispatch incoming WebSocket messages."""
        async for message in self._get_websocket():
            if isinstance(message, bytes):
                # Server should not push binary frames for STT; ignore.
                continue
            try:
                data = json.loads(message)
                await self._process_message(data)
            except json.JSONDecodeError:
                logger.warning(f"{self}: received non-JSON message: {message}")
            except Exception as e:
                logger.error(f"{self}: error processing message: {e}")

    async def _process_message(self, data: dict[str, Any]):
        """Dispatch a decoded server message.

        Handles messages emitted by the SLNG bridge, case-insensitively. The
        bridge typically emits ``Results`` (transcription) and ``Metadata``
        text frames, plus ``error`` payloads in either case.

        Args:
            data: Decoded JSON payload from the server.
        """
        msg_type = data.get("type") or ""
        type_lc = msg_type.lower() if isinstance(msg_type, str) else ""

        if type_lc == "ready":
            session_id = data.get("session_id", "")
            logger.debug(f"{self}: SLNG STT session ready (id={session_id})")
            self._ready_event.set()

        elif type_lc == "partial_transcript":
            await self._handle_transcript(data, is_final=False)

        elif type_lc == "final_transcript":
            await self._handle_transcript(data, is_final=True)

        elif type_lc == "utterance_end":
            logger.trace(f"{self}: SLNG STT utterance_end: {data}")

        elif type_lc == "error":
            err = data.get("data") if isinstance(data.get("data"), dict) else {}
            error_msg = (
                err.get("message")
                or data.get("message")
                or err.get("code")
                or data.get("code")
                or f"Unknown SLNG STT error (payload: {data})"
            )
            logger.error(f"{self}: SLNG STT error: {error_msg}")
            await self.push_error(error_msg=str(error_msg))
            await self.stop_all_metrics()

        else:
            logger.debug(f"{self}: unknown message: {data}")

    async def _handle_transcript(self, data: dict[str, Any], *, is_final: bool):
        """Handle a ``partial_transcript`` or ``final_transcript`` message.

        Per the Unmute bridge spec, ``transcript`` is at the top level. We also
        fall back to ``channel.alternatives[0].transcript`` because some
        upstream providers (e.g. Deepgram) include the full Deepgram payload
        passed through.
        """
        transcript = (data.get("transcript") or "").strip()
        if not transcript:
            channel = data.get("channel") or {}
            alternatives = channel.get("alternatives") or []
            if alternatives:
                transcript = (alternatives[0].get("transcript") or "").strip()
        if not transcript:
            return

        language: Language | None = None
        if raw_lang := data.get("language"):
            try:
                language = Language(raw_lang)
            except ValueError:
                pass

        if is_final:
            if data.get("from_finalize"):
                self.confirm_finalize()
            await self.push_frame(
                TranscriptionFrame(
                    transcript,
                    self._user_id,
                    time_now_iso8601(),
                    language,
                    result=data,
                )
            )
            await self._handle_transcription(transcript, True, language)
            await self.stop_processing_metrics()
        else:
            await self.push_frame(
                InterimTranscriptionFrame(
                    transcript,
                    self._user_id,
                    time_now_iso8601(),
                    language,
                    result=data,
                )
            )

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing.

        Args:
            transcript: The transcribed text.
            is_final: Whether this is a final (not interim) transcription.
            language: Detected or configured language.
        """
        pass

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta and reconnect if needed.

        Args:
            delta: A settings delta to apply.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)
        if not changed:
            return changed
        await self._request_reconnect()
        return changed


# ---------------------------------------------------------------------------
# HTTP batch STT (voiceai-sdk)
# ---------------------------------------------------------------------------

try:
    from voiceai_sdk import AsyncSlng
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use SLNG, you need to `pip install pipecat-ai[slng]`.")
    raise Exception(f"Missing module: {e}")


class SlngHttpSTTService(STTService):
    """Speech-to-text service using the SLNG HTTP API via voiceai-sdk.

    Buffers audio during a user's speech turn and submits it as a single
    batch request when ``VADUserStoppedSpeakingFrame`` is received.  This
    avoids the overhead of a persistent WebSocket connection and is suited
    for pipelines where low-latency streaming transcription is not required.

    The voiceai-sdk ``AsyncSlng`` client is created at construction time and
    can be replaced via ``stt._client`` for testing.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "slng/deepgram/nova:3-en",
        sample_rate: int | None = None,
        region_override: str | None = None,
        world_part_override: str | None = None,
        settings: STTSettings | None = None,
        **kwargs,
    ):
        """Initialize SlngHttpSTTService.

        Args:
            api_key: Authentication key for the SLNG API.
            model: The transcription model to use.  Defaults to
                ``"slng/deepgram/nova:3-en"``.
            sample_rate: Audio sample rate in Hz.  If ``None``, the value is
                taken from the pipeline ``StartFrame``.
            region_override: Pin requests to a specific datacenter. One of
                ``"ap-southeast-2"``, ``"eu-north-1"``, ``"us-east-1"``. Sets the
                ``X-Region-Override`` header (takes precedence over
                ``world_part_override``).
            world_part_override: Constrain routing to a broad geographic zone.
                One of ``"ap"``, ``"eu"``, ``"na"``. Sets the
                ``X-World-Part-Override`` header.
            settings: Runtime-updatable settings override.
            **kwargs: Additional arguments forwarded to the parent
                ``STTService``.
        """
        default_settings = STTSettings(
            model=model,
            language=Language.EN,
        )

        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            settings=default_settings,
            **kwargs,
        )

        routing_headers: dict[str, str] = {}
        if region_override:
            routing_headers["X-Region-Override"] = region_override
        if world_part_override:
            routing_headers["X-World-Part-Override"] = world_part_override

        self._client = AsyncSlng(api_key=api_key, default_headers=routing_headers)
        self._audio_buffer: list[bytes] = []

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate processing metrics.

        Returns:
            True, indicating metrics are supported.
        """
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Buffer audio data for deferred batch transcription.

        Audio is accumulated in ``_audio_buffer`` and transcribed only when
        ``VADUserStoppedSpeakingFrame`` arrives.  No frame is yielded here.

        Args:
            audio: Raw PCM audio bytes to buffer.

        Yields:
            None — transcription results are delivered via
            :meth:`_transcribe_buffer`.
        """
        self._audio_buffer.append(audio)
        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process pipeline frames and trigger transcription on VAD stop.

        Calls ``start_processing_metrics`` when the user starts speaking and
        invokes :meth:`_transcribe_buffer` when they stop.

        Args:
            frame: The frame to process.
            direction: Direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self.start_processing_metrics()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._transcribe_buffer()

    async def _transcribe_buffer(self):
        """Submit the accumulated audio buffer to the SLNG HTTP STT API.

        If the buffer is empty the call is skipped entirely.  On success a
        :class:`~pipecat.frames.frames.TranscriptionFrame` is pushed
        downstream and processing metrics are stopped.  On failure the error
        is logged, pushed upstream via ``push_error``, and all metrics are
        stopped.
        """
        if not self._audio_buffer:
            await self.stop_processing_metrics()
            return

        audio_data = b"".join(self._audio_buffer)
        self._audio_buffer = []

        language_str: str | None = None
        if is_given(self._settings.language) and self._settings.language is not None:
            language_str = str(self._settings.language)

        language_value: Language | None = None
        if is_given(self._settings.language) and self._settings.language is not None:
            try:
                language_value = Language(str(self._settings.language))
            except ValueError:
                pass

        model = self._settings.model or "slng/deepgram/nova:3-en"

        try:
            response = await self._client.speech_to_text.create(
                model,
                audio=audio_data,
                language=language_str,
                enable_partials=False,
                sample_rate=self.sample_rate,
                encoding="linear16",
            )

            if response.alternatives:
                transcript = response.alternatives[0].transcript or ""
                if transcript:
                    await self.push_frame(
                        TranscriptionFrame(
                            transcript,
                            self._user_id,
                            time_now_iso8601(),
                            language_value,
                            result=response,
                        )
                    )
                    await self._handle_transcription(transcript, True, language_value)
            await self.stop_processing_metrics()

        except Exception as e:
            logger.error(f"{self}: SLNG HTTP STT error: {e}")
            await self.push_error(error_msg=f"SLNG HTTP STT error: {e}", exception=e)
            await self.stop_all_metrics()

    @traced_stt
    async def _handle_transcription(
        self, transcript: str, is_final: bool, language: Language | None = None
    ):
        """Handle a transcription result with tracing.

        Args:
            transcript: The transcribed text.
            is_final: Whether this is a final (not interim) transcription.
            language: Detected or configured language.
        """
        pass
