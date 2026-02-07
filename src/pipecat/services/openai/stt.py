#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""OpenAI Speech-to-Text service implementations.

Provides two STT services:

- ``OpenAISTTService``: REST-based transcription using the Audio API
  (Whisper / GPT-4o).
- ``OpenAIRealtimeSTTService``: WebSocket-based streaming transcription
  using the Realtime API in transcription-only mode.
"""

import base64
import json
from typing import AsyncGenerator, Literal, Optional, Union

from loguru import logger

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_latency import OPENAI_REALTIME_TTFS_P99, OPENAI_TTFS_P99
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.services.whisper.base_stt import BaseWhisperSTTService, Transcription
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt

try:
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.protocol import State
except ModuleNotFoundError:
    websocket_connect = None
    State = None


class OpenAISTTService(BaseWhisperSTTService):
    """OpenAI Speech-to-Text service that generates text from audio.

    Uses OpenAI's transcription API to convert audio to text. Requires an OpenAI API key
    set via the api_key parameter or OPENAI_API_KEY environment variable.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o-transcribe",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        language: Optional[Language] = Language.EN,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        ttfs_p99_latency: Optional[float] = OPENAI_TTFS_P99,
        **kwargs,
    ):
        """Initialize OpenAI STT service.

        Args:
            model: Model to use — either gpt-4o or Whisper. Defaults to "gpt-4o-transcribe".
            api_key: OpenAI API key. Defaults to None.
            base_url: API base URL. Defaults to None.
            language: Language of the audio input. Defaults to English.
            prompt: Optional text to guide the model's style or continue a previous segment.
            temperature: Optional sampling temperature between 0 and 1. Defaults to 0.0.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to BaseWhisperSTTService.
        """
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            language=language,
            prompt=prompt,
            temperature=temperature,
            ttfs_p99_latency=ttfs_p99_latency,
            **kwargs,
        )

    async def _transcribe(self, audio: bytes) -> Transcription:
        assert self._language is not None  # Assigned in the BaseWhisperSTTService class

        # Build kwargs dict with only set parameters
        kwargs = {
            "file": ("audio.wav", audio, "audio/wav"),
            "model": self.model_name,
            "language": self._language,
        }

        if self._include_prob_metrics:
            # GPT-4o-transcribe models only support logprobs (not verbose_json)
            if self.model_name in ("gpt-4o-transcribe", "gpt-4o-mini-transcribe"):
                kwargs["response_format"] = "json"
                kwargs["include"] = ["logprobs"]
            else:
                # Whisper models support verbose_json
                kwargs["response_format"] = "verbose_json"

        if self._prompt is not None:
            kwargs["prompt"] = self._prompt

        if self._temperature is not None:
            kwargs["temperature"] = self._temperature

        return await self._client.audio.transcriptions.create(**kwargs)


_OPENAI_SAMPLE_RATE = 24000


class OpenAIRealtimeSTTService(WebsocketSTTService):
    """OpenAI Realtime Speech-to-Text service using WebSocket transcription sessions.

    Uses OpenAI's Realtime API in transcription-only mode for real-time streaming
    speech recognition with optional server-side VAD and noise reduction. The model
    does not generate conversational responses — only transcription output.

    This service supports two VAD modes:

    **Local VAD** (default): Disable server-side VAD and use
    a local VAD processor in the pipeline instead. When a
    ``VADUserStoppedSpeakingFrame`` is received, the service commits the
    audio buffer so that the server begins transcription for the completed
    speech segment.

    **Server-side VAD** (``turn_detection=None``): The OpenAI server performs voice-activity
    detection. The service broadcasts ``UserStartedSpeakingFrame`` and
    ``UserStoppedSpeakingFrame`` when the server detects speech boundaries.
    Do **not** use a separate VAD processor in the pipeline in this mode.

    Audio is sent as 24 kHz 16-bit mono PCM as required by the OpenAI Realtime
    API. If the pipeline runs at a different sample rate (e.g. 16 kHz for Silero
    VAD compatibility), audio is automatically upsampled before sending.

    Example::

        stt = OpenAIRealtimeSTTService(
            api_key="sk-...",
            model="gpt-4o-transcribe",
            noise_reduction="near_field",
        )
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-4o-transcribe",
        base_url: str = "wss://api.openai.com/v1/realtime",
        language: Optional[Language] = Language.EN,
        prompt: Optional[str] = None,
        turn_detection: Optional[Union[dict, Literal[False]]] = False,
        noise_reduction: Optional[Literal["near_field", "far_field"]] = None,
        should_interrupt: bool = True,
        ttfs_p99_latency: Optional[float] = OPENAI_REALTIME_TTFS_P99,
        **kwargs,
    ):
        """Initialize the OpenAI Realtime STT service.

        Args:
            api_key: OpenAI API key for authentication.
            model: Transcription model. Supported values are
                ``"gpt-4o-transcribe"`` and ``"gpt-4o-mini-transcribe"``.
                Defaults to ``"gpt-4o-transcribe"``.
            base_url: WebSocket base URL for the Realtime API.
                Defaults to ``"wss://api.openai.com/v1/realtime"``.
            language: Language of the audio input. Defaults to English.
            prompt: Optional prompt text to guide transcription style
                or provide keyword hints.
            turn_detection: Server-side VAD configuration. Defaults to
                ``False`` (disabled), which relies on a local VAD
                processor in the pipeline. Pass ``None`` to use server
                defaults (``server_vad``), or a dict with custom
                settings (e.g. ``{"type": "server_vad", "threshold": 0.5}``).
            noise_reduction: Noise reduction mode. ``"near_field"`` for
                close microphones, ``"far_field"`` for distant
                microphones, or ``None`` to disable.
            should_interrupt: Whether to interrupt bot output when
                speech is detected by server-side VAD. Only applies when
                turn detection is enabled. Defaults to True.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to parent
                WebsocketSTTService.
        """
        if websocket_connect is None:
            raise ImportError(
                "websockets is required for OpenAIRealtimeSTTService. "
                "Install it with: pip install pipecat-ai[openai]"
            )

        super().__init__(
            ttfs_p99_latency=ttfs_p99_latency,
            **kwargs,
        )

        self._api_key = api_key
        self._base_url = base_url
        self.set_model_name(model)

        self._language_code = self._language_to_code(language) if language else None
        self._prompt = prompt
        self._turn_detection = turn_detection
        self._noise_reduction = noise_reduction
        self._should_interrupt = should_interrupt

        self._receive_task = None
        self._session_ready = False
        self._resampler = create_stream_resampler()

        # Server-side VAD is disabled by default (turn_detection=False).
        # Set to None or a dict to enable server-side VAD.
        self._server_vad_enabled = turn_detection is not False

    @staticmethod
    def _language_to_code(language: Language) -> str:
        """Convert a Language enum value to an ISO-639-1 code.

        Args:
            language: The Language enum value.

        Returns:
            Two-letter ISO-639-1 language code.
        """
        # Language.value is e.g. "en", "en-US", "fr", "zh".
        return language.value.split("-")[0].lower()

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate processing metrics.

        Returns:
            True, as this service supports metrics generation.
        """
        return True

    async def set_language(self, language: Language):
        """Set the language for speech recognition.

        If the session is already active, sends an updated configuration
        to the server.

        Args:
            language: The language to use for speech recognition.
        """
        self._language_code = self._language_to_code(language)
        if self._session_ready:
            await self._send_session_update()

    async def start(self, frame: StartFrame):
        """Start the service and establish WebSocket connection.

        Args:
            frame: The start frame triggering service initialization.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service and close WebSocket connection.

        Args:
            frame: The end frame triggering service shutdown.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and close WebSocket connection.

        Args:
            frame: The cancel frame triggering service cancellation.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Send audio data to the transcription session.

        Audio is streamed over the WebSocket. Transcription results arrive
        asynchronously via the receive task and are pushed as
        ``InterimTranscriptionFrame`` or ``TranscriptionFrame``.

        Args:
            audio: Raw audio bytes (16-bit mono PCM at the pipeline
                sample rate). Automatically resampled to 24 kHz.

        Yields:
            None — results are delivered via the WebSocket receive task.
        """
        await self._send_audio(audio)
        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames from the pipeline.

        Extends the base STT service to handle local VAD events when
        server-side VAD is disabled. On ``VADUserStoppedSpeakingFrame``,
        commits the audio buffer so the server begins transcription for
        the completed speech segment.

        Args:
            frame: The frame to process.
            direction: The direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        # Handle local VAD events when server-side VAD is disabled.
        if not self._server_vad_enabled:
            if isinstance(frame, VADUserStartedSpeakingFrame):
                await self.start_processing_metrics()
            elif isinstance(frame, VADUserStoppedSpeakingFrame):
                await self._commit_audio_buffer()

    # ------------------------------------------------------------------
    # WebSocket connection management
    # ------------------------------------------------------------------

    async def _connect(self):
        """Connect to the transcription endpoint and start receiving."""
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        """Disconnect and clean up background tasks."""
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task, timeout=1.0)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Establish the WebSocket connection to the transcription endpoint."""
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            self._session_ready = False
            url = f"{self._base_url}?intent=transcription"
            self._websocket = await websocket_connect(
                uri=url,
                additional_headers={
                    "Authorization": f"Bearer {self._api_key}",
                },
            )
            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(
                error_msg=f"Error connecting to OpenAI Realtime STT: {e}",
                exception=e,
            )
            self._websocket = None

    async def _disconnect_websocket(self):
        """Close the WebSocket connection."""
        try:
            self._session_ready = False
            if self._websocket:
                await self._websocket.close()
        except Exception as e:
            await self.push_error(
                error_msg=f"Error disconnecting: {e}",
                exception=e,
            )
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    async def _ws_send(self, message: dict):
        """Send a JSON message over the WebSocket.

        Args:
            message: The message dict to serialize and send.
        """
        try:
            if not self._disconnecting and self._websocket:
                await self._websocket.send(json.dumps(message))
        except Exception as e:
            if self._disconnecting or not self._websocket:
                return
            await self.push_error(
                error_msg=f"Error sending message: {e}",
                exception=e,
            )

    # ------------------------------------------------------------------
    # Client events
    # ------------------------------------------------------------------

    async def _send_session_update(self):
        """Send ``session.update`` to configure the transcription session."""
        transcription: dict = {"model": self.model_name}

        if self._language_code:
            transcription["language"] = self._language_code

        if self._prompt:
            transcription["prompt"] = self._prompt

        input_audio: dict = {
            "format": {
                "type": "audio/pcm",
                "rate": _OPENAI_SAMPLE_RATE,
            },
            "transcription": transcription,
        }

        # Turn detection
        if self._turn_detection is False:
            input_audio["turn_detection"] = None
        elif self._turn_detection is not None:
            input_audio["turn_detection"] = self._turn_detection

        # Noise reduction
        if self._noise_reduction:
            input_audio["noise_reduction"] = {
                "type": self._noise_reduction,
            }

        await self._ws_send(
            {
                "type": "session.update",
                "session": {
                    "type": "transcription",
                    "audio": {
                        "input": input_audio,
                    },
                },
            }
        )

    async def _send_audio(self, audio: bytes):
        """Send audio data via ``input_audio_buffer.append``.

        Resamples from the pipeline sample rate to 24 kHz if needed.

        Args:
            audio: Raw audio bytes at the pipeline sample rate.
        """
        audio = await self._resampler.resample(audio, self.sample_rate, _OPENAI_SAMPLE_RATE)
        if not audio:
            return
        payload = base64.b64encode(audio).decode("utf-8")
        await self._ws_send(
            {
                "type": "input_audio_buffer.append",
                "audio": payload,
            }
        )

    async def _commit_audio_buffer(self):
        """Commit the current audio buffer for transcription."""
        await self._ws_send({"type": "input_audio_buffer.commit"})

    async def _clear_audio_buffer(self):
        """Clear the current audio buffer."""
        await self._ws_send({"type": "input_audio_buffer.clear"})

    # ------------------------------------------------------------------
    # Server event handling
    # ------------------------------------------------------------------

    async def _receive_messages(self):
        """Receive and dispatch server events from the transcription session.

        Called by ``WebsocketService._receive_task_handler`` which wraps
        this method with automatic reconnection on connection errors.
        """
        async for message in self._websocket:
            try:
                evt = json.loads(message)
            except json.JSONDecodeError:
                logger.warning("Failed to parse WebSocket message")
                continue

            evt_type = evt.get("type", "")

            if evt_type == "session.created":
                await self._handle_session_created(evt)
            elif evt_type == "session.updated":
                await self._handle_session_updated(evt)
            elif evt_type == "conversation.item.input_audio_transcription.delta":
                await self._handle_transcription_delta(evt)
            elif evt_type == "conversation.item.input_audio_transcription.completed":
                await self._handle_transcription_completed(evt)
            elif evt_type == "conversation.item.input_audio_transcription.failed":
                await self._handle_transcription_failed(evt)
            elif evt_type == "input_audio_buffer.speech_started":
                await self._handle_speech_started(evt)
            elif evt_type == "input_audio_buffer.speech_stopped":
                await self._handle_speech_stopped(evt)
            elif evt_type == "input_audio_buffer.committed":
                logger.trace(f"Audio buffer committed: item_id={evt.get('item_id', '')}")
            elif evt_type == "error":
                await self._handle_error(evt)
            else:
                logger.trace(f"Unhandled event: {evt_type}")

    async def _handle_session_created(self, evt: dict):
        """Handle ``session.created``.

        Sent immediately after connecting. We respond by configuring the
        session with our desired settings.

        Args:
            evt: The session created event from the server.
        """
        logger.debug("Transcription session created, sending configuration")
        await self._send_session_update()

    async def _handle_session_updated(self, evt: dict):
        """Handle ``session.updated``.

        The session is now fully configured and ready to transcribe.

        Args:
            evt: The session updated event from the server.
        """
        logger.debug("Transcription session configured and ready")
        self._session_ready = True

    async def _handle_transcription_delta(self, evt: dict):
        """Handle incremental transcription text.

        For ``gpt-4o-transcribe`` and ``gpt-4o-mini-transcribe``, deltas
        contain streaming partial text. For ``whisper-1``, each delta
        contains the full turn transcript.

        Args:
            evt: The delta event from the server.
        """
        delta = evt.get("delta", "")
        if delta:
            await self.push_frame(
                InterimTranscriptionFrame(
                    delta,
                    self._user_id,
                    time_now_iso8601(),
                    result=evt,
                )
            )

    async def _handle_transcription_completed(self, evt: dict):
        """Handle a completed transcription for a speech segment.

        Pushes a ``TranscriptionFrame`` and records the result for
        tracing.

        Args:
            evt: The completed event containing the full transcript.
        """
        transcript = evt.get("transcript", "")
        if transcript:
            await self.push_frame(
                TranscriptionFrame(
                    transcript,
                    self._user_id,
                    time_now_iso8601(),
                    result=evt,
                )
            )
            await self._handle_transcription_trace(transcript, True)
            await self.stop_processing_metrics()

    @traced_stt
    async def _handle_transcription_trace(
        self,
        transcript: str,
        is_final: bool,
        language: Optional[Language] = None,
    ):
        """Record transcription result for tracing.

        Args:
            transcript: The transcribed text.
            is_final: Whether this is a final transcription result.
            language: Optional language of the transcription.
        """
        pass

    async def _handle_speech_started(self, evt: dict):
        """Handle server-side VAD speech start.

        Broadcasts ``UserStartedSpeakingFrame`` and optionally triggers
        interruption of current bot output.

        Args:
            evt: The ``input_audio_buffer.speech_started`` event.
        """
        logger.debug("Server VAD: speech started")
        await self.broadcast_frame(UserStartedSpeakingFrame)
        if self._should_interrupt:
            await self.push_interruption_task_frame_and_wait()
        await self.start_processing_metrics()

    async def _handle_speech_stopped(self, evt: dict):
        """Handle server-side VAD speech stop.

        Broadcasts ``UserStoppedSpeakingFrame``. The audio buffer is
        automatically committed by the server when VAD is enabled.

        Args:
            evt: The ``input_audio_buffer.speech_stopped`` event.
        """
        logger.debug("Server VAD: speech stopped")
        await self.broadcast_frame(UserStoppedSpeakingFrame)

    async def _handle_transcription_failed(self, evt: dict):
        """Handle a transcription failure for a speech segment.

        Logs the error but does not treat it as fatal — the session
        remains active for subsequent turns.

        Args:
            evt: The failed event containing error details.
        """
        error = evt.get("error", {})
        error_msg = error.get("message", "Transcription failed")
        await self.push_error(error_msg=f"OpenAI Realtime STT error: {error_msg}")

    async def _handle_error(self, evt: dict):
        """Handle a fatal error from the transcription session.

        Raises an exception so that ``WebsocketService`` can decide
        whether to attempt reconnection.

        Args:
            evt: The error event.
        """
        error = evt.get("error", {})
        error_msg = error.get("message", "Unknown error")
        error_code = error.get("code", "")
        msg = f"OpenAI Realtime STT error [{error_code}]: {error_msg}"
        await self.push_error(error_msg=msg)
        raise Exception(msg)
