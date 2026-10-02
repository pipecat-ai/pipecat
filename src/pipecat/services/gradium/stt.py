#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gradium's speech-to-text service implementation.

This module provides integration with Gradium's real-time speech-to-text
WebSocket API for streaming audio transcription, with optional turn detection
from the server's end-pointing signal.
"""

import asyncio
import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, cast

from loguru import logger
from pydantic import BaseModel
from websockets.protocol import State

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessorSetup
from pipecat.services.settings import STTSettings
from pipecat.services.stt_latency import GRADIUM_TTFS_P99
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language, resolve_language
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.deprecation import deprecated
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given

# Seconds to wait after a "flushed" message for trailing text tokens to arrive
# before finalizing the transcription.
TRANSCRIPT_AGGREGATION_DELAY = 0.1

# Turn detection defaults: the end-pointing horizon watched, the inactivity
# probability that opens and closes a turn, and how many step messages to let
# pass after a flush before the signal is trusted again.
DEFAULT_EOT_HORIZON_S = 3.0
DEFAULT_EOT_THRESHOLD = 0.5
DEFAULT_POST_FLUSH_COOLDOWN_FRAMES = 8

# Settings read on the client as step messages arrive; a change to any of
# them takes effect on the next step without a reconnect.
_TURN_DETECTION_FIELDS = frozenset({"eot_horizon_s", "eot_threshold", "post_flush_cooldown_frames"})


class _TurnPhase(Enum):
    """Where turn detection is in a turn's lifecycle.

    IDLE waits for the signal to read inactive, ARMED opens a turn on the next
    dip, OPEN is a turn in progress, and ENDING has proposed the stop and
    flushed the server, and ignores the signal until the flush is
    acknowledged.
    """

    IDLE = auto()
    ARMED = auto()
    OPEN = auto()
    ENDING = auto()


# Gradium's language code asking it to detect the language rather than being
# grounded to one. It is not a Language enum member because it names a mode
# rather than a language, so it is passed through as a plain string.
_GRADIUM_ANY_LANGUAGE = "any"


def _input_format_from_encoding(encoding: str, sample_rate: int) -> str:
    """Build Gradium input_format from encoding type and sample rate.

    For PCM encoding, appends the sample rate (e.g., "pcm_16000").
    For other encodings (wav, opus), returns the encoding as-is.

    Args:
        encoding: Base encoding type ("pcm", "wav", or "opus").
        sample_rate: Audio sample rate in Hz.

    Returns:
        The full input_format string for the Gradium API.
    """
    if encoding == "pcm":
        match sample_rate:
            case 8000:
                return "pcm_8000"
            case 16000:
                return "pcm_16000"
            case 24000:
                return "pcm_24000"
        logger.warning(
            f"GradiumSTTService: unsupported sample rate {sample_rate} for PCM encoding, using pcm_16000"
        )
        return "pcm_16000"
    return encoding


def language_to_gradium_language(language: Language | str) -> str:
    """Convert a Language enum to Gradium's language code format.

    Args:
        language: The Language enum value to convert, or ``"any"``.

    Returns:
        The corresponding Gradium language code. If ``language`` is not in
        the verified mapping, falls back to the base language code (e.g.,
        ``en`` from ``en-US``) and logs a warning (via
        ``resolve_language(..., use_base_code=True)``).
    """
    if language == _GRADIUM_ANY_LANGUAGE:
        return _GRADIUM_ANY_LANGUAGE

    LANGUAGE_MAP = {
        Language.DE: "de",
        Language.EN: "en",
        Language.ES: "es",
        Language.FR: "fr",
        Language.PT: "pt",
    }

    # A raw string that is not a Language falls through to the base-code
    # branch of resolve_language, which handles it as-is.
    return resolve_language(cast("Language", language), LANGUAGE_MAP, use_base_code=True)


@dataclass
class GradiumSTTSettings(STTSettings):
    """Settings for GradiumSTTService.

    The ``eot_*`` and ``post_flush_cooldown_frames`` fields tune turn
    detection. They are ``None`` unless the service is constructed with
    ``enable_turn_detection=True``, which gives them their defaults.

    Parameters:
        delay_in_frames: Delay in audio frames (80ms each) before text is
            generated. Higher delays allow more context but increase latency.
            Allowed values: 7, 8, 10, 12, 14, 16, 20, 24, 36, 48.
            Default is 12 (960ms). Lower values like 7-8 give faster response.
        eot_horizon_s: Which end-pointing horizon drives turn decisions; the
            step entry with the closest ``horizon_s`` is used. Default 3.0.
        eot_threshold: Inactivity probability on that horizon at or above
            which an open turn ends, and below which a turn opens once the
            signal has read inactive. Default 0.5.
        post_flush_cooldown_frames: Number of step messages to ignore after a
            flush completes before the next turn may start or end. A flush
            feeds the model ``delay_in_frames`` of silence, and the
            end-pointing signal is unreliable on the frames that follow: it
            can dip as if speech resumed, then fire again on the first real
            frames. Default 8.
    """

    delay_in_frames: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    eot_horizon_s: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    eot_threshold: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    post_flush_cooldown_frames: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class GradiumSTTService(WebsocketSTTService):
    """Gradium real-time speech-to-text service.

    Provides real-time speech transcription using Gradium's WebSocket API.
    Supports both interim and final transcriptions with configurable parameters
    for audio processing and connection management.

    Transcribes English by default. Set ``settings.language`` to one of the
    other supported languages (German, Spanish, French, Portuguese), or to
    ``"any"`` to have Gradium detect the language.

    By default the pipeline's VAD closes each utterance: a
    ``VADUserStoppedSpeakingFrame`` flushes the server and the accumulated
    text finalizes into a :class:`TranscriptionFrame`.

    With ``enable_turn_detection=True`` the server's end-pointing signal
    drives turns instead. Every "step" message carries, per horizon, the
    probability that speech stays inactive over the next ``horizon_s``
    seconds, and the service watches the horizon closest to
    ``eot_horizon_s``::

        step(inactivity >= threshold) -> step(inactivity < threshold: turn opens)
            -> text* -> step(inactivity >= threshold: turn ends)
            -> ProposedUserStoppedSpeakingFrame -> flush -> TranscriptionFrame

    A turn opens on the signal falling below the threshold, not on it being
    there: the estimate starts low when the model has heard nothing yet, and
    again after each flush resets it, then climbs as silence accumulates.

    A turn start broadcasts a :class:`ProposedUserStartedSpeakingFrame`; a
    turn end broadcasts a :class:`ProposedUserStoppedSpeakingFrame` and
    flushes the server, and the final :class:`TranscriptionFrame` follows
    once the flush is acknowledged. Local VAD frames are ignored, and
    ``service_metadata_frame()`` recommends
    :class:`~pipecat.turns.user_turn_strategies.ExternalUserTurnStrategies`,
    which resolve the proposals into the user turn frames, own the
    interruption, and hold the turn open until that transcript arrives.

    Event handlers available (in addition to ``on_connected`` /
    ``on_disconnected``), fired only with turn detection on:

    - on_turn_start(service): the end-pointing signal opened a turn
    - on_turn_end(service): the end-pointing signal closed the turn

    Example::

        stt = GradiumSTTService(
            api_key="...",
            enable_turn_detection=True,
            settings=GradiumSTTService.Settings(eot_horizon_s=3.0, eot_threshold=0.5),
        )
    """

    Settings = GradiumSTTSettings
    _settings: Settings

    @deprecated(
        "`GradiumSTTService.InputParams` is deprecated since 0.0.105 and will be removed in "
        "2.0.0. Use `GradiumSTTService.Settings` instead."
    )
    class InputParams(BaseModel):
        """Configuration parameters for Gradium STT API.

        .. deprecated:: 0.0.105
            Use ``settings=GradiumSTTService.Settings(...)`` instead.
            Will be removed in 2.0.0.

        Parameters:
            language: Expected language of the audio (e.g., "en", "es", "fr").
                This helps ground the model to a specific language and improve
                transcription quality. Defaults to ``Language.EN``; ``"any"``
                asks Gradium to detect the language.
            delay_in_frames: Delay in audio frames (80ms each) before text is
                generated. Higher delays allow more context but increase latency.
                Allowed values: 7, 8, 10, 12, 14, 16, 20, 24, 36, 48.
                Default is 10 (800ms). Lower values like 7-8 give faster response.
        """

        language: Language | None = None
        delay_in_frames: int | None = None

    def __init__(
        self,
        *,
        api_key: str,
        api_endpoint_base_url: str = "wss://api.gradium.ai/api/speech/asr",
        encoding: str = "pcm",
        sample_rate: int | None = None,
        params: InputParams | None = None,
        json_config: str | None = None,
        enable_turn_detection: bool = False,
        settings: Settings | None = None,
        ttfs_p99_latency: float | None = GRADIUM_TTFS_P99,
        **kwargs,
    ):
        """Initialize the Gradium STT service.

        Args:
            api_key: Gradium API key for authentication.
            api_endpoint_base_url: WebSocket endpoint URL.
            encoding: Base audio encoding type. One of "pcm", "wav", or "opus".
                For PCM, the sample rate is appended automatically from the
                pipeline's audio_in_sample_rate (e.g., "pcm" becomes "pcm_16000").
                Defaults to "pcm".
            sample_rate: Audio sample rate in Hz. If None, uses the pipeline
                sample rate.
            params: Configuration parameters for language and delay settings.

                .. deprecated:: 0.0.105
                    Use ``settings=GradiumSTTService.Settings(...)`` instead.
                    Will be removed in 2.0.0.

            json_config: Optional JSON configuration string for additional model settings.

                .. deprecated:: 0.0.101
                    Use `params` instead for type-safe configuration.
                    Will be removed in 2.0.0.

            enable_turn_detection: Whether the server's end-pointing signal
                decides when user turns start and end, instead of the
                pipeline's VAD. Off by default.
            settings: Runtime-updatable settings. When provided alongside deprecated
                parameters, ``settings`` values take precedence.
            ttfs_p99_latency: P99 latency from speech end to final transcript in seconds.
                Override for your deployment. See https://github.com/pipecat-ai/stt-benchmark
            **kwargs: Additional arguments passed to parent STTService class.
        """
        if json_config is not None:
            import warnings

            warnings.warn(
                "Parameter 'json_config' is deprecated and will be removed in 2.0.0, use 'params' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # 1. Initialize default_settings with hardcoded defaults
        default_settings = self.Settings(
            model="default",
            language=Language.EN,
            delay_in_frames=12,
            eot_horizon_s=DEFAULT_EOT_HORIZON_S if enable_turn_detection else None,
            eot_threshold=DEFAULT_EOT_THRESHOLD if enable_turn_detection else None,
            post_flush_cooldown_frames=(
                DEFAULT_POST_FLUSH_COOLDOWN_FRAMES if enable_turn_detection else None
            ),
        )

        # 2. (No step 2, as there are no deprecated direct args)

        # 3. Apply params overrides — only if settings not provided
        if params is not None:
            self._warn_init_param_moved_to_settings("params")
            if not settings:
                if params.language is not None:
                    default_settings.language = params.language
                if params.delay_in_frames is not None:
                    default_settings.delay_in_frames = params.delay_in_frames

        # 4. Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            ttfs_p99_latency=ttfs_p99_latency,
            settings=default_settings,
            **kwargs,
        )

        self._api_key = api_key
        self._api_endpoint_base_url = api_endpoint_base_url
        self._encoding = encoding
        self._enable_turn_detection = enable_turn_detection
        self._websocket = None
        self._json_config = json_config

        self._receive_task = None

        self._input_format = ""

        self._audio_buffer = bytearray()
        self._chunk_size_ms = 80
        self._chunk_size_bytes = 0

        # Accumulates text fragments within a turn. Each "text" message
        # appends to this list. On "flushed" a short aggregation delay
        # allows trailing tokens to arrive before the full text is joined
        # and pushed as a TranscriptionFrame.
        self._accumulated_text: list[str] = []
        self._flush_counter = 0
        self._transcript_aggregation_task: asyncio.Task | None = None

        # Turn detection state: the phase, and how many steps remain in the
        # post-flush cooldown, which outlives the ENDING phase.
        self._turn_phase = _TurnPhase.IDLE
        self._flush_cooldown = 0

        self._register_event_handler("on_turn_start")
        self._register_event_handler("on_turn_end")

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            True if metrics generation is supported.
        """
        return True

    @property
    def supports_ttfs(self) -> bool:
        """TTFS applies only while the pipeline's VAD ends utterances.

        With turn detection on, the server defines the turn boundary, so there
        is no separate speech-end to final-transcript interval to measure.
        """
        return not self._enable_turn_detection

    def service_metadata_frame(self) -> STTMetadataFrame:
        """Recommend external turn strategies when turns are detected server-side.

        With turn detection on, the service proposes turn boundaries
        (``ProposedUserStarted/StoppedSpeakingFrame``), so the user aggregator
        resolves those rather than running local VAD/smart-turn. Otherwise the
        defaults are left in place. Applied unless the user passed their own
        ``user_turn_strategies``.
        """
        frame = super().service_metadata_frame()
        if self._enable_turn_detection:
            frame.user_turn_strategies = ExternalUserTurnStrategies()
        return frame

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Apply a settings delta, sync params, and reconnect.

        The model, language and decoder delay are bound to the connection's
        setup message, so a change to any of them is applied by reconnecting.
        The turn detection fields are read as step messages arrive and need
        no reconnect.

        Args:
            delta: A :class:`STTSettings` (or ``GradiumSTTService.Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)
        if not (changed.keys() - _TURN_DETECTION_FIELDS):
            return changed

        if self._websocket:
            await self._disconnect()
            await self._connect()
        return changed

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._input_format = _input_format_from_encoding(self._encoding, self.sample_rate)
        self._chunk_size_bytes = int(self._chunk_size_ms * self.sample_rate * 2 / 1000)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the speech-to-text service.

        Args:
            frame: End frame to stop processing.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the speech-to-text service.

        Args:
            frame: Cancel frame to abort processing.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and handle speech events.

        A VAD stop flushes the server, unless turn detection is on: then the
        server's end-pointing signal decides when a turn ends.

        Args:
            frame: The frame to process.
            direction: Direction of frame flow in the pipeline.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStoppedSpeakingFrame) and not self._enable_turn_detection:
            await self._send_flush()

    async def _send_flush(self):
        """Send a flush request to process any buffered audio immediately.

        Sends a flush message to tell the server to process buffered audio.
        The server responds with text fragments followed by a "flushed"
        acknowledgment, which triggers finalization.

        Returns:
            Whether the flush was sent, so a caller waiting on the "flushed"
            acknowledgment knows one is coming.
        """
        if not self._websocket or self._websocket.state is not State.OPEN:
            return False

        self._flush_counter += 1
        flush_id = str(self._flush_counter)
        msg = {"type": "flush", "flush_id": flush_id}
        try:
            await self._websocket.send(json.dumps(msg))
        except Exception as e:
            logger.warning(f"Failed to send flush: {e}")
            return False
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Process audio data for speech-to-text conversion.

        Args:
            audio: Raw audio bytes to process.

        Yields:
            None (processing handled via WebSocket messages).
        """
        self._audio_buffer.extend(audio)

        while len(self._audio_buffer) >= self._chunk_size_bytes:
            chunk = bytes(self._audio_buffer[: self._chunk_size_bytes])
            self._audio_buffer = self._audio_buffer[self._chunk_size_bytes :]
            chunk = base64.b64encode(chunk).decode("utf-8")
            msg = {"type": "audio", "audio": chunk}
            if self._websocket and self._websocket.state is State.OPEN:
                try:
                    await self._websocket.send(json.dumps(msg))
                except Exception as e:
                    logger.warning(f"{self}: send failed: {e}")
                    break

        yield None

    @traced_stt
    async def _trace_transcription(self, transcript: str, is_final: bool, language: Language):
        """Record transcription event for tracing."""
        pass

    async def _connect(self):
        await super()._connect()

        await self._connect_websocket()

        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return

            logger.debug("Connecting to Gradium STT")

            ws_url = self._api_endpoint_base_url
            headers = {
                "x-api-key": self._api_key,
                "x-api-source": "pipecat",
            }
            websocket = await self._websocket_connect(
                ws_url,
                additional_headers=headers,
            )
            self._websocket = websocket
            await self._call_event_handler("on_connected")
            setup_msg = {
                "type": "setup",
                "model_name": self._settings.model,
                "input_format": self._input_format,
            }
            # Build json_config: start with deprecated json_config, then override with params
            json_config = {}
            if self._json_config:
                json_config = json.loads(self._json_config)
            # Technically `_settings.language` could be a raw string, but
            # Language is a StrEnum so downstream handles either.
            language = cast("Language | None", assert_given(self._settings.language))
            if language is not None:
                gradium_language = language_to_gradium_language(language)
                if gradium_language:
                    json_config["language"] = gradium_language
            if self._settings.delay_in_frames:
                json_config["delay_in_frames"] = self._settings.delay_in_frames
            if json_config:
                setup_msg["json_config"] = json_config
            await websocket.send(json.dumps(setup_msg))
            ready_msg = await websocket.recv()
            ready_msg = json.loads(ready_msg)
            if ready_msg["type"] == "error":
                raise Exception(f"received error {ready_msg['message']}")
            if ready_msg["type"] != "ready":
                raise Exception(f"unexpected first message type {ready_msg['type']}")

            logger.debug("Connected to Gradium STT")

        except Exception as e:
            self._websocket = None
            await self.push_error(error_msg=f"Unable to connect to Gradium: {e}", exception=e)

    async def _disconnect(self):
        await super()._disconnect()

        if self._transcript_aggregation_task:
            await self.cancel_task(self._transcript_aggregation_task)
            self._transcript_aggregation_task = None

        self._accumulated_text.clear()
        self._flush_counter = 0
        self._turn_phase = _TurnPhase.IDLE
        self._flush_cooldown = 0

        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        await self._disconnect_websocket()

    async def _disconnect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                logger.debug("Disconnecting from Gradium STT")
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Unknown error occurred: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        async for message in self._get_websocket():
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"Received non-JSON message: {message}")
                continue

            type_ = msg.get("type", "")
            if type_ == "step":
                if self._enable_turn_detection:
                    await self._handle_step(msg)
            elif type_ == "text":
                await self._handle_text(msg["text"])
            elif type_ == "flushed":
                if self._enable_turn_detection:
                    self._turn_phase = _TurnPhase.IDLE
                    self._flush_cooldown = (
                        assert_given(self._settings.post_flush_cooldown_frames) or 0
                    )
                    self.confirm_finalize()
                await self._handle_flushed()
            elif type_ == "end_of_stream":
                logger.debug("Received end_of_stream message from server")
            elif type_ == "error":
                await self.push_error(error_msg=f"Error: {msg}")

    async def _handle_text(self, text: str):
        """Handle streaming transcription fragment.

        Accumulates text and pushes an InterimTranscriptionFrame with the
        full accumulated text so far.
        """
        self._accumulated_text.append(text)
        accumulated = " ".join(self._accumulated_text)
        # Technically `_settings.language` could be a raw string, but Language
        # is a StrEnum so downstream handles either.
        await self.push_frame(
            InterimTranscriptionFrame(
                text=accumulated,
                user_id=self._user_id,
                timestamp=time_now_iso8601(),
                language=cast("Language | None", assert_given(self._settings.language)),
            )
        )

    async def _handle_flushed(self):
        """Handle flush completion by starting a transcript aggregation timer.

        The "flushed" message confirms that buffered audio has been processed,
        but text tokens may still arrive after this point. A short timer allows
        trailing tokens to accumulate before finalizing the transcription.
        """
        if self._transcript_aggregation_task:
            await self.cancel_task(self._transcript_aggregation_task)
        self._transcript_aggregation_task = self.create_task(
            self._transcript_aggregation_handler(), "transcript_aggregation"
        )

    async def _transcript_aggregation_handler(self):
        """Wait for trailing tokens then finalize the accumulated transcription."""
        await asyncio.sleep(TRANSCRIPT_AGGREGATION_DELAY)
        await self._finalize_accumulated_text()

    async def _finalize_accumulated_text(self):
        """Join accumulated text, push TranscriptionFrame, and clear state."""
        if not self._accumulated_text:
            return
        self._transcript_aggregation_task = None

        text = " ".join(self._accumulated_text)
        self._accumulated_text.clear()
        logger.debug(f"Final transcription: [{text}]")
        # Technically `_settings.language` could be a raw string, but Language
        # is a StrEnum so downstream handles either.
        language = cast("Language | None", assert_given(self._settings.language))
        # Report usage before the transcription frame so tracing can attach
        # it to the STT span the frame closes.
        await self.emit_stt_usage_metrics()
        await self.push_frame(
            TranscriptionFrame(
                text,
                self._user_id,
                time_now_iso8601(),
                language,
            )
        )
        await self._trace_transcription(text, is_final=True, language=language)

    async def _handle_step(self, msg: dict):
        """Derive turn boundaries from the server's end-pointing signal.

        Each step carries, per horizon, the probability that speech stays
        inactive over the next ``horizon_s`` seconds; the entry closest to
        ``eot_horizon_s`` is watched. A step at or above ``eot_threshold``
        arms the detector; the next step below it opens a turn. At or above
        the threshold while a turn is open, the turn ends.
        """
        vad = msg.get("vad") or []
        if not vad:
            return
        if self._flush_cooldown > 0:
            self._flush_cooldown -= 1
            return

        horizon = assert_given(self._settings.eot_horizon_s)
        threshold = assert_given(self._settings.eot_threshold)
        if horizon is None or threshold is None:
            return
        entry = min(vad, key=lambda e: abs(e.get("horizon_s", float("inf")) - horizon))
        inactivity = entry.get("inactivity_prob")
        if inactivity is None:
            return
        logger.trace(f"Gradium turn detection: inactivity {inactivity:.2f} over {horizon}s")

        inactive = inactivity >= threshold
        match self._turn_phase:
            case _TurnPhase.IDLE if inactive:
                self._turn_phase = _TurnPhase.ARMED
            case _TurnPhase.ARMED if not inactive:
                await self._start_turn()
            case _TurnPhase.OPEN if inactive:
                await self._end_turn()
            # ENDING ignores the signal: the previous turn's TranscriptionFrame
            # and stop proposal must not arrive after the next turn's start
            # proposal.

    async def _start_turn(self):
        logger.debug("Gradium turn detection: start of turn")
        self._turn_phase = _TurnPhase.OPEN
        await self.broadcast_frame(ProposedUserStartedSpeakingFrame)
        await self._call_event_handler("on_turn_start")

    async def _end_turn(self):
        logger.debug("Gradium turn detection: end of turn")
        self._turn_phase = _TurnPhase.ENDING
        await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)
        await self._call_event_handler("on_turn_end")
        # The flush pushes the decoder past its lookahead so the turn's tail
        # tokens arrive and the transcript finalizes on the "flushed" ack.
        # Without a flush no ack is coming, so the transcript finalizes on
        # what has arrived.
        self.request_finalize()
        if await self._send_flush():
            return
        self._turn_phase = _TurnPhase.IDLE
        await self._finalize_accumulated_text()
