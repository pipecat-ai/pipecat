#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Deepgram Flux TTS base class shared across transports (WebSocket, SageMaker, etc.)."""

from abc import abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlencode

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameProcessorSetup
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TextAggregationMode, TTSService
from pipecat.utils.tracing.service_decorators import traced_tts
from pipecat.utils.types import NOT_GIVEN, NotGiven


@dataclass
class DeepgramFluxTTSSettings(TTSSettings):
    """Settings for Deepgram Flux TTS services.

    The Flux voice is a single ``flux-{voice}-{language}`` identifier (e.g.
    ``flux-alexis-en``), carried by ``voice``. Deepgram's API passes it as
    its ``model`` query parameter, so ``model`` is kept in sync with
    ``voice`` and is not directly settable.

    Parameters:
        speed: Speech-rate multiplier, from 0.85 to 1.15 in steps of 0.05.
            ``None`` leaves Flux at its default rate. Applied to the open
            connection, so a speed change keeps the cross-turn acoustic state.
        expressivity: Expressive range on a calm-to-animated axis. ``None``
            leaves Flux at its default range. Flux fixes expressivity when the
            connection opens, so a change reconnects.
    """

    speed: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    expressivity: Literal[-2, -1, 0, 1, 2] | None | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )


class DeepgramFluxTTSBase(TTSService):
    """Base class for Deepgram Flux TTS services across transports.

    Contains all shared Flux protocol logic (message handling, turn lifecycle,
    metrics, settings). Concrete subclasses implement the transport layer by
    providing ``_transport_send_json``, ``_transport_is_active``, ``_connect``
    and ``_disconnect``.

    Flux keeps acoustic state across turns on a single connection, so prosody
    and pacing stay consistent throughout a conversation.

    Event handlers:

    - on_connected: Called when the connection is established.
    - on_disconnected: Called when the connection is closed.
    - on_connection_error: Called when a connection error occurs.
    """

    Settings = DeepgramFluxTTSSettings
    _settings: Settings

    # Audio is always requested as linear16 (raw PCM), the format Pipecat
    # pipelines use internally.
    SUPPORTED_SAMPLE_RATES = (8000, 16000, 24000, 32000, 44100, 48000)

    # Settings Flux can change on an open connection, via `Configure`. Every
    # other setting is a query parameter, fixed for the life of the connection.
    _CONFIGURE_FIELDS = frozenset({"speed"})

    def __init__(
        self,
        *,
        mip_opt_out: bool | None = None,
        tag: list[str] | None = None,
        text_aggregation_mode: TextAggregationMode = TextAggregationMode.TOKEN,
        settings: Settings,
        **kwargs,
    ):
        """Initialize the Deepgram Flux TTS base service.

        Args:
            mip_opt_out: Opt out of the Deepgram Model Improvement Program. See
                https://dpgr.am/deepgram-mip for pricing impacts before setting to True.
            tag: Tags to label requests for identification during usage reporting.
            text_aggregation_mode: How to aggregate incoming text before synthesis.
                Defaults to ``TextAggregationMode.TOKEN``, streaming LLM tokens
                straight to Flux for the lowest latency.
            settings: Fully resolved settings instance (built by concrete subclass).
            **kwargs: Additional arguments passed to the parent TTSService (e.g.
                ``sample_rate``).
        """
        # Deepgram passes the voice identifier as its `model` query parameter,
        # so keep `model` in sync with `voice` for metrics.
        settings.model = settings.voice

        super().__init__(
            pause_frame_processing=True,
            push_stop_frames=False,
            push_start_frame=True,
            text_aggregation_mode=text_aggregation_mode,
            # Flux never inserts or strips whitespace between Speak messages,
            # so consecutive sentences would otherwise glue together. Applies
            # in sentence mode only; when streaming tokens, the LLM's own
            # whitespace is used as-is.
            append_trailing_space=True,
            settings=settings,
            **kwargs,
        )

        self._mip_opt_out = mip_opt_out
        self._tag = tag or []

    def can_generate_metrics(self) -> bool:
        """Check if the service can generate metrics.

        Returns:
            True, as Deepgram Flux TTS supports metrics generation.
        """
        return True

    @property
    def supports_processing_metrics(self) -> bool:
        """Whether this service has a meaningful processing-time metric.

        False: ``run_tts`` sends the text and returns, and audio arrives later
        on the receive task, so there is no synthesis inside the measured
        window.
        """
        return False

    # ------------------------------------------------------------------
    # Abstract transport interface — implemented by each concrete subclass
    # ------------------------------------------------------------------

    @abstractmethod
    async def _transport_send_json(self, message: dict):
        """Serialize and send a JSON control message over the transport."""
        pass

    @abstractmethod
    def _transport_is_active(self) -> bool:
        """Return True if the transport connection is currently active."""
        pass

    @abstractmethod
    async def _connect(self):
        """Establish the transport connection."""
        pass

    @abstractmethod
    async def _disconnect(self):
        """Tear down the transport connection."""
        pass

    # ------------------------------------------------------------------
    # Service lifecycle
    # ------------------------------------------------------------------

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the service and connect.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Stop the service on a graceful end.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service immediately.

        Disconnecting here is the prompt teardown: the receive loop runs
        independently of the audio-context task, so it keeps reading from Flux
        until the connection is closed.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def cleanup(self):
        """Release the connection at teardown."""
        await super().cleanup()
        await self._disconnect()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _validate_sample_rate(self):
        """Warn if the pipeline sample rate is one Flux doesn't synthesize."""
        if self.sample_rate not in self.SUPPORTED_SAMPLE_RATES:
            logger.warning(
                f"{self}: sample rate {self.sample_rate} is not supported. "
                f"Supported rates: {self.SUPPORTED_SAMPLE_RATES}."
            )

    def _build_query_string(self) -> str:
        """Build query string from current settings and init-only connection config."""
        params = [
            f"model={self._settings.voice}",
            "encoding=linear16",
            f"sample_rate={self.sample_rate}",
        ]

        if self._settings.speed is not None:
            params.append(f"speed={self._settings.speed}")

        if self._settings.expressivity is not None:
            params.append(f"expressivity={self._settings.expressivity}")

        if self._mip_opt_out is not None:
            params.append(f"mip_opt_out={str(self._mip_opt_out).lower()}")

        # Add tag parameters (can have multiple)
        for tag_value in self._tag:
            params.append(urlencode({"tag": tag_value}))

        return "&".join(params)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta.

        A speed change is sent to Flux with a ``Configure`` message. Every other
        setting is a query parameter, so a change reconnects.

        Args:
            delta: A :class:`TTSSettings` (or ``Settings``) delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)

        # Deepgram uses voice as the model, so keep them in sync for metrics
        if "voice" in changed:
            self._settings.model = self._settings.voice
            self._sync_model_name_to_metrics()

        if changed.keys() - self._CONFIGURE_FIELDS:
            await self._disconnect()
            await self._connect()
        elif changed:
            await self._send_configure()

        return changed

    async def _send_configure(self):
        """Apply the current speed to the open connection with a Configure message."""
        if not self._transport_is_active():
            return

        # Flux's own default is 1.0, so clearing the setting restores that rate
        # explicitly — there is no way to ask Flux to forget a configured speed.
        speed = self._settings.speed if self._settings.speed is not None else 1.0

        try:
            await self._transport_send_json({"type": "Configure", "speed": speed})
        except Exception as e:
            logger.error(f"{self} error sending Configure message: {e}")

    # ------------------------------------------------------------------
    # Turn lifecycle
    # ------------------------------------------------------------------

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the turn on the Flux connection when the user barges in.

        Flux's ``Interrupt`` message ends the active turn without closing the
        connection, so the cross-turn acoustic state survives a barge-in.

        Args:
            context_id: The ID of the audio context that was interrupted.
        """
        if self._transport_is_active():
            try:
                await self._transport_send_json({"type": "Interrupt"})
            except Exception as e:
                logger.error(f"{self} error sending Interrupt message: {e}")

        await super().on_audio_context_interrupted(context_id)

    async def flush_audio(self, context_id: str | None = None):
        """Flush any pending audio synthesis by sending a Flush message.

        This ends the active turn: the server generates any remaining audio
        and reports the turn's ``SpeechMetadata``.
        """
        if self._transport_is_active():
            try:
                await self._transport_send_json({"type": "Flush"})
            except Exception as e:
                logger.error(f"{self} error sending Flush message: {e}")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using Deepgram Flux.

        Args:
            text: The text to synthesize into speech.
            context_id: The context ID for tracking audio frames.

        Yields:
            Frame: Audio frames containing the synthesized speech, plus start/stop frames.
        """
        try:
            if not self._transport_is_active():
                # Full disconnect/connect cycle: after a server-initiated close
                # (e.g. Flux's one-hour session cap) the transport's receive
                # task has completed but is still set, so a plain _connect()
                # would not restart the receive loop.
                await self._disconnect()
                await self._connect()

                if not self._transport_is_active():
                    yield ErrorFrame(error=f"{self} is not connected")
                    return

            await self._transport_send_json({"type": "Speak", "text": text})

            await self.start_tts_usage_metrics(text)

            # The audio frames will be handled by the receive task
            yield None
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    async def _handle_audio(self, audio: bytes):
        """Append a chunk of synthesized audio to the active audio context.

        Audio carries no speech_id, so it is attributed to the active audio
        context. This is safe because ``pause_frame_processing=True``
        serializes turns, and an interruption leaves no active context, so
        audio the server had already generated for the abandoned turn is
        discarded.
        """
        ctx_id = self.get_active_audio_context_id()
        frame = TTSAudioRawFrame(audio, self.sample_rate, 1, context_id=ctx_id)
        await self.append_to_audio_context(ctx_id, frame)

    async def _handle_message(self, msg: dict[str, Any]):
        """Handle a JSON control message from Flux."""
        msg_type = msg.get("type")

        if msg_type == "Connected":
            logger.debug(
                f"{self}: connected (request_id: {msg.get('request_id')}, "
                f"model: {msg.get('model_name')})"
            )
        elif msg_type == "SpeechStarted":
            logger.trace(f"Received SpeechStarted: {msg}")
        elif msg_type == "Flushed":
            # Not end-of-turn: Flux acknowledges the flush and may still send
            # audio afterwards. SpeechMetadata is the definitive end-of-turn
            # signal.
            logger.trace(f"Received Flushed: {msg}")
        elif msg_type == "SpeechMetadata":
            # Sent once per turn after all of its audio.
            logger.debug(
                f"{self}: speech complete (speech_id: {msg.get('speech_id')}, "
                f"duration: {msg.get('audio_duration_ms')}ms, "
                f"billable characters: {msg.get('billable_character_count')})"
            )
            ctx_id = self.get_active_audio_context_id()
            await self.append_to_audio_context(ctx_id, TTSStoppedFrame(context_id=ctx_id))
            await self.remove_audio_context(ctx_id)
        elif msg_type == "SpeechInterrupted":
            # Acknowledges an Interrupt. The reported total covers audio the
            # server generated, not audio the user heard, since the request
            # carries no playback offset.
            logger.trace(
                f"{self}: speech interrupted (speech_id: {msg.get('speech_id')}, "
                f"audio played: {msg.get('audio_played_ms')}ms)"
            )
        elif msg_type == "SessionMetadata":
            logger.debug(f"{self}: session totals: {msg}")
        elif msg_type == "ConfigureSuccess":
            logger.debug(f"{self}: configuration applied: {msg.get('applied')}")
        elif msg_type == "ConfigureFailure":
            # Non-fatal: synthesis continues with the previous configuration, so
            # application code decides what a rejected settings update means for
            # the conversation.
            error_msg = (
                f"{self} configuration rejected {msg.get('code')} "
                f"({msg.get('field')}={msg.get('value')}): "
                f"{msg.get('description', 'Unknown failure')}"
            )
            await self.push_error(error_msg=error_msg)
        elif msg_type == "Warning":
            code = msg.get("code")
            if code == "NO_ACTIVE_SPEECH":
                # Routine: an audio context outlives the turn that filled it, so
                # a barge-in late in playback sends an Interrupt the server has
                # nothing left to cancel.
                logger.trace(f"{self}: no active turn to interrupt")
            else:
                logger.warning(
                    f"{self} warning {code}: {msg.get('description', 'Unknown warning')}"
                )
        elif msg_type == "Error":
            error_msg = f"{self} error {msg.get('code')}: {msg.get('description', 'Unknown error')}"
            logger.error(error_msg)
            await self.push_error(error_msg=error_msg)
        else:
            logger.debug(f"Received unknown message type: {msg}")
