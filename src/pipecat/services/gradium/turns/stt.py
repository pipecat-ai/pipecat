#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gradium turn-based speech-to-text service.

Derives turn-taking from Gradium's server-side end-pointing signal, in the
spirit of the Deepgram Flux / Cartesia Ink-2 turn protocols but without the
eager end-of-turn machinery.

Gradium's ASR websocket streams "step" messages carrying end-pointing
predictions: for several horizons, the probability that speech stays inactive
over the next ``horizon_s`` seconds. This service watches the horizon closest
to ``eot_horizon_s`` (default 3s):

- The first step whose inactivity probability drops below ``eot_threshold``
  opens a turn: a ``UserStartedSpeakingFrame`` is broadcast (plus an
  interruption unless disabled).
- While a turn is open, the first step at or above the threshold ends it: the
  server is flushed, the turn's transcript finalizes into a
  ``TranscriptionFrame``, and a ``UserStoppedSpeakingFrame`` follows it.

Turn boundaries are decided entirely by the server signal: local VAD frames
are ignored, and ``service_metadata_frame()`` recommends
``ExternalUserTurnStrategies`` so the user aggregator defers to the frames
this service emits.
"""

import json
from dataclasses import dataclass, field

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    STTMetadataFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.gradium.stt import GradiumSTTService, GradiumSTTSettings
from pipecat.services.settings import NOT_GIVEN, _NotGiven, assert_given
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies

DEFAULT_EOT_HORIZON_S = 3.0
DEFAULT_EOT_THRESHOLD = 0.5
DEFAULT_POST_FLUSH_COOLDOWN_FRAMES = 8


@dataclass
class GradiumTurnsSTTSettings(GradiumSTTSettings):
    """Settings for GradiumTurnsSTTService.

    Parameters:
        eot_horizon_s: Which end-pointing horizon drives turn decisions; the
            step entry with the closest ``horizon_s`` is used. Default 3.0.
        eot_threshold: Inactivity probability on that horizon at or above
            which an open turn ends -- and below which a turn opens.
            Default 0.5.
        post_flush_cooldown_frames: Number of step messages to receive after a
            flush completes before another turn may end. A flush feeds the
            model ``delay_in_frames`` of silence, which can leave the
            end-pointer primed to fire again on the first real frames.
            Default 8.
    """

    eot_horizon_s: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    eot_threshold: float | _NotGiven = field(default_factory=lambda: NOT_GIVEN)
    post_flush_cooldown_frames: int | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class GradiumTurnsSTTService(GradiumSTTService):
    """Gradium speech-to-text service with server-signal turn detection.

    The conversation is driven by the server's end-pointing signal::

        step*(inactivity < threshold: turn opens) -> text*
            -> step(inactivity >= threshold: turn ends) -> flush
            -> TranscriptionFrame -> UserStoppedSpeakingFrame

    Each turn start broadcasts a :class:`UserStartedSpeakingFrame` (and an
    interruption unless ``should_interrupt=False``); each turn end pushes the
    final :class:`TranscriptionFrame` followed by a
    :class:`UserStoppedSpeakingFrame`. There is no eager end-of-turn: a turn
    ends exactly once.

    Event handlers available (in addition to the base
    ``on_connected`` / ``on_disconnected``):

    - on_turn_start(service): the end-pointing signal opened a turn
    - on_turn_end(service, transcript): the turn ended with its final text

    Example::

        stt = GradiumTurnsSTTService(
            api_key="...",
            settings=GradiumTurnsSTTService.Settings(
                language=Language.FR, eot_horizon_s=3.0, eot_threshold=0.5
            ),
        )
    """

    Settings = GradiumTurnsSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        should_interrupt: bool = True,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Gradium turn-based STT service.

        Args:
            should_interrupt: Whether to broadcast an interruption when the
                server signal opens a new turn. Defaults to True.
            settings: Runtime-updatable settings, including ``eot_horizon_s``
                and ``eot_threshold``.
            **kwargs: Arguments passed to :class:`GradiumSTTService`
                (``api_key``, ``api_endpoint_base_url``, ``encoding``,
                ``sample_rate``, ...).
        """
        turns_defaults = self.Settings(
            eot_horizon_s=DEFAULT_EOT_HORIZON_S,
            eot_threshold=DEFAULT_EOT_THRESHOLD,
            post_flush_cooldown_frames=DEFAULT_POST_FLUSH_COOLDOWN_FRAMES,
        )
        if settings is not None:
            turns_defaults.apply_update(settings)

        super().__init__(settings=turns_defaults, **kwargs)

        self._should_interrupt = should_interrupt
        self._in_turn = False
        self._turn_end_pending = False
        self._flushing = False
        self._flush_cooldown = 0

        self._register_event_handler("on_turn_start")
        self._register_event_handler("on_turn_end")

    @property
    def supports_ttfs(self) -> bool:
        """TTFS doesn't apply: the server signal defines turn boundaries."""
        return False

    def service_metadata_frame(self) -> STTMetadataFrame:
        """Recommend external turn strategies: turns are detected server-side.

        This service emits ``UserStarted/StoppedSpeakingFrame`` from the
        server's end-pointing signal, so the user aggregator defers to those
        rather than running local VAD/smart-turn. Applied unless the user
        passed their own ``user_turn_strategies``.
        """
        frame = super().service_metadata_frame()
        frame.user_turn_strategies = ExternalUserTurnStrategies()
        return frame

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames, ignoring local VAD events.

        Turn boundaries and flushes are driven by the server's end-pointing
        signal, so GradiumSTTService's metrics-on-VAD-start and
        flush-on-VAD-stop behavior is skipped.
        """
        await super(GradiumSTTService, self).process_frame(frame, direction)

    async def _disconnect(self):
        self._in_turn = False
        self._turn_end_pending = False
        self._flushing = False
        self._flush_cooldown = 0
        await super()._disconnect()

    async def _receive_messages(self):
        # Same dispatch as GradiumSTTService._receive_messages, plus "step"
        # handling (the parent ignores step messages entirely).
        async for message in self._get_websocket():
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                logger.warning(f"Received non-JSON message: {message}")
                continue

            type_ = msg.get("type", "")
            if type_ == "step":
                await self._handle_step(msg)
            elif type_ == "text":
                await self._handle_text(msg["text"])
            elif type_ == "flushed":
                self._flushing = False
                self._flush_cooldown = assert_given(self._settings.post_flush_cooldown_frames)
                self.confirm_finalize()
                await self._handle_flushed()
            elif type_ == "end_of_stream":
                logger.debug("Received end_of_stream message from server")
            elif type_ == "error":
                await self.push_error(error_msg=f"Error: {msg}")

    async def _handle_step(self, msg: dict):
        """Derive turn boundaries from the server's end-pointing signal.

        Each step carries, per horizon, the probability that speech stays
        inactive over the next ``horizon_s`` seconds; we watch the entry
        closest to ``eot_horizon_s``. Below ``eot_threshold`` speech is
        likely happening -- the first such step opens a turn. At or above
        the threshold while a turn is open, the turn ends.
        """
        vad = msg.get("vad") or []
        if not vad:
            return
        cooling_down = self._flush_cooldown > 0
        if cooling_down:
            self._flush_cooldown -= 1

        horizon = assert_given(self._settings.eot_horizon_s)
        threshold = assert_given(self._settings.eot_threshold)
        entry = min(vad, key=lambda e: abs(e.get("horizon_s", float("inf")) - horizon))
        inactivity = entry.get("inactivity_prob")
        if inactivity is None:
            return

        if not self._in_turn:
            # While a turn end is still finalizing, suppress a new start so
            # the previous turn's TranscriptionFrame/UserStoppedSpeakingFrame
            # can't arrive after the next turn's UserStartedSpeakingFrame.
            if inactivity < threshold and not self._turn_end_pending:
                await self._start_turn()
        elif inactivity >= threshold and not cooling_down:
            await self._end_turn()

    async def _start_turn(self):
        logger.debug("Gradium turns: start of turn")
        self._in_turn = True
        await self.broadcast_frame(UserStartedSpeakingFrame)
        if self._should_interrupt:
            await self.broadcast_interruption()
        await self._start_metrics()
        await self._call_event_handler("on_turn_start")

    async def _end_turn(self):
        logger.debug("Gradium turns: end of turn")
        self._in_turn = False
        self._turn_end_pending = True
        # The flush pushes the decoder past its lookahead so the turn's tail
        # tokens arrive; the "flushed" ack finalizes the transcript and
        # _finalize_accumulated_text completes the turn.
        await self._send_flush()

    async def _send_flush(self):
        """Flush and request finalization; no-op while a flush is in flight."""
        if self._flushing:
            return
        self._flushing = True
        self.request_finalize()
        await super()._send_flush()

    async def _finalize_accumulated_text(self):
        """Finalize the transcript, then close the pending turn.

        The parent pushes the final TranscriptionFrame (or nothing, if no
        text accumulated -- e.g. a turn opened by non-speech noise); the
        UserStoppedSpeakingFrame follows it either way.
        """
        text = " ".join(self._accumulated_text)
        await super()._finalize_accumulated_text()
        if self._turn_end_pending:
            self._turn_end_pending = False
            await self.broadcast_frame(UserStoppedSpeakingFrame)
            await self._call_event_handler("on_turn_end", text)
