#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn stop strategy that answers a turn analyzer's eager end of turn speculatively."""

import asyncio

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.types import ProcessFrameResult, UserTurnSpeculation
from pipecat.turns.user_stop.eager_match_policy import EagerMatchPolicy, NormalizedMatch
from pipecat.turns.user_stop.turn_analyzer_user_turn_stop_strategy import (
    TurnAnalyzerUserTurnStopStrategy,
)


class TurnAnalyzerEagerUserTurnStopStrategy(TurnAnalyzerUserTurnStopStrategy):
    """Answers a turn analyzer's eager end of turn while the turn is still open.

    A turn detection model trades false alarms for latency: a verdict committed
    early ends turns the user had only paused in, one committed late leaves the
    user waiting for a response. An analyzer that reaches an *eager* end of turn
    (:meth:`~pipecat.audio.turn.base_turn_analyzer.BaseTurnAnalyzer.pop_eager`)
    separates the two costs. This strategy starts generating a response on the
    eager verdict, leaving the turn open, and the committed verdict releases
    that response, so the gap between the two is spent generating rather than
    waiting. The analyzer can then commit conservatively, on strong evidence,
    without the user paying for the extra silence.

    The eager verdict can be wrong in three ways, and each withdraws the response:

    - the user resumes speaking, as reported by VAD
    - the committed transcript differs from the eager one, per ``match_policy``
    - the turn never commits within ``speculation_timeout``, or ends some other
      way (the stop watchdog, an interruption)

    Every one of those leaves as an
    :class:`~pipecat.frames.frames.EagerEndOfTurnCancelFrame` from here, so a
    response is never dropped somewhere the rest of the pipeline cannot see.

    Nothing the speculation produces reaches the user or the context. The
    inference runs against a provisional context carrying the latest transcript,
    final or interim, and its response is held by the LLM service's
    :class:`~pipecat.turns.speculation_gate.SpeculationGate` until the turn is
    confirmed. The turn ends as it does with
    :class:`~pipecat.turns.user_stop.TurnAnalyzerUserTurnStopStrategy`: the user
    message written to the context is always the committed transcript.

    A speculation needs a transcript to run on. An eager end of turn reached
    before any transcript for the turn has arrived is held until one does, and
    dropped if the user resumes speaking or the turn commits first.

    An analyzer that never reaches an eager end of turn (the default
    ``pop_eager``) gets exactly the base strategy's behavior.

    Example::

        user_turn_strategies=UserTurnStrategies(
            stop=[TurnAnalyzerEagerUserTurnStopStrategy(turn_analyzer=analyzer)]
        )
    """

    def __init__(
        self,
        *,
        match_policy: EagerMatchPolicy | None = None,
        speculation_timeout: float = 5.0,
        **kwargs,
    ):
        """Initialize the eager turn analyzer user turn stop strategy.

        Args:
            match_policy: Decides whether the committed transcript is close
                enough to the eager one to keep the speculative response.
                Defaults to :class:`~pipecat.turns.user_stop.NormalizedMatch`,
                which ignores the capitalization and punctuation STT services
                commonly add when they commit a transcript. Pass
                :class:`~pipecat.turns.user_stop.ExactMatch` to require the two
                to be identical.
            speculation_timeout: Seconds an eager end of turn may go unresolved
                before its response is withdrawn. The analyzer's own silence
                backstop normally commits or the user resumes long before this;
                it guards against an analyzer that raises an eager end of turn
                and then never commits.
            **kwargs: Additional keyword arguments forwarded to
                :class:`~pipecat.turns.user_stop.TurnAnalyzerUserTurnStopStrategy`,
                including ``turn_analyzer`` and ``wait_for_transcript``.
        """
        super().__init__(**kwargs)
        self._match_policy = match_policy or NormalizedMatch()
        self._speculation_timeout = speculation_timeout
        self._speculation_timeout_task: asyncio.Task | None = None
        self._speculation: UserTurnSpeculation | None = None
        # An eager end of turn collected from the analyzer and not yet answered,
        # because no transcript has arrived to run the speculation on.
        self._eager_pending = False
        self._interim_text = ""
        self._turn_open = False

    @property
    def match_policy(self) -> EagerMatchPolicy:
        """The policy deciding whether a speculative response still applies."""
        return self._match_policy

    @property
    def speculating(self) -> bool:
        """Whether a speculative response is in flight for the current turn."""
        return self._speculation is not None

    async def handle_user_turn_started(self):
        """Ready the strategy for the turn now starting."""
        await super().handle_user_turn_started()
        self._turn_open = True

    async def handle_user_turn_stopped(self):
        """Close the turn, withdrawing a speculation it left unresolved."""
        self._turn_open = False
        await super().handle_user_turn_stopped()

    async def cleanup(self):
        """Cleanup the strategy."""
        await self._take_speculation()
        await super().cleanup()

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Process an incoming frame, answering a held eager end of turn once there is text.

        Args:
            frame: The frame to be analyzed.

        Returns:
            Always CONTINUE, so subsequent stop strategies are evaluated.
        """
        if isinstance(frame, InterimTranscriptionFrame):
            self._interim_text = frame.text
        elif isinstance(frame, TranscriptionFrame):
            self._interim_text = ""

        result = await super().process_frame(frame)

        if isinstance(frame, InterimTranscriptionFrame | TranscriptionFrame):
            await self._maybe_speculate()

        return result

    async def trigger_user_turn_stopped(self, *, enable_user_speaking_frames: bool | None = None):
        """End the turn, keeping the speculative response only if it still applies.

        Args:
            enable_user_speaking_frames: Whether to emit
                :class:`~pipecat.frames.frames.UserStoppedSpeakingFrame` for this
                turn.
        """
        self._eager_pending = False
        speculation = await self._take_speculation()
        if not speculation:
            await super().trigger_user_turn_stopped(
                enable_user_speaking_frames=enable_user_speaking_frames
            )
            return

        if self._match_policy.matches(speculation.text, self._text):
            logger.debug(f"{self}: eager end of turn held, keeping the speculative response")
            # Inference already ran, on the eager transcript. Only finalize:
            # the UserStoppedSpeakingFrame it emits is what releases the
            # response, and the flag stops a second inference answering the
            # same turn.
            await self.trigger_user_turn_finalized(
                enable_user_speaking_frames=enable_user_speaking_frames,
                confirms_speculation=True,
            )
            return

        logger.debug(
            f"{self}: eager end of turn missed, discarding the speculative response "
            f"(eager: [{speculation.text}], committed: [{self._text}])"
        )
        await self.trigger_user_turn_speculation_cancelled()
        # Inference has to run again, on the committed transcript, so fire both
        # events rather than just finalizing.
        await super().trigger_user_turn_stopped(
            enable_user_speaking_frames=enable_user_speaking_frames
        )

    async def _reset(self):
        """Clear per-turn state. Runs at both turn boundaries."""
        speculation = await self._take_speculation()
        await super()._reset()
        self._eager_pending = False
        self._interim_text = ""
        if speculation:
            # The turn ended without resolving the speculation — the stop
            # watchdog, an interruption, session end. Nothing else will withdraw
            # it, so the response would be held until the gate times out.
            logger.debug(f"{self}: turn ended unresolved, discarding the speculative response")
            await self.trigger_user_turn_speculation_cancelled()

    async def _handle_input_audio(self, frame: InputAudioRawFrame):
        """Feed the analyzer, then collect an eager end of turn it may have reached."""
        await super()._handle_input_audio(frame)
        await self._poll_eager()

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        """Ask the analyzer for its verdict, then collect an eager end of turn it may have reached."""
        await super()._handle_vad_user_stopped_speaking(frame)
        await self._poll_eager()

    async def _handle_vad_user_started_speaking(self, frame: VADUserStartedSpeakingFrame):
        """Withdraw the eager end of turn: the user was only pausing."""
        await super()._handle_vad_user_started_speaking(frame)
        self._eager_pending = False
        if await self._take_speculation():
            logger.debug(f"{self}: user resumed speaking, discarding the speculative response")
            await self.trigger_user_turn_speculation_cancelled()

    async def _poll_eager(self):
        """Collect the analyzer's eager end of turn, if it reached one."""
        if self._turn_analyzer.pop_eager():
            self._eager_pending = True
            await self._maybe_speculate()

    async def _maybe_speculate(self):
        """Answer a collected eager end of turn, once there is a transcript to run it on."""
        if not self._eager_pending:
            return

        # An eager end of turn has nothing to add once the turn has ended or
        # committed, while a speculation is already in flight, or while the
        # user is speaking (a stale one, collected after the turn was closed on
        # the same frame, lands here too).
        if (
            not self._turn_open
            or self._turn_complete
            or self._vad_user_speaking
            or self._speculation
        ):
            self._eager_pending = False
            return

        text = self._text or self._interim_text
        if not text:
            # Held until a transcript arrives; VAD start or the commit drop it.
            return

        self._eager_pending = False
        await self._speculate(text)

    async def _speculate(self, text: str):
        """Answer an eager end of turn, leaving the turn open."""
        speculation = UserTurnSpeculation(text=text)
        self._speculation = speculation
        self._speculation_timeout_task = self.task_manager.create_task(
            self._speculation_timeout_handler(speculation),
            f"{self}::_speculation_timeout_handler",
        )
        logger.debug(f"{self}: speculating on eager end of turn: [{speculation.text}]")
        await self.trigger_user_turn_inference_triggered(speculation=speculation)

    async def _speculation_timeout_handler(self, speculation: UserTurnSpeculation):
        """Withdraw an eager end of turn the analyzer never resolved."""
        await asyncio.sleep(self._speculation_timeout)
        # Cleared here rather than by a cancellation, since this timer fired.
        self._speculation_timeout_task = None

        if self._speculation is not speculation:
            return

        logger.debug(
            f"{self}: eager end of turn unresolved after {self._speculation_timeout}s, "
            "discarding the speculative response"
        )
        self._speculation = None
        await self.trigger_user_turn_speculation_cancelled()

    async def _take_speculation(self) -> UserTurnSpeculation | None:
        """Take the speculation in flight, stopping the clock on it.

        The single exit, so a resolved speculation can never be withdrawn again
        by a timer still running for it.
        """
        speculation, self._speculation = self._speculation, None
        if self._speculation_timeout_task:
            task, self._speculation_timeout_task = self._speculation_timeout_task, None
            await self.cancel_task(task)
        return speculation
